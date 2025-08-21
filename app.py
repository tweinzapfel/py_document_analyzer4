import os
import io
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader
from docx import Document as DocxDocument
from openai import OpenAI
import pandas as pd

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="RFP Analyzer (MVP)", page_icon="📄", layout="wide")

# Configure OpenAI client with flexible secrets/env lookup + sidebar diagnostics
def _get_openai_key() -> str:
    candidates = [
        st.secrets.get("api_key"),
        st.secrets.get("OPENAI_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OPENAI_KEY"),
        os.getenv("OPENAI_APIKEY"),
    ]
    for k in candidates:
        if k and isinstance(k, str) and k.strip():
            return k.strip()
    return ""

OPENAI_API_KEY = _get_openai_key()
try:
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add it to .streamlit/secrets.toml as 'api_key' or 'OPENAI_API_KEY', or set the OPENAI_API_KEY env var.")

# -----------------------------
# Sidebar diagnostics
# -----------------------------
with st.sidebar:
    st.header("Diagnostics")
    status = "✅ Ready" if (OPENAI_API_KEY and client) else "❌ Not configured"
    st.write(f"OpenAI client: {status}")

# -----------------------------
# Helpers: extraction
# -----------------------------

def extract_pdf_text_and_pages(file: io.BytesIO, doc_name: str) -> List[Dict[str, Any]]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append({"doc": doc_name, "page": i, "text": txt})
    return pages


def extract_docx_text_and_pages(file: io.BytesIO, doc_name: str) -> List[Dict[str, Any]]:
    d = DocxDocument(file)
    text = []
    for p in d.paragraphs:
        if p.text and p.text.strip():
            text.append(p.text.strip())
    pages = []
    buf, count, idx = [], 0, 1
    for para in text:
        buf.append(para)
        count += len(para) + 1
        if count > 8000:
            pages.append({"doc": doc_name, "page": idx, "text": "\n".join(buf)})
            buf, count = [], 0
            idx += 1
    if buf:
        pages.append({"doc": doc_name, "page": idx, "text": "\n".join(buf)})
    return pages


def load_files(files: List[io.BytesIO]) -> List[Dict[str, Any]]:
    all_pages = []
    for f in files:
        name = getattr(f, "name", "document")
        suffix = name.lower().split(".")[-1]
        try:
            if suffix == "pdf":
                pages = extract_pdf_text_and_pages(f, name)
            elif suffix in ("doc", "docx"):
                pages = extract_docx_text_and_pages(f, name)
            else:
                st.warning(f"Unsupported file type for {name}; skipping.")
                continue
            all_pages.extend(pages)
        except Exception as e:
            st.error(f"Failed to read {name}: {e}")
    return all_pages

# -----------------------------
# Chunking
# -----------------------------

def chunk_pages(pages: List[Dict[str, Any]], target_chars: int = 4000, overlap: int = 400) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        text = p["text"]
        if len(text) <= target_chars:
            chunks.append({**p, "chunk": 1, "text": text})
            continue
        paras = [x for x in text.split("\n") if x.strip()]
        buf, size, idx = [], 0, 1
        for para in paras:
            if size + len(para) + 1 > target_chars and buf:
                chunks.append({**p, "chunk": idx, "text": "\n".join(buf)})
                tail = "\n".join(buf)[-overlap:]
                buf = [tail, para]
                size = len(tail) + len(para) + 1
                idx += 1
            else:
                buf.append(para)
                size += len(para) + 1
        if buf:
            chunks.append({**p, "chunk": idx, "text": "\n".join(buf)})
    return chunks

# -----------------------------
# LLM Prompts & Calls
# -----------------------------

COMPLIANCE_TASK = """
You are a contracts analyst. From the provided RFP excerpt, extract every *mandatory* requirement and submission instruction explicitly stated.
Return strict JSON with two arrays: "requirements" and "instructions".
... (shortened for brevity) ...
"""

RISK_TASK = """..."""
QA_TASK = """..."""


def llm_json(task: str, content: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not client:
        raise RuntimeError("OpenAI client is not configured.")
    fallback_models = [model, "gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"]
    last_err = None
    for m in fallback_models:
        if not m:
            continue
        try:
            prompt = f"Task Instructions:\n{task}\n\nRFP Content:\n" + content
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a precise contracts analyst that returns STRICT JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            try:
                return json.loads(text)
            except Exception:
                import re
                mjson = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
                if mjson:
                    return json.loads(mjson.group(1))
                raise ValueError(f"Non-JSON output from {m}.")
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("All model attempts failed.")

# -----------------------------
# Aggregation across chunks
# -----------------------------

def aggregate_results(chunks: List[Dict[str, Any]], model: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req_rows, inst_rows, risk_rows = [], [], []
    for ch in chunks:
        content = f"Document: {ch['doc']} | Page: {ch['page']}\n\n{ch['text']}"
        try:
            data = llm_json(COMPLIANCE_TASK, content, model=model)
            for r in data.get("requirements", []):
                req_rows.append(r)
            for i in data.get("instructions", []):
                inst_rows.append(i)
        except Exception as e:
            st.info(f"Compliance extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
        try:
            risks = llm_json(RISK_TASK, content, model=model).get("risks", [])
            for rk in risks:
                risk_rows.append(rk)
        except Exception as e:
            st.info(f"Risk extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
    req_df = pd.DataFrame(req_rows).drop_duplicates(subset=["requirement_text","citation"]).reset_index(drop=True)
    inst_df = pd.DataFrame(inst_rows).drop_duplicates(subset=["topic","value","citation"]).reset_index(drop=True)
    risk_df = pd.DataFrame(risk_rows).drop_duplicates(subset=["rationale","citation"]).reset_index(drop=True)
    return req_df, inst_df, risk_df

# -----------------------------
# UI
# -----------------------------

st.title("📄 RFP Analyzer – MVP (Streamlit + OpenAI)")

uploaded = st.file_uploader("Upload one or more RFP files (PDF/DOCX)", type=["pdf","doc","docx"], accept_multiple_files=True)
st.session_state["uploaded_files"] = uploaded or []

model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"], index=0)
st.session_state["model"] = model

target_chars = st.slider("Chunk target size (chars)", 1500, 8000, 4000, 500)

if st.button("🔎 Analyze Documents", type="primary"):
    if not uploaded:
        st.warning("Please upload at least one PDF or DOCX file.")
        st.stop()
    if not client:
        st.error("OpenAI client not configured. Add your API key and try again.")
        st.stop()
    pages = load_files(uploaded)
    chunks = chunk_pages(pages, target_chars=target_chars)
    req_df, inst_df, risk_df = aggregate_results(chunks, model)
    st.subheader("📋 Compliance Matrix")
    st.dataframe(req_df)
    st.subheader("🗂️ Submission Instructions")
    st.dataframe(inst_df)
    st.subheader("⚠️ Risk Register")
    st.dataframe(risk_df)

st.markdown("---")
st.caption("MVP demo with robust model fallback.")
