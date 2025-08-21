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

# =============================
# OpenAI client + diagnostics
# =============================

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

with st.sidebar:
    st.header("Diagnostics")
    status = "✅ Ready" if (OPENAI_API_KEY and client) else "❌ Not configured"
    st.write(f"OpenAI client: {status}")

# -----------------------------
# Helpers: file extraction
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
    # chunk into pseudo-pages for citation stability
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
# LLM Prompts & Calls (full, schema-first)
# -----------------------------

COMPLIANCE_TASK = (
    "You are a contracts analyst. From the provided RFP excerpt, extract every mandatory requirement and submission "
    "instruction explicitly stated. Return STRICT JSON with two arrays: 'requirements' and 'instructions'.\n\n"
    "For each requirement object include: id (e.g., req_001), requirement_text (quote when short), category one of "
    "['technical','administrative','pricing','security','legal'], priority one of ['P1','P2','P3'], shall_must (true/false), "
    "evidence_needed (array of short strings), citation {doc: string, page: number}.\n\n"
    "For each instruction object include: id (e.g., inst_001), topic one of ['due_date','portal','format','naming','page_limit','font','margins','volumes','attachments'], "
    "value (raw text), normalized (object with fields like due_datetime ISO-8601 or units/limit when applicable), citation {doc, page}.\n\n"
    "Only output JSON. No commentary."
)

RISK_TASK = (
    "Identify notable compliance and business risks in the excerpt. Return STRICT JSON with array 'risks' of items: "
    "id (risk_###), type one of ['noncompliance','IP_rights','data_rights','security','schedule','pricing'], "
    "severity 'H'|'M'|'L', rationale, mitigation, citation {doc,page}."
)

QA_TASK = (
    "Answer the user question using ONLY the provided RFP content. Cite sources with {doc,page}. If unknown, say you cannot find it. "
    "Return JSON: { 'answer': string, 'citations': [ { 'doc': string, 'page': number } ] }."
)


def llm_json(task: str, content: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Call OpenAI and return strict JSON with model fallback and fenced extraction."""
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
            # Try direct JSON
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
# DataFrame builders with schema guards
# -----------------------------

REQ_COLS = ["id","requirement_text","category","priority","shall_must","evidence_needed","citation"]
INST_COLS = ["id","topic","value","normalized","citation"]
RISK_COLS = ["id","type","severity","rationale","mitigation","citation"]


def _df_from_rows(rows: List[Dict[str, Any]], expected_cols: List[str], dedupe_subset: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    # Ensure expected columns exist
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    # Safe dedupe: only use subset if all columns exist
    if all(c in df.columns for c in dedupe_subset) and len(df):
        df = df.drop_duplicates(subset=dedupe_subset)
    else:
        df = df.drop_duplicates()
    # Reorder columns
    return df[expected_cols].reset_index(drop=True)

# -----------------------------
# Aggregation across chunks (with progress & chunk cap)
# -----------------------------

def aggregate_results(chunks: List[Dict[str, Any]], model: str, max_chunks: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req_rows: List[Dict[str, Any]] = []
    inst_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []

    total = len(chunks) if max_chunks is None else min(len(chunks), max_chunks)
    prog = st.progress(0, text="Analyzing chunks…")

    for idx, ch in enumerate(chunks[:total], start=1):
        content = f"Document: {ch['doc']} | Page: {ch['page']}\n\n{ch['text']}"
        # Requirements & Instructions
        try:
            data = llm_json(COMPLIANCE_TASK, content, model=model)
            for r in data.get("requirements", []) if isinstance(data, dict) else []:
                req_rows.append(r)
            for i in data.get("instructions", []) if isinstance(data, dict) else []:
                inst_rows.append(i)
        except Exception as e:
            st.info(f"Compliance extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
        # Risks
        try:
            risks_obj = llm_json(RISK_TASK, content, model=model)
            for rk in risks_obj.get("risks", []) if isinstance(risks_obj, dict) else []:
                risk_rows.append(rk)
        except Exception as e:
            st.info(f"Risk extraction skipped for {ch['doc']} p.{ch['page']}: {e}")

        prog.progress(idx/total, text=f"Analyzing chunks… ({idx}/{total})")

    req_df = _df_from_rows(req_rows, REQ_COLS, ["requirement_text","citation"])
    inst_df = _df_from_rows(inst_rows, INST_COLS, ["topic","value","citation"])
    risk_df = _df_from_rows(risk_rows, RISK_COLS, ["rationale","citation"])
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
max_chunks = st.slider("Max chunks to analyze (for speed)", 5, 200, 40, 5)

if st.button("🔎 Analyze Documents", type="primary"):
    if not uploaded:
        st.warning("Please upload at least one PDF or DOCX file.")
        st.stop()
    if not client:
        st.error("OpenAI client not configured. Add your API key and try again.")
        st.stop()

    with st.status("Reading and chunking documents…", expanded=False):
        pages = load_files(uploaded)
        st.write(f"Loaded {len(pages)} pages across {len(uploaded)} file(s).")
        chunks = chunk_pages(pages, target_chars=target_chars)
        st.write(f"Created {len(chunks)} chunk(s) for analysis.")

    req_df, inst_df, risk_df = aggregate_results(chunks, model, max_chunks=max_chunks)

    st.success("Analysis complete.")

    st.subheader("📋 Compliance Matrix")
    st.data_editor(req_df, use_container_width=True, height=360)
    st.download_button("Download Requirements (CSV)", req_df.to_csv(index=False).encode("utf-8"), file_name="requirements.csv", mime="text/csv")

    st.subheader("🗂️ Submission Instructions")
    st.data_editor(inst_df, use_container_width=True, height=320)
    st.download_button("Download Instructions (CSV)", inst_df.to_csv(index=False).encode("utf-8"), file_name="instructions.csv", mime="text/csv")

    st.subheader("⚠️ Risk Register")
    st.data_editor(risk_df, use_container_width=True, height=320)
    st.download_button("Download Risks (CSV)", risk_df.to_csv(index=False).encode("utf-8"), file_name="risks.csv", mime="text/csv")

st.markdown("---")
st.caption("MVP demo with schema guards, progress, and model fallback. Next: add real retrieval + XLSX/DOCX/ICS exports.")
