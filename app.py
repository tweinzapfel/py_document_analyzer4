import os
import io
import json
from dataclasses import dataclass
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

# Read OpenAI key from Streamlit secrets or env var
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    st.warning("Add your OpenAI API key to .streamlit/secrets.toml as OPENAI_API_KEY or set the environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------------------
# Helpers: extraction
# -----------------------------

def extract_pdf_text_and_pages(file: io.BytesIO, doc_name: str) -> List[Dict[str, Any]]:
    """Return a list of {doc, page, text}. Keeps page boundaries for better citations."""
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
    """Treat each paragraph as a flowing page surrogate; we still index as page numbers for citation stability."""
    d = DocxDocument(file)
    text = []
    for p in d.paragraphs:
        if p.text and p.text.strip():
            text.append(p.text.strip())
    # group paragraphs into pseudo-pages of ~8000 chars
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
    """Return list of page dicts across all files with doc/page/text."""
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
    """Chunk at page granularity, then split long pages. Preserves doc/page metadata."""
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        text = p["text"]
        if len(text) <= target_chars:
            chunks.append({**p, "chunk": 1, "text": text})
            continue
        # sliding window on paragraphs
        paras = [x for x in text.split("\n") if x.strip()]
        buf, size, idx = [], 0, 1
        for para in paras:
            if size + len(para) + 1 > target_chars and buf:
                chunks.append({**p, "chunk": idx, "text": "\n".join(buf)})
                # start next buffer with overlap tail
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

For each requirement:
- id: string (e.g., req_001)
- requirement_text: the exact requirement sentence/phrase (quote when short)
- category: one of ["technical", "administrative", "pricing", "security", "legal"]
- priority: P1 (critical), P2, or P3
- shall_must: boolean (True if includes SHALL/MUST/REQUIRED)
- evidence_needed: array of short strings (forms, plans, resumes, past perf, certifications)
- citation: {doc: string, page: number}

For each instruction:
- id: string (e.g., inst_001)
- topic: one of ["due_date","portal","format","naming","page_limit","font","margins","volumes","attachments"]
- value: raw instruction value (e.g., "12:00 PM ET on Oct 12, 2025")
- normalized: include fields when applicable (e.g., {"due_datetime":"YYYY-MM-DDTHH:MM:SS-04:00","units":"pages","limit":n})
- citation: {doc: string, page: number}

Only output JSON. Do not include commentary.
"""

RISK_TASK = """
Identify notable compliance and business risks in the excerpt. Return JSON array "risks" with items:
- id (e.g., risk_001)
- type: one of ["noncompliance","IP_rights","data_rights","security","schedule","pricing"]
- severity: H/M/L
- rationale: short reason referencing the requirement/instruction
- mitigation: short actionable suggestion
- citation: {doc, page}
Only output JSON.
"""

QA_TASK = """
Answer the user question using ONLY the provided RFP content. Cite sources with {doc, page}. If unknown, say you cannot find it.
Return JSON: {"answer": "...", "citations": [{"doc":"","page":n}]}
Only output JSON.
"""


def llm_json(task: str, content: str, model: str = "gpt-5-think") -> Dict[str, Any]:
    """Call OpenAI and coerce to JSON. Falls back to best-effort JSON extraction."""
    if not client:
        raise RuntimeError("OpenAI client is not configured.")
    prompt = (
        f"Task Instructions:\n{task}\n\nRFP Content:\n" + content
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise contracts analyst that returns STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip()
    # try direct json
    try:
        return json.loads(text)
    except Exception:
        # try fence extraction
        import re
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # last resort
        raise ValueError(f"Model did not return valid JSON. Raw: {text[:400]}...")

# -----------------------------
# Aggregation across chunks
# -----------------------------

def aggregate_results(chunks: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req_rows: List[Dict[str, Any]] = []
    inst_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []

    for ch in chunks:
        content = f"Document: {ch['doc']} | Page: {ch['page']}\n\n{ch['text']}"
        try:
            data = llm_json(COMPLIANCE_TASK, content)
            for r in data.get("requirements", []):
                req_rows.append(r)
            for i in data.get("instructions", []):
                inst_rows.append(i)
        except Exception as e:
            st.info(f"Compliance extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
        try:
            risks = llm_json(RISK_TASK, content).get("risks", [])
            for rk in risks:
                risk_rows.append(rk)
        except Exception as e:
            st.info(f"Risk extraction skipped for {ch['doc']} p.{ch['page']}: {e}")

    req_df = pd.DataFrame(req_rows).drop_duplicates(subset=["requirement_text","citation"]).reset_index(drop=True)
    inst_df = pd.DataFrame(inst_rows).drop_duplicates(subset=["topic","value","citation"]).reset_index(drop=True)
    risk_df = pd.DataFrame(risk_rows).drop_duplicates(subset=["rationale","citation"]).reset_index(drop=True)

    # ensure expected columns exist
    def ensure_cols(df: pd.DataFrame, cols: List[str]):
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]

    req_df = ensure_cols(req_df, ["id","requirement_text","category","priority","shall_must","evidence_needed","citation"])  
    inst_df = ensure_cols(inst_df, ["id","topic","value","normalized","citation"])  
    risk_df = ensure_cols(risk_df, ["id","type","severity","rationale","mitigation","citation"])  

    return req_df, inst_df, risk_df

# -----------------------------
# UI
# -----------------------------

st.title("📄 RFP Analyzer – MVP (Streamlit + OpenAI)")
st.caption("Upload RFP PDFs/DOCXs, extract compliance items, submission instructions, and risks. All results include citations.")

col_left, col_right = st.columns([2,1])
with col_left:
    uploaded = st.file_uploader(
        "Upload one or more RFP files (PDF/DOCX)",
        type=["pdf","doc","docx"],
        accept_multiple_files=True,
    )
    model = st.selectbox("Model", ["gpt-5-think", "gpt-4.1", "gpt-4o-mini"], index=0)
    target_chars = st.slider("Chunk target size (chars)", 1500, 8000, 4000, 500)

with col_right:
    run_btn = st.button("🔎 Analyze Documents", type="primary", use_container_width=True)
    st.write("")
    st.write("⚙️ Tip: Keep OCR/scanned pages minimal for the MVP.")

if run_btn:
    if not uploaded:
        st.warning("Please upload at least one PDF or DOCX file.")
        st.stop()
    if not client:
        st.error("OpenAI client not configured. Add your API key and try again.")
        st.stop()

    with st.status("Reading and chunking documents…", expanded=False) as status:
        pages = load_files(uploaded)
        st.write(f"Loaded {len(pages)} pages across {len(uploaded)} file(s).")
        chunks = chunk_pages(pages, target_chars=target_chars)
        st.write(f"Created {len(chunks)} chunk(s) for analysis.")
        status.update(label="Calling OpenAI to extract requirements, instructions, and risks…", state="running")

    req_df, inst_df, risk_df = aggregate_results(chunks)

    st.success("Analysis complete.")

    st.subheader("📋 Compliance Matrix")
    st.data_editor(req_df, use_container_width=True, height=400)
    st.download_button("Download Requirements (CSV)", req_df.to_csv(index=False).encode("utf-8"), file_name="requirements.csv", mime="text/csv")

    st.subheader("🗂️ Submission Instructions")
    st.data_editor(inst_df, use_container_width=True, height=350)
    st.download_button("Download Instructions (CSV)", inst_df.to_csv(index=False).encode("utf-8"), file_name="instructions.csv", mime="text/csv")

    st.subheader("⚠️ Risk Register")
    st.data_editor(risk_df, use_container_width=True, height=350)
    st.download_button("Download Risks (CSV)", risk_df.to_csv(index=False).encode("utf-8"), file_name="risks.csv", mime="text/csv")

    st.divider()
    st.subheader("💬 Ask the RFP (Q&A)")
    q = st.text_input("Your question about the RFP")
    if q:
        # Build small context window from top-N longest chunks for now (simple MVP retrieval)
        top_chunks = sorted(chunks, key=lambda c: len(c["text"]))[-3:]
        context = "\n\n".join([f"[{c['doc']} p.{c['page']}]\n{c['text'][:4000]}" for c in top_chunks])
        try:
            qa = llm_json(QA_TASK, f"Question: {q}\n\nContext:\n{context}", model=model)
            st.write(qa.get("answer", "(no answer)"))
            cits = qa.get("citations", [])
            if cits:
                st.caption("Sources: " + ", ".join([f"{c['doc']} p.{c['page']}" for c in cits if 'doc' in c]))
        except Exception as e:
            st.error(f"Q&A failed: {e}")

# Footer
st.markdown("---")
st.caption("MVP demo. Next steps: real retrieval (FAISS/Chroma), robust JSON tooling, exports to XLSX/DOCX/ICS, OCR for scans.")
