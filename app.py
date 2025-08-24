import os
import io
import json
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus

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

with st.sidebar:
    st.header("Diagnostics")
    status = "✅ Ready" if (OPENAI_API_KEY and client) else "❌ Not configured"
    st.write(f"OpenAI client: {status}")
    if st.button("🧹 Clear results", use_container_width=True):
        for _k in ("req_df","inst_df","risk_df","lm_df","clause_df","fedrisk_df","clause_exp_df"):
            st.session_state.pop(_k, None)
        st.success("Cleared saved results.")
        st.rerun()

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
# Chunking & planning
# -----------------------------

def chunk_pages(pages: List[Dict[str, Any]], target_chars: int = 4000, overlap: int = 400) -> List[Dict[str, Any]]:
    """Split page texts into overlapping chunks while preserving doc/page metadata."""
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


def _files_key(files: List[io.BytesIO]) -> str:
    """Stable key for caching by names + sizes."""
    parts = []
    for f in files or []:
        nm = getattr(f, "name", "unknown")
        sz = getattr(f, "size", None)
        parts.append(f"{nm}:{sz}")
    return "|".join(parts)


def build_chunk_plan(pages: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Summarize chunking by document (pages, chars, est tokens, chunk count)."""
    rows = []
    docs = sorted({p["doc"] for p in pages})
    for doc in docs:
        doc_pages = [p for p in pages if p["doc"] == doc]
        doc_chars = sum(len(p["text"]) for p in doc_pages)
        est_tokens = int(doc_chars / 4)
        chunk_count = sum(1 for c in chunks if c["doc"] == doc)
        rows.append({
            "Document": doc,
            "Pages": len(doc_pages),
            "Characters": doc_chars,
            "Est. tokens": est_tokens,
            "Est. chunks": chunk_count,
            "Avg chars/page": int(doc_chars / max(1, len(doc_pages))),
            "Avg chars/chunk": int(doc_chars / max(1, chunk_count)) if chunk_count else 0,
        })
    df = pd.DataFrame(rows).sort_values(["Est. chunks","Document"], ascending=[False, True]).reset_index(drop=True)
    return df

# ---- Cost estimation helpers ----

def _estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))


def _model_price_defaults(model: str) -> Tuple[float, float]:
    if model == "gpt-4o-mini":
        return (0.0003, 0.0006)
    if model == "gpt-4.1":
        return (0.0050, 0.0150)
    if model == "gpt-3.5-turbo":
        return (0.0005, 0.0015)
    return (0.0, 0.0)


def estimate_costs(chunks: List[Dict[str, Any]],
                   calls_per_chunk: int,
                   overhead_tokens_per_call: int,
                   est_output_tokens_per_call: int,
                   price_in_per_1k: float,
                   price_out_per_1k: float,
                   limit_chunks: int | None = None) -> Dict[str, Any]:
    use_chunks = chunks if (limit_chunks is None) else chunks[:limit_chunks]
    n_chunks = len(use_chunks)
    chunk_tokens_total = sum(_estimate_tokens_for_text(c.get("text", "")) for c in use_chunks)
    input_tokens_total = n_chunks * calls_per_chunk * overhead_tokens_per_call + calls_per_chunk * chunk_tokens_total
    output_tokens_total = n_chunks * calls_per_chunk * est_output_tokens_per_call
    cost_in = (input_tokens_total / 1000.0) * price_in_per_1k if price_in_per_1k else 0.0
    cost_out = (output_tokens_total / 1000.0) * price_out_per_1k if price_out_per_1k else 0.0
    return {
        "n_chunks": n_chunks,
        "input_tokens": int(input_tokens_total),
        "output_tokens": int(output_tokens_total),
        "cost_in": float(cost_in),
        "cost_out": float(cost_out),
        "cost_total": float(cost_in + cost_out),
    }

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

# --- Federal smart checks prompts ---
LM_TASK = (
    "From the excerpt, extract items from Section L (Instructions, conditions, notices) and Section M (Evaluation factors).\n"
    "Return STRICT JSON with array 'lm_items'. Each item must include: \n"
    "- id (e.g., lm_001)\n- part: 'L' or 'M'\n- topic: one of ['submission_structure','volumes','page_limit','format','font','margins','naming','portal','q_and_a','due_date',\n"
    "            'evaluation_factors','subfactors','basis_of_award','rating_method','tradeoff','past_performance','key_personnel','small_business']\n"
    "- value: raw text\n- normalized: object (e.g., {'due_datetime': ISO8601, 'units':'pages','limit':n} when applicable)\n- citation: {doc: string, page: number}"
)

CLAUSE_TASK = (
    "Identify any FAR or DFARS clause references in the excerpt (e.g., 'FAR 52.204-21', 'DFARS 252.204-7012').\n"
    "Return STRICT JSON with array 'clauses'. Each item: id (cl_001), system: 'FAR'|'DFARS'|'OTHER', clause (e.g., '52.204-21' or '252.204-7012'),\n"
    "title (if present), alt (e.g., 'ALT II' if present), date (if present), requirement_summary (short plain-English), citation {doc,page}."
)

FED_CHECK_TASK = (
    "Flag federal-specific compliance signals found in the excerpt. Return STRICT JSON with array 'fed_risks'. Each item: \n"
    "- id (fed_001)\n- category: one of ['CMMC','NIST_800-171','ITAR','EAR','CUI','Data_Rights','IP_rights','OCI','SB_Setaside','Cybersecurity','Supply_Chain','Section508','Export_Control']\n"
    "- severity: 'H'|'M'|'L'\n- rationale: short reason referencing any clause/section\n- mitigation: short actionable suggestion\n- clause: clause number if applicable (e.g., '252.204-7012')\n- citation: {doc,page}"
)


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
# DataFrame builders with schema guards (dict-safe dedupe)
# -----------------------------

REQ_COLS = ["id","requirement_text","category","priority","shall_must","evidence_needed","citation"]
INST_COLS = ["id","topic","value","normalized","citation"]
RISK_COLS = ["id","type","severity","rationale","mitigation","citation"]
LM_COLS = ["id","part","topic","value","normalized","citation"]
CLAUSE_COLS = ["id","system","clause","title","alt","date","requirement_summary","citation"]
FEDRISK_COLS = ["id","category","severity","rationale","mitigation","clause","citation"]


def _norm_key(val: Any) -> str:
    if isinstance(val, (dict, list, tuple, set)):
        try:
            return json.dumps(val, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(val)
    if val is None:
        return ""
    return str(val)


def _df_from_rows(rows: List[Dict[str, Any]], expected_cols: List[str], dedupe_subset: List[str]) -> pd.DataFrame:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in (rows or []):
        key = tuple(_norm_key(r.get(c)) for c in dedupe_subset) if dedupe_subset else (str(r),)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    df = pd.DataFrame(uniq)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols].reset_index(drop=True)

# -----------------------------
# Clause library & explainers
# -----------------------------

CLAUSE_LIBRARY: Dict[str, Dict[str, Any]] = {
    # Cyber & safeguarding
    "FAR 52.204-21": {
        "summary": "Basic safeguarding for contractor info systems that process or store Federal contract information (FCI). 15 baseline controls.",
        "pitfalls": [
            "Assuming it only applies to CUI — it applies to FCI, too.",
            "No written proof of controls (policies, training records).",
        ],
        "actions": [
            "Confirm 15 safeguards (e.g., access control, patching, malware protection).",
            "Document in SSP; train staff; flow down to subs handling FCI.",
        ],
        "artifacts": ["SSP or policy set", "Training logs"],
    },
    "DFARS 252.204-7012": {
        "summary": "Safeguard Covered Defense Information; implement NIST SP 800-171; 72-hr cyber incident reporting; cloud providers must meet FedRAMP Moderate equivalent.",
        "pitfalls": [
            "No SPRS/NIST alignment; unclear incident reporting flow.",
            "For cloud use, lacking Gov/FedRAMP-Moderate-equivalent attestation.",
        ],
        "actions": [
            "Implement 800-171; maintain SSP/POA&M; flowdown; 72-hr DoD reporting.",
            "Ensure cloud terms meet -7012 paragraph (b)(2)(ii)(D).",
        ],
        "artifacts": ["SSP", "POA&M", "IR plan", "Sub flowdown"],
    },
    "DFARS 252.204-7019": {
        "summary": "DoD Assessment Requirements; contractor must have a current NIST 800-171 assessment score in SPRS.",
        "pitfalls": ["Bid submitted without posting Basic/Medium/High score in SPRS."],
        "actions": ["Post current score to SPRS; keep evidence for spot checks."],
        "artifacts": ["SPRS screenshot", "Assessment worksheet"],
    },
    "DFARS 252.204-7020": {
        "summary": "NIST assessment methodology; permits DoD to perform (and contractor to support) assessments; requires flowdown.",
        "pitfalls": ["Subcontractors not informed; missing flowdown."],
        "actions": ["Flow down to subs; prepare for DoD assessment requests."],
        "artifacts": ["Flowdown clause", "Assessment prep notes"],
    },
    "DFARS 252.204-7021": {
        "summary": "CMMC requirement clause specifying the level required and timing; ties to certification obligations.",
        "pitfalls": ["Wrong level assumed; not aligning scope and asset boundary."],
        "actions": ["Confirm required level; plan timeline; coordinate with assessor."],
        "artifacts": ["CMMC level statement", "Scope diagram"],
    },
    # Commercial items
    "FAR 52.212-1": {
        "summary": "Instructions to Offerors for commercial products/services; governs submission format, packaging, reps/certs.",
        "pitfalls": ["Ignoring addenda in the solicitation that change defaults."],
        "actions": ["Mirror submission structure and reps/certs exactly; follow addenda."],
        "artifacts": ["Submission checklist"],
    },
    "FAR 52.212-2": {
        "summary": "Evaluation factors for award on commercial items (price, technical, past performance, etc.).",
        "pitfalls": ["Not mapping proposal to stated factors/subfactors."],
        "actions": ["Build a factor-by-factor crosswalk; provide discriminators."],
        "artifacts": ["Evaluation crosswalk"],
    },
    # Set-asides
    "FAR 52.219-6": {
        "summary": "Notice of Total Small Business Set-Aside; size standard must be met for the NAICS code.",
        "pitfalls": ["SAM profile NAICS mismatch; subcontracting plan assumptions."],
        "actions": ["Verify size under NAICS; update SAM; consider teaming if needed."],
        "artifacts": ["SAM/DSBS screenshots"],
    },
    # IP/data rights
    "FAR 52.227-14": {
        "summary": "Rights in Data—General; sets Government rights in data first produced under the contract and delivered data.",
        "pitfalls": ["Failing to mark limited/restricted rights; losing rights by default."],
        "actions": ["Identify data categories; apply legends; include special license terms as needed."],
        "artifacts": ["Data rights matrix", "Marking plan"],
    },
    # Prohibitions
    "FAR 52.204-23": {
        "summary": "Prohibition on Kaspersky Lab products and services.",
        "pitfalls": ["Asset inventory not checked; suppliers not screened."],
        "actions": ["Scan inventory/SBOM; attest to non-use; update supply chain checks."],
        "artifacts": ["Supplier attestation"],
    },
    "FAR 52.204-24": {
        "summary": "Representation regarding use of covered telecommunications (Huawei/ZTE, etc.).",
        "pitfalls": ["Incomplete due diligence beyond Tier-1 vendors."],
        "actions": ["Run supply chain questionnaires; document determination; prepare alternate offer if applicable."],
        "artifacts": ["Representation form", "Supplier survey"],
    },
    "FAR 52.204-25": {
        "summary": "Prohibition on contracting for covered telecommunications equipment or services.",
        "pitfalls": ["No remediation plan if discovered post-award."],
        "actions": ["Implement detection/removal procedures; report promptly if found."],
        "artifacts": ["Removal SOP"],
    },
    # Cloud
    "DFARS 252.239-7010": {
        "summary": "Cloud Computing Services; requires DoD SRG compliance, incident reporting, forensic records, and data location/portability.",
        "pitfalls": ["Hosting outside approved impact levels; unclear incident timelines."],
        "actions": ["Match SRG IL to data sensitivity; include incident & data return terms."],
        "artifacts": ["Cloud SRG mapping", "IR clause addendum"],
    },
}

CLAUSE_EXPLAINER_TASK = (
    "You will receive a JSON object with fields: system (e.g., 'FAR' or 'DFARS'), clause (e.g., '52.204-21'), and title if known.
"
    "Explain the clause in plain English for proposal compliance. Return STRICT JSON with: 
"
    "{ 'summary': string, 'pitfalls': [string], 'actions': [string], 'artifacts': [string] } — concise and practical."
)


def _clause_key(system: Any, clause: Any) -> str:
    sys = (str(system or "").strip().upper())
    cl = (str(clause or "").strip())
    return f"{sys} {cl}".strip()


def _acqgov_link(system: Any, clause: Any) -> str:
    q = f"{str(system or '').strip()} {str(clause or '').strip()}".strip()
    return f"https://www.acquisition.gov/search?search={quote_plus(q)}"


def build_clause_explainers(clause_df: pd.DataFrame, use_llm: bool, model: str) -> pd.DataFrame:
    if clause_df is None or clause_df.empty:
        return pd.DataFrame(columns=list(CLAUSE_COLS) + ["summary","pitfalls","actions","artifacts","reference_url"])

    rows = []
    cache: Dict[str, Dict[str, Any]] = {}
    for _, r in clause_df.iterrows():
        row = r.to_dict()
        key = _clause_key(row.get("system"), row.get("clause"))
        link = _acqgov_link(row.get("system"), row.get("clause"))
        lib = CLAUSE_LIBRARY.get(key)
        if lib:
            expl = lib
        else:
            if key not in cache and use_llm and client:
                try:
                    payload = {"system": row.get("system"), "clause": row.get("clause"), "title": row.get("title")}
                    out = llm_json(CLAUSE_EXPLAINER_TASK, json.dumps(payload), model=model)
                    cache[key] = out if isinstance(out, dict) else {}
                except Exception:
                    cache[key] = {}
            expl = cache.get(key, {})
        rows.append({
            **row,
            "summary": expl.get("summary"),
            "pitfalls": expl.get("pitfalls"),
            "actions": expl.get("actions"),
            "artifacts": expl.get("artifacts"),
            "reference_url": link,
        })

    cols = list(CLAUSE_COLS) + ["summary","pitfalls","actions","artifacts","reference_url"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

# -----------------------------
# Aggregation across chunks (with progress & chunk cap)
# -----------------------------
# -----------------------------

def aggregate_results(chunks: List[Dict[str, Any]], model: str, max_chunks: int = None, enable_fed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req_rows: List[Dict[str, Any]] = []
    inst_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    lm_rows: List[Dict[str, Any]] = []
    clause_rows: List[Dict[str, Any]] = []
    fedrisk_rows: List[Dict[str, Any]] = []

    total = len(chunks) if max_chunks is None else min(len(chunks), max_chunks)
    prog = st.progress(0, text="Analyzing chunks…")

    for idx, ch in enumerate(chunks[:total], start=1):
        content = f"Document: {ch['doc']} | Page: {ch['page']}\n\n{ch['text']}"
        # Compliance & Instructions
        try:
            data = llm_json(COMPLIANCE_TASK, content, model=model)
            for r in data.get("requirements", []) if isinstance(data, dict) else []:
                req_rows.append(r)
            for i in data.get("instructions", []) if isinstance(data, dict) else []:
                inst_rows.append(i)
        except Exception as e:
            st.info(f"Compliance extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
        # Generic Risks
        try:
            risks_obj = llm_json(RISK_TASK, content, model=model)
            for rk in risks_obj.get("risks", []) if isinstance(risks_obj, dict) else []:
                risk_rows.append(rk)
        except Exception as e:
            st.info(f"Risk extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
        # Federal smart checks
        if enable_fed:
            try:
                lm_obj = llm_json(LM_TASK, content, model=model)
                for it in lm_obj.get("lm_items", []) if isinstance(lm_obj, dict) else []:
                    lm_rows.append(it)
            except Exception as e:
                st.info(f"L/M extraction skipped for {ch['doc']} p.{ch['page']}: {e}")
            try:
                cl_obj = llm_json(CLAUSE_TASK, content, model=model)
                for cl in cl_obj.get("clauses", []) if isinstance(cl_obj, dict) else []:
                    clause_rows.append(cl)
            except Exception as e:
                st.info(f"Clause spotting skipped for {ch['doc']} p.{ch['page']}: {e}")
            try:
                fr_obj = llm_json(FED_CHECK_TASK, content, model=model)
                for fr in fr_obj.get("fed_risks", []) if isinstance(fr_obj, dict) else []:
                    fedrisk_rows.append(fr)
            except Exception as e:
                st.info(f"Federal risk flags skipped for {ch['doc']} p.{ch['page']}: {e}")

        prog.progress(idx/total, text=f"Analyzing chunks… ({idx}/{total})")

    req_df = _df_from_rows(req_rows, REQ_COLS, ["requirement_text","citation"])
    inst_df = _df_from_rows(inst_rows, INST_COLS, ["topic","value","citation"])
    risk_df = _df_from_rows(risk_rows, RISK_COLS, ["rationale","citation"])
    lm_df = _df_from_rows(lm_rows, LM_COLS, ["part","topic","value","citation"]) if enable_fed else pd.DataFrame(columns=LM_COLS)
    clause_df = _df_from_rows(clause_rows, CLAUSE_COLS, ["system","clause","citation"]) if enable_fed else pd.DataFrame(columns=CLAUSE_COLS)
    fedrisk_df = _df_from_rows(fedrisk_rows, FEDRISK_COLS, ["category","rationale","citation"]) if enable_fed else pd.DataFrame(columns=FEDRISK_COLS)
    return req_df, inst_df, risk_df, lm_df, clause_df, fedrisk_df

# -----------------------------
# UI helpers
# -----------------------------

def render_results(req_df: pd.DataFrame, inst_df: pd.DataFrame, risk_df: pd.DataFrame):
    st.subheader("📋 Compliance Matrix")
    st.data_editor(req_df, use_container_width=True, height=360, key="editor_requirements")
    st.download_button(
        "Download Requirements (CSV)",
        req_df.to_csv(index=False).encode("utf-8"),
        file_name="requirements.csv",
        mime="text/csv",
        key="dl_requirements",
    )

    st.subheader("🗂️ Submission Instructions")
    st.data_editor(inst_df, use_container_width=True, height=320, key="editor_instructions")
    st.download_button(
        "Download Instructions (CSV)",
        inst_df.to_csv(index=False).encode("utf-8"),
        file_name="instructions.csv",
        mime="text/csv",
        key="dl_instructions",
    )

    st.subheader("⚠️ Risk Register")
    st.data_editor(risk_df, use_container_width=True, height=320, key="editor_risks")
    st.download_button(
        "Download Risks (CSV)",
        risk_df.to_csv(index=False).encode("utf-8"),
        file_name="risks.csv",
        mime="text/csv",
        key="dl_risks",
    )


def render_federal(lm_df: pd.DataFrame, clause_df: pd.DataFrame, fedrisk_df: pd.DataFrame):
    if lm_df.empty and clause_df.empty and fedrisk_df.empty:
        st.info("No federal-specific items detected (or federal checks disabled).")
        return

    st.header("🇺🇸 Federal Smart Checks")

    with st.expander("Section L/M Extract (instructions & evaluation)", expanded=True):
        st.data_editor(lm_df, use_container_width=True, height=320, key="editor_lm", hide_index=True)
        st.download_button(
            "Download L_M (CSV)",
            lm_df.to_csv(index=False).encode("utf-8"),
            file_name="section_LM.csv",
            mime="text/csv",
            key="dl_lm",
        )

    with st.expander("FAR/DFARS Clauses Detected", expanded=True):
        # Add a clickable reference column pointing to acquisition.gov search
        if clause_df is not None and not clause_df.empty:
            clause_view = clause_df.copy()
            try:
                clause_view["reference_url"] = clause_view.apply(lambda r: _acqgov_link(r.get("system"), r.get("clause")), axis=1)
            except Exception:
                clause_view["reference_url"] = None
        else:
            clause_view = clause_df
        st.data_editor(
            clause_view,
            use_container_width=True,
            height=320,
            key="editor_clauses",
            hide_index=True,
            column_config={
                "reference_url": st.column_config.LinkColumn(
                    "Reference",
                    help="Open the official text/search on acquisition.gov",
                    display_text="acquisition.gov",
                )
            },
        )
        st.download_button(
            "Download Clauses (CSV)",
            clause_df.to_csv(index=False).encode("utf-8"),
            file_name="clauses.csv",
            mime="text/csv",
            key="dl_clauses",
        )

    # Clause library & explainers (enriched view)
    clause_exp_df = st.session_state.get("clause_exp_df")
    if clause_exp_df is not None and not clause_exp_df.empty:
        with st.expander("📚 Clause Library & Explainers (plain-English + actions)", expanded=True):
            st.caption("Practical guidance only — not legal advice. Verify against the official clause text.")
            st.data_editor(
                clause_exp_df,
                use_container_width=True,
                height=360,
                key="editor_clause_exp",
                hide_index=True,
                column_config={
                    "reference_url": st.column_config.LinkColumn(
                        "Reference",
                        help="Open the official text/search on acquisition.gov",
                        display_text="View on acquisition.gov",
                    )
                },
            )
            st.download_button(
                "Download Clause Explainers (CSV)",
                clause_exp_df.to_csv(index=False).encode("utf-8"),
                file_name="clause_explainers.csv",
                mime="text/csv",
                key="dl_clause_explainers",
            )

# -----------------------------
# UI
# -----------------------------

st.title("📄 RFP Analyzer – MVP (Streamlit + OpenAI)")

uploaded = st.file_uploader("Upload one or more RFP files (PDF/DOCX)", type=["pdf","doc","docx"], accept_multiple_files=True)
st.session_state["uploaded_files"] = uploaded or []

model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"], index=0)
st.session_state["model"] = model

target_chars = st.slider("Chunk target size (chars)", 1500, 8000, 4000, 500)
overlap = st.slider("Chunk overlap (chars)", 0, 1000, 400, 50)

enable_fed = st.checkbox("Enable 🇺🇸 Federal smart checks (FAR/DFARS + Section L/M + CMMC/ITAR)", value=True)
st.session_state["enable_fed"] = enable_fed

use_llm_clause = st.checkbox("Use LLM to explain unknown clauses", value=False, help="If a clause isn't in the built-in library, generate a short explainer using the selected model.")
st.session_state["use_llm_clause"] = use_llm_clause

max_chunks = st.slider("Max chunks to analyze (for speed)", 5, 200, 40, 5)

# Invalidate saved results if files changed
current_key = _files_key(uploaded) if uploaded else ""
prev_key = st.session_state.get("last_files_key", None)
if uploaded and prev_key != current_key:
    for _k in ("req_df","inst_df","risk_df","lm_df","clause_df","fedrisk_df"):
        st.session_state.pop(_k, None)
    st.session_state["last_files_key"] = current_key

# ----- Chunk plan preview (no API) -----
if uploaded:
    key = _files_key(uploaded)
    cache = st.session_state.setdefault("_plan_cache", {})
    cached = cache.get((key, target_chars, overlap))

    with st.spinner("Pre-scanning documents to estimate chunks…"):
        if cached is None:
            pages = load_files(uploaded)
            chunks = chunk_pages(pages, target_chars=target_chars, overlap=overlap)
            cache[(key, target_chars, overlap)] = {"pages": pages, "chunks": chunks}
        else:
            pages = cached["pages"]
            chunks = cached["chunks"]

    plan_df = build_chunk_plan(pages, chunks)
    total_chunks = len(chunks)

    st.metric(label="Estimated total chunks (all docs)", value=total_chunks)
    if max_chunks < total_chunks:
        st.info(f"This run is limited to the first {max_chunks} of {total_chunks} chunks. Increase 'Max chunks' to analyze all.")

    with st.expander("🧮 Chunk Plan (preview — no API calls)", expanded=True):
        st.dataframe(plan_df, use_container_width=True, height=260)

    # ---- Cost estimation UI ----
    st.subheader("💵 Estimated Tokens & Cost")
    colA, colB, colC = st.columns(3)
    default_calls = 2 + (3 if enable_fed else 0)  # Compliance+Risk (+ L/M + clauses + fed_risks)
    with colA:
        calls_per_chunk = st.number_input("Calls per chunk", min_value=1, max_value=6, value=int(default_calls), help="Compliance + Risk (+ Federal checks if enabled)")
    with colB:
        overhead_tokens = st.number_input("Overhead tokens per call", min_value=0, max_value=3000, value=600, step=50, help="Prompt/system tokens per call")
    with colC:
        est_output_tokens = st.number_input("Est. output tokens per call", min_value=0, max_value=5000, value=400, step=50)

    default_in, default_out = _model_price_defaults(model)
    with st.expander("Set pricing (per 1K tokens)"):
        c1, c2 = st.columns(2)
        with c1:
            price_in = st.number_input("Input price $/1K tokens", min_value=0.0, value=float(default_in), step=0.0001, format="%.6f")
        with c2:
            price_out = st.number_input("Output price $/1K tokens", min_value=0.0, value=float(default_out), step=0.0001, format="%.6f")
        st.caption("These are editable placeholders — update them to your account's current pricing for accurate estimates.")

    est_run = estimate_costs(
        chunks,
        calls_per_chunk=calls_per_chunk,
        overhead_tokens_per_call=overhead_tokens,
        est_output_tokens_per_call=est_output_tokens,
        price_in_per_1k=price_in,
        price_out_per_1k=price_out,
        limit_chunks=min(total_chunks, max_chunks),
    )
    est_all = estimate_costs(
        chunks,
        calls_per_chunk=calls_per_chunk,
        overhead_tokens_per_call=overhead_tokens,
        est_output_tokens_per_call=est_output_tokens,
        price_in_per_1k=price_in,
        price_out_per_1k=price_out,
        limit_chunks=None,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Input tokens (this run)", f"{est_run['input_tokens']:,}")
    m2.metric("Output tokens (this run)", f"{est_run['output_tokens']:,}")
    m3.metric("Est. cost (this run)", f"${est_run['cost_total']:,.2f}")
    m4.metric("Est. cost (all chunks)", f"${est_all['cost_total']:,.2f}")

# --- Persisted results display (survives reruns like download_button) ---
if all(k in st.session_state for k in ("req_df","inst_df","risk_df")):
    st.success("Showing saved analysis results.")
    render_results(st.session_state["req_df"], st.session_state["inst_df"], st.session_state["risk_df"])
    # Build clause explainers on the fly if missing (library; LLM only if enabled)
    if "clause_df" in st.session_state and "clause_exp_df" not in st.session_state:
        st.session_state["clause_exp_df"] = build_clause_explainers(
            st.session_state.get("clause_df", pd.DataFrame(columns=CLAUSE_COLS)),
            use_llm=st.session_state.get("use_llm_clause", False),
            model=st.session_state.get("model", "gpt-4o-mini"),
        )
    render_federal(
        st.session_state.get("lm_df", pd.DataFrame(columns=LM_COLS)),
        st.session_state.get("clause_df", pd.DataFrame(columns=CLAUSE_COLS)),
        st.session_state.get("fedrisk_df", pd.DataFrame(columns=FEDRISK_COLS)),
    )

if st.button("🔎 Analyze Documents", type="primary"):

    if not uploaded:
        st.warning("Please upload at least one PDF or DOCX file.")
        st.stop()
    if not client:
        st.error("OpenAI client not configured. Add your API key and try again.")
        st.stop()

    key = _files_key(uploaded)
    cache = st.session_state.setdefault("_plan_cache", {})
    cached = cache.get((key, target_chars, overlap))
    if cached is None:
        with st.status("Reading and chunking documents…", expanded=False):
            pages = load_files(uploaded)
            st.write(f"Loaded {len(pages)} pages across {len(uploaded)} file(s).")
            chunks = chunk_pages(pages, target_chars=target_chars, overlap=overlap)
            st.write(f"Created {len(chunks)} chunk(s) for analysis.")
            cache[(key, target_chars, overlap)] = {"pages": pages, "chunks": chunks}
    else:
        pages = cached["pages"]
        chunks = cached["chunks"]

    req_df, inst_df, risk_df, lm_df, clause_df, fedrisk_df = aggregate_results(chunks, model, max_chunks=max_chunks, enable_fed=enable_fed)

    # Save results to session state so they survive reruns (e.g., download_button)
    st.session_state["req_df"] = req_df
    st.session_state["inst_df"] = inst_df
    st.session_state["risk_df"] = risk_df
    st.session_state["lm_df"] = lm_df
    st.session_state["clause_df"] = clause_df
    st.session_state["fedrisk_df"] = fedrisk_df
    # Build & store enriched clause explainers (library + optional LLM)
    st.session_state["clause_exp_df"] = build_clause_explainers(
        clause_df,
        use_llm=st.session_state.get("use_llm_clause", False),
        model=model,
    )

    st.success("Analysis complete.")
    render_results(req_df, inst_df, risk_df)
    render_federal(lm_df, clause_df, fedrisk_df)

st.markdown("---")
st.caption("MVP demo with chunk planning, cost estimates, schema guards, progress, model fallback, persistent results, and 🇺🇸 Federal smart checks.")
