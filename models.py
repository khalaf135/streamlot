"""OCR + QA adapters for Mistral OCR, Gemini, and Qwen.

Each provider exposes two capabilities:
  * OCR: extract raw text from a PDF
  * QA : answer a question given retrieved context

OCR results are cached on disk by (model, file-hash) so re-runs and
evaluation sweeps are free after the first call.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def _get_api_key(key_name: str) -> str | None:
    val = os.getenv(key_name)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key_name)
    except Exception:
        return None

def require_api_key(key_name: str) -> str:
    key = _get_api_key(key_name)
    if not key:
        available = []
        try:
            import streamlit as st
            available = list(st.secrets.keys())
        except Exception:
            pass
        raise RuntimeError(f"{key_name} not set. Streamlit sees these secret keys: {available}")
    return key

_db_initialized = False

def _get_db_connection():
    global _db_initialized
    import psycopg2
    from pgvector.psycopg2 import register_vector
    
    db_url = require_api_key("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    
    if not _db_initialized:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(1024)
                );
            """)
            cur.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS filename TEXT;")
            cur.execute("CREATE INDEX IF NOT EXISTS doc_chunks_hash_idx ON document_chunks (document_hash);")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    pdf_hash TEXT NOT NULL,
                    extracted_text TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS ocr_hash_idx ON ocr_results(pdf_hash, model_name);")
            
        conn.commit()
        _db_initialized = True
        
    try:
        register_vector(conn)
    except Exception as e:
        print("Warning: failed to register pgvector.", e)
        
    return conn

MISTRAL_OCR_MODEL = "mistral-ocr-latest"
GEMINI_OCR_MODEL = os.getenv("GEMINI_OCR_MODEL", "gemini-2.5-pro")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", GEMINI_OCR_MODEL)
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
NEBIUS_CHAT_MODEL = os.getenv("NEBIUS_CHAT_MODEL", "Qwen/Qwen2.5-32B-Instruct")
NEBIUS_VL_MODEL = os.getenv("NEBIUS_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"

OCR_MODELS = [
    "Mistral OCR (mistral-ocr-latest)",
    "Gemini (best)",
    "Nebius VL (Qwen2.5-VL-7B)",
    "Surya OCR (local)",
    "EasyOCR (local)",
    "Docling (local)",
]

QA_MODELS = [
    "Nebius (Qwen2.5-32B)",
    "Gemini (best)",
    "Mistral (mistral-large)",
]

def _add_tokens(category: str, amount: int):
    try:
        import streamlit as st
        if "token_usage" not in st.session_state:
            st.session_state["token_usage"] = {"ocr": 0, "embed": 0, "qa": 0}
        st.session_state["token_usage"][category] += amount
    except Exception:
        pass

OCR_PROMPT = (
    "You are an OCR engine. Extract ALL text from this document exactly as it "
    "appears, preserving reading order. Keep Arabic text as Arabic and English "
    "text as English — do NOT translate. Output plain text only, no commentary."
)

QA_SYSTEM = (
    "You answer questions about a company legal document using ONLY the "
    "provided context. Read the ENTIRE context carefully before answering. "
    "Look for exact matches AND related terms (e.g. 'License No' matches "
    "'license number', 'Share Capital' matches 'capital'). "
    "Extract and quote exact names, numbers, dates, and identifiers. "
    "Only say you don't know if the information is truly absent from "
    "every part of the context."
)


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pdf_to_png_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[bytes] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(pix.tobytes("png"))
    return pages


# --------------------------------------------------------------------------- #
# Mistral OCR
# --------------------------------------------------------------------------- #
def _run_mistral_ocr(pdf_bytes: bytes, filename: str) -> str:
    from mistralai.client import Mistral

    api_key = require_api_key("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    uploaded = client.files.upload(
        file={"file_name": filename, "content": pdf_bytes},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id)
    resp = client.ocr.process(
        model=MISTRAL_OCR_MODEL,
        document={"type": "document_url", "document_url": signed.url},
        include_image_base64=False,
    )
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            _add_tokens("ocr", getattr(usage, "total_tokens", 0))
    except Exception:
        pass
    return "\n\n".join(page.markdown for page in resp.pages)


def _answer_with_mistral(context: str, question: str) -> str:
    from mistralai.client import Mistral

    client = Mistral(api_key=require_api_key("MISTRAL_API_KEY"))
    resp = client.chat.complete(
        model=MISTRAL_CHAT_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            _add_tokens("qa", getattr(usage, "total_tokens", 0))
    except Exception:
        pass
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def _gemini_client():
    from google import genai

    api_key = _get_api_key("GOOGLE_API_KEY") or _get_api_key("GEMINI_API_KEY")
    if not api_key:
        available = []
        try:
            import streamlit as st
            available = list(st.secrets.keys())
        except Exception:
            pass
        raise RuntimeError(f"GOOGLE_API_KEY not set. Streamlit sees these secret keys: {available}")
    return genai.Client(api_key=api_key)


def _run_gemini_ocr(pdf_bytes: bytes, filename: str) -> str:
    client = _gemini_client()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        uploaded = client.files.upload(file=path)
        # Wait for the file to be ACTIVE before using it in a request.
        for _ in range(60):
            info = client.files.get(name=uploaded.name)
            state = getattr(info.state, "name", str(info.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini file upload failed")
            time.sleep(1)
        resp = client.models.generate_content(
            model=GEMINI_OCR_MODEL,
            contents=[uploaded, OCR_PROMPT],
        )
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            _add_tokens("ocr", getattr(resp.usage_metadata, "total_token_count", 0))
        return resp.text or ""
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _answer_with_gemini(context: str, question: str) -> str:
    client = _gemini_client()
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[
                    f"{QA_SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}"
                ],
            )
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                _add_tokens("qa", getattr(resp.usage_metadata, "total_token_count", 0))
            return resp.text or ""
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                wait = 30 * (attempt + 1)
                print(f"QA rate-limited, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("QA failed after max retries")


# --------------------------------------------------------------------------- #
# Qwen (DashScope)
# --------------------------------------------------------------------------- #
def _run_qwen_ocr(pdf_bytes: bytes, filename: str) -> str:
    import dashscope
    from dashscope import MultiModalConversation

    api_key = require_api_key("DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    page_pngs = _pdf_to_png_pages(pdf_bytes)
    texts: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        for i, png in enumerate(page_pngs):
            p = Path(td) / f"page_{i:03d}.png"
            p.write_bytes(png)
            resp = MultiModalConversation.call(
                model=QWEN_OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": f"file://{p}"},
                            {"text": OCR_PROMPT},
                        ],
                    }
                ],
            )
            content = resp["output"]["choices"][0]["message"]["content"]
            if isinstance(content, list):
                page_text = "".join(c.get("text", "") for c in content)
            else:
                page_text = str(content)
            texts.append(page_text)
    return "\n\n".join(texts)


def _answer_with_qwen(context: str, question: str) -> str:
    import dashscope
    from dashscope import Generation

    dashscope.api_key = require_api_key("DASHSCOPE_API_KEY")
    resp = Generation.call(
        model=QWEN_CHAT_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        result_format="message",
    )
    usage = resp.get("usage", {})
    _add_tokens("qa", usage.get("total_tokens", 0))
    return resp["output"]["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Nebius (OpenAI-compatible)
# --------------------------------------------------------------------------- #
def _nebius_client():
    from openai import OpenAI

    api_key = require_api_key("NEBIUS_API_KEY")
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)


def _answer_with_nebius(context: str, question: str) -> str:
    client = _nebius_client()
    resp = client.chat.completions.create(
        model=NEBIUS_CHAT_MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            _add_tokens("qa", getattr(usage, "total_tokens", 0))
    except Exception:
        pass
    return resp.choices[0].message.content or ""


def _run_nebius_ocr(pdf_bytes: bytes, filename: str) -> str:
    """OCR via Nebius Qwen2.5-VL-72B: convert PDF pages to images, send to VL model."""
    import base64

    client = _nebius_client()
    page_pngs = _pdf_to_png_pages(pdf_bytes)
    texts: list[str] = []
    for i, png in enumerate(page_pngs):
        b64 = base64.b64encode(png).decode()
        resp = client.chat.completions.create(
            model=NEBIUS_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
            max_tokens=4096,
        )
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                _add_tokens("ocr", getattr(usage, "total_tokens", 0))
        except Exception:
            pass
        texts.append(resp.choices[0].message.content or "")
    return "\n\n".join(texts)


# --------------------------------------------------------------------------- #
# Surya OCR (local, 90+ languages, excellent Arabic support)
# --------------------------------------------------------------------------- #
def _run_surya_ocr(pdf_bytes: bytes, filename: str) -> str:
    """OCR via Surya: high-quality local OCR with 90+ language support.

    Uses FoundationPredictor + RecognitionPredictor + DetectionPredictor.
    Models download automatically on first run (~1-2 GB).
    """
    from PIL import Image
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    page_pngs = _pdf_to_png_pages(pdf_bytes, dpi=300)
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()

    texts: list[str] = []
    for png_bytes in page_pngs:
        import io
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        predictions = recognition_predictor(
            [image], det_predictor=detection_predictor
        )
        # Each prediction is a page result with text_lines
        page_lines: list[str] = []
        for pred in predictions:
            if hasattr(pred, "text_lines"):
                for line in pred.text_lines:
                    page_lines.append(line.text)
            elif hasattr(pred, "text"):
                page_lines.append(pred.text)
        texts.append("\n".join(page_lines))
    return "\n\n".join(texts)


# --------------------------------------------------------------------------- #
# EasyOCR (local, lightweight, Arabic + English)
# --------------------------------------------------------------------------- #
def _run_easyocr(pdf_bytes: bytes, filename: str) -> str:
    """OCR via EasyOCR: lightweight local OCR supporting Arabic + English.

    Models download automatically on first run (~100-200 MB).
    """
    import easyocr
    import numpy as np
    from PIL import Image

    reader = easyocr.Reader(["ar", "en"], gpu=False)
    page_pngs = _pdf_to_png_pages(pdf_bytes, dpi=300)
    texts: list[str] = []
    for png_bytes in page_pngs:
        import io
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        texts.append("\n".join(results))
    return "\n\n".join(texts)


# --------------------------------------------------------------------------- #
# Docling (IBM — structured document parsing → Markdown)
# --------------------------------------------------------------------------- #
def _run_docling_ocr(pdf_bytes: bytes, filename: str) -> str:
    """OCR via Docling: IBM's document intelligence toolkit.

    Converts PDFs to structured Markdown preserving tables, headers,
    and reading order. Best for digitally-created PDFs; uses system
    OCR for scanned content on macOS (Apple Vision).
    """
    from docling.document_converter import DocumentConverter

    # Docling needs a file path — write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        return result.document.export_to_markdown()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
_OCR_FUNCS = {
    "Mistral OCR (mistral-ocr-latest)": _run_mistral_ocr,
    "Gemini (best)": _run_gemini_ocr,
    "Nebius VL (Qwen2.5-VL-72B)": _run_nebius_ocr,
    "Surya OCR (local)": _run_surya_ocr,
    "EasyOCR (local)": _run_easyocr,
    "Docling (local)": _run_docling_ocr,
}

_QA_FUNCS = {
    "Nebius (Qwen2.5-32B)": _answer_with_nebius,
    "Gemini (best)": _answer_with_gemini,
    "Mistral (mistral-large)": _answer_with_mistral,
}


def perform_ocr(
    model_name: str,
    pdf_bytes: bytes,
    filename: str,
    page_string: str = "",
    use_cache: bool = True,
) -> str:
    if model_name not in _OCR_FUNCS:
        raise ValueError(f"Unknown OCR model: {model_name}")
        
    from pdf_utils import trim_pdf_pages
    trimmed_pdf = trim_pdf_pages(pdf_bytes, page_string)
        
    pdf_hash = _file_hash(trimmed_pdf)
    
    if use_cache:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT extracted_text FROM ocr_results 
                WHERE pdf_hash = %s AND model_name = %s 
                LIMIT 1
            """, (pdf_hash, model_name))
            result = cur.fetchone()
        conn.close()
        
        if result:
            return result[0]
            
    # Not cached, perform OCR
    text = _OCR_FUNCS[model_name](trimmed_pdf, filename)
    
    # Save to database
    if use_cache:
        conn = _get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ocr_results (filename, model_name, pdf_hash, extracted_text)
                VALUES (%s, %s, %s, %s)
            """, (filename, model_name, pdf_hash, text))
        conn.commit()
        conn.close()
        
    return text


def get_answer(model_name: str, context: str, question: str) -> str:
    if model_name not in _QA_FUNCS:
        raise ValueError(f"Unknown QA model: {model_name}")
    return _QA_FUNCS[model_name](context, question)
