"""Streamlit UI: upload PDF → OCR → Arabic/English split → RAG Q&A → eval."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from evaluation import EVAL_QUESTIONS, evaluate_answer
from models import OCR_MODELS, QA_MODELS, get_answer, perform_ocr
from rag import cosine_topk
from text_utils import chunk_text, split_by_language


def _build_context(chunks: list[str], top: list[tuple[int, float]]) -> str:
    return "\n\n".join(chunks[i] for i, _ in top)


def run() -> None:
    st.set_page_config(page_title="RAG OCR", page_icon="📄", layout="wide")
    st.title("RAG OCR — Arabic / English")
    st.caption(
        "Read → OCR → language split → cosine-similarity retrieval → "
        "answer → evaluation"
    )

    with st.sidebar:
        st.header("Settings")
        ocr_model = st.selectbox("OCR model", OCR_MODELS)
        qa_model = st.selectbox("QA model", QA_MODELS)
        st.divider()
        top_k = st.slider("Top-k chunks", 1, 20, 8)
        target_chars = st.slider("Chunk size (chars)", 200, 1500, 400, step=50)
        overlap = st.slider("Chunk overlap (chars)", 0, 200, 100, step=10)
        
        st.divider()
        st.header("Token Usage")
        if "token_usage" not in st.session_state:
            st.session_state["token_usage"] = {"ocr": 0, "embed": 0, "qa": 0}
        
        st.metric("OCR Tokens", f"{st.session_state['token_usage']['ocr']:,}")
        st.metric("Embed/Rerank Tokens", f"{st.session_state['token_usage']['embed']:,}")
        st.metric("QA Tokens", f"{st.session_state['token_usage']['qa']:,}")
        total = sum(st.session_state['token_usage'].values())
        st.metric("Total Tokens", f"{total:,}")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    page_range = st.text_input("Page Range (Optional)", placeholder="e.g. 1-3, 5, 7-9 (leave blank for all)")

    if uploaded and st.button("Run OCR", type="primary"):
        st.session_state["last_request_tokens"] = {"ocr": 0, "embed": 0, "qa": 0}
        pdf_bytes = uploaded.read()
        with st.spinner(f"Running OCR with {ocr_model}..."):
            import time
            start_time = time.time()
            try:
                text = perform_ocr(ocr_model, pdf_bytes, uploaded.name, page_range)
            except Exception as exc:  # noqa: BLE001
                st.error(f"OCR failed: {exc}")
                return
            duration = time.time() - start_time
        st.session_state.update(
            ocr_text=text, ocr_model=ocr_model, ocr_file=uploaded.name
        )
        tokens_used = st.session_state["last_request_tokens"]["ocr"]
        st.success(f"OCR completed in {duration:.1f}s! (Used {tokens_used:,} tokens)")

    if "ocr_text" not in st.session_state:
        st.info("Upload a PDF and click **Run OCR** to begin.")
        return

    text: str = st.session_state["ocr_text"]
    stored_ocr_model: str = st.session_state["ocr_model"]

    english, arabic = split_by_language(text)

    st.divider()
    st.subheader(f"OCR output — {st.session_state['ocr_file']}  ·  {stored_ocr_model}")
    tab_full, tab_en, tab_ar = st.tabs(
        [f"Full ({len(text)} chars)", f"English ({len(english)})", f"Arabic ({len(arabic)})"]
    )
    with tab_full:
        st.text_area("full", text, height=260, label_visibility="collapsed")
    with tab_en:
        st.text_area("en", english, height=260, label_visibility="collapsed")
    with tab_ar:
        st.text_area("ar", arabic, height=260, label_visibility="collapsed")

    chunks = chunk_text(text, target_chars=target_chars, overlap=overlap)
    st.caption(f"Indexed **{len(chunks)}** chunks for retrieval.")

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input("Your question (Arabic or English)")
    if question and st.button("Get answer"):
        st.session_state["last_request_tokens"] = {"ocr": 0, "embed": 0, "qa": 0}
        top = cosine_topk(question, chunks, filename=st.session_state['ocr_file'], k=top_k)
        with st.expander("Retrieved chunks"):
            for rank, (idx, sim) in enumerate(top, 1):
                st.markdown(f"**#{rank} · chunk {idx} · sim={sim:.3f}**")
                st.text(chunks[idx])
        context = _build_context(chunks, top)
        with st.spinner(f"Answering with {qa_model}..."):
            import time
            start_time = time.time()
            try:
                answer = get_answer(qa_model, context, question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"QA failed: {exc}")
                return
            duration = time.time() - start_time
        st.success(answer)
        qa_tok = st.session_state["last_request_tokens"]["qa"]
        embed_tok = st.session_state["last_request_tokens"]["embed"]
        st.caption(f"**Tokens used for this request:** QA = {qa_tok:,} | Embed/Rerank = {embed_tok:,}  ·  **Time:** {duration:.1f}s")

    st.divider()
    st.subheader("Evaluation — 22 standard MOA questions")
    st.caption(
        "Runs each question through retrieval + QA and scores the answer "
        "against a reference. Metrics: keyword coverage, semantic (cosine) "
        "similarity, and their mean (overall)."
    )

    if st.button("Run evaluation suite"):
        rows = []
        progress = st.progress(0.0, text="Evaluating...")
        for i, q in enumerate(EVAL_QUESTIONS):
            top = cosine_topk(q["question"], chunks, filename=st.session_state['ocr_file'], k=top_k)
            context = _build_context(chunks, top)
            try:
                answer = get_answer(qa_model, context, q["question"])
            except Exception as exc:  # noqa: BLE001
                answer = f"[error: {exc}]"
            metrics = evaluate_answer(answer, q["reference"], q["keywords"])
            rows.append(
                {
                    "id": q["id"],
                    "question": q["name"],
                    "answer": (answer[:400] + "…") if len(answer) > 400 else answer,
                    **metrics,
                }
            )
            progress.progress((i + 1) / len(EVAL_QUESTIONS), text=q["name"])
        progress.empty()

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean keyword coverage", round(df["keyword_coverage"].mean(), 3))
        col2.metric("Mean semantic similarity", round(df["semantic_similarity"].mean(), 3))
        col3.metric("Mean overall", round(df["overall"].mean(), 3))

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"eval_{qa_model.replace(' ', '_')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    run()
