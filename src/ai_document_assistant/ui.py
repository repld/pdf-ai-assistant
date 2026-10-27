"""Streamlit user interface for uploading PDFs and asking grounded questions."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAIError

from ai_document_assistant.application import build_rag_assistant
from ai_document_assistant.assistant import ConfigurationError
from ai_document_assistant.documents import DocumentError, chunk_pages, extract_pdf_pages
from ai_document_assistant.rag import RAGAnswer, RAGAssistant

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


def _get_assistant() -> RAGAssistant:
    """Return the assistant for the current browser session."""

    assistant = st.session_state.get("assistant")
    if assistant is None:
        assistant = build_rag_assistant()
        st.session_state["assistant"] = assistant
    return assistant


def _index_upload(uploaded_file: Any) -> str:
    """Index a new uploaded PDF and return its original display name."""

    filename = Path(uploaded_file.name or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise DocumentError("only PDF files are supported")

    content = uploaded_file.getvalue()
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("PDF exceeds the 20 MB limit")

    file_hash = hashlib.sha256(content).hexdigest()
    if file_hash == st.session_state.get("indexed_file_hash"):
        return str(st.session_state["indexed_source"])

    assistant = build_rag_assistant()
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name

        pages = extract_pdf_pages(Path(temporary_path), source_name=filename)
        chunks = chunk_pages(pages)
        assistant.index(chunks)
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)

    st.session_state["assistant"] = assistant
    st.session_state["indexed_file_hash"] = file_hash
    st.session_state["indexed_source"] = filename
    st.session_state["indexed_page_count"] = len(pages)
    st.session_state["messages"] = []
    return filename


def _display_answer(result: RAGAnswer) -> None:
    """Display an answer and its page citations."""

    st.markdown(result.answer)
    if result.citations:
        st.caption("Sources")
        for citation in result.citations:
            st.caption(f"{citation.source}, page {citation.page_number}")


def main() -> None:
    """Render the Streamlit document chat application."""

    st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
    st.title("AI Document Assistant")
    st.write("Upload a PDF, then ask questions that are answered using its contents.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        try:
            source = _index_upload(uploaded_file)
        except ConfigurationError as error:
            st.error(str(error))
        except DocumentError as error:
            st.error(str(error))
        except (OpenAIError, RuntimeError):
            logger.exception("Failed to index uploaded document")
            st.error("The AI provider failed while processing the document.")
        else:
            st.success(
                f"Ready: {source} ({st.session_state['indexed_page_count']} pages indexed)."
            )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.caption("Sources")
                for citation in message["citations"]:
                    st.caption(f"{citation['source']}, page {citation['page_number']}")

    if st.session_state.get("indexed_source") is None:
        st.info("Upload a PDF to start asking questions.")
        return

    question = st.chat_input("Ask a question about the PDF")
    if question is None:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner(
        "Searching the document and preparing an answer..."
    ):
        try:
            result = _get_assistant().answer(question)
        except (OpenAIError, RuntimeError, ValueError):
            logger.exception("Failed to answer a UI question")
            st.error("The AI provider failed while answering the question.")
            return
        _display_answer(result)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": [
                {
                    "source": citation.source,
                    "page_number": citation.page_number,
                }
                for citation in result.citations
            ],
        }
    )


if __name__ == "__main__":
    main()
