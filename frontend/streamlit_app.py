"""Streamlit user interface for the AI Employee Onboarding Assistant."""

from __future__ import annotations

import os
import base64
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _load_cloud_secrets() -> None:
    """Expose Streamlit Cloud secrets to the shared backend configuration."""
    try:
        for name in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL"):
            value = st.secrets.get(name)
            if value:
                os.environ.setdefault(name, str(value))
    except Exception:
        # Local development may not have a Streamlit secrets file.
        pass


_load_cloud_secrets()

from backend.chat import LLMUnavailableError
from backend.rag import ConversationTurn, answer_question
from backend.utils import get_knowledge_base_status
from backend.vector_store import VectorStoreError


MODEL_NAME = os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free")
LOGO_PATH = PROJECT_DIR / "assets" / "company_logo.png"
HEADER_LOGO_PATH = PROJECT_DIR / "assets" / "header_logo.png"
SUGGESTED_QUESTIONS = [
    "How many annual leaves do I receive?",
    "How do I connect to the VPN?",
    "How does payroll work?",
    "How do I claim travel reimbursement?",
    "What health insurance is available?",
    "What are the work-from-home rules?",
]


def initialise_session() -> None:
    """Create the session-scoped conversation state on first page load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_data(ttl=15, show_spinner=False)
def get_backend_status() -> dict[str, Any] | None:
    """Get locally bundled knowledge-base readiness information."""
    try:
        status = get_knowledge_base_status()
        return {
            "document_count": status.document_count,
            "indexed_chunk_count": status.indexed_chunk_count,
            "is_ready": status.is_ready,
        }
    except Exception:
        return None


def request_answer(question: str) -> dict[str, Any]:
    """Run RAG locally in Streamlit and send only context to OpenRouter."""
    try:
        history = [
            ConversationTurn(role=item["role"], content=item["content"])
            for item in st.session_state.messages[-6:]
        ]
        response = answer_question(question, history)
        return {
            "answer": response.answer,
            "sources": [
                {"document": source.document, "page": source.page}
                for source in response.sources
            ],
        }
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    except (VectorStoreError, LLMUnavailableError) as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError("The assistant could not process the question.") from exc


def add_message(role: str, content: str, sources: list[dict[str, Any]] | None = None) -> None:
    """Store one rendered message and its timestamp in the active session."""
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render citations returned by the RAG backend."""
    if sources:
        citations = " · ".join(f"{item['document']} (p. {item['page']})" for item in sources)
        st.caption(f"Sources: {citations}")


def render_message(message: dict[str, Any]) -> None:
    """Render a single chat bubble with timestamp and optional citations."""
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        render_sources(message["sources"])
        st.caption(message["timestamp"])


def submit_question(question: str) -> None:
    """Add an employee question, fetch its answer, and persist both messages."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return

    add_message("user", cleaned_question)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching company documents…"):
            try:
                response = request_answer(cleaned_question)
            except RuntimeError as exc:
                answer = str(exc)
                sources: list[dict[str, Any]] = []
            else:
                answer = str(response["answer"])
                sources = list(response.get("sources", []))
        st.markdown(answer)
        render_sources(sources)
        st.caption(datetime.now().strftime("%I:%M %p"))
    add_message("assistant", answer, sources)


def apply_style() -> None:
    """Apply a compact, accessible dashboard visual treatment."""
    header_logo = ""
    if HEADER_LOGO_PATH.is_file():
        encoded_logo = base64.b64encode(HEADER_LOGO_PATH.read_bytes()).decode("ascii")
        header_logo = (
            "background-image: url('data:image/png;base64,"
            f"{encoded_logo}'); background-repeat: no-repeat; "
            "background-position: 92px center; background-size: 220px auto;"
        )
    st.markdown(
        f"""
        <style>
            .stApp {{ background: #f7f9fc; }}
            header[data-testid="stHeader"] {{
                height: 92px;
                background-color: #ffffff;
                {header_logo}
            }}
            [data-testid="stSidebar"] {{ background: #102a43; }}
            [data-testid="stSidebar"] * {{ color: #f0f4f8; }}
            .hero {{ padding: 0.4rem 0 1.1rem; }}
            .hero h1 {{ color: #102a43; margin-bottom: 0.2rem; }}
            .hero p {{ color: #627d98; margin-top: 0; }}
            .sidebar-title {{ margin: 1.2rem 0 1.5rem; }}
            .sidebar-title h2 {{ margin: 0; color: #ffffff; font-size: 1.35rem; letter-spacing: -0.025em; white-space: nowrap; }}
            .sidebar-title p {{ margin: .3rem 0 0; color: #b8c7d9; font-size: .86rem; }}
            .status-card {{ background: linear-gradient(145deg, rgba(61,106,143,.78), rgba(38,73,105,.9)); border: 1px solid rgba(170,204,232,.17); border-radius: 16px; padding: 1rem; margin: 0 0 1rem; box-shadow: 0 8px 20px rgba(3,20,36,.18); }}
            .status-card h3 {{ margin: 0 0 .75rem; font-size: .9rem; letter-spacing: .05em; text-transform: uppercase; color: #c9d9e8; }}
            .readiness {{ display: inline-flex; padding: .3rem .6rem; border-radius: 999px; font-weight: 700; font-size: .86rem; }}
            .readiness.ready {{ background: rgba(42,202,141,.16); color: #9ef1c8; }}
            .readiness.offline {{ background: rgba(255,185,102,.16); color: #ffd18d; }}
            .status-note {{ margin: .8rem 0 0; color: #dbe8f4; font-size: .84rem; line-height: 1.45; }}
            .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; margin: 0 0 1rem; }}
            .metric {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: .7rem; }}
            .metric strong {{ display: block; color: #ffffff; font-size: 1.25rem; line-height: 1.1; }}
            .metric span {{ display: block; color: #b8c7d9; font-size: .73rem; margin-top: .25rem; }}
            .model-chip {{ color: #b8c7d9; font-size: .82rem; padding: .7rem .8rem; border-left: 3px solid #37b7ff; background: rgba(255,255,255,.05); border-radius: 0 10px 10px 0; margin-bottom: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_company_logo() -> None:
    """Display the administrator-provided company logo when available."""
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=280)
    else:
        st.markdown("## ◈")


def main() -> None:
    """Render and operate the employee onboarding chat dashboard."""
    st.set_page_config(page_title="AI Employee Onboarding", page_icon="🤖", layout="wide")
    initialise_session()
    apply_style()

    with st.sidebar:
        render_company_logo()
        st.markdown(
            "<div class='sidebar-title'><h2>AI Onboarding Assistant</h2><p>Company policy guidance, made simple.</p></div>",
            unsafe_allow_html=True,
        )
        index_status = get_backend_status()
        is_ready = bool(index_status and index_status.get("is_ready"))
        indicator = "● Ready" if is_ready else "● Not ready"
        st.markdown(
            f"""
            <section class='status-card'>
              <h3>Knowledge base</h3>
              <span class='readiness {'ready' if is_ready else 'offline'}'>{indicator}</span>
              <p class='status-note'>Approved company documents, securely indexed for grounded answers.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        if index_status:
            st.markdown(
                f"""
                <div class='metric-grid'>
                  <div class='metric'><strong>{index_status['document_count']}</strong><span>Documents</span></div>
                  <div class='metric'><strong>{index_status['indexed_chunk_count']}</strong><span>Indexed chunks</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='model-chip'>Backend unavailable</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='model-chip'>Local model<br><b>{MODEL_NAME}</b></div>", unsafe_allow_html=True)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        """
        <div class="hero">
          <h1>How can I help with your onboarding?</h1>
          <p>Answers are grounded only in your company’s approved documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.subheader("Suggested questions")
        columns = st.columns(3)
        for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
            if columns[index % 3].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                submit_question(suggestion)
                st.rerun()
    else:
        for message in st.session_state.messages:
            render_message(message)

    user_question = st.chat_input("Ask about leave, payroll, benefits, VPN, travel, or policies…")
    if user_question:
        submit_question(user_question)
        st.rerun()


if __name__ == "__main__":
    main()
