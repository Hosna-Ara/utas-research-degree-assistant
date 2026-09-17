"""Streamlit interface for the local UTAS Research Degree Assistant."""

import logging
from html import escape
import os
from pathlib import Path

import streamlit as st

from utas_research_assistant.service import create_service
from utas_research_assistant.chat_history import (
    DEFAULT_DB_PATH, add_message, create_conversation, get_conversation,
    list_conversations,
)
from utas_research_assistant.ui.components import render_assistant_result
from utas_research_assistant.ui.styles import APP_CSS

LOGGER = logging.getLogger(__name__)

SUGGESTIONS = [
    ("⌁", "Find AI & machine learning PhD projects"),
    ("Aa", "What English score do I need for a PhD?"),
    ("◇", "Find funded ICT projects for international students"),
    ("⌂", "Show Master by Research projects in Hobart"),
    ("RA", "Tell me about Soonja Yeom"),
    ("AI", "Which supervisors work in artificial intelligence?"),
]


@st.cache_resource(show_spinner=False)
def get_service():
    return create_service()


def _queue_question(question: str) -> None:
    st.session_state.pending_question = question


def _queue_chat_input() -> None:
    question = str(st.session_state.get("composer_input") or "").strip()
    if question:
        st.session_state.pending_question = question


def _history_enabled() -> bool:
    """Public deployment uses ephemeral session state, never local personal DB."""
    return os.environ.get("UTAS_DEPLOYMENT_MODE", "local").casefold() != "public"


def _load_session_conversation(conversation_id: int) -> None:
    if not _history_enabled():
        return
    record = get_conversation(conversation_id, DEFAULT_DB_PATH)
    if record is None:
        st.session_state.conversation_id = None
        st.session_state.messages = []
        return
    messages = []
    for item in record["messages"]:
        if item["role"] == "assistant" and isinstance(item.get("metadata", {}).get("payload"), dict):
            messages.append({"role": "assistant", **item["metadata"]["payload"]})
        elif item["role"] == "user":
            messages.append({"role": "user", "content": item["content"]})
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = messages


def _new_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.conversation_id = None
    st.session_state.view = "chat"


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">✦</div><div><strong>UTAS Research</strong><span>Degree Assistant</span></div></div>', unsafe_allow_html=True)
        if st.button("＋  New chat", use_container_width=True, type="primary", key="new_chat"):
            _new_chat()
            st.rerun()
        st.markdown('<div class="nav-section">Recent chats</div>', unsafe_allow_html=True)
        conversations = list_conversations(DEFAULT_DB_PATH) if _history_enabled() else []
        if conversations:
            for conversation in conversations[:8]:
                selected = conversation["id"] == st.session_state.get("conversation_id")
                label = str(conversation["title"] or "New chat")
                if len(label) > 34:
                    label = label[:31].rstrip() + "…"
                if st.button(("●  " if selected else "○  ") + label, key=f"conversation_{conversation['id']}",
                             use_container_width=True, type="secondary"):
                    _load_session_conversation(int(conversation["id"]))
                    st.session_state.view = "chat"
                    st.rerun()
        else:
            st.caption("No saved conversations yet.")
        st.markdown('<div class="nav-section">Workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item nav-item-active"><span>◌</span> Chat</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-item nav-item-muted"><span>⌕</span><span>Explore research <small>Coming soon</small></span></div>', unsafe_allow_html=True)
        if st.button("ⓘ  About this assistant", key="about_nav", use_container_width=True):
            st.session_state.view = "about"
            st.rerun()
        st.markdown('<div class="nav-section">System features</div><div class="feature-list">• Hybrid RAG<br>• Semantic + BM25 retrieval<br>• Knowledge Graph<br>• SPARQL reasoning<br>• Local Qwen LLM<br>• Grounded citations</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-section snapshot-heading">Knowledge snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="snapshot-date">14 Sep 2026</div>', unsafe_allow_html=True)
        st.markdown('<div class="snapshot-stats"><div><b>219</b><span>Projects</span></div><div><b>174</b><span>Supervisors</span></div><div><b>25</b><span>Research areas</span></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="status-list"><div><i></i> Local AI</div><div><i></i> Grounded sources</div><div><i></i> Knowledge graph</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-footer">Student project prototype developed by <strong>Hosna and team</strong> as part of a course unit group project.<br><span>Always verify important information with official University of Tasmania sources.</span></div>', unsafe_allow_html=True)


def _render_about() -> None:
    st.markdown('<div class="about-view"><div class="eyebrow">About this assistant</div><h1>Grounded research discovery for UTAS HDR study</h1><p>This unofficial prototype helps explore research projects, supervisors, scholarships, entry requirements and research-degree information using locally stored official UTAS sources.</p><div class="about-grid"><div><strong>What powers it</strong><p>Hybrid RAG, semantic and BM25 retrieval, Knowledge Graph reasoning, SPARQL, and a local open-source Qwen model.</p></div><div><strong>Knowledge snapshot</strong><p>219 projects, 174 supervisor identities, 140 resolved profiles and 25 research areas, dated 14 September 2026.</p></div><div><strong>Limitations</strong><p>The local corpus is a dated snapshot. Always verify important information with official University of Tasmania sources.</p></div></div></div>', unsafe_allow_html=True)


def _render_empty_state() -> None:
    st.markdown('<div class="empty-state"><div class="eyebrow">Research discovery assistant</div><h1>What would you like to explore?</h1><p>Search research opportunities, supervisors, scholarships, entry requirements and HDR information.</p></div>', unsafe_allow_html=True)
    columns = st.columns(2)
    for index, (icon, suggestion) in enumerate(SUGGESTIONS):
        with columns[index % 2]:
            st.button(f"{icon}   {suggestion}", key=f"suggestion_{index}", use_container_width=True, on_click=_queue_question, args=(suggestion,))


def _friendly_error(exc: Exception) -> str:
    LOGGER.exception("Question processing failed", exc_info=True)
    return "Something went wrong while processing your question. Please try again."


def _process_question(question: str) -> None:
    if _history_enabled() and not st.session_state.get("conversation_id"):
        conversation = create_conversation(first_question=question, path=DEFAULT_DB_PATH)
        st.session_state.conversation_id = int(conversation["id"])
    if _history_enabled():
        add_message(st.session_state.conversation_id, "user", question, path=DEFAULT_DB_PATH)
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        # Use Streamlit's role defaults. Passing decorative strings here makes
        # Streamlit treat them as image paths on some supported versions.
        with st.chat_message("user"):
            st.markdown(f'<div class="message-label user-label">You</div><div class="user-bubble">{escape(question).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            progress = st.empty()
            progress.markdown('<div class="finding-state"><span class="finding-dot"></span>Finding answer<span class="finding-ellipsis">...</span></div>', unsafe_allow_html=True)
            result = get_service().answer_with_evidence(question)
            progress.empty()
            st.markdown('<div class="message-label assistant-label">Research assistant</div>', unsafe_allow_html=True)
            payload = {"response": result.response.model_dump(mode="json"), "evidence": result.evidence, "generation_notice": result.generation_notice}
            render_assistant_result(payload)
            st.session_state.messages.append({"role": "assistant", **payload})
            if _history_enabled():
                add_message(st.session_state.conversation_id, "assistant", result.response.answer,
                            {"payload": payload, "citation_ids": result.response.citations,
                             "project_ids": result.response.project_ids,
                             "reasoning_method": result.response.reasoning_method}, path=DEFAULT_DB_PATH)
    except Exception as exc:
        message = _friendly_error(exc)
        # Keep the user-facing error small while retaining the traceback in the
        # terminal log for diagnosis.
        with st.chat_message("assistant"):
            st.info(message)
        st.session_state.messages.append({"role": "assistant", "error": message})
        if _history_enabled() and st.session_state.get("conversation_id"):
            add_message(st.session_state.conversation_id, "assistant", message,
                        {"payload": {"error": message}}, path=DEFAULT_DB_PATH)
    # Re-render the complete thread so the composer remains below the newest
    # answer instead of leaving the just-created answer under the form.
    st.rerun()


def _render_composer() -> str | None:
    # Native chat_input is rendered in Streamlit's stable bottom composer
    # layer, avoiding the form being inserted between thread messages.
    st.chat_input("Ask about UTAS research degrees, projects or supervisors...",
                  key="composer_input", on_submit=_queue_chat_input)
    return None


def main() -> None:
    st.set_page_config(page_title="UTAS Research Degree Assistant", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "view" not in st.session_state:
        st.session_state.view = "chat"
    _render_sidebar()
    if st.session_state.view == "about":
        _render_about()
        return
    st.markdown('<div class="workspace-top"><div><div class="workspace-title"><span class="workspace-icon">✦</span><span>UTAS Research Degree Assistant</span></div><div class="workspace-tagline">Explore research projects, supervisors, scholarships and HDR information through grounded AI-powered search.</div></div><span class="assignment-label">welcome to KIT848</span></div>', unsafe_allow_html=True)
    if not st.session_state.messages:
        _render_empty_state()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                content = escape(message["content"]).replace(chr(10), "<br>")
                st.markdown(f'<div class="message-label user-label">You</div><div class="user-bubble">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="message-label assistant-label">Research assistant</div>', unsafe_allow_html=True)
                render_assistant_result(message)
    if st.session_state.messages:
        st.markdown('<div id="latest-message" tabindex="-1"></div><div class="latest-hint">Latest answer ↓</div>', unsafe_allow_html=True)
    # Process queued suggestion submissions before drawing the composer so the
    # transient user/loading state remains part of the thread above it.
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        _process_question(question)
    entered_question = _render_composer()
    if entered_question:
        _process_question(entered_question)


if __name__ == "__main__":
    main()
