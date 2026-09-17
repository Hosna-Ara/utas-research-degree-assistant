"""Exercise actual chat callbacks, cached construction, routing and generation."""
import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import utas_research_assistant as package
from utas_research_assistant import chat_history, service as service_module
from utas_research_assistant.query import planner as planner_module
from utas_research_assistant.generation.answer_generator import OllamaAnswerProvider

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    ("What type of applicant is Applicant B?", "LOCAL-BUSINESS-PERSONA-02"),
    ("What does my private applicant profile say about my research background?", "LOCAL-HOSNA-PROFILE-01"),
    ("What does my private project decision journal say?", "LOCAL-DECISION-JOURNAL-07"),
    ("Applicant B is the business-focused synthetic persona in my local documents. Summarise Applicant B's background and research preferences.", "LOCAL-BUSINESS-PERSONA-02"),
    ("Using my private applicant profile, summarise my background and main research interests.", "LOCAL-HOSNA-PROFILE-01"),
]


@pytest.mark.parametrize('model_available', [True, False])
def test_fresh_streamlit_messages(monkeypatch, tmp_path, model_available):
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'local')
    monkeypatch.delenv('UTAS_DATA_DIR', raising=False)
    monkeypatch.setattr(chat_history, 'DEFAULT_DB_PATH', tmp_path / 'history.sqlite3')
    monkeypatch.setattr(planner_module.OllamaReasoningPlanner, 'is_available', lambda self: model_available)
    # Reproduce the valid but incorrect scope returned by live Qwen. Keep the
    # real planner, router, hybrid retriever, corpus and AnswerGenerator.
    monkeypatch.setattr(planner_module.OllamaReasoningPlanner, 'generate', lambda self, question: json.dumps({
        'method': 'retrieval', 'scope': 'general', 'search_query': 'applicant background research',
        'intent': 'applicant_profile_summary', 'planner_method': 'llm',
    }))
    monkeypatch.setattr(OllamaAnswerProvider, 'generate', lambda self, prompt: json.dumps({
        'answer': '', 'insufficient_evidence': True,
    }))
    plans = []
    original_plan = planner_module.ReasoningPlanner.plan

    def capture_plan(self, question):
        plan = original_plan(self, question)
        plans.append(plan)
        return plan

    monkeypatch.setattr(planner_module.ReasoningPlanner, 'plan', capture_plan)
    calls = []
    original = service_module.QuestionAnswerService.answer_with_evidence

    def capture(self, question):
        result = original(self, question)
        calls.append((self, question, result, dict(self.router.planner.last_diagnostics)))
        return result

    monkeypatch.setattr(service_module.QuestionAnswerService, 'answer_with_evidence', capture)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        assert not app.exception
        for index, (question, expected_id) in enumerate(QUESTIONS, 1):
            app.button(key='new_chat').click().run()
            app.chat_input[0].set_value(question).run()
            assert not app.exception
            assert len(calls) == index  # A fresh message calls the current service exactly once.
            service, submitted, result, diagnostics = calls[-1]
            assert service is calls[0][0]  # One cached UI service across fresh chats.
            assert submitted == question
            assert service.processed_dir == ROOT / 'data/processed'
            local_ids = {item.metadata['document_id'] for item in service.router.retriever.corpus.items
                         if item.item_type == 'local_document'}
            assert len(local_ids) == 10
            assert {item[1] for item in QUESTIONS} <= local_ids
            assert len(plans) == index
            plan = plans[-1]
            assert plan.intent == 'local_document_lookup'
            assert plan.scope == 'all'
            assert plan.search_query == question
            evidence = result.evidence
            assert evidence['interpreted_intent'] == plan.intent
            assert evidence['scope'] == 'all'
            assert evidence['candidate_count_after'] == len(service.router.retriever.corpus.items)
            top5 = [(row['item_type'], row.get('document_id'))
                    for row in evidence['ranked_retrieval_evidence'][:5]]
            assert top5[0] == ('local_document', expected_id)
            assert not result.response.insufficient_evidence
            assert 'active knowledge base does not contain' not in result.response.answer
            assert app.session_state.messages[-1]['response'] == result.response.model_dump(mode='json')
            app.run()  # Rendering a stored answer must not call the service again.
            assert len(calls) == index
            if model_available:
                assert diagnostics['raw_model_response']
                assert not diagnostics['qwen_plan_accepted']
                assert 'local-document routing' in diagnostics['reason_for_fallback']
        assert Path(package.__file__).resolve() == ROOT / 'src/utas_research_assistant/__init__.py'
        assert Path(service_module.__file__).resolve() == ROOT / 'src/utas_research_assistant/service.py'
        assert Path(planner_module.__file__).resolve() == ROOT / 'src/utas_research_assistant/query/planner.py'
        assert Path.cwd() == ROOT
    finally:
        st.cache_resource.clear()
