"""Synthetic CVs only; exercise extraction, scoring, privacy and real Streamlit turns."""
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from rdflib import Graph

from utas_research_assistant.applicant import ApplicantProfile, extract_profile, extract_upload, MAX_BYTES
from utas_research_assistant.personalisation import ConversationContext, Personalisation, score_project, suggestions
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.service import QuestionAnswerService, session_answer
from utas_research_assistant.query.reasoning_router import ReasoningRouter
from utas_research_assistant.generation.answer_generator import AnswerGenerator

ROOT = Path(__file__).resolve().parents[1]
CV = '''Education:
Bachelor of Computer Science
Research interests: machine learning; artificial intelligence; cybersecurity
Skills: Python; SQL; data analytics
Methods: quantitative; simulation
Projects:
Built a machine learning classifier in Python
Location preference: Hobart
Degree preference: PhD
'''

class OfflineSemantic:
    def search(self, *args, **kwargs):
        return []

@pytest.fixture
def service():
    corpus = load_corpus(ROOT / 'deployment_data')
    graph = Graph().parse(ROOT / 'deployment_data/utas_research_graph.ttl', format='turtle')
    return QuestionAnswerService(ReasoningRouter(None, HybridRetriever(corpus, None, OfflineSemantic()), graph), AnswerGenerator(), ROOT / 'deployment_data')


def pdf_bytes(text=CV):
    with fitz.open() as doc:
        doc.new_page().insert_text((50, 50), text)
        return doc.tobytes()


def docx_bytes(text=CV):
    from xml.sax.saxutils import escape
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as doc:
        doc.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + ''.join('<w:p><w:r><w:t>' + escape(line) + '</w:t></w:r></w:p>' for line in text.splitlines()) + '</w:body></w:document>')
    return data.getvalue()


def test_pdf_and_docx_extraction():
    assert 'Bachelor of Computer Science' in extract_upload('cv.pdf', pdf_bytes())
    assert extract_upload('cv.docx', docx_bytes()) == CV.strip()

@pytest.mark.parametrize('name,data', [('x.pdf', b''), ('x.docx', b'bad'), ('x.pdf', b'bad'), ('x.txt', b'text'), ('x.pdf', b'x' * (MAX_BYTES + 1)), ('x.pdf', pdf_bytes(''))])
def test_invalid_empty_large_upload(name, data):
    with pytest.raises(ValueError):
        extract_upload(name, data)


def test_deterministic_profile_evidence_and_preferences():
    profile = extract_profile(CV + '\nContact:\nsynthetic@example.invalid\n')
    assert 'Python' in profile.values('programming_tools')
    assert profile.values('location_preferences') == ['Hobart']
    assert not profile.values('student_type', 'funding_preference', 'publications')
    assert 'example.invalid' not in profile.model_dump_json()
    assert not extract_profile('Bachelor in Hobart. International marketing degree.').values('location_preferences', 'student_type')


def test_optional_qwen_quotes_and_public_no_network(monkeypatch):
    calls = []
    def post(self, url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'response': json.dumps({'research_interests': ['machine learning', 'invented subject'], 'student_type': ['international']})})
    monkeypatch.setattr('utas_research_assistant.applicant.requests.Session.post', post)
    profile = extract_profile(CV, use_qwen=True)
    assert calls[0][0].startswith('http://127.0.0.1:')
    assert 'invented subject' not in profile.model_dump_json()
    assert not profile.values('student_type')
    calls.clear()
    extract_profile(CV, use_qwen=True, public=True)
    assert not calls
    monkeypatch.setattr('utas_research_assistant.applicant.requests.Session.post', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unavailable')))
    assert extract_profile(CV, use_qwen=True) == extract_profile(CV)


def test_score_missing_evidence_and_reproducibility():
    item = SimpleNamespace(metadata={'description': 'Machine learning using Python and interviews'}, text='Machine learning using Python and interviews')
    p = extract_profile('Research interests: machine learning\nSkills: Python')
    result = score_project(p, item)
    assert result == score_project(p, item)
    assert result['coverage'] == 55
    assert result['score'] == 100
    assert result['breakdown']['Supervisor fit']['points'] is None
    assert any('interviews' in s and 'not proof' in s for s in result['unknowns'])
    assert result['gaps'] == []
    empty = score_project(ApplicantProfile(), item)
    assert empty['score'] == empty['coverage'] == 0


def test_project_supervisor_matching_and_followups(service):
    engine, profile, ctx = Personalisation(service), extract_profile(CV), ConversationContext()
    result = engine.answer('Find the 3 UTAS research projects most aligned with my CV', profile, ctx)
    assert len(result.response.project_ids) == 3
    assert len(result.evidence['recommendations']) == 3
    assert all(r['coverage'] <= 100 for r in result.evidence['recommendations'])
    assert all(s['source_url'].startswith('https://') for s in result.response.sources)
    first = ctx.project_ids[0]
    assert engine.resolve('project 1', ctx) == [first]
    assert engine.resolve('this project', ctx) == [first]
    sup = engine.answer('Show me the supervisor for Project 1', profile, ctx)
    assert sup.response.sources[0]['item_type'] == 'supervisor_profile'
    assert ctx.stage == 'supervisor'
    interests = engine.answer("Show this supervisor's research interests", profile, ctx)
    assert 'Research interests:' in interests.response.answer
    assert engine.answer('What gaps do I have for this project?', profile, ctx).response.project_ids == [first]
    compared = engine.answer('Compare project 1 and project 2 for my background', profile, ctx)
    assert len(compared.response.project_ids) == 2
    ranked = engine.answer('Which supervisors are most aligned with my CV?', profile, ctx)
    assert 3 <= len(ranked.response.sources) <= 5
    sid = ctx.supervisor_ids[0]
    engine.resolve('the first supervisor', ctx)
    assert ctx.active_supervisor == sid
    assert 'Upload' in engine.answer('What gaps do I have for this project?', None, ctx).response.answer


def test_dynamic_suggestions():
    ctx = ConversationContext(project_ids=['1', '2'], active_project='1', stage='projects')
    project = suggestions('Find matches', context=ctx, has_profile=True)
    ctx.stage = 'supervisor'
    supervisor = suggestions('Show supervisor', context=ctx, has_profile=True)
    research = suggestions("Show this supervisor's research interests", context=ctx, has_profile=True)
    assert project != supervisor != research
    for choices in (project, supervisor, research, suggestions('funding'), suggestions('entry requirements'), suggestions()):
        assert 3 <= len(choices) <= 4
        assert len(choices) == len(set(choices))
    assert 'Show me the supervisor for Project 1' in project


def test_session_general_queries_keep_graph_and_guards(service, monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **k: pytest.fail('Network model called'))
    result = session_answer(service, 'Which projects does Soonja Yeom supervise?')
    assert result.response.tool_used == 'SPARQL Knowledge Graph'
    result = session_answer(service, 'What English score do I need for a PhD?')
    assert result.response.citations


@pytest.mark.parametrize('mode', ['local', 'public'])
def test_streamlit_cv_acceptance_privacy_and_rerun(service, monkeypatch, tmp_path, mode):
    from utas_research_assistant import service as module, chat_history
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', mode)
    monkeypatch.setattr(module, 'create_service', lambda: service)
    db = tmp_path / 'chat_history.db'
    monkeypatch.setattr(chat_history, 'DEFAULT_DB_PATH', db)
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **k: pytest.fail('CV session contacted a model'))
    # AppTest lacks an upload setter. Supply an in-memory file beside the real
    # sidebar uploader, then exercise actual Save, actions and chat callbacks.
    script = '''
import io
import streamlit as st
import fitz
from utas_research_assistant.ui.applicant import stage_upload
import app
if not st.session_state.get('seeded'):
    with fitz.open() as doc:
        doc.new_page().insert_text((50, 50), 'Education: Bachelor of Computer Science\\nResearch interests: machine learning\\nSkills: Python')
        upload = io.BytesIO(doc.tobytes())
    upload.name = 'synthetic.pdf'
    st.session_state.cv_upload_0 = upload
    stage_upload()
    st.session_state.seeded = True
from unittest.mock import patch
original_uploader = st.file_uploader
def selected_upload(*args, **kwargs):
    original_uploader(*args, **{**kwargs, 'key': kwargs['key'] + '_display'})
    return st.session_state.get(kwargs['key'])
with patch('streamlit.file_uploader', selected_upload):
    app.main()
'''
    st.cache_resource.clear()
    try:
        app = AppTest.from_string(script, default_timeout=90).run()
        assert not app.exception
        assert 'applicant_profile' not in app.session_state
        assert len(app.sidebar.get('file_uploader')) == 1
        assert len(app.main.get('file_uploader')) == 0
        assert not app.sidebar.button(key='save_cv').disabled
        app.run()
        assert 'applicant_profile' not in app.session_state
        app.sidebar.button(key='save_cv').click().run()
        assert not app.exception
        assert app.session_state.applicant_profile
        assert app.sidebar.success
        assert not app.main.text_input
        assert len(app.sidebar.button) >= 5
        app.button(key='match_cv').click().run()
        assert not app.exception
        messages = app.session_state.messages
        assert len(messages[-1]['response']['project_ids']) == 3
        first = messages[-1]['response']['project_ids'][0]
        choices = messages[-1]['suggestions']
        app.run()
        assert app.session_state.messages[-1]['suggestions'] == choices
        assert len(app.session_state.messages) == 2
        app.button(key='next_question_0').click().run()
        assert not app.exception
        assert len(app.session_state.messages) == 4
        assert app.session_state.messages[-2]['content'] == 'Show me the supervisor for Project 1'
        assert 'Research interests:' in app.session_state.messages[-1]['response']['answer']
        app.button(key='next_question_0').click().run()
        assert not app.exception
        assert app.session_state.messages[-1]['suggestions'] != choices
        app.chat_input[0].set_value('What gaps do I have for this project?').run()
        assert app.session_state.messages[-1]['response']['project_ids'] == [first]
        app.sidebar.button(key='match_cv_supervisors').click().run()
        assert app.session_state.messages[-1]['evidence']['supervisor_ids']
        app.sidebar.button(key='match_cv_gaps').click().run()
        assert len(app.session_state.messages[-1]['response']['project_ids']) == 3
        app.sidebar.button(key='match_cv_best').click().run()
        assert app.session_state.messages[-1]['response']['project_ids']
        app.button(key='clear_cv').click().run()
        assert not app.exception
        assert not app.sidebar.success
        assert not any(b.key == 'match_cv' for b in app.sidebar.button)
        assert app.sidebar.button(key='save_cv').disabled
        assert 'applicant_profile' not in app.session_state
        app.chat_input[0].set_value('Find projects aligned with my CV').run()
        assert 'Upload' in app.session_state.messages[-1]['response']['answer']
        if db.exists():
            import sqlite3
            with sqlite3.connect(db) as conn:
                assert conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 0
    finally:
        st.cache_resource.clear()


def test_interests_are_not_skills_and_preferences_are_not_qualifications():
    profile = extract_profile('Research interests: machine learning and Python\nDegree preference: PhD\nLocation preference: Hobart')
    assert not profile.values('technical_skills', 'programming_tools', 'degrees', 'academic_disciplines')
    assert profile.values('research_interests')


def test_partial_constraint_evidence_and_aliases():
    profile = extract_profile('Degree preference: Doctor of Philosophy\nLocation preference: Hobart\nFunding preference: scholarship')
    item = SimpleNamespace(metadata={'degree_types': ['PhD'], 'location': 'Hobart', 'funding_status': 'unknown'}, text='')
    result = score_project(profile, item)
    assert result['coverage'] == 10  # Two of three explicit constraints supported.
    assert result['score'] == 100
    assert result['gaps'] == []
    assert any('funding status' in u for u in result['unknowns'])
    profile = extract_profile('Funding preference: self-funded')
    assert score_project(profile, item)['coverage'] == 0


def test_related_projects_and_invalid_references(service):
    engine, ctx, profile = Personalisation(service), ConversationContext(), extract_profile(CV)
    assert 'cannot resolve' in engine.answer('Show project 9', None, ctx).response.answer
    engine.answer('Find 3 projects aligned with my CV', profile, ctx)
    engine.answer('Show the supervisor for project 1', profile, ctx)
    sid = ctx.active_supervisor
    related = engine.answer('Which projects does this supervisor lead?', profile, ctx)
    assert related.response.project_ids == engine.related_projects(sid)[:5]
    ranked = engine.answer('Which supervisors are most aligned with my CV?', profile, ctx)
    assert len([s for s in ranked.response.sources if s['item_type'] == 'supervisor_profile']) == 5
    assert 'select a project' in engine.answer('What gaps do I have for this project?', profile, ctx).response.answer


@pytest.mark.parametrize('mode,local_count', [('public', 0), ('local', 10)])
def test_real_runtime_cv_matching_without_ollama_and_semantic_model(monkeypatch, mode, local_count):
    from utas_research_assistant.service import create_service
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', mode)
    monkeypatch.setattr('utas_research_assistant.retrieval.semantic.load_model', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    monkeypatch.setattr('requests.sessions.Session.request', lambda *a, **k: pytest.fail('Unexpected external request'))
    runtime = create_service()
    assert runtime.processed_dir.name == ('deployment_data' if mode == 'public' else 'processed')
    corpus = runtime.router.retriever.corpus
    assert len({i.metadata['document_id'] for i in corpus.items if i.item_type == 'local_document'}) == local_count
    result = Personalisation(runtime).answer('Find 5 projects aligned with my CV', extract_profile(CV), ConversationContext())
    assert len(result.response.project_ids) == 5
    assert all(s['item_type'] == 'research_project' for s in result.response.sources)
    # Existing graph and private-document guards remain usable in a CV session.
    private = session_answer(runtime, 'What type of applicant is Applicant B?')
    assert private.response.insufficient_evidence == (mode == 'public')


def test_cv_pipeline_does_not_write_log_or_modify_shared_service(service, monkeypatch, caplog):
    import builtins
    original_open = builtins.open
    def guarded_open(file, mode='r', *args, **kwargs):
        if any(c in mode for c in 'wax+'):
            pytest.fail('CV pipeline attempted a file write')
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    marker = 'SESSION_ONLY_SYNTHETIC_SENTINEL_8821'
    text = extract_upload('synthetic.docx', docx_bytes(CV + '\nProjects: ' + marker))
    profile = extract_profile(text)
    Personalisation(service).answer('Find 3 projects aligned with my CV', profile, ConversationContext())
    assert marker not in caplog.text
    assert marker not in repr(vars(service))
    assert marker not in repr(vars(service.router))
    assert marker not in repr(vars(service.router.retriever))
    assert not hasattr(service, 'applicant_profile')


def test_qwen_cannot_follow_redirects_or_use_proxies(monkeypatch):
    def post(self, url, **kwargs):
        assert self.trust_env is False
        assert kwargs['allow_redirects'] is False
        assert kwargs['json']['format']['additionalProperties'] is False
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'response': '{}'})
    monkeypatch.setattr('requests.Session.post', post)
    assert extract_profile(CV, use_qwen=True) == extract_profile(CV)


def test_reupload_invalid_file_and_new_session(monkeypatch, service):
    from utas_research_assistant import service as module
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    monkeypatch.setattr(module, 'create_service', lambda: service)
    script = '''
import io
import streamlit as st
import app
from utas_research_assistant.ui.applicant import stage_upload
if st.session_state.get('test_upload') is not None:
    name, data = st.session_state.pop('test_upload')
    upload = io.BytesIO(data)
    upload.name = name
    st.session_state.cv_upload_0 = upload
    stage_upload()
from unittest.mock import patch
original_uploader = st.file_uploader
def selected_upload(*args, **kwargs):
    original_uploader(*args, **{**kwargs, 'key': kwargs['key'] + '_display'})
    return st.session_state.get(kwargs['key'])
with patch('streamlit.file_uploader', selected_upload):
    app.main()
'''
    st.cache_resource.clear()
    try:
        app = AppTest.from_string(script).run()
        assert 'applicant_profile' not in app.session_state
        app.session_state['test_upload'] = ('first.pdf', pdf_bytes())
        app.run()
        assert 'applicant_profile' not in app.session_state
        app.sidebar.button(key='save_cv').click().run()
        assert app.session_state.applicant_profile.values('programming_tools')
        app.session_state['test_upload'] = ('second.docx', docx_bytes('Education: Bachelor of Biology\nResearch interests: ecology'))
        app.run()
        assert 'Python' in app.session_state.applicant_profile.model_dump_json()
        app.sidebar.button(key='save_cv').click().run()
        assert 'Python' not in app.session_state.applicant_profile.model_dump_json()
        app.session_state['test_upload'] = ('bad.pdf', b'bad')
        app.run()
        app.sidebar.button(key='save_cv').click().run()
        assert 'applicant_profile' not in app.session_state
        assert app.warning
        fresh = AppTest.from_string(script).run()
        assert 'applicant_profile' not in fresh.session_state
        assert not fresh.session_state.messages
    finally:
        st.cache_resource.clear()


def test_no_source_method_evidence_does_not_inflate_coverage():
    profile = extract_profile('Research interests: ecology\nSkills: Python')
    item = SimpleNamespace(metadata={'description': 'Ecology and coastal habitats'}, text='Ecology and coastal habitats')
    score = score_project(profile, item)
    assert score['coverage'] == 30
    assert score['breakdown']['Skills/methods']['points'] is None


def test_negated_experience_is_not_a_skill_even_from_qwen(monkeypatch):
    text = 'Skills: No experience in Python'
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'response': json.dumps({'technical_skills': ['Python']})})
    monkeypatch.setattr('requests.Session.post', lambda *a, **k: response)
    profile = extract_profile(text, use_qwen=True)
    assert not profile.values('technical_skills', 'programming_tools')
    assert profile.values('additional_constraints')


def test_ordinary_named_project_preserves_existing_router(service):
    engine = Personalisation(service)
    pid = next(iter(engine.projects))
    assert engine.answer(f'Tell me about project {pid}', None, ConversationContext()) is None


def test_sidebar_status_does_not_expose_profile_content(monkeypatch):
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    script = '''
import streamlit as st
from utas_research_assistant.applicant import ApplicantProfile
from utas_research_assistant.ui.applicant import render_cv
st.session_state.applicant_profile = ApplicantProfile(facts={'research_interests': ['![tracker](https://example.invalid/private-cv)']})
with st.sidebar:
    render_cv()
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not any('![tracker]' in t.value for t in [*app.text, *app.markdown, *app.caption])
    assert app.sidebar.success
    assert not app.main.get('file_uploader')
    assert not app.text_input
