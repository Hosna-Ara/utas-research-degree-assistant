import sys
from types import SimpleNamespace

import pytest

from utas_research_assistant.retrieval import semantic


@pytest.mark.parametrize('mode, local_only', [('local', True), ('public', False)])
def test_model_loading_policy(monkeypatch, mode, local_only):
    calls = []
    model = object()

    def constructor(name, **kwargs):
        calls.append((name, kwargs))
        if mode == 'public' and kwargs['local_files_only']:
            raise OSError('Model not cached')
        return model

    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', mode)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    assert semantic.load_model(semantic.DEFAULT_MODEL) is model
    assert calls[-1] == (semantic.DEFAULT_MODEL, {'device': 'cpu', 'local_files_only': local_only})
    assert len(calls) == (1 if local_only else 2)


def test_local_build_can_explicitly_download(monkeypatch):
    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'local')
    calls = []
    def constructor(name, **kwargs):
        calls.append(kwargs)
        if kwargs['local_files_only']:
            raise OSError('Model not cached')
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    semantic.load_model(semantic.DEFAULT_MODEL, allow_download=True)
    assert calls[-1]['local_files_only'] is False


@pytest.mark.parametrize('failure', [OSError, ConnectionError, RuntimeError])
def test_public_service_uses_bm25_on_model_failure(monkeypatch, caplog, failure):
    from utas_research_assistant.service import create_service
    from utas_research_assistant.query.planner import OllamaReasoningPlanner

    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        raise failure('download failed: sensitive diagnostic detail')

    monkeypatch.setenv('UTAS_DEPLOYMENT_MODE', 'public')
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=unavailable))
    monkeypatch.setattr(OllamaReasoningPlanner, 'is_available', lambda self: False)
    service = create_service()
    question = 'Find AI & machine learning PhD projects'
    for _ in range(2):
        result = service.answer_with_evidence(question)
        rows = result.evidence['ranked_retrieval_evidence']
        assert rows
        assert not result.response.insufficient_evidence
        assert all(row['semantic_rank'] is None for row in rows)
        assert all(row['bm25_rank'] is not None for row in rows)
        for row in rows:
            assert row['rrf_score'] == pytest.approx(1 / (60 + row['bm25_rank']))
    assert len(calls) == 2
    assert calls[-1]['local_files_only'] is False
    warnings = [r for r in caplog.records if r.name == semantic.__name__]
    assert len(warnings) == 1
    assert 'BM25-only' in warnings[0].message
    assert warnings[0].exc_info is None
    assert 'sensitive diagnostic detail' not in caplog.text
