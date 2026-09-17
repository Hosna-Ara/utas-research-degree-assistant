"""Thin application facade for the existing reasoning and answer pipeline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdflib import Graph

from utas_research_assistant.generation.answer_generator import AnswerGenerator
from utas_research_assistant.generation.models import AnswerResponse
from utas_research_assistant.query.planner import ReasoningPlanner
from utas_research_assistant.query.reasoning_router import ReasoningRouter
from utas_research_assistant.retrieval.corpus import load_corpus
from utas_research_assistant.retrieval.hybrid import HybridRetriever
from utas_research_assistant.retrieval.semantic import SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class AnswerResult:
    """Answer plus router evidence needed for citations and transparent UI panels."""

    response: AnswerResponse
    evidence: dict
    generation_notice: str | None = None


class QuestionAnswerService:
    def __init__(self, router: ReasoningRouter, answer_generator: AnswerGenerator):
        self.router = router
        self.answer_generator = answer_generator

    def answer_with_evidence(self, question: str) -> AnswerResult:
        evidence = self.router.route(question)
        response = self.answer_generator.generate(question, evidence)
        notice = None
        if self.answer_generator.last_error:
            notice = "The local answer model was unavailable or returned an unusable response; a grounded fallback was used."
        return AnswerResult(response=response, evidence=evidence, generation_notice=notice)

    def answer_question(self, question: str) -> AnswerResponse:
        return self.answer_with_evidence(question).response


def create_service(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> QuestionAnswerService:
    """Load immutable local assets once and compose the already-tested pipeline."""
    corpus = load_corpus(processed_dir)
    graph = Graph().parse(processed_dir / "utas_research_graph.ttl", format="turtle")
    semantic = SemanticRetriever(corpus, processed_dir)
    router = ReasoningRouter(ReasoningPlanner(), HybridRetriever(corpus, None, semantic), graph)
    return QuestionAnswerService(router, AnswerGenerator())


@lru_cache(maxsize=1)
def _default_service() -> QuestionAnswerService:
    return create_service()


def answer_question(question: str) -> AnswerResponse:
    """Convenience end-to-end API used by non-UI callers."""
    return _default_service().answer_question(question)
