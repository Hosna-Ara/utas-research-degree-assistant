"""Connect explicit query plans to hybrid retrieval and retain evidence metadata."""

from dataclasses import dataclass

from utas_research_assistant.query.models import RetrievalPlan
from utas_research_assistant.query.planner import QueryPlanner
from utas_research_assistant.retrieval.hybrid import HybridRetriever


@dataclass
class RetrievalRouter:
    planner: QueryPlanner
    retriever: HybridRetriever

    def route(self, question: str, *, top_k: int = 5) -> dict:
        plan = self.planner.plan(question)
        # A degree mentioned in an HDR information question is not a project filter.
        filters = plan.project_filters() if plan.scope in {"projects", "all"} else {}
        before = len(self.retriever.candidate_indices(plan.scope))
        candidates = self.retriever.candidate_indices(plan.scope, filters or None)
        results = self.retriever.search(plan.search_query, top_k, plan.scope, filters or None)
        return {
            "question": question,
            "plan": plan,
            "candidate_count_before_filters": before,
            "candidate_count_after_filters": len(candidates),
            "results": results,
            "planner_error": self.planner.last_llm_error,
        }
