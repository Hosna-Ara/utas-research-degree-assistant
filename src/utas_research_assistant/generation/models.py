"""Public response contract for generated answers."""

from pydantic import BaseModel, ConfigDict, StrictBool


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    reasoning_method: str
    planner_method: str
    citations: list[str]
    sources: list[dict]
    tool_used: str | None = None
    tool_result: str | None = None
    insufficient_evidence: StrictBool
    project_ids: list[str]
    generation_model: str
