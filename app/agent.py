"""Grafo LangGraph seguro para investigar divergencias sem executar financeiro."""
from __future__ import annotations

import operator
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session
from typing_extensions import TypedDict

from app import config
from app.agent_tools import (
    AgentToolContext,
    AgentToolError,
    InvestigationRecommendation,
    get_audit_trail,
    get_document_context,
    get_group,
    get_reconciliation,
    get_transaction,
    search_similar_cases,
)

ALLOWED_TOOLS = frozenset(
    {
        "get_reconciliation",
        "get_transaction",
        "get_group",
        "search_similar_cases",
        "get_audit_trail",
        "get_document_context",
    }
)


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)


class PlanDecision(BaseModel):
    action: Literal["TOOLS", "COMPOSE"]
    tool_calls: list[ToolRequest] = Field(default_factory=list, max_length=8)
    usage: ModelUsage


class ComposeDecision(BaseModel):
    recommendation: InvestigationRecommendation
    usage: ModelUsage


class ModelGateway(Protocol):
    def plan(
        self, state: InvestigationState, available_tools: tuple[str, ...]
    ) -> PlanDecision: ...

    def compose(self, state: InvestigationState) -> ComposeDecision: ...


@dataclass(frozen=True, slots=True)
class AgentBudget:
    max_cycles: int = config.AGENT_MAX_CYCLES
    max_tool_calls: int = config.AGENT_MAX_TOOL_CALLS
    max_input_tokens: int = config.AGENT_MAX_INPUT_TOKENS
    max_output_tokens: int = config.AGENT_MAX_OUTPUT_TOKENS
    max_cost_usd: Decimal = config.AGENT_MAX_COST_USD


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    session_factory: Any
    model_gateway: ModelGateway
    scope: AgentToolContext
    budget: AgentBudget = AgentBudget()


class InvestigationState(TypedDict, total=False):
    run_id: str
    thread_id: str
    actor_id: str
    daily_reconciliation_id: int | None
    reconciliation_group_id: int | None
    bank_account_id: int
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    planned_tool_calls: list[dict[str, Any]]
    attempt: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: str | None
    cost_complete: bool
    prompt_version: str
    model: str
    retrieval_version: str
    recommendation: dict[str, Any] | None
    human_feedback: dict[str, Any] | None
    status: str
    errors: Annotated[list[dict[str, str]], operator.add]
    safety_violations: Annotated[list[str], operator.add]


class TransactionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: int = Field(gt=0)


class GroupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_id: int = Field(gt=0)


class ReconciliationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reconciliation_id: int = Field(gt=0)


class SimilarCasesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class AuditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_type: str = Field(min_length=1, max_length=40)
    entity_id: int = Field(gt=0)
    limit: int = Field(default=50, ge=1, le=100)


class DocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_id: int = Field(gt=0)
    group_item_id: int = Field(gt=0)


def initial_state(
    *,
    run_id: str,
    thread_id: str,
    actor_id: str,
    scope: AgentToolContext,
    model: str = config.AGENT_MODEL,
    prompt_version: str = config.AGENT_PROMPT_VERSION,
    retrieval_version: str = config.AGENT_RETRIEVAL_VERSION,
) -> InvestigationState:
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "actor_id": actor_id,
        "daily_reconciliation_id": scope.daily_reconciliation_id,
        "reconciliation_group_id": scope.reconciliation_group_id,
        "bank_account_id": scope.bank_account_id,
        "messages": [],
        "tool_calls": [],
        "evidence": [],
        "planned_tool_calls": [],
        "attempt": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": "0",
        "cost_complete": True,
        "prompt_version": prompt_version,
        "model": model,
        "retrieval_version": retrieval_version,
        "recommendation": None,
        "human_feedback": None,
        "status": "RUNNING",
        "errors": [],
        "safety_violations": [],
    }


def _usage_update(
    state: InvestigationState, usage: ModelUsage, budget: AgentBudget
) -> dict[str, Any]:
    input_tokens = state.get("input_tokens", 0) + usage.input_tokens
    output_tokens = state.get("output_tokens", 0) + usage.output_tokens
    current_cost = Decimal(state.get("estimated_cost_usd") or 0)
    cost_complete = state.get("cost_complete", True)
    if usage.estimated_cost_usd is None:
        cost_complete = False
        cost = current_cost
    else:
        cost = current_cost + usage.estimated_cost_usd
    exceeded = (
        input_tokens > budget.max_input_tokens
        or output_tokens > budget.max_output_tokens
        or (cost_complete and cost > budget.max_cost_usd)
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": str(cost) if cost_complete or cost else None,
        "cost_complete": cost_complete,
        "status": "BUDGET_EXCEEDED" if exceeded else state.get("status", "RUNNING"),
    }


def _evidence(source_type: str, source_id: str, payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return {"source_type": source_type, "source_id": source_id, "payload": payload}


def _load_case(
    state: InvestigationState, runtime: Runtime[AgentRuntimeContext]
) -> dict[str, Any]:
    scope = runtime.context.scope
    if (
        state.get("bank_account_id") != scope.bank_account_id
        or state.get("daily_reconciliation_id") != scope.daily_reconciliation_id
        or state.get("reconciliation_group_id") != scope.reconciliation_group_id
    ):
        return {
            "status": "FAILED",
            "safety_violations": ["runtime_scope_mismatch"],
            "errors": [{"code": "SCOPE_MISMATCH", "message": "Caso fora do runtime"}],
        }
    with runtime.context.session_factory() as session:
        if scope.daily_reconciliation_id is not None:
            record = get_reconciliation(session, scope.daily_reconciliation_id, scope)
            evidence = [_evidence("reconciliation", f"rec:{record.id}", record)]
        else:
            group = get_group(session, scope.reconciliation_group_id, scope)
            evidence = [_evidence("group", f"group:{group.id}", group)]
    return {"evidence": evidence, "messages": [{"role": "system", "event": "case_loaded"}]}


def _plan(
    state: InvestigationState, runtime: Runtime[AgentRuntimeContext]
) -> dict[str, Any]:
    if state.get("status") != "RUNNING":
        return {}
    if state.get("attempt", 0) >= runtime.context.budget.max_cycles:
        return {
            "status": "BUDGET_EXCEEDED",
            "errors": [{"code": "MAX_CYCLES", "message": "Limite de ciclos atingido"}],
        }
    decision = runtime.context.model_gateway.plan(state, tuple(sorted(ALLOWED_TOOLS)))
    update = _usage_update(state, decision.usage, runtime.context.budget)
    update["attempt"] = state.get("attempt", 0) + 1
    if update["status"] == "BUDGET_EXCEEDED":
        update["planned_tool_calls"] = []
        return update
    update["planned_tool_calls"] = [call.model_dump(mode="json") for call in decision.tool_calls]
    update["messages"] = [{"role": "assistant", "event": decision.action.lower()}]
    if decision.action == "TOOLS" and not decision.tool_calls:
        update["status"] = "FAILED"
        update["errors"] = [{"code": "EMPTY_TOOL_PLAN", "message": "Plano sem tools"}]
    return update


def _execute_tool(
    session: Session,
    request: ToolRequest,
    scope: AgentToolContext,
) -> list[dict[str, Any]]:
    args = request.arguments
    if request.name == "get_reconciliation":
        parsed = ReconciliationArgs.model_validate(args)
        result = get_reconciliation(session, parsed.reconciliation_id, scope)
        return [_evidence("reconciliation", f"rec:{result.id}", result)]
    if request.name == "get_transaction":
        parsed = TransactionArgs.model_validate(args)
        result = get_transaction(session, parsed.transaction_id, scope)
        return [_evidence("transaction", f"tx:{result.id}", result)]
    if request.name == "get_group":
        parsed = GroupArgs.model_validate(args)
        result = get_group(session, parsed.group_id, scope)
        return [_evidence("group", f"group:{result.id}", result)]
    if request.name == "search_similar_cases":
        parsed = SimilarCasesArgs.model_validate(args)
        result = search_similar_cases(session, parsed.query, scope, limit=parsed.limit)
        evidence = [
            _evidence("memory", f"mem:{case.memory_id}", case)
            for case in result.cases
        ]
        if result.degraded:
            evidence.append(_evidence("memory", "memory:degraded", {"degraded": True}))
        return evidence
    if request.name == "get_audit_trail":
        parsed = AuditArgs.model_validate(args)
        result = get_audit_trail(
            session,
            parsed.entity_type,
            parsed.entity_id,
            scope,
            limit=parsed.limit,
        )
        return [_evidence("audit", f"audit:{event.id}", event) for event in result]
    if request.name == "get_document_context":
        parsed = DocumentArgs.model_validate(args)
        result = get_document_context(
            session, parsed.group_id, parsed.group_item_id, scope
        )
        return [_evidence("document", f"doc:{result.document_id}", result)]
    raise ValueError("Ferramenta fora da allowlist")


def _tools(
    state: InvestigationState, runtime: Runtime[AgentRuntimeContext]
) -> dict[str, Any]:
    planned = [ToolRequest.model_validate(call) for call in state.get("planned_tool_calls", [])]
    current_count = len(state.get("tool_calls", []))
    if current_count + len(planned) > runtime.context.budget.max_tool_calls:
        return {
            "status": "BUDGET_EXCEEDED",
            "errors": [{"code": "MAX_TOOL_CALLS", "message": "Limite de tools atingido"}],
            "planned_tool_calls": [],
        }
    forbidden = [call.name for call in planned if call.name not in ALLOWED_TOOLS]
    if forbidden:
        return {
            "status": "FAILED",
            "safety_violations": [f"forbidden_tool:{name}" for name in forbidden],
            "errors": [{"code": "FORBIDDEN_TOOL", "message": name} for name in forbidden],
            "planned_tool_calls": [],
        }

    evidence: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with runtime.context.session_factory() as session:
        for request in planned:
            try:
                tool_evidence = _execute_tool(session, request, runtime.context.scope)
                evidence.extend(tool_evidence)
                calls.append({"name": request.name, "status": "COMPLETED"})
            except (AgentToolError, ValidationError, ValueError) as exc:
                calls.append({"name": request.name, "status": "FAILED"})
                errors.append({"code": exc.__class__.__name__, "message": str(exc)[:500]})
    return {
        "evidence": evidence,
        "tool_calls": calls,
        "errors": errors,
        "planned_tool_calls": [],
    }


def _grade_evidence(state: InvestigationState) -> dict[str, Any]:
    if state.get("status") != "RUNNING":
        return {}
    return {
        "messages": [
            {
                "role": "system",
                "event": "evidence_graded",
                "evidence_count": len(state.get("evidence", [])),
            }
        ]
    }


def _compose(
    state: InvestigationState, runtime: Runtime[AgentRuntimeContext]
) -> dict[str, Any]:
    if state.get("status") != "RUNNING":
        return {}
    decision = runtime.context.model_gateway.compose(state)
    update = _usage_update(state, decision.usage, runtime.context.budget)
    if update["status"] == "BUDGET_EXCEEDED":
        return update
    recommendation = decision.recommendation
    available = {item["source_id"] for item in state.get("evidence", [])}
    cited = {item.source_id for item in recommendation.evidence}
    claim_citations = {
        evidence_id
        for claim in recommendation.claims
        for evidence_id in claim.evidence_ids
    }
    if not cited or not cited <= available or not claim_citations <= cited:
        return {
            **update,
            "status": "FAILED",
            "safety_violations": ["ungrounded_recommendation"],
            "errors": [
                {"code": "UNGROUNDED", "message": "Citacao fora das evidencias recuperadas"}
            ],
        }
    return {
        **update,
        "recommendation": recommendation.model_dump(mode="json"),
        "status": "RECOMMENDATION_READY",
    }


def _human_review(state: InvestigationState) -> dict[str, Any]:
    # Task 5 troca este marcador por interrupt() persistente e tipado.
    if state.get("status") != "RECOMMENDATION_READY":
        return {}
    return {"status": "WAITING_HUMAN"}


def _record_feedback(state: InvestigationState) -> dict[str, Any]:
    # Node declarado desde ja; Task 5 o liga ao resume idempotente.
    return {"status": "COMPLETED" if state.get("human_feedback") else state["status"]}


def _route_after_plan(state: InvestigationState) -> str:
    if state.get("status") != "RUNNING":
        return END
    return "tools" if state.get("planned_tool_calls") else "compose"


def _route_after_grade(state: InvestigationState) -> str:
    return END if state.get("status") != "RUNNING" else "plan"


def build_graph(*, checkpointer=None):
    builder = StateGraph(InvestigationState, context_schema=AgentRuntimeContext)
    builder.add_node("load_case", _load_case)
    builder.add_node("plan", _plan)
    builder.add_node("tools", _tools)
    builder.add_node("grade_evidence", _grade_evidence)
    builder.add_node("compose", _compose)
    builder.add_node("human_review", _human_review)
    builder.add_node("record_feedback", _record_feedback)
    builder.add_edge(START, "load_case")
    builder.add_edge("load_case", "plan")
    builder.add_conditional_edges("plan", _route_after_plan, ["tools", "compose", END])
    builder.add_edge("tools", "grade_evidence")
    builder.add_conditional_edges("grade_evidence", _route_after_grade, ["plan", END])
    builder.add_edge("compose", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)
