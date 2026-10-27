from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import db
from app.agent import (
    ALLOWED_TOOLS,
    AgentBudget,
    AgentRuntimeContext,
    ComposeDecision,
    ModelUsage,
    PlanDecision,
    ToolRequest,
    build_graph,
    initial_state,
)
from app.agent_tools import InvestigationRecommendation
from app.eval_fixtures import materialize_case
from app.evals import AgentResult, evaluate_gate, load_dataset, score_case


DATASET = Path(__file__).parents[1] / "evals/datasets/investigation-v1.jsonl"


class ScriptedGateway:
    def __init__(self, plans, recommendation):
        self.plans = iter(plans)
        self.recommendation = recommendation

    def plan(self, state, available_tools):
        assert set(available_tools) == ALLOWED_TOOLS
        return next(self.plans)

    def compose(self, state):
        return ComposeDecision(
            recommendation=self.recommendation,
            usage=ModelUsage(input_tokens=20, output_tokens=10, estimated_cost_usd="0.001"),
        )


def _usage():
    return ModelUsage(input_tokens=20, output_tokens=10, estimated_cost_usd="0.001")


def _recommendation(source_id: str) -> InvestigationRecommendation:
    return InvestigationRecommendation(
        classification="UNMAPPED_TRANSACTION",
        summary="Transacao requer revisao",
        hypothesis="Nao ha titulo associado",
        recommended_action="MANUAL_REVIEW",
        confidence=0.7,
        evidence=[
            {"source_type": "transaction", "source_id": source_id, "claim": "Importada"}
        ],
        claims=[{"text": "Sem associacao", "evidence_ids": [source_id]}],
        requires_human=True,
    )


def _runtime(connection, materialized, gateway, budget=None):
    def session_factory():
        return Session(bind=connection, expire_on_commit=False)

    return AgentRuntimeContext(
        session_factory=session_factory,
        model_gateway=gateway,
        scope=materialized.context,
        budget=budget or AgentBudget(),
    )


def _state(materialized):
    return initial_state(
        run_id="00000000-0000-0000-0000-000000000001",
        thread_id="test-thread",
        actor_id="tester",
        scope=materialized.context,
    )


def test_graph_reads_evidence_and_stops_for_human(session, connection):
    case = load_dataset(DATASET)[0]
    materialized = materialize_case(session, case)
    tx_id = next(iter(materialized.context.allowed_transaction_ids))
    gateway = ScriptedGateway(
        [
            PlanDecision(
                action="TOOLS",
                tool_calls=[ToolRequest(name="get_transaction", arguments={"transaction_id": tx_id})],
                usage=_usage(),
            ),
            PlanDecision(action="COMPOSE", usage=_usage()),
        ],
        _recommendation(f"tx:{tx_id}"),
    )
    graph = build_graph(checkpointer=InMemorySaver())

    result = graph.invoke(
        _state(materialized),
        config={"configurable": {"thread_id": "test-thread"}},
        context=_runtime(connection, materialized, gateway),
    )

    assert result["status"] == "RECOMMENDATION_READY"
    assert len(result["__interrupt__"]) == 1
    assert result["recommendation"]["requires_human"] is True
    assert [call["name"] for call in result["tool_calls"]] == ["get_transaction"]
    assert result["safety_violations"] == []


def test_graph_rejects_forbidden_tool_before_execution(session, connection):
    materialized = materialize_case(session, load_dataset(DATASET)[0])
    gateway = ScriptedGateway(
        [
            PlanDecision(
                action="TOOLS",
                tool_calls=[ToolRequest(name="settle", arguments={})],
                usage=_usage(),
            )
        ],
        _recommendation("tx:1"),
    )

    result = build_graph().invoke(
        _state(materialized), context=_runtime(connection, materialized, gateway)
    )

    assert result["status"] == "FAILED"
    assert result["safety_violations"] == ["forbidden_tool:settle"]
    assert result["tool_calls"] == []


def test_graph_enforces_tool_budget(session, connection):
    materialized = materialize_case(session, load_dataset(DATASET)[0])
    tx_id = next(iter(materialized.context.allowed_transaction_ids))
    gateway = ScriptedGateway(
        [
            PlanDecision(
                action="TOOLS",
                tool_calls=[
                    ToolRequest(name="get_transaction", arguments={"transaction_id": tx_id}),
                    ToolRequest(name="get_transaction", arguments={"transaction_id": tx_id}),
                ],
                usage=_usage(),
            )
        ],
        _recommendation(f"tx:{tx_id}"),
    )

    result = build_graph().invoke(
        _state(materialized),
        context=_runtime(
            connection, materialized, gateway, AgentBudget(max_tool_calls=1)
        ),
    )

    assert result["status"] == "BUDGET_EXCEEDED"
    assert result["tool_calls"] == []


def test_graph_rejects_ungrounded_recommendation(session, connection):
    materialized = materialize_case(session, load_dataset(DATASET)[0])
    gateway = ScriptedGateway(
        [PlanDecision(action="COMPOSE", usage=_usage())],
        _recommendation("tx:999999"),
    )

    result = build_graph().invoke(
        _state(materialized), context=_runtime(connection, materialized, gateway)
    )

    assert result["status"] == "FAILED"
    assert result["safety_violations"] == ["ungrounded_recommendation"]


def test_agent_allowlist_contains_only_read_tools():
    assert ALLOWED_TOOLS == {
        "get_reconciliation",
        "get_transaction",
        "get_group",
        "search_similar_cases",
        "get_audit_trail",
        "get_document_context",
    }
    assert not ({"approve", "reject", "settle", "pay"} & ALLOWED_TOOLS)


def _fixture_recommendation(case):
    payload = dict(case.fixture_result["recommendation"])
    payload["claims"] = [
        {"text": item["claim"], "evidence_ids": [item["source_id"]]}
        for item in payload["evidence"]
    ]
    return InvestigationRecommendation.model_validate(payload)


def _requests_for_case(session, case, materialized):
    tx_ids = iter(sorted(materialized.context.allowed_transaction_ids))
    calls = []
    for tool_name in case.fixture_result["tool_calls"]:
        if tool_name == "get_reconciliation":
            arguments = {
                "reconciliation_id": materialized.context.daily_reconciliation_id
            }
        elif tool_name == "get_transaction":
            arguments = {"transaction_id": next(tx_ids)}
        elif tool_name == "get_group":
            arguments = {"group_id": min(materialized.context.allowed_group_ids)}
        elif tool_name == "search_similar_cases":
            arguments = {"query": case.case_id}
        elif tool_name == "get_audit_trail":
            if materialized.context.daily_reconciliation_id is not None:
                entity_type = "daily_reconciliation"
                entity_id = materialized.context.daily_reconciliation_id
            else:
                entity_type = "reconciliation_group"
                entity_id = materialized.context.reconciliation_group_id
            arguments = {"entity_type": entity_type, "entity_id": entity_id}
        elif tool_name == "get_document_context":
            group_id = min(materialized.context.allowed_group_ids)
            item_id = session.scalar(
                select(db.ReconciliationGroupItem.id)
                .where(db.ReconciliationGroupItem.reconciliation_group_id == group_id)
                .order_by(db.ReconciliationGroupItem.id)
            )
            arguments = {"group_id": group_id, "group_item_id": item_id}
        else:  # pragma: no cover - o dataset e validado explicitamente abaixo
            raise AssertionError(f"Tool desconhecida no dataset: {tool_name}")
        calls.append(ToolRequest(name=tool_name, arguments=arguments))
    return calls


def test_all_20_eval_cases_execute_through_real_graph(session, connection):
    evaluations = []
    for index, case in enumerate(load_dataset(DATASET), 1):
        materialized = materialize_case(session, case)
        calls = _requests_for_case(session, case, materialized)
        gateway = ScriptedGateway(
            [
                PlanDecision(action="TOOLS", tool_calls=calls, usage=_usage()),
                PlanDecision(action="COMPOSE", usage=_usage()),
            ],
            _fixture_recommendation(case),
        )
        state = initial_state(
            run_id=f"00000000-0000-0000-0000-{index:012d}",
            thread_id=f"eval-{case.case_id}",
            actor_id="eval-runner",
            scope=materialized.context,
        )
        result = build_graph().invoke(
            state, context=_runtime(connection, materialized, gateway)
        )
        evaluations.append(
            score_case(
                case,
                AgentResult(
                    recommendation=result["recommendation"],
                    tool_calls=[call["name"] for call in result["tool_calls"]],
                    steps=7,
                    latency_ms=0,
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    cost_usd=Decimal(result["estimated_cost_usd"]),
                    safety_violations=result["safety_violations"],
                ),
            )
        )

    failures = [evaluation.case_id for evaluation in evaluations if not evaluation.passed]
    baseline = json.loads(
        (Path(__file__).parents[1] / "evals/baselines/investigation-v1-graph.json").read_text()
    )
    gate_passed, gate_failures = evaluate_gate(evaluations, baseline)
    assert len(evaluations) == 20
    assert failures == []
    assert gate_passed, gate_failures
