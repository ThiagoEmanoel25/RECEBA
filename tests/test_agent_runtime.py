from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import db
from app.agent import ModelUsage, PlanDecision, ToolRequest
from app.agent_runtime import AgentRuntime, postgres_checkpointer_factory
from app.eval_fixtures import materialize_case
from app.evals import load_dataset
from tests.test_agent import ScriptedGateway, _fixture_recommendation


DATASET = Path(__file__).parents[1] / "evals/datasets/investigation-v1.jsonl"


def _session_factory(connection):
    def factory():
        return Session(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

    return factory


def _checkpointer_factory(checkpointer):
    @contextmanager
    def factory():
        yield checkpointer

    return factory


def _gateway(case, materialized):
    usage = ModelUsage(input_tokens=10, output_tokens=5, estimated_cost_usd="0.001")
    transaction_id = min(materialized.context.allowed_transaction_ids)
    return ScriptedGateway(
        [
            PlanDecision(
                action="TOOLS",
                tool_calls=[
                    ToolRequest(
                        name="get_transaction",
                        arguments={"transaction_id": transaction_id},
                    )
                ],
                usage=usage,
            ),
            PlanDecision(action="COMPOSE", usage=usage),
        ],
        _fixture_recommendation(case),
    )


@pytest.mark.parametrize("decision", ["accepted", "edited", "rejected"])
def test_resume_survives_runtime_restart_and_is_idempotent(
    session, connection, decision
):
    case = load_dataset(DATASET)[0]
    materialized = materialize_case(session, case)
    checkpointer = InMemorySaver()
    factory = _session_factory(connection)
    checkpointer_factory = _checkpointer_factory(checkpointer)
    first_runtime = AgentRuntime(
        session_factory=factory,
        model_gateway=_gateway(case, materialized),
        checkpointer_factory=checkpointer_factory,
    )

    paused = first_runtime.start(
        actor_id="analyst-start",
        scope=materialized.context,
        run_id=f"00000000-0000-0000-0000-00000000000{len(decision)}",
        thread_id=f"restart-{decision}",
    )

    assert paused["status"] == "RECOMMENDATION_READY"
    request = paused["__interrupt__"][0]
    assert request.value["editable_fields"]
    assert request.response_schema["properties"]["decision"]
    with factory() as persisted:
        run = persisted.get(db.AgentRun, paused["run_id"])
        assert run.status == "WAITING_HUMAN"

    # Novo objeto simula outro processo atendendo a retomada.
    restarted_runtime = AgentRuntime(
        session_factory=factory,
        model_gateway=_gateway(case, materialized),
        checkpointer_factory=checkpointer_factory,
    )
    feedback = {"decision": decision, "note": "revisado no teste"}
    if decision == "edited":
        edited = _fixture_recommendation(case).model_copy(
            update={"summary": "Resumo corrigido pelo analista"}
        )
        feedback["recommendation"] = edited.model_dump(mode="json")

    completed = restarted_runtime.resume(
        paused["run_id"], actor_id="analyst-review", feedback=feedback
    )
    repeated = restarted_runtime.resume(
        paused["run_id"], actor_id="analyst-review", feedback=feedback
    )

    assert completed["status"] == "COMPLETED"
    assert repeated["status"] == "COMPLETED"
    assert completed["human_feedback"]["decision"] == decision
    with factory() as persisted:
        run = persisted.get(db.AgentRun, paused["run_id"])
        audit_count = persisted.scalar(
            select(func.count(db.AuditEvent.id)).where(
                db.AuditEvent.entity_type == "agent_run",
                db.AuditEvent.entity_id == paused["run_id"],
                db.AuditEvent.event_type == "HUMAN_FEEDBACK_RECORDED",
            )
        )
        assert run.status == "COMPLETED"
        assert run.human_feedback["actor_id"] == "analyst-review"
        assert audit_count == 1
        if decision == "edited":
            assert run.recommendation["summary"] == "Resumo corrigido pelo analista"


def test_resume_rejects_invalid_edit_before_touching_checkpoint(session, connection):
    case = load_dataset(DATASET)[0]
    materialized = materialize_case(session, case)
    checkpointer = InMemorySaver()
    runtime = AgentRuntime(
        session_factory=_session_factory(connection),
        model_gateway=_gateway(case, materialized),
        checkpointer_factory=_checkpointer_factory(checkpointer),
    )
    paused = runtime.start(actor_id="starter", scope=materialized.context)

    with pytest.raises(ValueError, match="edited exige recommendation"):
        runtime.resume(
            paused["run_id"],
            actor_id="reviewer",
            feedback={"decision": "edited", "note": "sem payload editado"},
        )

    with _session_factory(connection)() as persisted:
        assert persisted.get(db.AgentRun, paused["run_id"]).status == "WAITING_HUMAN"


def test_postgres_checkpointer_resumes_after_connection_restart(session, connection):
    case = load_dataset(DATASET)[0]
    materialized = materialize_case(session, case)
    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://receba:receba@localhost:5433/receba_test",
    )
    checkpointer_factory = postgres_checkpointer_factory(database_url)
    thread_id = f"postgres-restart-{uuid.uuid4()}"
    factory = _session_factory(connection)
    first_runtime = AgentRuntime(
        session_factory=factory,
        model_gateway=_gateway(case, materialized),
        checkpointer_factory=checkpointer_factory,
    )
    try:
        paused = first_runtime.start(
            actor_id="starter", scope=materialized.context, thread_id=thread_id
        )

        # resume abre outro PostgresSaver e outra conexao.
        second_runtime = AgentRuntime(
            session_factory=factory,
            model_gateway=_gateway(case, materialized),
            checkpointer_factory=checkpointer_factory,
        )
        completed = second_runtime.resume(
            paused["run_id"],
            actor_id="reviewer",
            feedback={"decision": "accepted", "note": "persistiu"},
        )
        assert completed["status"] == "COMPLETED"
    finally:
        with checkpointer_factory() as checkpointer:
            checkpointer.delete_thread(thread_id)
