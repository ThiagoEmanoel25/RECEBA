"""Runtime persistente do agente: start, interrupt e resume idempotente."""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app import config, db
from app.agent import (
    AgentRuntimeContext,
    HumanFeedback,
    InvestigationRecommendation,
    ModelGateway,
    build_graph,
    initial_state,
)
from app.agent_tools import AgentToolContext, resolve_investigation_scope

SessionFactory = Callable[[], Session]
CheckpointerFactory = Callable[[], Any]


class ResumeInput(BaseModel):
    """Campos que a API podera aceitar; o ator vem da autenticacao."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["accepted", "edited", "rejected"]
    note: str | None = Field(default=None, max_length=2000)
    recommendation: InvestigationRecommendation | None = None


def _psycopg_url(database_url: str) -> str:
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        raise ValueError("Checkpointer de producao exige PostgreSQL")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def postgres_checkpointer_factory(database_url: str) -> CheckpointerFactory:
    connection_string = _psycopg_url(database_url)

    @contextmanager
    def factory() -> Iterator[PostgresSaver]:
        with PostgresSaver.from_conn_string(connection_string) as checkpointer:
            checkpointer.setup()
            yield checkpointer

    return factory


class AgentRuntime:
    """Orquestra o grafo sem guardar estado de execucao na memoria do processo."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        model_gateway: ModelGateway,
        checkpointer_factory: CheckpointerFactory | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.model_gateway = model_gateway
        self.checkpointer_factory = checkpointer_factory or postgres_checkpointer_factory(
            config.DATABASE_URL
        )

    def _context(self, scope: AgentToolContext) -> AgentRuntimeContext:
        return AgentRuntimeContext(
            session_factory=self.session_factory,
            model_gateway=self.model_gateway,
            scope=scope,
        )

    @staticmethod
    def _graph_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def start(
        self,
        *,
        actor_id: str,
        scope: AgentToolContext,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or str(uuid.uuid4())
        thread_id = thread_id or str(uuid.uuid4())
        with self.session_factory() as session, session.begin():
            session.add(
                db.AgentRun(
                    run_id=run_id,
                    thread_id=thread_id,
                    actor_id=actor_id,
                    bank_account_id=scope.bank_account_id,
                    daily_reconciliation_id=scope.daily_reconciliation_id,
                    reconciliation_group_id=scope.reconciliation_group_id,
                    status="RUNNING",
                    model=config.AGENT_MODEL,
                    prompt_version=config.AGENT_PROMPT_VERSION,
                    retrieval_version=config.AGENT_RETRIEVAL_VERSION,
                    cost_complete=True,
                )
            )

        with self.checkpointer_factory() as checkpointer:
            result = build_graph(checkpointer=checkpointer).invoke(
                initial_state(
                    run_id=run_id,
                    thread_id=thread_id,
                    actor_id=actor_id,
                    scope=scope,
                ),
                config=self._graph_config(thread_id),
                context=self._context(scope),
            )

        interrupted = bool(result.get("__interrupt__"))
        with self.session_factory() as session, session.begin():
            run = session.get(db.AgentRun, run_id)
            if run is None:
                raise RuntimeError("Run desapareceu durante a execucao")
            run.recommendation = result.get("recommendation")
            run.status = "WAITING_HUMAN" if interrupted else result.get("status", "FAILED")
        return result

    def resume(
        self,
        run_id: str,
        *,
        actor_id: str,
        feedback: ResumeInput | dict[str, Any],
    ) -> dict[str, Any]:
        requested = ResumeInput.model_validate(feedback)
        validated = HumanFeedback.model_validate(
            {**requested.model_dump(mode="json"), "actor_id": actor_id}
        )
        with self.session_factory() as session:
            run = session.get(db.AgentRun, run_id)
            if run is None:
                raise ValueError("Run inexistente")
            if run.status == "COMPLETED":
                if run.human_feedback != validated.model_dump(
                    mode="json", exclude_none=True
                ):
                    raise ValueError("Run ja concluido com outra decisao")
                return {
                    "run_id": run.run_id,
                    "thread_id": run.thread_id,
                    "status": run.status,
                    "recommendation": run.recommendation,
                    "human_feedback": run.human_feedback,
                }
            if run.status != "WAITING_HUMAN":
                raise ValueError("Run nao esta aguardando revisao humana")
            thread_id = run.thread_id
            if run.daily_reconciliation_id is not None:
                scope = resolve_investigation_scope(
                    session, daily_reconciliation_id=run.daily_reconciliation_id
                )
            else:
                scope = resolve_investigation_scope(
                    session, reconciliation_group_id=run.reconciliation_group_id
                )

        with self.checkpointer_factory() as checkpointer:
            result = build_graph(checkpointer=checkpointer).invoke(
                Command(resume=validated.model_dump(mode="json", exclude_none=True)),
                config=self._graph_config(thread_id),
                context=self._context(scope),
            )
        if result.get("status") != "COMPLETED":
            raise RuntimeError("Retomada nao concluiu o registro do feedback")
        return result

