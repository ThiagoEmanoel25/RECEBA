"""Observabilidade persistente do agente em transacao independente do dominio."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import AgentRun, AgentStep

TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "BUDGET_EXCEEDED"})
logger = logging.getLogger(__name__)
SessionFactory = Callable[[], Session]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def payload_hash(payload: Any) -> str | None:
    """Hash estavel para correlacao sem persistir o conteudo original."""
    if payload is None:
        return None
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal


@dataclass(frozen=True, slots=True)
class PricingTable:
    version: str
    models: dict[str, ModelPrice]

    def estimate(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal | None:
        price = self.models.get(model)
        if price is None:
            return None
        million = Decimal(1_000_000)
        return (
            Decimal(input_tokens) * price.input_per_million_usd
            + Decimal(output_tokens) * price.output_per_million_usd
        ) / million


class SafeStepAttributes(BaseModel):
    """Atributos de baixa cardinalidade; texto financeiro e PII nao sao aceitos."""

    model_config = ConfigDict(extra="forbid")

    case_type: str | None = Field(default=None, max_length=40)
    attempt: int | None = Field(default=None, ge=0)
    degraded: bool | None = None
    result_count: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=80)


@dataclass(slots=True)
class StepRecorder:
    step: AgentStep
    output_payload: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def record_output(self, payload: Any) -> None:
        self.output_payload = payload

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Contagem de tokens nao pode ser negativa")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AgentObservability:
    """Sink que persiste telemetria fora da transacao do trabalho observado."""

    def __init__(self, session_factory: SessionFactory, pricing: PricingTable):
        self.session_factory = session_factory
        self.pricing = pricing

    def start_run(
        self,
        *,
        actor_id: str,
        bank_account_id: int,
        model: str,
        prompt_version: str,
        retrieval_version: str,
        daily_reconciliation_id: int | None = None,
        reconciliation_group_id: int | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentRun:
        if (daily_reconciliation_id is None) == (reconciliation_group_id is None):
            raise ValueError("Informe exatamente um caso: conciliacao diaria ou grupo")
        with self.session_factory() as session, session.begin():
            run = AgentRun(
                run_id=run_id or str(uuid.uuid4()),
                thread_id=thread_id or str(uuid.uuid4()),
                actor_id=actor_id,
                bank_account_id=bank_account_id,
                daily_reconciliation_id=daily_reconciliation_id,
                reconciliation_group_id=reconciliation_group_id,
                status="RUNNING",
                model=model,
                prompt_version=prompt_version,
                retrieval_version=retrieval_version,
                pricing_version=self.pricing.version,
                estimated_cost_usd=None,
                cost_complete=True,
            )
            session.add(run)
            session.flush()
            session.expunge(run)
            return run

    def _start_step(
        self,
        run_id: str,
        *,
        name: str,
        kind: str,
        input_payload: Any,
        safe_attributes: SafeStepAttributes | dict[str, Any] | None,
    ) -> AgentStep:
        attributes = (
            SafeStepAttributes.model_validate(safe_attributes).model_dump(
                mode="json", exclude_none=True
            )
            if safe_attributes is not None
            else None
        )
        with self.session_factory() as session, session.begin():
            run = session.scalar(
                select(AgentRun).where(AgentRun.run_id == run_id).with_for_update()
            )
            if run is None:
                raise ValueError("Run inexistente")
            if run.status != "RUNNING":
                raise ValueError("So e permitido abrir etapa em run RUNNING")
            sequence = run.next_step_sequence
            run.next_step_sequence += 1
            step = AgentStep(
                run_id=run.run_id,
                sequence=sequence,
                name=name,
                kind=kind,
                status="RUNNING",
                input_hash=payload_hash(input_payload),
                safe_attributes=attributes,
            )
            session.add(step)
            session.flush()
            session.expunge(step)
            return step

    def _finish_step(
        self,
        recorder: StepRecorder,
        *,
        status: str,
        error_class: str | None,
        latency_ms: int,
    ) -> None:
        with self.session_factory() as session, session.begin():
            step = session.get(AgentStep, recorder.step.id)
            run = session.scalar(
                select(AgentRun)
                .where(AgentRun.run_id == recorder.step.run_id)
                .with_for_update()
            )
            if step is None or run is None:
                raise RuntimeError("Run ou step de observabilidade desapareceu")
            step.status = status
            step.error_class = error_class
            step.finished_at = _now()
            step.latency_ms = latency_ms
            step.output_hash = payload_hash(recorder.output_payload)
            step.input_tokens = recorder.input_tokens
            step.output_tokens = recorder.output_tokens

            usage_missing = recorder.input_tokens is None or recorder.output_tokens is None
            if step.kind == "llm" and usage_missing:
                step.estimated_cost_usd = None
                run.cost_complete = False
            elif usage_missing:
                step.estimated_cost_usd = None
            else:
                step.estimated_cost_usd = self.pricing.estimate(
                    run.model, recorder.input_tokens, recorder.output_tokens
                )
                run.input_tokens += recorder.input_tokens
                run.output_tokens += recorder.output_tokens
                if step.kind == "llm":
                    if step.estimated_cost_usd is None:
                        run.cost_complete = False
                    else:
                        run.estimated_cost_usd = (
                            run.estimated_cost_usd or Decimal(0)
                        ) + step.estimated_cost_usd

    @contextmanager
    def record_step(
        self,
        run_id: str,
        *,
        name: str,
        kind: str,
        input_payload: Any = None,
        safe_attributes: SafeStepAttributes | dict[str, Any] | None = None,
    ) -> Iterator[StepRecorder]:
        step = self._start_step(
            run_id,
            name=name,
            kind=kind,
            input_payload=input_payload,
            safe_attributes=safe_attributes,
        )
        recorder = StepRecorder(step=step)
        started = time.monotonic()
        original_error: BaseException | None = None
        try:
            yield recorder
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                self._finish_step(
                    recorder,
                    status="FAILED" if original_error else "COMPLETED",
                    error_class=(
                        original_error.__class__.__name__ if original_error else None
                    ),
                    latency_ms=max(0, round((time.monotonic() - started) * 1000)),
                )
            except Exception:
                if original_error is None:
                    raise
                logger.exception(
                    "falha ao fechar observabilidade; preservando erro original"
                )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        recommendation: dict[str, Any] | None = None,
        human_feedback: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> AgentRun:
        if status not in TERMINAL_STATUSES | {"WAITING_HUMAN"}:
            raise ValueError("Status final invalido")
        with self.session_factory() as session, session.begin():
            run = session.get(AgentRun, run_id)
            if run is None:
                raise ValueError("Run inexistente")
            now = _now()
            run.status = status
            run.recommendation = recommendation
            run.human_feedback = human_feedback
            run.error_code = error_code
            run.finished_at = now if status in TERMINAL_STATUSES else None
            run.latency_ms = max(
                0, round((now - run.started_at).total_seconds() * 1000)
            )
            session.flush()
            session.expunge(run)
            return run
