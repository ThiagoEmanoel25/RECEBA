"""Persistencia, custo e isolamento transacional da observabilidade."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import agent_observability as obs
from app.db import AgentRun, AgentStep, DailyReconciliation
from tests.conftest import TODAY


@pytest.fixture()
def pricing():
    return obs.PricingTable(
        version="test-v1",
        models={
            "model-known": obs.ModelPrice(
                input_per_million_usd=Decimal("2"),
                output_per_million_usd=Decimal("8"),
            )
        },
    )


@pytest.fixture()
def observation_factory(connection):
    return sessionmaker(
        bind=connection,
        class_=Session,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )


@pytest.fixture()
def store(observation_factory, pricing):
    return obs.AgentObservability(observation_factory, pricing)


def _start(session, company, store, *, model="model-known", day_offset=0) -> AgentRun:
    reconciliation = DailyReconciliation(
        bank_account_id=company["account"].id,
        reconciliation_date=TODAY + dt.timedelta(days=day_offset),
        opening_balance=Decimal("0"),
        total_credits=Decimal("0"),
        total_debits=Decimal("0"),
        expected_balance=Decimal("0"),
        real_balance=Decimal("0"),
        difference=Decimal("0"),
        status="INVESTIGATION_REQUIRED",
    )
    session.add(reconciliation)
    session.commit()
    return store.start_run(
        actor_id="analyst-key",
        bank_account_id=company["account"].id,
        daily_reconciliation_id=reconciliation.id,
        model=model,
        prompt_version="prompt-v1",
        retrieval_version="rrf-v1",
    )


def test_step_mede_hash_tokens_custo_e_ordem(
    session, company, store, observation_factory, monkeypatch
):
    ticks = iter([10.0, 10.125, 20.0, 20.050])
    monkeypatch.setattr(obs.time, "monotonic", lambda: next(ticks))
    run = _start(session, company, store)

    with store.record_step(
        run.run_id,
        name="plan",
        kind="llm",
        input_payload={"description": "PIX sigiloso", "amount": "100.00"},
        safe_attributes={"case_type": "daily_reconciliation"},
    ) as step:
        step.record_usage(input_tokens=1_000, output_tokens=500)
        step.record_output({"classification": "UNKNOWN", "secret": "nao persistir"})

    # Step local sem usage nao apaga o custo conhecido do LLM.
    with store.record_step(run.run_id, name="tools", kind="tool"):
        pass

    with observation_factory() as read:
        rows = read.scalars(
            select(AgentStep)
            .where(AgentStep.run_id == run.run_id)
            .order_by(AgentStep.sequence)
        ).all()
        persisted_run = read.get(AgentRun, run.run_id)
        assert [row.sequence for row in rows] == [1, 2]
        assert rows[0].latency_ms == 125
        assert rows[0].input_hash == obs.payload_hash(
            {"description": "PIX sigiloso", "amount": "100.00"}
        )
        assert rows[0].estimated_cost_usd == Decimal("0.00600000")
        assert persisted_run.estimated_cost_usd == Decimal("0.00600000")
        assert persisted_run.cost_complete is True
        assert persisted_run.next_step_sequence == 3
        persisted = str(rows[0].__dict__)
        assert "PIX sigiloso" not in persisted
        assert "nao persistir" not in persisted


def test_step_sobrevive_ao_rollback_da_transacao_de_dominio(
    session, company, store, observation_factory
):
    run = _start(session, company, store)
    session.add(DailyReconciliation(
        bank_account_id=company["account"].id,
        reconciliation_date=TODAY + dt.timedelta(days=10),
        opening_balance=0,
        total_credits=0,
        total_debits=0,
        expected_balance=0,
        real_balance=0,
        difference=0,
        status="TEMP",
    ))
    with store.record_step(run.run_id, name="load_case", kind="node"):
        pass
    session.rollback()

    with observation_factory() as read:
        step = read.scalar(select(AgentStep).where(AgentStep.run_id == run.run_id))
        assert step.status == "COMPLETED"


def test_step_fecha_com_erro_e_propaga_excecao(
    session, company, store, observation_factory
):
    run = _start(session, company, store)
    with pytest.raises(RuntimeError, match="provider caiu"):
        with store.record_step(run.run_id, name="llm", kind="llm"):
            raise RuntimeError("provider caiu")

    with observation_factory() as read:
        step = read.scalar(select(AgentStep).where(AgentStep.run_id == run.run_id))
        assert step.status == "FAILED"
        assert step.error_class == "RuntimeError"
        assert step.finished_at is not None


def test_falha_da_telemetria_nao_mascara_erro_original(
    session, company, store, monkeypatch
):
    run = _start(session, company, store)
    monkeypatch.setattr(
        store, "_finish_step", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("telemetria caiu")
        )
    )
    with pytest.raises(ValueError, match="erro de dominio"):
        with store.record_step(run.run_id, name="llm", kind="llm"):
            raise ValueError("erro de dominio")


def test_custo_desconhecido_e_usage_ausente_sao_null(
    session, company, store, observation_factory
):
    unknown = _start(session, company, store, model="model-sem-preco")
    with store.record_step(unknown.run_id, name="llm", kind="llm") as step:
        step.record_usage(input_tokens=100, output_tokens=50)

    missing = _start(session, company, store, day_offset=1)
    with store.record_step(missing.run_id, name="llm", kind="llm"):
        pass

    with observation_factory() as read:
        for run_id in (unknown.run_id, missing.run_id):
            persisted = read.get(AgentRun, run_id)
            assert persisted.estimated_cost_usd is None
            assert persisted.cost_complete is False


def test_safe_attributes_rejeitam_texto_arbitrario(session, company, store):
    run = _start(session, company, store)
    with pytest.raises(ValidationError):
        with store.record_step(
            run.run_id,
            name="tool",
            kind="tool",
            safe_attributes={"description": "PIX com PII"},
        ):
            pass


def test_finish_run_usa_tempo_de_parede(
    session, company, store, observation_factory, monkeypatch
):
    run = _start(session, company, store)
    started = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)
    with observation_factory() as write, write.begin():
        persisted = write.get(AgentRun, run.run_id)
        persisted.started_at = started
    monkeypatch.setattr(obs, "_now", lambda: started + dt.timedelta(seconds=2))

    finished = store.finish_run(
        run.run_id,
        status="COMPLETED",
        recommendation={"classification": "UNKNOWN", "requires_human": True},
    )
    assert finished.latency_ms == 2000
    assert finished.finished_at is not None


def test_start_run_exige_exatamente_um_tipo_de_caso(store, company):
    with pytest.raises(ValueError, match="exatamente um caso"):
        store.start_run(
            actor_id="analyst",
            bank_account_id=company["account"].id,
            model="model-known",
            prompt_version="v1",
            retrieval_version="v1",
        )
