"""Dataset, scorers, gate de regressao e persistencia das evals."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sqlalchemy import select

from app import eval_fixtures, evals
from app.db import EvalCaseResult, EvalRun

DATASET = Path("evals/datasets/investigation-v1.jsonl")


def test_dataset_v1_tem_20_casos_e_distribuicao_planejada():
    cases = evals.load_dataset(DATASET)
    counts = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    assert len(cases) == 20
    assert counts == {
        "unmapped_transaction": 4,
        "group_difference": 4,
        "possible_duplicate": 3,
        "partial_adjustment": 3,
        "prompt_injection": 3,
        "degradation": 3,
    }
    assert all(case.fixture.scenario == case.category for case in cases)


@pytest.mark.parametrize("case_index", range(20))
def test_cada_caso_materializa_um_mundo_executavel(session, case_index):
    case = evals.load_dataset(DATASET)[case_index]
    materialized = eval_fixtures.materialize_case(session, case)

    assert materialized.context.bank_account_id > 0
    assert materialized.input.keys() >= case.input.keys()
    assert materialized.context.allowed_transaction_ids
    assert case.expected.required_evidence_ids <= materialized.aliases.keys()


def test_fixture_adapter_passa_gate_completo():
    report = evals.run_evaluation(
        evals.load_dataset(DATASET), evals.FixtureAgentAdapter()
    )
    assert report["passed"] is True
    assert report["metrics"]["passed_cases"] == 20
    assert report["metrics"]["schema_valid"] == 1
    assert report["metrics"]["safety_violation_count"] == 0


def _first_case():
    return evals.load_dataset(DATASET)[0]


def _fixture_result(case):
    return evals.FixtureAgentAdapter().run(case).model_dump(mode="json")


@pytest.mark.parametrize(
    ("mutation", "metric"),
    [
        (
            lambda result: result["recommendation"].update(requires_human=False),
            "schema_valid",
        ),
        (
            lambda result: result["recommendation"].update(
                classification="GROUP_DIFFERENCE"
            ),
            "classification_accuracy",
        ),
        (
            lambda result: result["recommendation"].update(
                recommended_action="APPROVE"
            ),
            "action_accuracy",
        ),
        (lambda result: result["recommendation"].update(evidence=[]), "evidence_recall"),
        (
            lambda result: result["recommendation"]["evidence"][0].update(
                source_id="inventado"
            ),
            "citation_precision",
        ),
        (
            lambda result: result["recommendation"]["claims"][0].update(
                evidence_ids=["inventado"]
            ),
            "grounded_claim_ratio",
        ),
        (lambda result: result.update(tool_calls=[]), "required_tools_recall"),
        (lambda result: result["tool_calls"].append("settle"), "forbidden_tools_count"),
        (
            lambda result: result["safety_violations"].append("financial_write"),
            "safety_violation_count",
        ),
        (lambda result: result.update(steps=99), "within_step_limit"),
    ],
)
def test_cada_scorer_detecta_mutacao(mutation, metric):
    case = _first_case()
    raw = copy.deepcopy(_fixture_result(case))
    mutation(raw)
    evaluation = evals.score_case(case, evals.AgentResult.model_validate(raw))
    assert evaluation.passed is False
    if metric.endswith("_count"):
        assert evaluation.scores[metric] > 0
    else:
        assert evaluation.scores[metric] < 1


def test_gate_detecta_regressao_de_caso_critico():
    cases = evals.load_dataset(DATASET)
    baseline = evals.run_evaluation(cases, evals.FixtureAgentAdapter())
    target = next(case for case in cases if case.critical)
    target.fixture_result["safety_violations"] = ["unsafe"]
    report = evals.run_evaluation(cases, evals.FixtureAgentAdapter(), baseline=baseline)
    assert report["passed"] is False
    assert any("caso critico regrediu" in failure for failure in report["failures"])


def test_runner_persiste_resultados_sem_resposta_crua(session):
    cases = evals.load_dataset(DATASET)
    report = evals.run_evaluation(cases, evals.FixtureAgentAdapter())
    run = evals.persist_report(
        session,
        report,
        dataset="investigation-v1",
        model="fixture-agent-v1",
        configuration={
            "prompt_version": "fixture-v1",
            "prompt_content": "fixture deterministic agent",
            "retrieval_version": "rrf-v1",
            "temperature": 0,
        },
    )

    persisted = session.get(EvalRun, run.id)
    results = session.scalars(
        select(EvalCaseResult).where(EvalCaseResult.eval_run_id == run.id)
    ).all()
    assert persisted.status == "PASSED"
    assert persisted.metrics["cases"] == 20
    assert persisted.dataset_hash == report["dataset_hash"]
    assert persisted.finished_at is not None
    assert "prompt_content" not in persisted.configuration
    assert len(results) == 20
    assert all(len(result.result_hash) == 64 for result in results)


def test_cli_retorna_zero_para_dataset_aprovado(capsys):
    exit_code = evals.main(
        ["run", "--dataset", "investigation-v1", "--no-persist"]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"passed": true' in output
