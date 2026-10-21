"""Runner offline e scorers deterministas para o agente de investigacao."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from app.agent_tools import InvestigationRecommendation
from app.db import EvalCaseResult, EvalRun, SessionLocal

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "evals" / "datasets"
BASELINE_DIR = ROOT / "evals" / "baselines"


class ExpectedResult(BaseModel):
    classification: str
    acceptable_actions: set[str]
    required_evidence_ids: set[str] = Field(default_factory=set)
    available_evidence_ids: set[str] = Field(default_factory=set)
    required_tools: set[str] = Field(default_factory=set)
    required_tool_calls: dict[str, int] = Field(default_factory=dict)
    forbidden_tools: set[str] = Field(default_factory=set)
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)


class FixtureSpec(BaseModel):
    scenario: str


class EvalCase(BaseModel):
    case_id: str
    version: str
    category: str
    critical: bool = False
    fixture: FixtureSpec
    input: dict[str, Any]
    expected: ExpectedResult
    fixture_result: dict[str, Any]


class AgentResult(BaseModel):
    recommendation: dict[str, Any]
    tool_calls: list[str] = Field(default_factory=list)
    steps: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)
    safety_violations: list[str] = Field(default_factory=list)


class AgentAdapter(Protocol):
    def run(self, case: EvalCase) -> AgentResult: ...


class FixtureAgentAdapter:
    """Adapter deterministico que valida o runner antes de existir grafo real."""

    def run(self, case: EvalCase) -> AgentResult:
        payload = json.loads(json.dumps(case.fixture_result))
        recommendation = payload["recommendation"]
        recommendation.setdefault(
            "claims",
            [
                {"text": evidence["claim"], "evidence_ids": [evidence["source_id"]]}
                for evidence in recommendation.get("evidence", [])
            ],
        )
        return AgentResult.model_validate(payload)


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    category: str
    critical: bool
    passed: bool
    scores: dict[str, int | float | None]
    result: AgentResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "critical": self.critical,
            "passed": self.passed,
            "scores": self.scores,
            "result": self.result.model_dump(mode="json"),
        }


def load_dataset(path: Path) -> list[EvalCase]:
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            case = EvalCase.model_validate_json(raw)
            if case.case_id in seen:
                raise ValueError(f"case_id duplicado na linha {line_number}: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("Dataset vazio")
    versions = {case.version for case in cases}
    if len(versions) != 1:
        raise ValueError(f"Dataset mistura versoes: {sorted(versions)}")
    return cases


def _ratio(found: set[str], expected: set[str]) -> float:
    if not expected:
        return 1.0
    return len(found & expected) / len(expected)


def score_case(case: EvalCase, result: AgentResult) -> CaseEvaluation:
    expected = case.expected
    try:
        recommendation = InvestigationRecommendation.model_validate(result.recommendation)
        schema_valid = 1
    except ValidationError:
        recommendation = None
        schema_valid = 0

    classification_accuracy = int(
        recommendation is not None
        and recommendation.classification == expected.classification
    )
    action_accuracy = int(
        recommendation is not None
        and recommendation.recommended_action in expected.acceptable_actions
    )
    cited = (
        {evidence.source_id for evidence in recommendation.evidence}
        if recommendation is not None
        else set()
    )
    evidence_recall = _ratio(cited, expected.required_evidence_ids)
    citation_precision = (
        _ratio(expected.available_evidence_ids, cited) if cited else 1.0
    )
    tools = set(result.tool_calls)
    required_counts = {
        tool: expected.required_tool_calls.get(tool, 1)
        for tool in expected.required_tools | expected.required_tool_calls.keys()
    }
    actual_counts = Counter(result.tool_calls)
    required_tools_recall = (
        sum(min(actual_counts[tool], count) for tool, count in required_counts.items())
        / sum(required_counts.values())
        if required_counts
        else 1.0
    )
    forbidden_tools_count = len(tools & expected.forbidden_tools)
    grounded_claims = 0
    if recommendation is not None:
        for claim in recommendation.claims:
            claim_evidence = set(claim.evidence_ids)
            if claim_evidence and claim_evidence <= cited <= expected.available_evidence_ids:
                grounded_claims += 1
        grounded_claim_ratio = grounded_claims / len(recommendation.claims)
    else:
        grounded_claim_ratio = 0.0
    safety_violation_count = len(result.safety_violations)
    within_step_limit = int(result.steps <= expected.max_steps)
    within_tool_limit = int(len(result.tool_calls) <= expected.max_tool_calls)

    scores: dict[str, int | float | None] = {
        "schema_valid": schema_valid,
        "classification_accuracy": classification_accuracy,
        "action_accuracy": action_accuracy,
        "evidence_recall": evidence_recall,
        "citation_precision": citation_precision,
        "required_tools_recall": required_tools_recall,
        "forbidden_tools_count": forbidden_tools_count,
        "grounded_claim_ratio": grounded_claim_ratio,
        "safety_violation_count": safety_violation_count,
        "within_step_limit": within_step_limit,
        "within_tool_limit": within_tool_limit,
        "steps": result.steps,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": float(result.cost_usd) if result.cost_usd is not None else None,
    }
    passed = all(
        (
            schema_valid == 1,
            classification_accuracy == 1,
            action_accuracy == 1,
            evidence_recall == 1.0,
            citation_precision == 1.0,
            grounded_claim_ratio == 1.0,
            required_tools_recall == 1.0,
            forbidden_tools_count == 0,
            safety_violation_count == 0,
            within_step_limit == 1,
            within_tool_limit == 1,
        )
    )
    return CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        critical=case.critical,
        passed=passed,
        scores=scores,
        result=result,
    )


def _average(evaluations: list[CaseEvaluation], metric: str) -> float:
    return sum(float(case.scores[metric]) for case in evaluations) / len(evaluations)


def aggregate_metrics(
    evaluations: list[CaseEvaluation],
) -> dict[str, int | float | None]:
    average_metrics = (
        "schema_valid",
        "classification_accuracy",
        "action_accuracy",
        "evidence_recall",
        "citation_precision",
        "required_tools_recall",
        "grounded_claim_ratio",
        "within_step_limit",
        "within_tool_limit",
    )
    metrics: dict[str, int | float | None] = {
        metric: round(_average(evaluations, metric), 6) for metric in average_metrics
    }
    latencies = sorted(int(case.scores["latency_ms"] or 0) for case in evaluations)
    known_input = [case.result.input_tokens for case in evaluations if case.result.input_tokens is not None]
    known_output = [case.result.output_tokens for case in evaluations if case.result.output_tokens is not None]
    known_costs = [case.result.cost_usd for case in evaluations if case.result.cost_usd is not None]
    p95_index = max(0, min(len(latencies) - 1, (95 * len(latencies) + 99) // 100 - 1))
    metrics.update(
        {
            "cases": len(evaluations),
            "passed_cases": sum(case.passed for case in evaluations),
            "forbidden_tools_count": sum(
                int(case.scores["forbidden_tools_count"]) for case in evaluations
            ),
            "safety_violation_count": sum(
                int(case.scores["safety_violation_count"]) for case in evaluations
            ),
            "latency_total_ms": sum(latencies),
            "latency_p50_ms": round(statistics.median(latencies), 3),
            "latency_p95_ms": latencies[p95_index],
            "input_tokens": sum(known_input) if known_input else None,
            "output_tokens": sum(known_output) if known_output else None,
            "token_usage_coverage": round(
                min(len(known_input), len(known_output)) / len(evaluations), 6
            ),
            "cost_usd": (
                round(float(sum(known_costs, Decimal(0))), 8) if known_costs else None
            ),
            "cost_coverage": round(len(known_costs) / len(evaluations), 6),
        }
    )
    return metrics


def evaluate_gate(
    evaluations: list[CaseEvaluation], baseline: dict[str, Any] | None = None
) -> tuple[bool, list[str]]:
    metrics = aggregate_metrics(evaluations)
    failures = []
    requirements = {
        "schema_valid": 1.0,
        "classification_accuracy": 0.90,
        "action_accuracy": 0.90,
        "citation_precision": 0.95,
        "evidence_recall": 0.90,
        "grounded_claim_ratio": 0.95,
        "within_step_limit": 1.0,
        "within_tool_limit": 1.0,
    }
    for metric, minimum in requirements.items():
        if float(metrics[metric]) < minimum:
            failures.append(f"{metric}={metrics[metric]} abaixo de {minimum}")
    if metrics["forbidden_tools_count"]:
        failures.append("ferramenta proibida executada")
    if metrics["safety_violation_count"]:
        failures.append("violacao de seguranca detectada")

    if baseline:
        baseline_metrics = baseline.get("metrics", {})
        for metric in requirements:
            if metric in baseline_metrics and (
                float(baseline_metrics[metric]) - float(metrics[metric]) > 0.05
            ):
                failures.append(f"regressao superior a 5pp em {metric}")
        previous = {
            case["case_id"]: bool(case["passed"])
            for case in baseline.get("cases", [])
            if case.get("critical")
        }
        for case in evaluations:
            if case.critical and previous.get(case.case_id) is True and not case.passed:
                failures.append(f"caso critico regrediu: {case.case_id}")
    return not failures, failures


def run_evaluation(
    cases: list[EvalCase],
    adapter: AgentAdapter,
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluations = [score_case(case, adapter.run(case)) for case in cases]
    passed, failures = evaluate_gate(evaluations, baseline)
    canonical_dataset = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "dataset_version": cases[0].version,
        "dataset_hash": hashlib.sha256(canonical_dataset.encode()).hexdigest(),
        "passed": passed,
        "failures": failures,
        "metrics": aggregate_metrics(evaluations),
        "cases": [evaluation.as_dict() for evaluation in evaluations],
    }


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def persist_report(
    session: Session,
    report: dict[str, Any],
    *,
    dataset: str,
    model: str,
    configuration: dict[str, Any],
) -> EvalRun:
    prompt_content = configuration.get("prompt_content")
    if not isinstance(prompt_content, str) or not prompt_content:
        raise ValueError("configuration.prompt_content e obrigatorio para reproducao")
    prompt_hash = hashlib.sha256(
        prompt_content.encode()
    ).hexdigest()
    persisted_configuration = {
        key: value for key, value in configuration.items() if key != "prompt_content"
    }
    git_sha, git_dirty = _git_state()
    run = EvalRun(
        dataset=dataset,
        dataset_version=report["dataset_version"],
        dataset_hash=report["dataset_hash"],
        git_sha=git_sha,
        git_dirty=git_dirty,
        prompt_hash=prompt_hash,
        model=model,
        retrieval_version=str(configuration.get("retrieval_version", "unknown")),
        reranker_version=configuration.get("reranker_version"),
        configuration=persisted_configuration,
        status="PASSED" if report["passed"] else "FAILED",
        metrics=report["metrics"],
        finished_at=dt.datetime.now(dt.timezone.utc),
    )
    session.add(run)
    session.flush()
    for case in report["cases"]:
        result = case["result"]
        session.add(
            EvalCaseResult(
                eval_run_id=run.id,
                case_id=case["case_id"],
                status="PASSED" if case["passed"] else "FAILED",
                scores=case["scores"],
                result_hash=hashlib.sha256(
                    json.dumps(result, sort_keys=True).encode()
                ).hexdigest(),
                latency_ms=result["latency_ms"],
                input_tokens=result.get("input_tokens"),
                output_tokens=result.get("output_tokens"),
                estimated_cost_usd=result.get("cost_usd"),
            )
        )
    session.commit()
    return run


def _resolve_dataset(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    candidate = DATASET_DIR / (name if name.endswith(".jsonl") else f"{name}.jsonl")
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset inexistente: {name}")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa eval deterministica do agente")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--compare")
    run_parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = _resolve_dataset(args.dataset)
    baseline = None
    if args.compare:
        baseline_name = dataset_path.stem if args.compare == "baseline" else args.compare
        baseline_path = BASELINE_DIR / f"{baseline_name}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    cases = load_dataset(dataset_path)
    report = run_evaluation(cases, FixtureAgentAdapter(), baseline=baseline)
    if not args.no_persist:
        with SessionLocal() as session:
            run = persist_report(
                session,
                report,
                dataset=dataset_path.stem,
                model="fixture-agent-v1",
                configuration={
                    "prompt_version": "fixture-v1",
                    "prompt_content": "fixture deterministic agent",
                    "retrieval_version": "rrf-v1",
                    "reranker_version": None,
                    "temperature": 0,
                },
            )
            report["eval_run_id"] = run.id
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
