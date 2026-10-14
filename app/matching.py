"""Matching Engine (spec secoes 15-25). Deterministico: score, lote, parcial, tarifa."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.db import (
    BankTransaction,
    Customer,
    MatchingCandidate,
    Payable,
    Receivable,
    ReconciliationGroup,
    ReconciliationGroupItem,
    Supplier,
    audit,
)
from app.parser import classify_adjustment, digits, normalize, parse_observation, same_ref

AUTO_MATCHED = "AUTO_MATCHED"
MATCHED_BATCH = "MATCHED_BATCH"
MATCH_WITH_ADJUSTMENT = "MATCH_WITH_ADJUSTMENT"
PARTIALLY_PAID = "PARTIALLY_PAID"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
UNMATCHED = "UNMATCHED"

# Referencias de peso alto: so elas liberam baixa parcial/ajuste automatico.
STRONG_SIGNALS = {"invoice", "bl", "container"}


@dataclass(slots=True)
class Scored:
    entity_type: str
    entity: object
    score: int
    reasons: list[str] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.entity.id

    @property
    def amount_open(self) -> Decimal:
        return Decimal(self.entity.amount_open)

    @property
    def strong(self) -> bool:
        return any(r in STRONG_SIGNALS for r in self.reasons)


def _party(session: Session, scored_entity) -> Supplier | Customer | None:
    if isinstance(scored_entity, Payable):
        return session.get(Supplier, scored_entity.supplier_id)
    return session.get(Customer, scored_entity.customer_id)


def score(
    transaction: BankTransaction, title, party, tokens: dict[str, str]
) -> tuple[int, list[str]]:
    """Soma de sinais (spec secao 18). Sem IA, sem heuristica escondida."""
    reasons: list[str] = []
    weights = config.WEIGHTS
    description = transaction.description_normalized or ""

    invoice_sources = [tokens.get("invoice"), transaction.document_number, transaction.reference]
    if any(same_ref(title.invoice_number, src) for src in invoice_sources):
        reasons.append("invoice")

    party_cnpj = digits(getattr(party, "cnpj", None))
    if party_cnpj and (party_cnpj == digits(tokens.get("cnpj")) or party_cnpj in digits(description)):
        reasons.append("cnpj")

    if abs(Decimal(transaction.amount) - Decimal(title.amount_open)) <= config.AMOUNT_TOLERANCE:
        reasons.append("amount")

    name = normalize(getattr(party, "name", None))
    name_words = [w for w in name.split() if len(w) >= 3]
    if name_words and any(w in description.split() for w in name_words):
        reasons.append("party")

    if title.due_date and abs((transaction.transaction_date - title.due_date).days) <= (
        config.DUE_DATE_TOLERANCE_DAYS
    ):
        reasons.append("date")

    if tokens.get("bl") and same_ref(getattr(title, "bill_of_lading", None), tokens["bl"]):
        reasons.append("bl")
    if tokens.get("container") and same_ref(
        getattr(title, "container", None), tokens["container"]
    ):
        reasons.append("container")

    total = min(100, sum(weights.get(r, 0) for r in reasons))
    return total, reasons


def candidates(session: Session, transaction: BankTransaction) -> list[Scored]:
    """Filtra por conta/moeda/janela de datas antes de pontuar (spec secao 22)."""
    tokens = parse_observation(transaction.description_raw or transaction.description_normalized)
    model = Payable if transaction.direction == "DEBIT" else Receivable
    window = dt.timedelta(days=config.DATE_WINDOW_DAYS)
    rows = session.scalars(
        select(model).where(
            model.status.in_(("OPEN", "PARTIALLY_PAID")),
            model.amount_open > 0,
            model.currency == transaction.currency,
            model.due_date.is_(None)
            | (
                model.due_date.between(
                    transaction.transaction_date - window, transaction.transaction_date + window
                )
            ),
        )
    ).all()

    entity_type = "PAYABLE" if model is Payable else "RECEIVABLE"
    scored = []
    for row in rows:
        value, reasons = score(transaction, row, _party(session, row), tokens)
        if value > 0:
            scored.append(Scored(entity_type, row, value, reasons))
    scored.sort(key=lambda s: (-s.score, s.amount_open))
    return scored


def find_batches(target: Decimal, pool: list[Scored]) -> list[tuple[Scored, ...]]:
    """Subset-sum limitado. Devolve TODAS as combinacoes exatas encontradas."""
    # ponytail: busca combinatoria com teto fixo; se o pool crescer, trocar por DP/ILP.
    pool = pool[: config.BATCH_MAX_CANDIDATES]
    found: list[tuple[Scored, ...]] = []
    for size in range(2, min(config.BATCH_MAX_ITEMS, len(pool)) + 1):
        for group in combinations(pool, size):
            total = sum((item.amount_open for item in group), Decimal(0))
            if abs(total - target) <= config.AMOUNT_TOLERANCE:
                found.append(group)
        if len(found) > 1:  # ja ambiguo, nao vale continuar explodindo
            break
    return found


@dataclass(slots=True)
class MatchResult:
    status: str
    group_type: str | None = None
    items: list[tuple[str, int | None, Decimal]] = field(default_factory=list)
    confidence: int = 0
    reasons: list[str] = field(default_factory=list)
    difference: Decimal = Decimal("0.00")


def evaluate(session: Session, transaction: BankTransaction) -> tuple[MatchResult, list[Scored]]:
    """Decide o destino da transacao. Nao grava nada."""
    pool = candidates(session, transaction)
    amount = Decimal(transaction.amount)
    adjustment = classify_adjustment(transaction.description_raw)

    if not pool:
        return MatchResult(UNMATCHED, reasons=["sem candidatos"]), pool

    best = pool[0]
    tied = [c for c in pool if c.score == best.score]

    # 1) Exact Match: referencia forte unica + valor exato (spec secoes 15 e 16).
    exact = [c for c in pool if c.strong and "amount" in c.reasons]
    if len(exact) == 1:
        hit = exact[0]
        return (
            MatchResult(AUTO_MATCHED, "SINGLE", [(hit.entity_type, hit.id, amount)],
                        hit.score, hit.reasons),
            pool,
        )
    if len(exact) > 1:
        return (
            MatchResult(REVIEW_REQUIRED, confidence=best.score,
                        reasons=[f"{len(exact)} titulos com referencia e valor identicos"]),
            pool,
        )

    # 2) score alto com valor batendo
    if best.score >= config.AUTO_MATCH_MIN and "amount" in best.reasons:
        if len(tied) > 1:
            return (
                MatchResult(REVIEW_REQUIRED, reasons=["empate entre candidatos de mesmo score"],
                            confidence=best.score),
                pool,
            )
        return (
            MatchResult(AUTO_MATCHED, "SINGLE", [(best.entity_type, best.id, amount)],
                        best.score, best.reasons),
            pool,
        )

    # 3) referencia forte unica com diferenca explicavel: ajuste ou pagamento parcial
    strongs = [c for c in pool if c.strong]
    if len(strongs) == 1:
        best = strongs[0]
        difference = amount - best.amount_open
        if difference > 0 and difference <= config.FEE_MAX:
            kind = adjustment if adjustment in {"FEE", "INTEREST", "PENALTY"} else "FEE"
            return (
                MatchResult(
                    MATCH_WITH_ADJUSTMENT, "ADJUSTMENT",
                    [(best.entity_type, best.id, best.amount_open), (kind, None, difference)],
                    best.score, best.reasons + [kind.lower()], difference,
                ),
                pool,
            )
        if difference < 0 and adjustment == "DISCOUNT" and -difference <= config.FEE_MAX:
            return (
                MatchResult(
                    MATCH_WITH_ADJUSTMENT, "ADJUSTMENT",
                    [(best.entity_type, best.id, best.amount_open),
                     ("DISCOUNT", None, difference)],
                    best.score, best.reasons + ["discount"], difference,
                ),
                pool,
            )
        if difference < 0:
            return (
                MatchResult(PARTIALLY_PAID, "PARTIAL", [(best.entity_type, best.id, amount)],
                            best.score, best.reasons, difference),
                pool,
            )

    # 4) lote
    batches = find_batches(amount, pool)
    if len(batches) == 1:
        group = batches[0]
        group_type = "BATCH_PAYMENT" if transaction.direction == "DEBIT" else "BATCH_RECEIPT"
        return (
            MatchResult(
                MATCHED_BATCH, group_type,
                [(item.entity_type, item.id, item.amount_open) for item in group],
                min(item.score for item in group),
                sorted({r for item in group for r in item.reasons}),
            ),
            pool,
        )
    if len(batches) > 1:
        return (
            MatchResult(REVIEW_REQUIRED, reasons=[f"{len(batches)} combinacoes de lote validas"],
                        confidence=best.score),
            pool,
        )

    # 5) revisao humana
    if best.score >= config.REVIEW_MIN:
        return MatchResult(REVIEW_REQUIRED, confidence=best.score, reasons=best.reasons), pool
    return MatchResult(UNMATCHED, confidence=best.score, reasons=best.reasons), pool


def _settle(session: Session, result: MatchResult) -> None:
    """Aplica a baixa. Unico ponto que mexe em amount_open."""
    for entity_type, entity_id, allocated in result.items:
        if entity_type not in {"PAYABLE", "RECEIVABLE"} or entity_id is None:
            continue
        title = session.get(Payable if entity_type == "PAYABLE" else Receivable, entity_id)
        title.amount_open = Decimal(title.amount_open) - Decimal(allocated)
        title.status = "PAID" if title.amount_open <= config.AMOUNT_TOLERANCE else "PARTIALLY_PAID"


def run(session: Session, transaction: BankTransaction, auto_settle: bool = True):
    """Roda o matching, persiste candidatos, grupo e auditoria."""
    result, pool = evaluate(session, transaction)
    audit(session, "bank_transaction", transaction.id, "MATCHING_STARTED")

    session.query(MatchingCandidate).filter_by(bank_transaction_id=transaction.id).delete()
    for candidate in pool[: config.BATCH_MAX_CANDIDATES]:
        session.add(
            MatchingCandidate(
                bank_transaction_id=transaction.id,
                candidate_type=candidate.entity_type,
                candidate_id=candidate.id,
                score=candidate.score,
                match_reasons={"reasons": candidate.reasons},
            )
        )

    group = None
    if result.items:
        allocated = sum(
            (amount for kind, _, amount in result.items if kind in {"PAYABLE", "RECEIVABLE"}),
            Decimal(0),
        )
        decided = result.status in {AUTO_MATCHED, MATCHED_BATCH, MATCH_WITH_ADJUSTMENT,
                                    PARTIALLY_PAID}
        group = ReconciliationGroup(
            bank_transaction_id=transaction.id,
            group_type=result.group_type or "SINGLE",
            total_transaction_amount=Decimal(transaction.amount),
            total_allocated_amount=allocated,
            difference=Decimal(transaction.amount) - allocated,
            status="APPROVED" if (decided and auto_settle) else "REVIEW_REQUIRED",
            confidence_score=result.confidence,
            match_reasons={"reasons": result.reasons},
        )
        session.add(group)
        session.flush()
        for entity_type, entity_id, amount in result.items:
            session.add(
                ReconciliationGroupItem(
                    reconciliation_group_id=group.id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    amount_allocated=amount,
                )
            )
        if decided and auto_settle:
            _settle(session, result)

    transaction.status = result.status
    audit(
        session, "bank_transaction", transaction.id, result.status,
        score=result.confidence, reasons=result.reasons,
        group_id=group.id if group is not None else None,
    )
    session.commit()
    return result, group
