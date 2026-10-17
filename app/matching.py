"""Matching Engine (spec secoes 15-25). Deterministico: score, lote, parcial, tarifa."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config, memory
from app.db import (
    BankTransaction,
    Customer,
    MATCH_DECISION,
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

# Razoes que vem so da memoria: nunca entram em baixa automatica (nao decidiram nada).
HISTORY_REASONS = {"history", "history_rejected"}

logger = logging.getLogger(__name__)


def _base_reasons(reasons: list[str]) -> list[str]:
    """Razoes do documento, sem as da memoria. Usado onde o sistema baixa sozinho."""
    return [r for r in reasons if r not in HISTORY_REASONS]


def _base_order(pool: list["Scored"]) -> list["Scored"]:
    """Ordem total pelo score base (id desempata): a memoria nunca decide posicao aqui."""
    return sorted(pool, key=lambda c: (-c.score, c.amount_open, c.id))


@dataclass(slots=True)
class Scored:
    entity_type: str
    entity: object
    score: int  # score base: so sinais do documento. O teto da memoria mora aqui.
    reasons: list[str] = field(default_factory=list)
    history: int = 0  # ajuste vindo da memoria; nunca entra no score base
    memory_ids: list[int] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.entity.id

    @property
    def amount_open(self) -> Decimal:
        return Decimal(self.entity.amount_open)

    @property
    def strong(self) -> bool:
        return any(r in STRONG_SIGNALS for r in self.reasons)

    @property
    def effective(self) -> int:
        """Score com a memoria. Usado para ordenar, desempatar e promover revisao."""
        return min(100, max(0, self.score + self.history))


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


def apply_history(session: Session, transaction: BankTransaction, pool: list[Scored]) -> None:
    """Aplica memoria episodica ao pool. Muta no lugar; nunca mexe no score base.

    A memoria e consultiva: so reordena e promove revisao. Falha dela nao derruba o matching.
    """
    weight = config.WEIGHTS.get("history", 0)
    if not pool or not weight:
        return
    query = transaction.description_normalized or transaction.description_raw or ""
    if not query:
        return

    try:
        with session.begin_nested():  # savepoint: erro de banco nao suja a transacao externa
            hits = memory.retrieve(
                session,
                MATCH_DECISION,
                query,
                filters={"bank_account_id": transaction.bank_account_id},
                k=config.HISTORY_HITS,
            )
    except Exception:  # noqa: BLE001 - memoria e consultiva, nunca bloqueia o matching
        logger.warning("falha ao buscar memoria de match; seguindo sem history", exc_info=True)
        return

    # A memoria vale pela parte envolvida, nao pelo titulo: titulo pago nao reaparece.
    # Uma decisao por parte (setdefault): rejeicao repetida grava memoria duplicada e o
    # lote aprovado grava varias memorias com o mesmo texto; nada disso pode acumular peso.
    # So conta hit lexical: o canal semantico devolve vizinhos mesmo sem relacao real.
    # ponytail: plainto_tsquery faz AND de todos os termos da descricao atual; com extrato real
    # (NF/NSU/data variando) quase nunca dispara. Upgrade: Hit levar a distancia cosseno para
    # aceitar hit semantico acima de um limiar, e/ou websearch_to_tsquery no lado lexical.
    # Conflito APPROVED x REJECTED na mesma parte: rejeicao vence (conservador), nao a mais antiga.
    by_party: dict[tuple[str, int], tuple[str, int]] = {}
    for hit in hits:
        if "lexical" not in hit.via:
            continue
        decision = (hit.item.item_metadata or {}).get("decision")
        if decision not in {"APPROVED", "REJECTED"}:
            continue
        for column, value in (("supplier", hit.item.supplier_id),
                              ("customer", hit.item.customer_id)):
            key = (column, value)
            if value is not None and (decision == "REJECTED" or key not in by_party):
                by_party[key] = (decision, hit.item.id)
    if not by_party:
        return

    for candidate in pool:
        party = _party(session, candidate.entity)
        if party is None:
            continue
        key = ("supplier" if isinstance(candidate.entity, Payable) else "customer", party.id)
        remembered = by_party.get(key)
        if remembered is None:
            continue
        decision, memory_id = remembered
        candidate.memory_ids.append(memory_id)
        if decision == "APPROVED":
            candidate.history = weight
            candidate.reasons.append("history")
        elif decision == "REJECTED":
            candidate.history = -weight
            candidate.reasons.append("history_rejected")


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
    apply_history(session, transaction, scored)
    scored.sort(key=lambda s: (-s.effective, s.amount_open, s.id))
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
    # Auto-match olha so o score base: a memoria nao escolhe nem desempata baixa automatica.
    base_order = _base_order(pool)
    base_best = base_order[0]
    tied = [c for c in pool if c.score == base_best.score]

    # 1) Exact Match: referencia forte unica + valor exato (spec secoes 15 e 16).
    exact = [c for c in pool if c.strong and "amount" in c.reasons]
    if len(exact) == 1:
        hit = exact[0]
        return (
            MatchResult(AUTO_MATCHED, "SINGLE", [(hit.entity_type, hit.id, amount)],
                        hit.score, _base_reasons(hit.reasons)),
            pool,
        )
    if len(exact) > 1:
        return (
            MatchResult(REVIEW_REQUIRED, confidence=best.effective,
                        reasons=[f"{len(exact)} titulos com referencia e valor identicos"]),
            pool,
        )

    # 2) score alto com valor batendo. Teto da memoria: score base, sem history.
    if base_best.score >= config.AUTO_MATCH_MIN and "amount" in base_best.reasons:
        if len(tied) > 1:
            return (
                MatchResult(REVIEW_REQUIRED, reasons=["empate entre candidatos de mesmo score"],
                            confidence=best.effective),
                pool,
            )
        return (
            MatchResult(AUTO_MATCHED, "SINGLE", [(base_best.entity_type, base_best.id, amount)],
                        base_best.score, _base_reasons(base_best.reasons)),
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
                    best.score, _base_reasons(best.reasons) + [kind.lower()], difference,
                ),
                pool,
            )
        if difference < 0 and adjustment == "DISCOUNT" and -difference <= config.FEE_MAX:
            return (
                MatchResult(
                    MATCH_WITH_ADJUSTMENT, "ADJUSTMENT",
                    [(best.entity_type, best.id, best.amount_open),
                     ("DISCOUNT", None, difference)],
                    best.score, _base_reasons(best.reasons) + ["discount"], difference,
                ),
                pool,
            )
        if difference < 0:
            return (
                MatchResult(PARTIALLY_PAID, "PARTIAL", [(best.entity_type, best.id, amount)],
                            best.score, _base_reasons(best.reasons), difference),
                pool,
            )

    # 4) lote
    # Lote baixa sozinho: o corte por BATCH_MAX_CANDIDATES segue a ordem do score base.
    batches = find_batches(amount, base_order)
    if len(batches) == 1:
        group = batches[0]
        group_type = "BATCH_PAYMENT" if transaction.direction == "DEBIT" else "BATCH_RECEIPT"
        return (
            MatchResult(
                MATCHED_BATCH, group_type,
                [(item.entity_type, item.id, item.amount_open) for item in group],
                min(item.score for item in group),
                sorted({r for item in group for r in _base_reasons(item.reasons)}),
            ),
            pool,
        )
    if len(batches) > 1:
        return (
            MatchResult(REVIEW_REQUIRED, reasons=[f"{len(batches)} combinacoes de lote validas"],
                        confidence=best.effective),
            pool,
        )

    # 5) revisao humana
    # History pode promover UNMATCHED -> REVIEW_REQUIRED, nunca liberar baixa.
    if best.effective >= config.REVIEW_MIN:
        return MatchResult(REVIEW_REQUIRED, confidence=best.effective, reasons=best.reasons), pool
    return MatchResult(UNMATCHED, confidence=best.effective, reasons=best.reasons), pool


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
        memory_ids=sorted({item for candidate in pool for item in candidate.memory_ids}),
    )
    session.commit()
    return result, group
