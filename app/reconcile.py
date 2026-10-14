"""Conciliacao diaria e investigacao de divergencias (spec secoes 28 e 29)."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import AMOUNT_TOLERANCE
from app.db import (
    BankBalance,
    BankTransaction,
    DailyReconciliation,
    Payable,
    Receivable,
    ReconciliationGroup,
    audit,
)
from app.matching import REVIEW_REQUIRED, UNMATCHED
from app.parser import same_ref


def investigate(session: Session, bank_account_id: int, day: dt.date) -> list[dict]:
    """Lista pistas para o humano. Nao corrige nada."""
    findings: list[dict] = []
    transactions = session.scalars(
        select(BankTransaction).where(
            BankTransaction.bank_account_id == bank_account_id,
            BankTransaction.transaction_date == day,
        )
    ).all()

    for transaction in transactions:
        if transaction.status in {UNMATCHED, "IMPORTED"}:
            findings.append({
                "type": "TRANSACTION_NOT_MAPPED",
                "bank_transaction_id": transaction.id,
                "amount": str(transaction.amount),
                "description": transaction.description_raw,
            })
        elif transaction.status == REVIEW_REQUIRED:
            findings.append({
                "type": "PENDING_HUMAN_REVIEW",
                "bank_transaction_id": transaction.id,
                "amount": str(transaction.amount),
            })

    # pagamentos duplicados: mesma referencia saindo duas vezes no mesmo dia
    debits = [t for t in transactions if t.direction == "DEBIT"]
    for first, second in ((a, b) for i, a in enumerate(debits) for b in debits[i + 1:]):
        if first.amount == second.amount and (
            same_ref(first.document_number, second.document_number)
            or first.description_normalized == second.description_normalized
        ):
            findings.append({
                "type": "POSSIBLE_DUPLICATE_PAYMENT",
                "bank_transaction_ids": [first.id, second.id],
                "amount": str(first.amount),
            })

    for group in session.scalars(
        select(ReconciliationGroup).where(
            ReconciliationGroup.status == "REVIEW_REQUIRED",
            func.abs(ReconciliationGroup.difference) > AMOUNT_TOLERANCE,
        )
    ):
        findings.append({
            "type": "GROUP_DIFFERENCE",
            "group_id": group.id,
            "difference": str(group.difference),
        })

    for model, kind in ((Payable, "PAYABLE"), (Receivable, "RECEIVABLE")):
        for title in session.scalars(
            select(model).where(model.status == "PARTIALLY_PAID", model.amount_open > 0)
        ):
            findings.append({
                "type": "PARTIALLY_SETTLED_TITLE",
                "entity_type": kind,
                "entity_id": title.id,
                "amount_open": str(title.amount_open),
            })
    return findings


def run_daily(session: Session, bank_account_id: int, day: dt.date) -> DailyReconciliation:
    balance = session.scalar(
        select(BankBalance).where(
            BankBalance.bank_account_id == bank_account_id, BankBalance.balance_date == day
        )
    )
    if balance is None:
        raise ValueError(f"Sem saldo bancario informado para {day}")

    def total(direction: str) -> Decimal:
        value = session.scalar(
            select(func.coalesce(func.sum(BankTransaction.amount), 0)).where(
                BankTransaction.bank_account_id == bank_account_id,
                BankTransaction.transaction_date == day,
                BankTransaction.direction == direction,
            )
        )
        return Decimal(value or 0)

    credits, debits = total("CREDIT"), total("DEBIT")
    expected = Decimal(balance.opening_balance) + credits - debits
    difference = expected - Decimal(balance.closing_balance)
    findings = investigate(session, bank_account_id, day)
    reconciled = abs(difference) <= AMOUNT_TOLERANCE and not findings

    existing = session.scalar(
        select(DailyReconciliation).where(
            DailyReconciliation.bank_account_id == bank_account_id,
            DailyReconciliation.reconciliation_date == day,
        )
    )
    record = existing or DailyReconciliation(
        bank_account_id=bank_account_id, reconciliation_date=day
    )
    record.opening_balance = balance.opening_balance
    record.total_credits = credits
    record.total_debits = debits
    record.expected_balance = expected
    record.real_balance = balance.closing_balance
    record.difference = difference
    record.status = "RECONCILED" if reconciled else "INVESTIGATION_REQUIRED"
    record.findings = {"items": findings}
    session.add(record)
    session.flush()
    audit(session, "daily_reconciliation", record.id, record.status,
          difference=str(difference), findings=len(findings))
    session.commit()
    return record
