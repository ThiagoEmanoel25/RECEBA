"""Materializacao deterministica dos casos de eval no Postgres de teste."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from app import db, memory
from app.agent_tools import AgentToolContext, resolve_investigation_scope

if TYPE_CHECKING:
    from app.evals import EvalCase

FIXTURE_DATE = dt.date(2026, 10, 20)


@dataclass(frozen=True, slots=True)
class MaterializedCase:
    input: dict[str, Any]
    context: AgentToolContext
    aliases: dict[str, int]


def _ids(case: EvalCase, prefix: str) -> list[tuple[str, int]]:
    aliases = sorted(
        case.expected.available_evidence_ids | case.expected.required_evidence_ids
    )
    result = []
    for alias in aliases:
        kind, separator, raw_id = alias.partition(":")
        if kind == prefix and separator and raw_id.isdigit():
            result.append((alias, int(raw_id)))
    return result


def materialize_case(session: Session, case: EvalCase) -> MaterializedCase:
    """Cria um mundo minimo, isolado e executavel para uma linha do dataset."""
    if case.fixture.scenario != case.category:
        raise ValueError("Fixture e categoria devem representar o mesmo cenario")

    account = db.BankAccount(company=f"EVAL {case.case_id}", currency="BRL")
    supplier = db.Supplier(name=f"FORNECEDOR {case.case_id}")
    session.add_all([account, supplier])
    session.flush()
    aliases: dict[str, int] = {}

    documents: list[db.Document] = []
    for alias, document_id in _ids(case, "doc"):
        document = db.Document(
            id=document_id,
            source="UPLOAD",
            document_type="INVOICE",
            filename=f"{case.case_id}.pdf",
            storage_key=f"eval/{case.case_id}.pdf",
            status="VALIDATED",
            content_hash=f"{document_id:064x}"[-64:],
        )
        session.add(document)
        documents.append(document)
        aliases[alias] = document_id
    session.flush()

    payables: list[db.Payable] = []
    for index, document in enumerate(documents, 1):
        payable = db.Payable(
            supplier_id=supplier.id,
            document_id=document.id,
            invoice_number=f"EVAL-{document.id}",
            amount_original=Decimal("100.00"),
            amount_open=Decimal("100.00"),
            status="OPEN",
        )
        session.add(payable)
        session.flush()
        payables.append(payable)
        session.add_all(
            [
                db.ExtractedDocument(
                    document_id=document.id,
                    supplier_name=supplier.name,
                    invoice_number=payable.invoice_number,
                    amount=Decimal("100.00"),
                    currency="BRL",
                    observation=str(case.input.get("untrusted_text") or "fixture de eval"),
                    model_name="fixture",
                ),
                db.ValidationResult(
                    document_id=document.id,
                    rule_code="AMOUNT_POSITIVE",
                    status="PASS",
                    actual_value="100.00",
                ),
            ]
        )

    transactions: list[db.BankTransaction] = []
    transaction_aliases = _ids(case, "tx")
    if not transaction_aliases:
        fallback_id = 70_000 + sum(ord(char) for char in case.case_id)
        transaction_aliases = [(f"fixture-tx:{fallback_id}", fallback_id)]
    for index, (alias, transaction_id) in enumerate(transaction_aliases):
        transaction = db.BankTransaction(
            id=transaction_id,
            bank_account_id=account.id,
            external_id=f"{case.case_id}-{transaction_id}",
            transaction_date=FIXTURE_DATE,
            amount=Decimal("100.00"),
            direction="DEBIT",
            description_raw=str(
                case.input.get("untrusted_text") or f"TRANSACAO {case.case_id} {index}"
            ),
            description_normalized=f"TRANSACAO {case.case_id} {index}".upper(),
            status="IMPORTED",
        )
        session.add(transaction)
        transactions.append(transaction)
        aliases[alias] = transaction_id
    session.flush()

    group_aliases = _ids(case, "group")
    requested_group_id = case.input.get("reconciliation_group_id")
    if isinstance(requested_group_id, int) and all(
        group_id != requested_group_id for _, group_id in group_aliases
    ):
        group_aliases.append((f"fixture-group:{requested_group_id}", requested_group_id))
    needs_group = bool(group_aliases or documents or requested_group_id)
    groups: list[db.ReconciliationGroup] = []
    if needs_group:
        if not group_aliases:
            hidden_id = 80_000 + sum(ord(char) for char in case.case_id)
            group_aliases = [(f"fixture-group:{hidden_id}", hidden_id)]
        for alias, group_id in group_aliases:
            group = db.ReconciliationGroup(
                id=group_id,
                bank_transaction_id=transactions[0].id,
                group_type="PARTIAL" if case.category == "partial_adjustment" else "SINGLE",
                total_transaction_amount=Decimal("100.00"),
                total_allocated_amount=Decimal("90.00"),
                difference=Decimal("10.00"),
                status="REVIEW_REQUIRED",
                confidence_score=70,
                match_reasons={"reasons": ["eval_fixture"]},
            )
            session.add(group)
            session.flush()
            groups.append(group)
            aliases[alias] = group_id
            for payable in payables:
                session.add(
                    db.ReconciliationGroupItem(
                        reconciliation_group_id=group.id,
                        entity_type="PAYABLE",
                        entity_id=payable.id,
                        amount_allocated=Decimal("90.00"),
                    )
                )

    for alias, memory_id in _ids(case, "mem"):
        content = f"Caso semelhante para {case.case_id}"
        vectors = memory.embed([content])
        item = db.MemoryItem(
            id=memory_id,
            kind=db.DIVERGENCE_CASE,
            content=content,
            embedding=vectors[0] if vectors else None,
            item_metadata={"resolution": "revisao humana"},
            source_entity="eval_fixture",
            source_id=memory_id,
            bank_account_id=account.id,
        )
        session.add(item)
        session.flush()
        aliases[alias] = memory_id

    daily_reconciliation_id = case.input.get("daily_reconciliation_id")
    if isinstance(daily_reconciliation_id, int):
        if case.category == "possible_duplicate":
            finding: dict[str, Any] = {
                "type": "POSSIBLE_DUPLICATE_PAYMENT",
                "bank_transaction_ids": [transaction.id for transaction in transactions],
            }
            if groups:
                finding["group_id"] = groups[0].id
        elif groups:
            finding = {"type": "GROUP_DIFFERENCE", "group_id": groups[0].id}
        else:
            finding = {
                "type": "TRANSACTION_NOT_MAPPED",
                "bank_transaction_id": transactions[0].id,
            }
        reconciliation = db.DailyReconciliation(
            id=daily_reconciliation_id,
            bank_account_id=account.id,
            reconciliation_date=FIXTURE_DATE,
            opening_balance=Decimal("1000.00"),
            total_credits=Decimal("0.00"),
            total_debits=Decimal("100.00"),
            expected_balance=Decimal("900.00"),
            real_balance=Decimal("890.00"),
            difference=Decimal("10.00"),
            status="INVESTIGATION_REQUIRED",
            findings={"items": [finding]},
        )
        session.add(reconciliation)
        session.flush()
        aliases[f"rec:{reconciliation.id}"] = reconciliation.id
        materialized_input = {
            **case.input,
            "daily_reconciliation_id": reconciliation.id,
        }
        context = resolve_investigation_scope(
            session, daily_reconciliation_id=reconciliation.id
        )
        audit_entity = "daily_reconciliation"
        audit_entity_id = reconciliation.id
    elif groups:
        group = next(
            (
                candidate
                for candidate in groups
                if candidate.id == requested_group_id
            ),
            groups[0],
        )
        materialized_input = {
            **case.input,
            "reconciliation_group_id": group.id,
        }
        context = resolve_investigation_scope(session, reconciliation_group_id=group.id)
        audit_entity = "reconciliation_group"
        audit_entity_id = group.id
    else:
        raise ValueError("Fixture precisa materializar conciliacao ou grupo")

    for alias, audit_id in _ids(case, "audit"):
        session.add(
            db.AuditEvent(
                id=audit_id,
                entity_type=audit_entity,
                entity_id=str(audit_entity_id),
                event_type="EVAL_FIXTURE",
                event_metadata={"case_id": case.case_id},
            )
        )
        aliases[alias] = audit_id
    session.flush()
    return MaterializedCase(
        input=materialized_input,
        context=context,
        aliases=aliases,
    )
