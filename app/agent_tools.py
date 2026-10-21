"""Contratos e ferramentas read-only do agente de investigacao (Spec 2)."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import memory
from app.db import (
    AuditEvent,
    BankTransaction,
    DailyReconciliation,
    DIVERGENCE_CASE,
    Document,
    ExtractedDocument,
    Payable,
    ReconciliationGroup,
    ReconciliationGroupItem,
    ValidationResult,
)

MAX_AUDIT_EVENTS = 100
MAX_SIMILAR_CASES = 20
MAX_CONTENT_CHARS = 2000
MAX_RESULT_ITEMS = 50
MAX_FIELD_CHARS = 2000
MAX_PAYLOAD_DEPTH = 4
ALLOWED_AUDIT_ENTITIES = frozenset(
    {"daily_reconciliation", "bank_transaction", "reconciliation_group", "document"}
)


class AgentToolError(RuntimeError):
    """Erro de dominio seguro para devolver ao runtime do agente."""


class ToolNotFoundError(AgentToolError):
    """O recurso pedido nao existe."""


class ToolScopeError(AgentToolError):
    """O recurso existe, mas nao pertence ao caso autorizado."""


class InvalidToolInputError(AgentToolError):
    """O argumento nao pertence a allowlist da ferramenta."""


class AgentToolContext(BaseModel):
    """Capacidade imutavel resolvida pelo servidor para um unico caso."""

    bank_account_id: int = Field(gt=0)
    daily_reconciliation_id: int | None = Field(default=None, gt=0)
    reconciliation_group_id: int | None = Field(default=None, gt=0)
    allowed_transaction_ids: frozenset[int] = Field(default_factory=frozenset)
    allowed_group_ids: frozenset[int] = Field(default_factory=frozenset)
    allowed_document_ids: frozenset[int] = Field(default_factory=frozenset)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def exactly_one_case(self) -> AgentToolContext:
        if (self.daily_reconciliation_id is None) == (
            self.reconciliation_group_id is None
        ):
            raise ValueError("Contexto exige exatamente um tipo de caso")
        return self


class Evidence(BaseModel):
    source_type: Literal[
        "reconciliation", "transaction", "group", "memory", "audit", "document"
    ]
    source_id: str
    claim: str


class GroundedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)


class InvestigationRecommendation(BaseModel):
    classification: Literal[
        "UNMAPPED_TRANSACTION",
        "GROUP_DIFFERENCE",
        "POSSIBLE_DUPLICATE",
        "PARTIAL",
        "UNKNOWN",
    ]
    summary: str = Field(min_length=1, max_length=1000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    recommended_action: Literal["APPROVE", "REJECT", "REQUEST_DATA", "MANUAL_REVIEW"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1, max_length=20)
    claims: list[GroundedClaim] = Field(min_length=1, max_length=20)
    missing_information: list[str] = Field(default_factory=list)
    requires_human: Literal[True] = True


class ReconciliationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    bank_account_id: int
    reconciliation_date: dt.date
    opening_balance: Decimal
    total_credits: Decimal
    total_debits: Decimal
    expected_balance: Decimal
    real_balance: Decimal
    difference: Decimal
    status: str
    findings: list[dict[str, Any]]


class TransactionView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    bank_account_id: int
    transaction_date: dt.date
    amount: Decimal
    direction: str
    description: str | None
    document_number: str | None
    reference: str | None
    currency: str
    status: str


class GroupItemView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    entity_type: str
    entity_id: int | None
    amount_allocated: Decimal


class GroupView(BaseModel):
    id: int
    bank_transaction_id: int
    bank_account_id: int
    group_type: str
    status: str
    total_transaction_amount: Decimal
    total_allocated_amount: Decimal
    difference: Decimal
    confidence_score: int
    reasons: list[str]
    items: list[GroupItemView]


class SimilarCaseView(BaseModel):
    memory_id: int
    content: str
    resolution: str | None
    score: float
    via: list[str]
    source_entity: str | None
    source_id: int | None


class SimilarCasesResult(BaseModel):
    cases: list[SimilarCaseView]
    degraded: bool


class AuditEventView(BaseModel):
    id: int
    created_at: dt.datetime
    event_type: str
    actor_type: str
    actor_id: str | None
    metadata: dict[str, Any] | None


class ExtractedDocumentView(BaseModel):
    supplier_name: str | None
    supplier_cnpj: str | None
    invoice_number: str | None
    amount: Decimal | None
    currency: str
    due_date: dt.date | None
    bank_code: str | None
    bank_agency: str | None
    bank_account: str | None
    purchase_order: str | None
    bill_of_lading: str | None
    container: str | None
    cost_center: str | None
    expense_category: str | None
    observation: str | None
    structured_payload: dict[str, Any] | None
    model_name: str | None


class ValidationView(BaseModel):
    rule_code: str
    status: str
    expected_value: str | None
    actual_value: str | None
    message: str | None


class DocumentContextView(BaseModel):
    document_id: int
    filename: str
    document_type: str | None
    status: str
    extracted: ExtractedDocumentView | None
    validations: list[ValidationView]


def _clip(value: str | None, limit: int = MAX_FIELD_CHARS) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Limita texto e colecoes vindos de documento/audit nao confiavel."""
    if depth >= MAX_PAYLOAD_DEPTH:
        return "[max-depth]"
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, dict):
        return {
            str(key)[:120]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:MAX_RESULT_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_value(item, depth=depth + 1)
            for item in value[:MAX_RESULT_ITEMS]
        ]
    return value


def _require_account(actual: int, context: AgentToolContext) -> None:
    if actual != context.bank_account_id:
        raise ToolScopeError("Recurso fora da conta bancaria autorizada")


def _require_allowed(entity_id: int, allowed: frozenset[int], label: str) -> None:
    if entity_id not in allowed:
        raise ToolScopeError(f"{label} fora do caso autorizado")


def _group_resources(
    session: Session, group_ids: set[int], bank_account_id: int
) -> tuple[set[int], set[int]]:
    """Resolve transacoes e documentos alcancaveis pelos grupos do caso."""
    transaction_ids: set[int] = set()
    document_ids: set[int] = set()
    for group_id in group_ids:
        group = session.get(ReconciliationGroup, group_id)
        if group is None:
            raise ToolNotFoundError(f"Grupo de conciliacao inexistente: {group_id}")
        transaction = session.get(BankTransaction, group.bank_transaction_id)
        if transaction is None:
            raise ToolNotFoundError("Transacao do grupo inexistente")
        if transaction.bank_account_id != bank_account_id:
            raise ToolScopeError("Grupo fora da conta bancaria do caso")
        transaction_ids.add(transaction.id)
        items = session.scalars(
            select(ReconciliationGroupItem).where(
                ReconciliationGroupItem.reconciliation_group_id == group.id,
                ReconciliationGroupItem.entity_type == "PAYABLE",
            )
        ).all()
        payable_ids = [item.entity_id for item in items if item.entity_id is not None]
        if payable_ids:
            document_ids.update(
                document_id
                for document_id in session.scalars(
                    select(Payable.document_id).where(
                        Payable.id.in_(payable_ids), Payable.document_id.is_not(None)
                    )
                )
                if document_id is not None
            )
    return transaction_ids, document_ids


def resolve_investigation_scope(
    session: Session,
    *,
    daily_reconciliation_id: int | None = None,
    reconciliation_group_id: int | None = None,
) -> AgentToolContext:
    """Cria a capacidade do run a partir do caso, nunca de IDs escolhidos pelo modelo."""
    if (daily_reconciliation_id is None) == (reconciliation_group_id is None):
        raise InvalidToolInputError("Informe exatamente um tipo de caso")

    group_ids: set[int] = set()
    transaction_ids: set[int] = set()
    document_ids: set[int] = set()
    if daily_reconciliation_id is not None:
        record = session.get(DailyReconciliation, daily_reconciliation_id)
        if record is None:
            raise ToolNotFoundError("Conciliacao inexistente")
        bank_account_id = record.bank_account_id
        for finding in (record.findings or {}).get("items", []):
            transaction_id = finding.get("bank_transaction_id")
            if isinstance(transaction_id, int):
                transaction_ids.add(transaction_id)
            transaction_ids.update(
                value
                for value in finding.get("bank_transaction_ids", [])
                if isinstance(value, int)
            )
            group_id = finding.get("group_id")
            if isinstance(group_id, int):
                group_ids.add(group_id)
    else:
        group = session.get(ReconciliationGroup, reconciliation_group_id)
        if group is None:
            raise ToolNotFoundError("Grupo de conciliacao inexistente")
        transaction = session.get(BankTransaction, group.bank_transaction_id)
        if transaction is None:
            raise ToolNotFoundError("Transacao do grupo inexistente")
        bank_account_id = transaction.bank_account_id
        group_ids.add(group.id)

    group_transactions, group_documents = _group_resources(
        session, group_ids, bank_account_id
    )
    transaction_ids.update(group_transactions)
    document_ids.update(group_documents)
    for transaction_id in transaction_ids:
        transaction = session.get(BankTransaction, transaction_id)
        if transaction is None:
            raise ToolNotFoundError(f"Transacao inexistente: {transaction_id}")
        if transaction.bank_account_id != bank_account_id:
            raise ToolScopeError("Finding referencia transacao fora da conta do caso")

    return AgentToolContext(
        bank_account_id=bank_account_id,
        daily_reconciliation_id=daily_reconciliation_id,
        reconciliation_group_id=reconciliation_group_id,
        allowed_transaction_ids=frozenset(transaction_ids),
        allowed_group_ids=frozenset(group_ids),
        allowed_document_ids=frozenset(document_ids),
    )


def _transaction(
    session: Session, transaction_id: int, context: AgentToolContext
) -> BankTransaction:
    transaction = session.get(BankTransaction, transaction_id)
    if transaction is None:
        raise ToolNotFoundError("Transacao inexistente")
    _require_account(transaction.bank_account_id, context)
    _require_allowed(transaction.id, context.allowed_transaction_ids, "Transacao")
    return transaction


def _group_with_transaction(
    session: Session, group_id: int, context: AgentToolContext
) -> tuple[ReconciliationGroup, BankTransaction]:
    group = session.get(ReconciliationGroup, group_id)
    if group is None:
        raise ToolNotFoundError("Grupo de conciliacao inexistente")
    _require_allowed(group.id, context.allowed_group_ids, "Grupo")
    transaction = _transaction(session, group.bank_transaction_id, context)
    return group, transaction


def get_reconciliation(
    session: Session, reconciliation_id: int, context: AgentToolContext
) -> ReconciliationView:
    record = session.get(DailyReconciliation, reconciliation_id)
    if record is None:
        raise ToolNotFoundError("Conciliacao inexistente")
    _require_account(record.bank_account_id, context)
    if record.id != context.daily_reconciliation_id:
        raise ToolScopeError("Conciliacao fora do caso autorizado")
    return ReconciliationView(
        id=record.id,
        bank_account_id=record.bank_account_id,
        reconciliation_date=record.reconciliation_date,
        opening_balance=record.opening_balance,
        total_credits=record.total_credits,
        total_debits=record.total_debits,
        expected_balance=record.expected_balance,
        real_balance=record.real_balance,
        difference=record.difference,
        status=record.status,
        findings=[
            _bounded_value(finding)
            for finding in list((record.findings or {}).get("items", []))[
                :MAX_RESULT_ITEMS
            ]
        ],
    )


def get_transaction(
    session: Session, transaction_id: int, context: AgentToolContext
) -> TransactionView:
    transaction = _transaction(session, transaction_id, context)
    return TransactionView(
        id=transaction.id,
        bank_account_id=transaction.bank_account_id,
        transaction_date=transaction.transaction_date,
        amount=transaction.amount,
        direction=transaction.direction,
        description=_clip(transaction.description_raw),
        document_number=_clip(transaction.document_number),
        reference=_clip(transaction.reference),
        currency=transaction.currency,
        status=transaction.status,
    )


def get_group(
    session: Session, group_id: int, context: AgentToolContext
) -> GroupView:
    group, transaction = _group_with_transaction(session, group_id, context)
    items = session.scalars(
        select(ReconciliationGroupItem)
        .where(ReconciliationGroupItem.reconciliation_group_id == group.id)
        .order_by(ReconciliationGroupItem.id)
    ).all()
    return GroupView(
        id=group.id,
        bank_transaction_id=group.bank_transaction_id,
        bank_account_id=transaction.bank_account_id,
        group_type=group.group_type,
        status=group.status,
        total_transaction_amount=group.total_transaction_amount,
        total_allocated_amount=group.total_allocated_amount,
        difference=group.difference,
        confidence_score=group.confidence_score,
        reasons=[
            _clip(str(reason))
            for reason in list((group.match_reasons or {}).get("reasons", []))[
                :MAX_RESULT_ITEMS
            ]
        ],
        items=[
            GroupItemView.model_validate(item) for item in items[:MAX_RESULT_ITEMS]
        ],
    )


def search_similar_cases(
    session: Session,
    query: str,
    context: AgentToolContext,
    *,
    limit: int = 5,
) -> SimilarCasesResult:
    normalized = query.strip()
    if not normalized:
        raise InvalidToolInputError("Consulta de memoria vazia")
    if not 1 <= limit <= MAX_SIMILAR_CASES:
        raise InvalidToolInputError(
            f"Limite deve estar entre 1 e {MAX_SIMILAR_CASES}"
        )
    outcome = memory.retrieve_outcome(
        session,
        DIVERGENCE_CASE,
        normalized,
        filters={"bank_account_id": context.bank_account_id},
        k=limit,
        audit_degradation=False,
    )
    return SimilarCasesResult(
        degraded=outcome.degraded,
        cases=[
            SimilarCaseView(
                memory_id=hit.item.id,
                content=hit.item.content[:MAX_CONTENT_CHARS],
                resolution=_clip((hit.item.item_metadata or {}).get("resolution")),
                score=hit.score,
                via=sorted(hit.via),
                source_entity=hit.item.source_entity,
                source_id=hit.item.source_id,
            )
            for hit in outcome.hits
        ],
    )


def _authorize_audit_entity(
    session: Session, entity_type: str, entity_id: int, context: AgentToolContext
) -> None:
    if entity_type == "daily_reconciliation":
        get_reconciliation(session, entity_id, context)
    elif entity_type == "bank_transaction":
        _transaction(session, entity_id, context)
    elif entity_type == "reconciliation_group":
        _group_with_transaction(session, entity_id, context)
    elif entity_type == "document":
        document = session.get(Document, entity_id)
        if document is None:
            raise ToolNotFoundError("Documento inexistente")
        _require_allowed(document.id, context.allowed_document_ids, "Documento")
    else:
        raise InvalidToolInputError("Tipo de entidade nao permitido para audit trail")


def get_audit_trail(
    session: Session,
    entity_type: str,
    entity_id: int,
    context: AgentToolContext,
    *,
    limit: int = 50,
) -> list[AuditEventView]:
    if entity_type not in ALLOWED_AUDIT_ENTITIES:
        raise InvalidToolInputError("Tipo de entidade nao permitido para audit trail")
    if not 1 <= limit <= MAX_AUDIT_EVENTS:
        raise InvalidToolInputError(f"Limite deve estar entre 1 e {MAX_AUDIT_EVENTS}")
    _authorize_audit_entity(session, entity_type, entity_id, context)
    events = session.scalars(
        select(AuditEvent)
        .where(
            AuditEvent.entity_type == entity_type,
            AuditEvent.entity_id == str(entity_id),
        )
        .order_by(AuditEvent.id)
        .limit(limit)
    ).all()
    return [
        AuditEventView(
            id=event.id,
            created_at=event.created_at,
            event_type=event.event_type,
            actor_type=event.actor_type,
            actor_id=event.actor_id,
            metadata=_bounded_value(event.event_metadata),
        )
        for event in events
    ]


def get_document_context(
    session: Session,
    group_id: int,
    group_item_id: int,
    context: AgentToolContext,
) -> DocumentContextView:
    group, _ = _group_with_transaction(session, group_id, context)
    item = session.get(ReconciliationGroupItem, group_item_id)
    if item is None or item.reconciliation_group_id != group.id:
        raise ToolNotFoundError("Item do grupo inexistente")
    if item.entity_type != "PAYABLE" or item.entity_id is None:
        raise InvalidToolInputError("Contexto documental exige um item PAYABLE")

    payable = session.get(Payable, item.entity_id)
    if payable is None or payable.document_id is None:
        raise ToolNotFoundError("Titulo sem documento associado")
    document = session.get(Document, payable.document_id)
    if document is None:
        raise ToolNotFoundError("Documento associado inexistente")
    _require_allowed(document.id, context.allowed_document_ids, "Documento")

    extracted = session.scalar(
        select(ExtractedDocument)
        .where(ExtractedDocument.document_id == document.id)
        .order_by(ExtractedDocument.id.desc())
        .limit(1)
    )
    validations = session.scalars(
        select(ValidationResult)
        .where(ValidationResult.document_id == document.id)
        .order_by(ValidationResult.id)
        .limit(MAX_RESULT_ITEMS)
    ).all()
    extracted_view = None
    if extracted is not None:
        extracted_view = ExtractedDocumentView(
            supplier_name=_clip(extracted.supplier_name),
            supplier_cnpj=extracted.supplier_cnpj,
            invoice_number=_clip(extracted.invoice_number),
            amount=extracted.amount,
            currency=extracted.currency,
            due_date=extracted.due_date,
            bank_code=_clip(extracted.bank_code),
            bank_agency=_clip(extracted.bank_agency),
            bank_account=_clip(extracted.bank_account),
            purchase_order=_clip(extracted.purchase_order),
            bill_of_lading=_clip(extracted.bill_of_lading),
            container=_clip(extracted.container),
            cost_center=_clip(extracted.cost_center),
            expense_category=_clip(extracted.expense_category),
            observation=_clip(extracted.observation),
            structured_payload=_bounded_value(extracted.structured_payload),
            model_name=_clip(extracted.model_name),
        )
    return DocumentContextView(
        document_id=document.id,
        filename=_clip(document.filename),
        document_type=document.document_type,
        status=document.status,
        extracted=extracted_view,
        validations=[
            ValidationView(
                rule_code=result.rule_code,
                status=result.status,
                expected_value=_clip(result.expected_value),
                actual_value=_clip(result.actual_value),
                message=_clip(result.message),
            )
            for result in validations
        ],
    )
