"""Schema (spec secao 32) + sessao + audit trail."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import DATABASE_URL

Money = Numeric(18, 2)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Supplier(Base, TimestampMixin):
    __tablename__ = "suppliers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    cnpj: Mapped[str | None] = mapped_column(String(14), index=True)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    cnpj: Mapped[str | None] = mapped_column(String(14), index=True)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")


class SupplierBankAccount(Base):
    __tablename__ = "supplier_bank_accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), index=True)
    bank_code: Mapped[str | None] = mapped_column(String(10))
    agency: Mapped[str | None] = mapped_column(String(20))
    account: Mapped[str | None] = mapped_column(String(30))
    account_digit: Mapped[str | None] = mapped_column(String(5))
    pix_key: Mapped[str | None] = mapped_column(String(140))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    validated_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(30))  # UPLOAD / EMAIL / ERP / WEBHOOK
    document_type: Mapped[str | None] = mapped_column(String(30))
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(100))
    sender: Mapped[str | None] = mapped_column(String(255))
    received_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    status: Mapped[str] = mapped_column(String(20), default="RECEIVED")
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    raw_metadata: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ExtractedDocument(Base, TimestampMixin):
    __tablename__ = "extracted_documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    supplier_name: Mapped[str | None] = mapped_column(String(255))
    supplier_cnpj: Mapped[str | None] = mapped_column(String(14))
    invoice_number: Mapped[str | None] = mapped_column(String(60))
    amount: Mapped[Decimal | None] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(3), default="BRL")
    due_date: Mapped[dt.date | None] = mapped_column(Date)
    bank_code: Mapped[str | None] = mapped_column(String(10))
    bank_agency: Mapped[str | None] = mapped_column(String(20))
    bank_account: Mapped[str | None] = mapped_column(String(30))
    purchase_order: Mapped[str | None] = mapped_column(String(60))
    bill_of_lading: Mapped[str | None] = mapped_column(String(60))
    container: Mapped[str | None] = mapped_column(String(20))
    cost_center: Mapped[str | None] = mapped_column(String(40))
    expense_category: Mapped[str | None] = mapped_column(String(60))
    observation: Mapped[str | None] = mapped_column(Text)
    structured_payload: Mapped[dict | None] = mapped_column(JSON)
    model_name: Mapped[str | None] = mapped_column(String(80))


class PurchaseOrder(Base, TimestampMixin):
    """Pedido/embarque. Existe para as regras PURCHASE_ORDER_EXISTS / BL / CONTAINER."""

    __tablename__ = "purchase_orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    number: Mapped[str] = mapped_column(String(60), unique=True)
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"))
    amount: Mapped[Decimal | None] = mapped_column(Money)
    bill_of_lading: Mapped[str | None] = mapped_column(String(60), index=True)
    container: Mapped[str | None] = mapped_column(String(20), index=True)
    cost_center: Mapped[str | None] = mapped_column(String(40))


class Payable(Base, TimestampMixin):
    __tablename__ = "payables"
    id: Mapped[int] = mapped_column(primary_key=True)
    supplier_id: Mapped[int] = mapped_column(ForeignKey("suppliers.id"), index=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"))
    invoice_number: Mapped[str | None] = mapped_column(String(60), index=True)
    purchase_order: Mapped[str | None] = mapped_column(String(60))
    bill_of_lading: Mapped[str | None] = mapped_column(String(60))
    container: Mapped[str | None] = mapped_column(String(20))
    amount_original: Mapped[Decimal] = mapped_column(Money)
    amount_open: Mapped[Decimal] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(3), default="BRL")
    due_date: Mapped[dt.date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    cost_center: Mapped[str | None] = mapped_column(String(40))
    erp_reference: Mapped[str | None] = mapped_column(String(60))


class Receivable(Base, TimestampMixin):
    __tablename__ = "receivables"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    invoice_number: Mapped[str | None] = mapped_column(String(60), index=True)
    bill_of_lading: Mapped[str | None] = mapped_column(String(60))
    container: Mapped[str | None] = mapped_column(String(20))
    amount_original: Mapped[Decimal] = mapped_column(Money)
    amount_open: Mapped[Decimal] = mapped_column(Money)
    currency: Mapped[str] = mapped_column(String(3), default="BRL")
    due_date: Mapped[dt.date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    erp_reference: Mapped[str | None] = mapped_column(String(60))


class BankAccount(Base, TimestampMixin):
    __tablename__ = "bank_accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    company: Mapped[str] = mapped_column(String(120))
    bank_code: Mapped[str | None] = mapped_column(String(10))
    agency: Mapped[str | None] = mapped_column(String(20))
    account: Mapped[str | None] = mapped_column(String(30))
    currency: Mapped[str] = mapped_column(String(3), default="BRL")


class BankTransaction(Base, TimestampMixin):
    __tablename__ = "bank_transactions"
    __table_args__ = (UniqueConstraint("bank_account_id", "external_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    bank_account_id: Mapped[int] = mapped_column(ForeignKey("bank_accounts.id"), index=True)
    external_id: Mapped[str] = mapped_column(String(120))
    transaction_date: Mapped[dt.date] = mapped_column(Date, index=True)
    amount: Mapped[Decimal] = mapped_column(Money)  # sempre positivo
    direction: Mapped[str] = mapped_column(String(10))  # CREDIT / DEBIT
    description_raw: Mapped[str | None] = mapped_column(Text)
    description_normalized: Mapped[str | None] = mapped_column(Text)
    document_number: Mapped[str | None] = mapped_column(String(60))
    reference: Mapped[str | None] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(3), default="BRL")
    status: Mapped[str] = mapped_column(String(25), default="IMPORTED")
    raw_payload: Mapped[dict | None] = mapped_column(JSON)


class BankBalance(Base):
    """Saldo informado pelo banco, usado na conciliacao diaria (spec secao 28)."""

    __tablename__ = "bank_balances"
    __table_args__ = (UniqueConstraint("bank_account_id", "balance_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    bank_account_id: Mapped[int] = mapped_column(ForeignKey("bank_accounts.id"), index=True)
    balance_date: Mapped[dt.date] = mapped_column(Date)
    opening_balance: Mapped[Decimal] = mapped_column(Money)
    closing_balance: Mapped[Decimal] = mapped_column(Money)


class MatchingCandidate(Base, TimestampMixin):
    __tablename__ = "matching_candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    bank_transaction_id: Mapped[int] = mapped_column(
        ForeignKey("bank_transactions.id"), index=True
    )
    candidate_type: Mapped[str] = mapped_column(String(20))  # PAYABLE / RECEIVABLE
    candidate_id: Mapped[int] = mapped_column()
    score: Mapped[int] = mapped_column()
    match_reasons: Mapped[dict | None] = mapped_column(JSON)


class ValidationResult(Base, TimestampMixin):
    __tablename__ = "validation_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    rule_code: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(10))  # PASS / FAIL / SKIP
    expected_value: Mapped[str | None] = mapped_column(String(255))
    actual_value: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str | None] = mapped_column(Text)


class ReconciliationGroup(Base, TimestampMixin):
    __tablename__ = "reconciliation_groups"
    id: Mapped[int] = mapped_column(primary_key=True)
    bank_transaction_id: Mapped[int] = mapped_column(
        ForeignKey("bank_transactions.id"), index=True
    )
    group_type: Mapped[str] = mapped_column(String(20))
    total_transaction_amount: Mapped[Decimal] = mapped_column(Money)
    total_allocated_amount: Mapped[Decimal] = mapped_column(Money)
    difference: Mapped[Decimal] = mapped_column(Money)
    status: Mapped[str] = mapped_column(String(25))
    confidence_score: Mapped[int] = mapped_column(default=0)
    match_reasons: Mapped[dict | None] = mapped_column(JSON)


class ReconciliationGroupItem(Base, TimestampMixin):
    __tablename__ = "reconciliation_group_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    reconciliation_group_id: Mapped[int] = mapped_column(
        ForeignKey("reconciliation_groups.id"), index=True
    )
    entity_type: Mapped[str] = mapped_column(String(20))  # PAYABLE/RECEIVABLE/FEE/...
    entity_id: Mapped[int | None] = mapped_column()
    amount_allocated: Mapped[Decimal] = mapped_column(Money)


class DailyReconciliation(Base, TimestampMixin):
    __tablename__ = "daily_reconciliations"
    __table_args__ = (UniqueConstraint("bank_account_id", "reconciliation_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    bank_account_id: Mapped[int] = mapped_column(ForeignKey("bank_accounts.id"), index=True)
    reconciliation_date: Mapped[dt.date] = mapped_column(Date)
    opening_balance: Mapped[Decimal] = mapped_column(Money)
    total_credits: Mapped[Decimal] = mapped_column(Money)
    total_debits: Mapped[Decimal] = mapped_column(Money)
    expected_balance: Mapped[Decimal] = mapped_column(Money)
    real_balance: Mapped[Decimal] = mapped_column(Money)
    difference: Mapped[Decimal] = mapped_column(Money)
    status: Mapped[str] = mapped_column(String(25))
    findings: Mapped[dict | None] = mapped_column(JSON)


class AuditEvent(Base, TimestampMixin):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), index=True)
    entity_id: Mapped[str] = mapped_column(String(40), index=True)
    event_type: Mapped[str] = mapped_column(String(50))
    actor_type: Mapped[str] = mapped_column(String(20), default="SYSTEM")
    actor_id: Mapped[str | None] = mapped_column(String(80))
    event_metadata: Mapped[dict | None] = mapped_column(JSON)


def audit(
    session: Session,
    entity_type: str,
    entity_id: int | str,
    event_type: str,
    actor_type: str = "SYSTEM",
    actor_id: str | None = None,
    **metadata,
) -> AuditEvent:
    event = AuditEvent(
        entity_type=entity_type,
        entity_id=str(entity_id),
        event_type=event_type,
        actor_type=actor_type,
        actor_id=actor_id,
        event_metadata=metadata or None,
    )
    session.add(event)
    return event


engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False, future=True)


def init_db() -> None:
    # ponytail: create_all no lugar de Alembic; trocar por migrations antes do primeiro deploy.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def get_session():
    with SessionLocal() as session:
        yield session
