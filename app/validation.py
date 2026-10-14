"""Validation Engine (spec secao 10). Nenhum LLM aqui: regra e codigo."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import AMOUNT_TOLERANCE
from app.db import (
    Document,
    ExtractedDocument,
    Payable,
    PurchaseOrder,
    Supplier,
    SupplierBankAccount,
    ValidationResult,
    audit,
)
from app.parser import digits, same_ref

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Falha nessas regras bloqueia o titulo; nas outras, manda para revisao humana.
BLOCKING_RULES = {"BANK_ACCOUNT_MATCH", "INVOICE_NOT_DUPLICATE", "SUPPLIER_CNPJ_MATCH"}


@dataclass(slots=True)
class Check:
    rule_code: str
    status: str
    expected: str | None = None
    actual: str | None = None
    message: str | None = None


def valid_cnpj(raw: str | None) -> bool:
    """Digitos verificadores do CNPJ (modulo 11)."""
    cnpj = digits(raw)
    if len(cnpj) != 14 or cnpj == cnpj[0] * 14:
        return False
    for size in (12, 13):
        weights = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2][-size:]
        total = sum(int(d) * w for d, w in zip(cnpj[:size], weights))
        check = 11 - total % 11
        if int(cnpj[size]) != (0 if check >= 10 else check):
            return False
    return True


def _supplier(session: Session, data: ExtractedDocument) -> Supplier | None:
    cnpj = digits(data.supplier_cnpj)
    if cnpj:
        found = session.scalar(select(Supplier).where(Supplier.cnpj == cnpj))
        if found:
            return found
    if data.supplier_name:
        return session.scalar(
            select(Supplier).where(Supplier.name.ilike(data.supplier_name.strip()))
        )
    return None


class ValidationEngine:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -- regras -------------------------------------------------------------
    def supplier_exists(self, data, supplier) -> Check:
        if supplier is None:
            return Check("SUPPLIER_EXISTS", FAIL, actual=data.supplier_name,
                         message="Fornecedor nao cadastrado")
        return Check("SUPPLIER_EXISTS", PASS, actual=str(supplier.id))

    def supplier_cnpj_match(self, data, supplier) -> Check:
        cnpj = digits(data.supplier_cnpj)
        if not cnpj:
            return Check("SUPPLIER_CNPJ_MATCH", FAIL, message="CNPJ ausente no documento")
        if not valid_cnpj(cnpj):
            return Check("SUPPLIER_CNPJ_MATCH", FAIL, actual=cnpj,
                         message="CNPJ invalido (digito verificador)")
        if supplier is None:
            return Check("SUPPLIER_CNPJ_MATCH", SKIP, message="Sem fornecedor para comparar")
        if digits(supplier.cnpj) != cnpj:
            return Check("SUPPLIER_CNPJ_MATCH", FAIL, expected=supplier.cnpj, actual=cnpj,
                         message="CNPJ divergente do cadastro")
        return Check("SUPPLIER_CNPJ_MATCH", PASS, actual=cnpj)

    def bank_account_match(self, data, supplier) -> Check:
        informed = (digits(data.bank_code), digits(data.bank_agency), digits(data.bank_account))
        if not any(informed):
            return Check("BANK_ACCOUNT_MATCH", SKIP, message="Documento sem dados bancarios")
        if supplier is None:
            return Check("BANK_ACCOUNT_MATCH", FAIL, message="Sem cadastro para comparar conta")
        registered = self.session.scalars(
            select(SupplierBankAccount).where(
                SupplierBankAccount.supplier_id == supplier.id,
                SupplierBankAccount.active.is_(True),
            )
        ).all()
        for account in registered:
            candidate = (digits(account.bank_code), digits(account.agency),
                         digits(account.account))
            if candidate == informed:
                return Check("BANK_ACCOUNT_MATCH", PASS, actual="/".join(informed))
        return Check(
            "BANK_ACCOUNT_MATCH", FAIL,
            expected="; ".join(
                f"{digits(a.bank_code)}/{digits(a.agency)}/{digits(a.account)}"
                for a in registered
            ) or None,
            actual="/".join(informed),
            message="BANK_ACCOUNT_MISMATCH: conta nao confere com o cadastro",
        )

    def invoice_not_duplicate(self, document, data, supplier) -> Check:
        if not data.invoice_number:
            return Check("INVOICE_NOT_DUPLICATE", SKIP, message="Documento sem numero")
        query = select(Payable).where(Payable.invoice_number.is_not(None))
        if supplier is not None:
            query = query.where(Payable.supplier_id == supplier.id)
        for payable in self.session.scalars(query):
            if payable.document_id == document.id:
                continue
            if same_ref(payable.invoice_number, data.invoice_number):
                return Check("INVOICE_NOT_DUPLICATE", FAIL, expected=payable.invoice_number,
                             actual=data.invoice_number,
                             message=f"Invoice ja lancada no titulo {payable.id}")
        return Check("INVOICE_NOT_DUPLICATE", PASS, actual=data.invoice_number)

    def amount_match(self, data, order) -> Check:
        if data.amount is None or data.amount <= 0:
            return Check("AMOUNT_MATCH", FAIL, actual=str(data.amount),
                         message="Valor ausente ou invalido")
        if order is None or order.amount is None:
            return Check("AMOUNT_MATCH", SKIP, message="Sem pedido para comparar valor")
        if abs(Decimal(data.amount) - Decimal(order.amount)) > AMOUNT_TOLERANCE:
            return Check("AMOUNT_MATCH", FAIL, expected=str(order.amount),
                         actual=str(data.amount), message="Valor divergente do pedido")
        return Check("AMOUNT_MATCH", PASS, actual=str(data.amount))

    def due_date_valid(self, data) -> Check:
        if data.due_date is None:
            return Check("DUE_DATE_VALID", FAIL, message="Vencimento ausente")
        if data.due_date < dt.date.today() - dt.timedelta(days=365 * 2):
            return Check("DUE_DATE_VALID", FAIL, actual=str(data.due_date),
                         message="Vencimento implausivel")
        return Check("DUE_DATE_VALID", PASS, actual=str(data.due_date))

    def purchase_order_exists(self, data, order) -> Check:
        if not data.purchase_order:
            return Check("PURCHASE_ORDER_EXISTS", SKIP)
        if order is None:
            return Check("PURCHASE_ORDER_EXISTS", FAIL, actual=data.purchase_order,
                         message="Pedido inexistente")
        return Check("PURCHASE_ORDER_EXISTS", PASS, actual=order.number)

    def bl_exists(self, data, order) -> Check:
        if not data.bill_of_lading:
            return Check("BL_EXISTS", SKIP)
        found = self.session.scalar(
            select(PurchaseOrder).where(PurchaseOrder.bill_of_lading == data.bill_of_lading)
        )
        if found is None:
            return Check("BL_EXISTS", FAIL, actual=data.bill_of_lading, message="BL inexistente")
        if order is not None and found.id != order.id:
            return Check("BL_EXISTS", FAIL, expected=order.bill_of_lading,
                         actual=data.bill_of_lading, message="BL de outro embarque")
        return Check("BL_EXISTS", PASS, actual=data.bill_of_lading)

    def container_exists(self, data, order) -> Check:
        if not data.container:
            return Check("CONTAINER_EXISTS", SKIP)
        found = self.session.scalar(
            select(PurchaseOrder).where(PurchaseOrder.container == data.container)
        )
        if found is None:
            return Check("CONTAINER_EXISTS", FAIL, actual=data.container,
                         message="Container inexistente")
        if order is not None and found.id != order.id:
            return Check("CONTAINER_EXISTS", FAIL, expected=order.container,
                         actual=data.container, message="Container de outro embarque")
        return Check("CONTAINER_EXISTS", PASS, actual=data.container)

    # -- orquestracao -------------------------------------------------------
    def validate(self, document: Document, data: ExtractedDocument) -> tuple[str, list[Check]]:
        supplier = _supplier(self.session, data)
        order = (
            self.session.scalar(
                select(PurchaseOrder).where(PurchaseOrder.number == data.purchase_order)
            )
            if data.purchase_order
            else None
        )
        checks = [
            self.supplier_exists(data, supplier),
            self.supplier_cnpj_match(data, supplier),
            self.bank_account_match(data, supplier),
            self.invoice_not_duplicate(document, data, supplier),
            self.amount_match(data, order),
            self.due_date_valid(data),
            self.purchase_order_exists(data, order),
            self.bl_exists(data, order),
            self.container_exists(data, order),
        ]
        failures = [c for c in checks if c.status == FAIL]
        if any(c.rule_code in BLOCKING_RULES for c in failures):
            status = "BLOCKED"
        elif failures:
            status = "REVIEW_REQUIRED"
        else:
            status = "APPROVED"
        return status, checks

    def run(self, document: Document, data: ExtractedDocument) -> tuple[str, list[Check]]:
        """Valida, persiste resultados, atualiza status e registra auditoria."""
        status, checks = self.validate(document, data)
        for check in checks:
            self.session.add(
                ValidationResult(
                    document_id=document.id,
                    rule_code=check.rule_code,
                    status=check.status,
                    expected_value=check.expected,
                    actual_value=check.actual,
                    message=check.message,
                )
            )
        document.status = status
        audit(
            self.session, "document", document.id, f"VALIDATION_{status}",
            failed=[c.rule_code for c in checks if c.status == FAIL],
        )
        self.session.commit()
        return status, checks


def promote_to_payable(session: Session, document: Document, data: ExtractedDocument) -> Payable:
    """Documento APPROVED vira titulo aberto (spec secao 26)."""
    supplier = _supplier(session, data)
    if supplier is None:
        raise ValueError("Documento sem fornecedor identificado")
    if data.amount is None or data.amount <= 0:
        raise ValueError("Documento sem valor")
    existing = session.scalar(select(Payable).where(Payable.document_id == document.id))
    if existing:
        return existing
    payable = Payable(
        supplier_id=supplier.id,
        document_id=document.id,
        invoice_number=data.invoice_number,
        purchase_order=data.purchase_order,
        bill_of_lading=data.bill_of_lading,
        container=data.container,
        amount_original=data.amount,
        amount_open=data.amount,
        currency=data.currency,
        due_date=data.due_date,
        cost_center=data.cost_center,
    )
    session.add(payable)
    session.flush()
    audit(session, "payable", payable.id, "PAYABLE_CREATED", document_id=document.id)
    session.commit()
    return payable
