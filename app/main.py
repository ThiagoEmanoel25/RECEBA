"""API (spec secao 33). Sincrona: Celery entra quando o volume pedir."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from decimal import Decimal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile
from fastapi import File as FileParam
from fastapi.encoders import ENCODERS_BY_TYPE, jsonable_encoder
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import bank_import, matching, memory, reconcile
from app.config import ALLOWED_MIME, API_KEYS, MAX_UPLOAD_BYTES, STORAGE_DIR
from app.db import (
    BankTransaction,
    EXTRACTION_EXAMPLE,
    Document,
    ExtractedDocument,
    MATCH_DECISION,
    MatchingCandidate,
    MemoryItem,
    Payable,
    Receivable,
    ReconciliationGroup,
    ReconciliationGroupItem,
    Supplier,
    audit,
    get_session,
    init_db,
)
from app.extract import extract_with_llm, read_text
from app.parser import digits
from app.validation import ValidationEngine, promote_to_payable

# Dinheiro sai como string no JSON: float em API financeira perde centavo.
ENCODERS_BY_TYPE[Decimal] = str

app = FastAPI(title="RECEBA - Conciliacao Financeira")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    os.makedirs(STORAGE_DIR, exist_ok=True)


# ---- auth -----------------------------------------------------------------
class Actor(BaseModel):
    key: str
    role: str


def current_actor(x_api_key: str = Header(...)) -> Actor:
    role = API_KEYS.get(x_api_key)
    if role is None:
        raise HTTPException(401, "API key invalida")
    return Actor(key=x_api_key[:4] + "***", role=role)


def require(*roles: str):
    def guard(actor: Actor = Depends(current_actor)) -> Actor:
        if actor.role not in roles:
            raise HTTPException(403, f"Perfil {actor.role} nao autorizado")
        return actor

    return guard


WRITE = require("ADMIN", "ANALYST")
APPROVE = require("ADMIN", "FINANCIAL_APPROVER")
READ = current_actor


# ---- documentos -----------------------------------------------------------
@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile = FileParam(...),
    source: str = Query("UPLOAD"),
    sender: str | None = Query(None),
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"MIME nao aceito: {file.content_type}")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Arquivo vazio")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Arquivo acima do limite")

    content_hash = hashlib.sha256(content).hexdigest()
    existing = session.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing:
        audit(session, "document", existing.id, "DUPLICATE_UPLOAD_IGNORED",
              actor_type="USER", actor_id=actor.key)
        session.commit()
        return {"id": existing.id, "status": existing.status, "duplicate": True}

    storage_key = f"{content_hash}{os.path.splitext(file.filename or '')[1]}"
    with open(os.path.join(STORAGE_DIR, storage_key), "wb") as handle:
        handle.write(content)
    document = Document(
        source=source,
        filename=file.filename or storage_key,
        storage_key=storage_key,
        mime_type=file.content_type,
        sender=sender,
        content_hash=content_hash,
        raw_metadata={"size": len(content)},
    )
    session.add(document)
    session.flush()
    audit(session, "document", document.id, "DOCUMENT_STORED",
          actor_type="USER", actor_id=actor.key, source=source)
    session.commit()
    return {"id": document.id, "status": document.status, "duplicate": False}


@app.get("/documents")
def list_documents(
    status: str | None = None,
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    query = select(Document).order_by(Document.id.desc()).limit(limit)
    if status:
        query = query.where(Document.status == status)
    return [
        {"id": d.id, "filename": d.filename, "status": d.status, "type": d.document_type}
        for d in session.scalars(query)
    ]


@app.get("/documents/{document_id}")
def get_document(
    document_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Documento inexistente")
    extracted = session.scalar(
        select(ExtractedDocument).where(ExtractedDocument.document_id == document_id)
    )
    return {
        "id": document.id,
        "status": document.status,
        "filename": document.filename,
        "extracted": extracted.structured_payload if extracted else None,
    }


@app.post("/documents/{document_id}/process")
def process_document(
    document_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Documento inexistente")
    document.status = "PROCESSING"
    session.commit()
    try:
        text = read_text(os.path.join(STORAGE_DIR, document.storage_key))
        data, model_name, memory_ids = extract_with_llm(text, session=session)
    except Exception as exc:  # noqa: BLE001 - erro de extracao nao pode derrubar o doc
        document.status = "ERROR"
        audit(session, "document", document_id, "EXTRACTION_FAILED", error=str(exc)[:300])
        session.commit()
        raise HTTPException(422, f"Falha na extracao: {exc}") from exc

    session.query(ExtractedDocument).filter_by(document_id=document_id).delete()
    bank = data.bank_account
    extracted = ExtractedDocument(
        document_id=document_id,
        supplier_name=data.supplier_name,
        supplier_cnpj=data.supplier_cnpj,
        invoice_number=data.invoice_number,
        amount=data.amount,
        currency=data.currency,
        due_date=data.due_date,
        bank_code=bank.bank_code if bank else None,
        bank_agency=bank.agency if bank else None,
        bank_account=bank.account if bank else None,
        purchase_order=data.purchase_order,
        bill_of_lading=data.bill_of_lading,
        container=data.container,
        cost_center=data.cost_center,
        observation=data.observation,
        structured_payload=data.model_dump(mode="json"),
        model_name=model_name,
    )
    session.add(extracted)
    document.document_type = data.document_type
    document.status = "EXTRACTED"
    audit(session, "document", document_id, "DOCUMENT_EXTRACTED",
          model=model_name, memory_ids=memory_ids)
    session.commit()
    return {"id": document_id, "status": document.status, "extracted": extracted.structured_payload}


class ExtractionCorrectionIn(BaseModel):
    """Correcao humana dos campos extraidos. Tudo opcional: so o enviado e corrigido."""

    supplier_name: str | None = None
    supplier_cnpj: str | None = None
    invoice_number: str | None = None
    amount: Decimal | None = None
    due_date: dt.date | None = None
    bank_code: str | None = None
    bank_agency: str | None = None
    bank_account: str | None = None
    purchase_order: str | None = None
    bill_of_lading: str | None = None
    container: str | None = None
    cost_center: str | None = None
    observation: str | None = None


BANK_FLAT_TO_NESTED = {"bank_code": "bank_code", "bank_agency": "agency", "bank_account": "account"}


@app.patch("/documents/{document_id}/extracted")
def correct_extraction(
    document_id: int,
    payload: ExtractionCorrectionIn,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    """Corrige a extracao e guarda o par (texto -> campos corretos) como memoria."""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Documento inexistente")
    extracted = session.scalar(
        select(ExtractedDocument).where(ExtractedDocument.document_id == document_id)
    )
    if extracted is None:
        raise HTTPException(409, "Documento ainda nao processado")

    # Apenas o que o analista enviou: campo omitido nao sobrescreve valor bom com None.
    corrections = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not corrections:
        raise HTTPException(422, "Nenhum campo informado")
    if "supplier_cnpj" in corrections:
        corrections["supplier_cnpj"] = digits(corrections["supplier_cnpj"])

    before = extracted.structured_payload or {}
    for field_name, value in corrections.items():
        setattr(extracted, field_name, value)
    # No JSON o banco vai aninhado (bank_account.{bank_code,agency,account}); colunas ficam planas.
    flat = jsonable_encoder(corrections)
    bank = dict(before.get("bank_account") or {})
    for flat_key, nested_key in BANK_FLAT_TO_NESTED.items():
        if flat_key in flat:
            bank[nested_key] = flat.pop(flat_key)
    merged = {**before, **flat}
    if bank:
        merged["bank_account"] = bank
    extracted.structured_payload = merged
    extracted.model_name = f"{extracted.model_name or 'desconhecido'}+humano"

    cnpj = digits(extracted.supplier_cnpj)
    supplier = session.scalar(select(Supplier).where(Supplier.cnpj == cnpj)) if cnpj else None

    text = read_text(os.path.join(STORAGE_DIR, document.storage_key))
    item = memory.remember(
        session,
        EXTRACTION_EXAMPLE,
        text[:4000],
        item_metadata=extracted.structured_payload,
        source_entity="document",
        source_id=document_id,
        supplier_id=supplier.id if supplier else None,
        cnpj=cnpj or None,
        amount=extracted.amount,
    )

    # Campos mudaram: a validacao anterior nao vale mais, exige revalidar.
    # ponytail: correcao nao reverte Payable ja criado; estorno exige processo proprio.
    document.status = "EXTRACTED"
    audit(
        session,
        "document",
        document_id,
        "EXTRACTION_CORRECTED",
        actor_type="USER",
        actor_id=actor.key,
        fields=sorted(corrections),
        memory_id=item.id,
        previous_payload=before,
    )
    session.commit()
    return {
        "id": document_id,
        "status": document.status,
        "corrected_fields": sorted(corrections),
        "memory_id": item.id,
    }


@app.post("/documents/{document_id}/validate")
def validate_document(
    document_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Documento inexistente")
    extracted = session.scalar(
        select(ExtractedDocument).where(ExtractedDocument.document_id == document_id)
    )
    if extracted is None:
        raise HTTPException(409, "Documento ainda nao extraido")

    status, checks = ValidationEngine(session).run(document, extracted)
    payable_id = None
    if status == "APPROVED":
        try:
            payable_id = promote_to_payable(session, document, extracted).id
        except ValueError as exc:
            document.status = "REVIEW_REQUIRED"
            status = "REVIEW_REQUIRED"
            audit(session, "document", document_id, "PAYABLE_CREATION_FAILED", error=str(exc))
            session.commit()
    return {
        "id": document_id,
        "status": status,
        "payable_id": payable_id,
        "checks": [
            {"rule": c.rule_code, "status": c.status, "expected": c.expected,
             "actual": c.actual, "message": c.message}
            for c in checks
        ],
    }


# ---- transacoes bancarias -------------------------------------------------
@app.post("/bank-transactions/import")
async def import_transactions(
    bank_account_id: int = Query(...),
    file: UploadFile = FileParam(...),
    run_matching: bool = Query(True),
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Arquivo acima do limite")
    try:
        result = bank_import.import_csv(session, bank_account_id, content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if run_matching:
        outcomes: dict[str, int] = {}
        for transaction_id in result["ids"]:
            match, _ = matching.run(session, session.get(BankTransaction, transaction_id))
            outcomes[match.status] = outcomes.get(match.status, 0) + 1
        result["matching"] = outcomes
    return result


class BalanceIn(BaseModel):
    bank_account_id: int
    balance_date: dt.date
    opening_balance: Decimal
    closing_balance: Decimal


@app.post("/bank-accounts/balances")
def post_balance(
    payload: BalanceIn,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    balance = bank_import.set_balance(
        session, payload.bank_account_id, payload.balance_date,
        payload.opening_balance, payload.closing_balance,
    )
    return {"id": balance.id}


@app.get("/bank-transactions")
def list_transactions(
    status: str | None = None,
    bank_account_id: int | None = None,
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    query = select(BankTransaction).order_by(BankTransaction.id.desc()).limit(limit)
    if status:
        query = query.where(BankTransaction.status == status)
    if bank_account_id:
        query = query.where(BankTransaction.bank_account_id == bank_account_id)
    return [
        {"id": t.id, "date": t.transaction_date, "amount": t.amount, "direction": t.direction,
         "description": t.description_raw, "status": t.status}
        for t in session.scalars(query)
    ]


@app.get("/bank-transactions/{transaction_id}")
def get_transaction(
    transaction_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    transaction = session.get(BankTransaction, transaction_id)
    if transaction is None:
        raise HTTPException(404, "Transacao inexistente")
    group = session.scalar(
        select(ReconciliationGroup).where(
            ReconciliationGroup.bank_transaction_id == transaction_id
        )
    )
    return {
        "id": transaction.id,
        "date": transaction.transaction_date,
        "amount": transaction.amount,
        "direction": transaction.direction,
        "description_raw": transaction.description_raw,
        "description_normalized": transaction.description_normalized,
        "status": transaction.status,
        "group_id": group.id if group else None,
    }


# ---- matching -------------------------------------------------------------
@app.post("/matching/{transaction_id}/run")
def run_matching_endpoint(
    transaction_id: int,
    auto_settle: bool = Query(True),
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    transaction = session.get(BankTransaction, transaction_id)
    if transaction is None:
        raise HTTPException(404, "Transacao inexistente")
    result, group = matching.run(session, transaction, auto_settle=auto_settle)
    return {
        "status": result.status,
        "group_id": group.id if group else None,
        "group_type": result.group_type,
        "confidence": result.confidence,
        "reasons": result.reasons,
        "difference": result.difference,
        "items": [
            {"entity_type": kind, "entity_id": entity_id, "amount": amount}
            for kind, entity_id, amount in result.items
        ],
    }


@app.get("/matching/{transaction_id}/candidates")
def list_candidates(
    transaction_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    rows = session.scalars(
        select(MatchingCandidate)
        .where(MatchingCandidate.bank_transaction_id == transaction_id)
        .order_by(MatchingCandidate.score.desc())
    ).all()
    return [
        {"type": c.candidate_type, "id": c.candidate_id, "score": c.score,
         "reasons": (c.match_reasons or {}).get("reasons", [])}
        for c in rows
    ]


def _group_for_transaction(session: Session, transaction_id: int) -> ReconciliationGroup:
    group = session.scalar(
        select(ReconciliationGroup)
        .where(ReconciliationGroup.bank_transaction_id == transaction_id)
        .order_by(ReconciliationGroup.id.desc())
    )
    if group is None:
        raise HTTPException(404, "Sem grupo de conciliacao para a transacao")
    return group


@app.post("/matching/{transaction_id}/approve")
def approve_matching(
    transaction_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(APPROVE),
):
    return _approve_group(session, _group_for_transaction(session, transaction_id), actor)


@app.post("/matching/{transaction_id}/reject")
def reject_matching(
    transaction_id: int,
    reason: str = Query(...),
    session: Session = Depends(get_session),
    actor: Actor = Depends(APPROVE),
):
    group = _group_for_transaction(session, transaction_id)
    if group.status == "APPROVED":
        raise HTTPException(409, "Grupo ja aprovado; estorno exige processo proprio")
    group.status = "REJECTED"
    transaction = session.get(BankTransaction, transaction_id)
    transaction.status = matching.UNMATCHED
    _remember_match(session, group, "REJECTED", actor=actor, reason=reason)
    audit(session, "reconciliation_group", group.id, "HUMAN_REJECTED",
          actor_type="USER", actor_id=actor.key, reason=reason)
    session.commit()
    return {"group_id": group.id, "status": group.status}


# ---- grupos de conciliacao ------------------------------------------------
def _recalc(session: Session, group: ReconciliationGroup) -> None:
    items = session.scalars(
        select(ReconciliationGroupItem).where(
            ReconciliationGroupItem.reconciliation_group_id == group.id
        )
    ).all()
    allocated = sum(
        (Decimal(i.amount_allocated) for i in items
         if i.entity_type in {"PAYABLE", "RECEIVABLE"}),
        Decimal(0),
    )
    group.total_allocated_amount = allocated
    group.difference = Decimal(group.total_transaction_amount) - allocated


def _remember_match(
    session: Session,
    group: ReconciliationGroup,
    decision: str,
    *,
    actor: Actor,
    reason: str | None = None,
) -> list[MemoryItem]:
    """Decisao humana de match vira memoria episodica: uma por titulo, com a parte como chave."""
    transaction = session.get(BankTransaction, group.bank_transaction_id)
    if transaction is None:
        return []
    items = session.scalars(
        select(ReconciliationGroupItem).where(
            ReconciliationGroupItem.reconciliation_group_id == group.id,
            ReconciliationGroupItem.entity_type.in_(("PAYABLE", "RECEIVABLE")),
        ).order_by(ReconciliationGroupItem.id)
    ).all()

    memories = []
    for item in items:
        # a parte (fornecedor/cliente) e a chave que a pontuacao usa depois: o titulo ja
        # baixado nunca volta como candidato aberto. Sem parte, a memoria seria inutil.
        supplier_id = customer_id = None
        if item.entity_type == "PAYABLE":
            title = session.get(Payable, item.entity_id)
            supplier_id = title.supplier_id if title else None
        else:
            title = session.get(Receivable, item.entity_id)
            customer_id = title.customer_id if title else None
        if supplier_id is None and customer_id is None:
            continue
        memories.append(memory.remember(
            session,
            MATCH_DECISION,
            transaction.description_normalized or transaction.description_raw or "",
            item_metadata={
                "decision": decision,
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "group_type": group.group_type,
                "amount": str(item.amount_allocated),
                "reason": reason,
                "actor": actor.key,
            },
            source_entity="reconciliation_group",
            source_id=group.id,
            supplier_id=supplier_id,
            customer_id=customer_id,
            bank_account_id=transaction.bank_account_id,
            amount=Decimal(item.amount_allocated),
        ))
    return memories


@app.get("/reconciliation-groups/{group_id}")
def get_group(
    group_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    group = session.get(ReconciliationGroup, group_id)
    if group is None:
        raise HTTPException(404, "Grupo inexistente")
    items = session.scalars(
        select(ReconciliationGroupItem).where(
            ReconciliationGroupItem.reconciliation_group_id == group_id
        )
    ).all()
    return {
        "id": group.id,
        "bank_transaction_id": group.bank_transaction_id,
        "group_type": group.group_type,
        "status": group.status,
        "transaction_amount": group.total_transaction_amount,
        "allocated": group.total_allocated_amount,
        "difference": group.difference,
        "confidence": group.confidence_score,
        "reasons": (group.match_reasons or {}).get("reasons", []),
        "items": [
            {"id": i.id, "entity_type": i.entity_type, "entity_id": i.entity_id,
             "amount": i.amount_allocated}
            for i in items
        ],
    }


class GroupItemIn(BaseModel):
    entity_type: str
    entity_id: int | None = None
    amount_allocated: Decimal


@app.post("/reconciliation-groups/{group_id}/items", status_code=201)
def add_group_item(
    group_id: int,
    payload: GroupItemIn,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    group = session.get(ReconciliationGroup, group_id)
    if group is None:
        raise HTTPException(404, "Grupo inexistente")
    if group.status == "APPROVED":
        raise HTTPException(409, "Grupo aprovado nao aceita alteracao")
    if payload.entity_type not in {"PAYABLE", "RECEIVABLE", "FEE", "DISCOUNT", "INTEREST",
                                   "PENALTY"}:
        raise HTTPException(400, "entity_type invalido")
    item = ReconciliationGroupItem(
        reconciliation_group_id=group_id,
        entity_type=payload.entity_type,
        entity_id=payload.entity_id,
        amount_allocated=payload.amount_allocated,
    )
    session.add(item)
    session.flush()
    _recalc(session, group)
    audit(session, "reconciliation_group", group_id, "ITEM_ADDED",
          actor_type="USER", actor_id=actor.key, item_id=item.id)
    session.commit()
    return {"item_id": item.id, "difference": group.difference}


@app.delete("/reconciliation-groups/{group_id}/items/{item_id}")
def remove_group_item(
    group_id: int,
    item_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    group = session.get(ReconciliationGroup, group_id)
    item = session.get(ReconciliationGroupItem, item_id)
    if group is None or item is None or item.reconciliation_group_id != group_id:
        raise HTTPException(404, "Item inexistente")
    if group.status == "APPROVED":
        raise HTTPException(409, "Grupo aprovado nao aceita alteracao")
    session.delete(item)
    session.flush()
    _recalc(session, group)
    audit(session, "reconciliation_group", group_id, "ITEM_REMOVED",
          actor_type="USER", actor_id=actor.key, item_id=item_id)
    session.commit()
    return {"removed": item_id, "difference": group.difference}


def _approve_group(session: Session, group: ReconciliationGroup, actor: Actor) -> dict:
    if group.status == "APPROVED":
        return {"group_id": group.id, "status": group.status, "already": True}
    _recalc(session, group)
    if abs(Decimal(group.difference)) > Decimal("0.01") and group.group_type not in {
        "PARTIAL", "ADJUSTMENT"
    }:
        raise HTTPException(
            409, f"Grupo nao fecha: diferenca {group.difference}"
        )
    items = session.scalars(
        select(ReconciliationGroupItem).where(
            ReconciliationGroupItem.reconciliation_group_id == group.id
        )
    ).all()
    matching._settle(
        session,
        matching.MatchResult(
            status="APPROVED",
            items=[(i.entity_type, i.entity_id, Decimal(i.amount_allocated)) for i in items],
        ),
    )
    group.status = "APPROVED"
    transaction = session.get(BankTransaction, group.bank_transaction_id)
    transaction.status = "RECONCILED"
    _remember_match(session, group, "APPROVED", actor=actor)
    audit(session, "reconciliation_group", group.id, "HUMAN_APPROVED",
          actor_type="USER", actor_id=actor.key)
    session.commit()
    return {"group_id": group.id, "status": group.status, "difference": group.difference}


@app.post("/reconciliation-groups/{group_id}/approve")
def approve_group(
    group_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(APPROVE),
):
    group = session.get(ReconciliationGroup, group_id)
    if group is None:
        raise HTTPException(404, "Grupo inexistente")
    return _approve_group(session, group, actor)


# ---- conciliacao diaria ---------------------------------------------------
class ReconciliationIn(BaseModel):
    bank_account_id: int
    reconciliation_date: dt.date


@app.post("/reconciliations/run")
def run_reconciliation(
    payload: ReconciliationIn,
    session: Session = Depends(get_session),
    actor: Actor = Depends(WRITE),
):
    try:
        record = reconcile.run_daily(
            session, payload.bank_account_id, payload.reconciliation_date
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "id": record.id,
        "status": record.status,
        "expected_balance": record.expected_balance,
        "real_balance": record.real_balance,
        "difference": record.difference,
        "findings": (record.findings or {}).get("items", []),
    }


@app.get("/reconciliations")
def list_reconciliations(
    limit: int = Query(30, le=200),
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    from app.db import DailyReconciliation

    rows = session.scalars(
        select(DailyReconciliation).order_by(DailyReconciliation.id.desc()).limit(limit)
    ).all()
    return [
        {"id": r.id, "date": r.reconciliation_date, "status": r.status,
         "difference": r.difference}
        for r in rows
    ]


@app.get("/reconciliations/{reconciliation_id}")
def get_reconciliation(
    reconciliation_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    from app.db import DailyReconciliation

    record = session.get(DailyReconciliation, reconciliation_id)
    if record is None:
        raise HTTPException(404, "Conciliacao inexistente")
    return {
        "id": record.id,
        "date": record.reconciliation_date,
        "opening_balance": record.opening_balance,
        "credits": record.total_credits,
        "debits": record.total_debits,
        "expected_balance": record.expected_balance,
        "real_balance": record.real_balance,
        "difference": record.difference,
        "status": record.status,
        "findings": (record.findings or {}).get("items", []),
    }


@app.post("/reconciliations/{reconciliation_id}/approve")
def approve_reconciliation(
    reconciliation_id: int,
    session: Session = Depends(get_session),
    actor: Actor = Depends(APPROVE),
):
    from app.db import DailyReconciliation

    record = session.get(DailyReconciliation, reconciliation_id)
    if record is None:
        raise HTTPException(404, "Conciliacao inexistente")
    record.status = "APPROVED"
    audit(session, "daily_reconciliation", record.id, "HUMAN_APPROVED",
          actor_type="USER", actor_id=actor.key)
    session.commit()
    return {"id": record.id, "status": record.status}


# ---- metricas (spec secoes 37 e 42) ---------------------------------------
@app.get("/metrics")
def metrics(session: Session = Depends(get_session), actor: Actor = Depends(READ)):
    def count(model, **filters) -> int:
        query = select(func.count()).select_from(model)
        for column, value in filters.items():
            query = query.where(getattr(model, column) == value)
        return session.scalar(query) or 0

    total_transactions = count(BankTransaction)
    by_status = dict(
        session.execute(
            select(BankTransaction.status, func.count()).group_by(BankTransaction.status)
        ).all()
    )

    def rate(*statuses: str) -> float:
        if not total_transactions:
            return 0.0
        return round(sum(by_status.get(s, 0) for s in statuses) / total_transactions, 4)

    return {
        "documents_processed": count(Document),
        "documents_by_status": dict(
            session.execute(select(Document.status, func.count()).group_by(Document.status)).all()
        ),
        "transactions": total_transactions,
        "transactions_by_status": by_status,
        "auto_match_rate": rate(matching.AUTO_MATCHED),
        "batch_match_rate": rate(matching.MATCHED_BATCH),
        "partial_match_rate": rate(matching.PARTIALLY_PAID),
        "manual_review_rate": rate(matching.REVIEW_REQUIRED),
        "unmatched_rate": rate(matching.UNMATCHED, "IMPORTED"),
        "amount_reconciled": session.scalar(
            select(func.coalesce(func.sum(ReconciliationGroup.total_allocated_amount), 0)).where(
                ReconciliationGroup.status == "APPROVED"
            )
        ),
        "payables_open": session.scalar(
            select(func.coalesce(func.sum(Payable.amount_open), 0)).where(
                Payable.status != "PAID"
            )
        ),
        "receivables_open": session.scalar(
            select(func.coalesce(func.sum(Receivable.amount_open), 0)).where(
                Receivable.status != "PAID"
            )
        ),
    }


@app.get("/audit/{entity_type}/{entity_id}")
def audit_trail(
    entity_type: str,
    entity_id: str,
    session: Session = Depends(get_session),
    actor: Actor = Depends(READ),
):
    from app.db import AuditEvent

    rows = session.scalars(
        select(AuditEvent)
        .where(AuditEvent.entity_type == entity_type, AuditEvent.entity_id == entity_id)
        .order_by(AuditEvent.id)
    ).all()
    return [
        {"at": e.created_at, "event": e.event_type, "actor": e.actor_type,
         "actor_id": e.actor_id, "metadata": e.event_metadata}
        for e in rows
    ]
