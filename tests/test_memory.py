"""Memoria e RAG hibrido (Spec 1)."""
import datetime as dt
from decimal import Decimal

from sqlalchemy import select

from app import db


def test_memory_item_grava_tsvector_e_embedding(session, company):
    item = db.MemoryItem(
        kind=db.EXTRACTION_EXAMPLE,
        supplier_id=company["supplier"].id,
        cnpj="11222333000181",
        amount=Decimal("7500.00"),
        content="NOTA FISCAL ACME invoice 44120 container MSCU1234567",
        embedding=[0.1] * db.EMBEDDING_DIM,
        item_metadata={"invoice_number": "44120"},
        source_entity="document",
    )
    session.add(item)
    session.commit()

    saved = session.scalar(select(db.MemoryItem).where(db.MemoryItem.id == item.id))
    assert saved.content_tsv is not None, "coluna gerada deve ser preenchida pelo Postgres"
    assert len(saved.embedding) == db.EMBEDDING_DIM
    assert saved.item_metadata == {"invoice_number": "44120"}
    assert isinstance(saved.created_at, dt.datetime)


def test_bank_transaction_aceita_embedding(session, make_transaction):
    transaction = make_transaction("7500.00", "PAGAMENTO ACME INVOICE 44120")
    transaction.embedding = [0.2] * db.EMBEDDING_DIM
    session.commit()
    assert len(session.get(db.BankTransaction, transaction.id).embedding) == db.EMBEDDING_DIM
