"""Memoria e RAG hibrido (Spec 1)."""
import datetime as dt
from decimal import Decimal

from sqlalchemy import select

from app import db, memory
from app.config import EMBEDDING_DIM


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


def test_embed_fake_e_deterministico_e_normalizado():
    primeiro = memory.embed(["PAGAMENTO ACME INVOICE 44120"])
    segundo = memory.embed(["PAGAMENTO ACME INVOICE 44120"])
    assert primeiro == segundo, "sem API key o embedding deve ser reprodutivel"
    assert len(primeiro) == 1 and len(primeiro[0]) == EMBEDDING_DIM
    norma = sum(value * value for value in primeiro[0]) ** 0.5
    assert abs(norma - 1.0) < 1e-6, "vetor deve estar normalizado"


def test_embed_textos_diferentes_geram_vetores_diferentes():
    (um, dois) = memory.embed(["ACME INVOICE 44120", "BETA INVOICE 99999"])
    assert um != dois


def test_embed_lista_vazia():
    assert memory.embed([]) == []


def test_embed_retorna_none_quando_voyage_falha(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "chave-invalida")

    def explode(*args, **kwargs):
        raise RuntimeError("voyage fora do ar")

    monkeypatch.setattr(memory, "_voyage_client", explode)
    assert memory.embed(["qualquer texto"]) is None
    assert memory.embed_query("qualquer texto") is None


def test_embed_query_usa_fake_sem_api_key():
    vetor = memory.embed_query("ACME INVOICE 44120")
    assert vetor is not None and len(vetor) == EMBEDDING_DIM
