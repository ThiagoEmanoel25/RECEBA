"""Memoria e RAG hibrido (Spec 1)."""
import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select

from app import db, memory
from app.config import EMBEDDING_DIM
from app.db import MATCH_DECISION


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


def test_embed_lista_vazia_nao_chama_api(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "chave-valida")

    def nao_deve_chamar(*args, **kwargs):
        raise AssertionError("embed([]) nao deve chamar a API")

    monkeypatch.setattr(memory, "_voyage_client", nao_deve_chamar)
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


def _lembrar(session, content, **kwargs):
    item = memory.remember(session, db.EXTRACTION_EXAMPLE, content, **kwargs)
    session.commit()
    return item


def test_remember_grava_e_gera_embedding(session, company):
    item = _lembrar(
        session,
        "NOTA FISCAL ACME invoice 44120",
        item_metadata={"invoice_number": "44120"},
        source_entity="document",
        source_id=7,
        supplier_id=company["supplier"].id,
        cnpj="11222333000181",
    )
    assert item.id is not None
    assert len(item.embedding) == EMBEDDING_DIM
    assert item.source_entity == "document" and item.source_id == 7


def test_remember_rejeita_chave_desconhecida(session):
    with pytest.raises(ValueError, match="invoice_number"):
        memory.remember(session, db.EXTRACTION_EXAMPLE, "texto", invoice_number="44120")


def test_retrieve_filtra_por_kind(session):
    _lembrar(session, "documento da ACME com invoice 44120")
    memory.remember(session, MATCH_DECISION, "pagamento da ACME invoice 44120")
    session.commit()

    hits = memory.retrieve(session, db.EXTRACTION_EXAMPLE, "ACME invoice 44120")
    assert hits, "deve achar a memoria de extracao"
    assert all(hit.item.kind == db.EXTRACTION_EXAMPLE for hit in hits)


def test_retrieve_filtra_por_cnpj(session, company):
    _lembrar(session, "documento da ACME invoice 44120", cnpj="11222333000181")
    _lembrar(session, "documento da BETA invoice 44120", cnpj="99888777000166")

    hits = memory.retrieve(
        session, db.EXTRACTION_EXAMPLE, "invoice 44120", filters={"cnpj": "11222333000181"}
    )
    assert len(hits) == 1
    assert hits[0].item.cnpj == "11222333000181"


def test_retrieve_ignora_filtro_none(session, company):
    _lembrar(session, "documento da ACME invoice 44120", cnpj="11222333000181")
    hits = memory.retrieve(
        session, db.EXTRACTION_EXAMPLE, "invoice 44120", filters={"cnpj": None}
    )
    assert len(hits) == 1, "filtro None nao deve restringir nada"


def test_retrieve_acha_container_pelo_full_text(session):
    """Canal lexico salva a busca: o embedding fake nao tem semantica nenhuma."""
    for indice in range(6):
        _lembrar(session, f"pagamento generico numero {indice} sem referencia forte")
    alvo = _lembrar(session, "documento com container MSCU1234567 do embarque")

    hits = memory.retrieve(session, db.EXTRACTION_EXAMPLE, "MSCU1234567", k=3)
    assert hits[0].item.id == alvo.id
    assert "lexical" in hits[0].via


def test_retrieve_respeita_k(session):
    for indice in range(8):
        _lembrar(session, f"documento ACME invoice 4412{indice}")
    assert len(memory.retrieve(session, db.EXTRACTION_EXAMPLE, "ACME invoice", k=3)) == 3


def test_retrieve_degrada_sem_embedding_e_audita(session, monkeypatch):
    alvo = _lembrar(session, "documento com container MSCU1234567")
    monkeypatch.setattr(memory, "embed_query", lambda text: None)

    hits = memory.retrieve(session, db.EXTRACTION_EXAMPLE, "MSCU1234567")
    assert hits and hits[0].item.id == alvo.id
    assert hits[0].via == {"lexical"}, "sem vetor, so o canal lexico responde"

    eventos = session.scalars(
        select(db.AuditEvent).where(db.AuditEvent.event_type == "MEMORY_DEGRADED")
    ).all()
    assert len(eventos) == 1
    assert eventos[0].event_metadata["kind"] == db.EXTRACTION_EXAMPLE


def test_retrieve_sem_memoria_retorna_vazio(session):
    assert memory.retrieve(session, db.DIVERGENCE_CASE, "qualquer coisa") == []
