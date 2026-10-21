"""Memoria e RAG hibrido (Spec 1). Nucleo em SQL; LangChain so na borda."""
from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app import config
from app.db import MemoryItem, audit

logger = logging.getLogger(__name__)


def _fake_vector(text: str) -> list[float]:
    """Embedding deterministico para rodar offline. Nao tem semantica: so reprodutibilidade."""
    needed = config.EMBEDDING_DIM
    raw = bytearray()
    counter = 0
    while len(raw) < needed * 2:
        raw += hashlib.sha256(f"{text}|{counter}".encode()).digest()
        counter += 1
    values = [
        int.from_bytes(raw[i * 2 : i * 2 + 2], "big") / 65535.0 - 0.5 for i in range(needed)
    ]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


@functools.lru_cache(maxsize=None)  # um cliente por processo: reusa conexao HTTP
def _voyage_client():
    from langchain_voyageai import VoyageAIEmbeddings

    return VoyageAIEmbeddings(
        model=config.EMBEDDING_MODEL, output_dimension=config.EMBEDDING_DIM
    )


def embed(texts: list[str]) -> list[list[float]] | None:
    """Embeddings de documentos. None so quando a Voyage falha (o chamador degrada)."""
    if not texts:
        return []
    if not os.getenv("VOYAGE_API_KEY"):
        return [_fake_vector(text) for text in texts]
    try:
        return _voyage_client().embed_documents(texts)
    except Exception:  # noqa: BLE001 - falha de rede/API vira degradacao, mas logada
        logger.warning("Voyage falhou em embed, degradando", exc_info=True)
        return None


def embed_query(text: str) -> list[float] | None:
    """Embedding de consulta. A Voyage distingue query de documento; o fake nao precisa."""
    if not os.getenv("VOYAGE_API_KEY"):
        return _fake_vector(text)
    try:
        return _voyage_client().embed_query(text)
    except Exception:  # noqa: BLE001 - idem embed, degradacao logada
        logger.warning("Voyage falhou em embed_query, degradando", exc_info=True)
        return None


FILTER_COLUMNS = frozenset(
    {"supplier_id", "customer_id", "bank_account_id", "cnpj", "amount"}
)
RRF_K = 60  # constante classica da Reciprocal Rank Fusion


@dataclass(slots=True)
class Hit:
    item: MemoryItem
    score: float
    via: set[str] = field(default_factory=set)


@dataclass(slots=True)
class RetrievalOutcome:
    hits: list[Hit]
    degraded: bool = False


def remember(
    session: Session,
    kind: str,
    content: str,
    *,
    item_metadata: dict | None = None,
    source_entity: str | None = None,
    source_id: int | None = None,
    **keys,
) -> MemoryItem:
    """Grava uma memoria. Nao commita: quem chama decide a fronteira da transacao."""
    desconhecidas = set(keys) - FILTER_COLUMNS
    if desconhecidas:
        raise ValueError(f"Chaves de memoria invalidas: {sorted(desconhecidas)}")

    vectors = embed([content])
    item = MemoryItem(
        kind=kind,
        content=content,
        embedding=vectors[0] if vectors else None,
        item_metadata=item_metadata,
        source_entity=source_entity,
        source_id=source_id,
        **keys,
    )
    session.add(item)
    session.flush()
    return item


def _as_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def retrieve_outcome(
    session: Session,
    kind: str,
    query: str,
    *,
    filters: dict | None = None,
    k: int = 5,
    audit_degradation: bool = True,
) -> RetrievalOutcome:
    """Busca hibrida: filtro estruturado -> full-text + vetor -> fusao RRF.

    Pode gravar um audit MEMORY_DEGRADED (sem embedding); quem chama decide commit/rollback.
    """
    filters = filters or {}
    desconhecidas = set(filters) - FILTER_COLUMNS
    if desconhecidas:
        raise ValueError(f"Filtros de memoria invalidos: {sorted(desconhecidas)}")

    conditions = ["kind = :kind"]
    params: dict = {
        "kind": kind,
        "query": query,
        "k": k,
        "n": config.RETRIEVE_CANDIDATES,
    }
    for index, (column, value) in enumerate(filters.items()):
        if value is None:
            continue
        conditions.append(f"{column} = :filter_{index}")
        params[f"filter_{index}"] = value
    where = " AND ".join(conditions)

    vector = embed_query(query)
    if vector is None:
        logger.warning("memoria degradada para kind=%s: embedding indisponivel", kind)
        if audit_degradation:
            audit(
                session,
                "memory",
                kind,
                "MEMORY_DEGRADED",
                kind=kind,
                reason="embedding_indisponivel",
            )
    else:
        params["vector"] = _as_vector_literal(vector)

    semantic_cte = (
        """,
        semantic AS (
            SELECT id, row_number() OVER (
                ORDER BY embedding <=> CAST(:vector AS vector), id
            ) AS rank
            FROM base
            WHERE embedding IS NOT NULL
            ORDER BY rank
            LIMIT :n
        )"""
        if vector is not None
        else ""
    )
    semantic_join = (
        "LEFT JOIN semantic s ON s.id = b.id" if vector is not None else ""
    )
    semantic_score = (
        "COALESCE(1.0 / (:rrf + s.rank), 0)" if vector is not None else "0"
    )
    semantic_flag = "s.rank IS NOT NULL" if vector is not None else "false"
    semantic_filter = "OR s.rank IS NOT NULL" if vector is not None else ""
    params["rrf"] = RRF_K

    sql = text(
        f"""
        WITH base AS (
            SELECT id, content_tsv, embedding FROM memory_items WHERE {where}
        ),
        lexical AS (
            SELECT id, row_number() OVER (
                ORDER BY ts_rank(content_tsv, plainto_tsquery('portuguese', :query)) DESC, id
            ) AS rank
            FROM base
            WHERE content_tsv @@ plainto_tsquery('portuguese', :query)
            ORDER BY rank
            LIMIT :n
        ){semantic_cte}
        SELECT
            b.id AS id,
            COALESCE(1.0 / (:rrf + l.rank), 0) + {semantic_score} AS score,
            (l.rank IS NOT NULL) AS via_lexical,
            ({semantic_flag}) AS via_semantic
        FROM base b
        LEFT JOIN lexical l ON l.id = b.id
        {semantic_join}
        WHERE l.rank IS NOT NULL {semantic_filter}
        ORDER BY score DESC, b.id
        LIMIT :k
        """
    )

    rows = session.execute(sql, params).all()
    if not rows:
        return RetrievalOutcome(hits=[], degraded=vector is None)

    items = {
        item.id: item
        for item in session.scalars(
            select(MemoryItem).where(MemoryItem.id.in_([row.id for row in rows]))
        )
    }
    hits = []
    for row in rows:
        via = set()
        if row.via_lexical:
            via.add("lexical")
        if row.via_semantic:
            via.add("semantic")
        hits.append(Hit(item=items[row.id], score=float(row.score), via=via))
    return RetrievalOutcome(hits=hits, degraded=vector is None)


def retrieve(
    session: Session,
    kind: str,
    query: str,
    *,
    filters: dict | None = None,
    k: int = 5,
) -> list[Hit]:
    """Compatibilidade da Spec 1: busca e registra degradacao no audit do chamador."""
    return retrieve_outcome(
        session, kind, query, filters=filters, k=k, audit_degradation=True
    ).hits


class RecebaRetriever(BaseRetriever):
    """Ponte para a Spec 2: o agente LangGraph consome retrieve() por aqui."""

    session: Any
    kind: str
    filters: dict | None = None
    k: int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = retrieve(
            self.session, self.kind, query, filters=self.filters, k=self.k
        )
        documents = []
        for hit in hits:
            metadata = {
                **(hit.item.item_metadata or {}),
                "memory_id": hit.item.id,
                "kind": hit.item.kind,
                "score": hit.score,
                "via": sorted(hit.via),
            }
            if hit.item.source_entity is not None and hit.item.source_id is not None:
                metadata["source"] = f"{hit.item.source_entity}:{hit.item.source_id}"
            documents.append(Document(page_content=hit.item.content, metadata=metadata))
        return documents


def retriever_tool(
    session: Session,
    kind: str,
    *,
    filters: dict | None = None,
    k: int = 5,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool:
    """Ferramenta pronta para o agente. Nome e descricao em portugues, como o dominio."""
    retriever = RecebaRetriever(session=session, kind=kind, filters=filters, k=k)
    return retriever.as_tool(
        name=name or f"buscar_{kind.lower()}",
        description=description
        or f"Busca memorias do tipo {kind} parecidas com a consulta informada.",
    )
