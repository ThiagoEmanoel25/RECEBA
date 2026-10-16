"""Memoria e RAG hibrido (Spec 1). Nucleo em SQL; LangChain so na borda."""
from __future__ import annotations

import hashlib
import math
import os

from app import config


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
    except Exception:  # noqa: BLE001 - qualquer falha de rede/API vira degradacao
        return None


def embed_query(text: str) -> list[float] | None:
    """Embedding de consulta. A Voyage distingue query de documento; o fake nao precisa."""
    if not os.getenv("VOYAGE_API_KEY"):
        return _fake_vector(text)
    try:
        return _voyage_client().embed_query(text)
    except Exception:  # noqa: BLE001
        return None
