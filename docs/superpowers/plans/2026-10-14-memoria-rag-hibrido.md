# Memória e RAG Híbrido — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar ao RECEBA memória construída a partir de decisões humanas, recuperada por busca híbrida (filtro estruturado + full-text + vetorial com fusão RRF), usada como few-shot na extração, como peso com teto no matching e como evidência nas divergências.

**Architecture:** Uma tabela `memory_items` no Postgres com `pgvector` guarda três tipos de memória (`EXTRACTION_EXAMPLE`, `MATCH_DECISION`, `DIVERGENCE_CASE`). Um módulo novo `app/memory.py` expõe um núcleo em SQL puro (`embed`, `embed_query`, `remember`, `retrieve`) e, por cima dele, adaptadores LangChain finos (`RecebaRetriever`, `retriever_tool`) que a Spec 2 vai consumir. Memória só é gravada a partir de ação humana e nunca produz baixa automática sozinha.

**Tech Stack:** Python 3.12, FastAPI 0.115.6, SQLAlchemy 2.0.36, Pydantic 2.10.4, psycopg 3.2.3, Postgres 17 + pgvector, `pgvector==0.5.1`, `langchain-core==1.6.10`, `langchain-voyageai==0.4.1`, Voyage `voyage-3.5` (1024 dims), pytest 8.3.4.

**Spec:** `docs/superpowers/specs/2026-10-14-memoria-rag-hibrido-design.md`

## Global Constraints

- **Princípio do produto:** "IA interpreta. Código valida. Regras decidem. Humano resolve exceções." Memória nunca decide sozinha.
- **Teto do history:** a condição de `AUTO_MATCHED` em `app/matching.py` usa sempre o score **base** (sem history). History só ordena, desempata e pode promover `UNMATCHED → REVIEW_REQUIRED`.
- **Toda memória usada vai para o audit trail** via `audit()` de `app/db.py`.
- **Degradação nunca quebra o pipeline:** falha de embedding → `retrieve()` roda só estruturado + léxico e grava `MEMORY_DEGRADED`; falha no import → coluna `embedding` fica `NULL`.
- **Sem `ANTHROPIC_API_KEY`** a extração mantém o comportamento atual (só regex, sem memória).
- **Sem `VOYAGE_API_KEY`** usa embedding fake determinístico, para os testes rodarem offline.
- **Versões exatas** (já verificadas no PyPI): `pgvector==0.5.1`, `langchain-core==1.6.10`, `langchain-voyageai==0.4.1`.
- **Dimensão dos embeddings:** 1024 (default do `voyage-3.5`), em `config.EMBEDDING_DIM`.
- **Sem Alembic:** `create_all` continua, com comentário `ponytail:` marcando a troca antes do primeiro deploy real.
- **Nome do atributo de metadados:** `item_metadata` (o nome `metadata` é reservado no declarative do SQLAlchemy), seguindo o padrão de `AuditEvent.event_metadata`.
- **Comentários e docstrings em português sem acento**, como o resto do `app/`.
- **Todo commit roda a suíte inteira** (`pytest -q`); os 20 testes existentes nunca regridem.

---

### Task 1: Infra Postgres + pgvector e testes sobre Postgres

Troca o SQLite por Postgres com pgvector. Entrega: os 20 testes atuais passando sobre Postgres, sem nenhuma feature nova.

**Files:**
- Create: `docker-compose.yml`
- Create: `docker/init-test-db.sql`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `app/config.py`
- Modify: `app/db.py` (função `init_db`, linhas 297-300)
- Modify: `tests/conftest.py` (fixture `session`, linhas 17-24)
- Modify: `tests/test_api.py` (fixture `client`, linhas 25-40)

**Interfaces:**
- Consumes: nada.
- Produces: `app.config.EMBEDDING_DIM: int`, `app.config.EMBEDDING_MODEL: str`, `app.config.RETRIEVE_CANDIDATES: int`, `app.config.HISTORY_HITS: int`; `app.db.init_db()` passa a criar a extensão `vector`; fixture `engine` (escopo de sessão) e fixture `session` (rollback por teste) em `tests/conftest.py`.

- [ ] **Step 1: Criar o docker-compose**

Criar `docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_USER: receba
      POSTGRES_PASSWORD: receba
      POSTGRES_DB: receba
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/init-test-db.sql:/docker-entrypoint-initdb.d/01-init-test-db.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U receba -d receba"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgdata:
```

Criar `docker/init-test-db.sql`:

```sql
-- Banco separado para a suite de testes, criado uma vez na primeira subida do volume.
CREATE DATABASE receba_test OWNER receba;
```

- [ ] **Step 2: Subir o banco e confirmar que a extensão existe**

```bash
docker compose up -d db
sleep 10
docker compose exec -T db psql -U receba -d receba_test -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"
```

Esperado: uma linha com a versão da extensão (ex.: `0.8.1`).

- [ ] **Step 3: Adicionar as dependências**

Em `requirements.txt`, acrescentar depois de `psycopg[binary]==3.2.3`:

```
pgvector==0.5.1
langchain-core==1.6.10
langchain-voyageai==0.4.1
```

Instalar:

```bash
.venv/bin/pip install -r requirements.txt
```

- [ ] **Step 4: Atualizar `.env.example`**

Substituir a linha `DATABASE_URL=sqlite:///./receba.db` e acrescentar as novas variáveis, deixando o arquivo assim:

```
DATABASE_URL=postgresql+psycopg://receba:receba@localhost:5433/receba
TEST_DATABASE_URL=postgresql+psycopg://receba:receba@localhost:5433/receba_test
STORAGE_DIR=./storage
API_KEYS={"dev-key":"ADMIN"}
ANTHROPIC_API_KEY=
AMOUNT_TOLERANCE=0.01
FEE_MAX=150.00
VOYAGE_API_KEY=
EMBEDDING_MODEL=voyage-3.5
EMBEDDING_DIM=1024
RETRIEVE_CANDIDATES=20
HISTORY_HITS=5
```

- [ ] **Step 5: Atualizar `app/config.py`**

Trocar a linha 6 (`DATABASE_URL = ...`) por:

```python
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://receba:receba@localhost:5433/receba"
)
```

Acrescentar no fim do arquivo:

```python
# ---- memoria / RAG -------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", 20))  # top-N por canal antes do RRF
HISTORY_HITS = int(os.getenv("HISTORY_HITS", 5))  # memorias de match lidas por transacao
```

- [ ] **Step 6: Criar a extensão no `init_db`**

Em `app/db.py`, substituir a função `init_db` (linhas 297-300) por:

```python
def init_db() -> None:
    # ponytail: create_all no lugar de Alembic; trocar por migrations antes do primeiro deploy.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
```

E acrescentar `text` à lista de imports de `sqlalchemy` (linhas 7-20), em ordem alfabética depois de `String`:

```python
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
```

- [ ] **Step 7: Reescrever a fixture de sessão do conftest**

Em `tests/conftest.py`, substituir os imports e a fixture `session` (linhas 1-24) por:

```python
import datetime as dt
import os
import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import db

warnings.filterwarnings("ignore", message=".*does \\*not\\* support Decimal.*")

TODAY = dt.date(2026, 10, 9)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://receba:receba@localhost:5433/receba_test"
)
SKIP_MESSAGE = (
    f"Postgres com pgvector indisponivel em {TEST_DATABASE_URL}. "
    "Suba com: docker compose up -d db"
)


@pytest.fixture(scope="session")
def engine():
    """Schema criado uma vez por sessao de teste; cada teste roda em transacao propria."""
    eng = create_engine(TEST_DATABASE_URL, future=True)
    try:
        with eng.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except SQLAlchemyError as exc:
        eng.dispose()
        pytest.skip(f"{SKIP_MESSAGE} ({exc.__class__.__name__})")
    db.Base.metadata.drop_all(eng)
    db.Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def connection(engine):
    """Transacao externa com rollback no fim: os commits do app viram savepoints."""
    conn = engine.connect()
    transaction = conn.begin()
    yield conn
    transaction.rollback()
    conn.close()


@pytest.fixture()
def session(connection):
    with Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    ) as s:
        yield s
```

O resto do arquivo (fixtures `company`, `make_payable`, `make_transaction`) fica igual.

- [ ] **Step 8: Reescrever a fixture `client` do test_api**

Em `tests/test_api.py`, substituir os imports de SQLAlchemy (linhas 7-9) e a fixture `client` (linhas 25-40 e o que vem depois dela, até o `yield`) por:

```python
import datetime as dt
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app import db
from app.main import app
from tests.conftest import TODAY
```

```python
@pytest.fixture()
def client(connection, tmp_path, monkeypatch):
    Local = sessionmaker(
        bind=connection,
        class_=Session,
        expire_on_commit=False,
        future=True,
        join_transaction_mode="create_savepoint",
    )
    monkeypatch.setattr("app.main.STORAGE_DIR", str(tmp_path))
    monkeypatch.setitem(db.__dict__, "SessionLocal", Local)
    monkeypatch.setattr("app.main.init_db", lambda: None)

    def override():
        with Local() as session:
            yield session

    monkeypatch.setitem(app.dependency_overrides, db.get_session, override)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
```

Conferir o trecho original antes de substituir: manter qualquer linha extra que já exista entre `def override():` e o `yield` (por exemplo criação de chaves de API ou fixtures de dados). O que muda é só a origem do engine, que agora é a fixture `connection`.

- [ ] **Step 9: Rodar a suíte inteira sobre Postgres**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 20 passed. Se algum teste falhar por diferença de dialeto (ex.: ordenação sem `ORDER BY`, comparação de `Decimal`), corrigir o teste ou o código, não voltar para SQLite.

- [ ] **Step 10: Confirmar o skip quando não há Postgres**

```bash
TEST_DATABASE_URL=postgresql+psycopg://nobody:nobody@127.0.0.1:1/none .venv/bin/python -m pytest -q 2>&1 | tail -5
```

Esperado: todos os testes com status `skipped` e a mensagem "Postgres com pgvector indisponivel".

- [ ] **Step 11: Commit**

```bash
git add docker-compose.yml docker/init-test-db.sql requirements.txt .env.example app/config.py app/db.py tests/conftest.py tests/test_api.py
git commit -m "feat(infra): Postgres com pgvector no lugar do SQLite

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Tabela `memory_items` e coluna de embedding nas transações

**Files:**
- Modify: `app/db.py` (acrescentar modelo depois de `AuditEvent`, e coluna em `BankTransaction` na linha 179)
- Test: `tests/test_memory.py` (criar)

**Interfaces:**
- Consumes: `app.config.EMBEDDING_DIM` (Task 1).
- Produces: `app.db.MemoryItem` com os atributos `id`, `kind`, `supplier_id`, `customer_id`, `bank_account_id`, `cnpj`, `amount`, `content`, `content_tsv`, `embedding`, `item_metadata`, `source_entity`, `source_id`, `created_at`; as constantes `app.db.EXTRACTION_EXAMPLE`, `app.db.MATCH_DECISION`, `app.db.DIVERGENCE_CASE`; e `app.db.BankTransaction.embedding`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_memory.py`:

```python
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
        source_id=1,
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
```

- [ ] **Step 2: Rodar o teste para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_memory.py -q
```

Esperado: FAIL com `AttributeError: module 'app.db' has no attribute 'MemoryItem'`.

- [ ] **Step 3: Implementar o modelo**

Em `app/db.py`, acrescentar aos imports:

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from app.config import DATABASE_URL, EMBEDDING_DIM
```

(`Computed` e `Index` entram na lista de imports de `sqlalchemy` em ordem alfabética; a linha `from app.config import DATABASE_URL` passa a importar também `EMBEDDING_DIM`.)

Acrescentar na linha 179 de `BankTransaction`, depois de `raw_payload`:

```python
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
```

Acrescentar depois da classe `AuditEvent`, antes da função `audit`:

```python
# Tipos de memoria (Spec 1). Procedural: como extrair. Episodica: o que ja foi decidido.
EXTRACTION_EXAMPLE = "EXTRACTION_EXAMPLE"
MATCH_DECISION = "MATCH_DECISION"
DIVERGENCE_CASE = "DIVERGENCE_CASE"


class MemoryItem(Base, TimestampMixin):
    """Memoria vinda de decisao humana. Busca hibrida: estruturada + full-text + vetor."""

    __tablename__ = "memory_items"
    __table_args__ = (
        Index("ix_memory_items_tsv", "content_tsv", postgresql_using="gin"),
        Index(
            "ix_memory_items_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"), index=True)
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"), index=True)
    bank_account_id: Mapped[int | None] = mapped_column(
        ForeignKey("bank_accounts.id"), index=True
    )
    cnpj: Mapped[str | None] = mapped_column(String(14), index=True)
    amount: Mapped[Decimal | None] = mapped_column(Money)
    content: Mapped[str] = mapped_column(Text)
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('portuguese', content)", persisted=True)
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    item_metadata: Mapped[dict | None] = mapped_column(JSONB)
    source_entity: Mapped[str | None] = mapped_column(String(40))
    source_id: Mapped[int | None] = mapped_column()
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 22 passed. Se a tabela antiga tiver ficado no banco de teste, a fixture `engine` faz `drop_all` antes do `create_all`, então basta rodar de novo.

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_memory.py
git commit -m "feat(db): tabela memory_items e embedding em bank_transactions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `embed()` com Voyage e fallback fake determinístico

**Files:**
- Create: `app/memory.py`
- Test: `tests/test_memory.py` (acrescentar)

**Interfaces:**
- Consumes: `app.config.EMBEDDING_MODEL`, `app.config.EMBEDDING_DIM` (Task 1).
- Produces: `app.memory.embed(texts: list[str]) -> list[list[float]] | None` e `app.memory.embed_query(text: str) -> list[float] | None`. Ambas retornam `None` só quando a API da Voyage falha; sem `VOYAGE_API_KEY` retornam vetores fake determinísticos. `embed([])` retorna `[]`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_memory.py`:

```python
from app import memory
from app.config import EMBEDDING_DIM


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
```

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_memory.py -q
```

Esperado: FAIL com `ModuleNotFoundError: No module named 'app.memory'`.

- [ ] **Step 3: Implementar**

Criar `app/memory.py`:

```python
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
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 27 passed.

- [ ] **Step 5: Commit**

```bash
git add app/memory.py tests/test_memory.py
git commit -m "feat(memory): embed com Voyage e fallback deterministico offline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `remember()` e `retrieve()` com fusão RRF

Coração da spec: a busca híbrida numa query só.

**Files:**
- Modify: `app/memory.py`
- Test: `tests/test_memory.py` (acrescentar)

**Interfaces:**
- Consumes: `embed`, `embed_query` (Task 3); `app.db.MemoryItem`, `app.db.audit` (Task 2); `app.config.RETRIEVE_CANDIDATES` (Task 1).
- Produces:
  - `app.memory.Hit` — dataclass com `item: MemoryItem`, `score: float`, `via: set[str]`.
  - `app.memory.remember(session, kind, content, *, item_metadata=None, source_entity=None, source_id=None, **keys) -> MemoryItem`. `keys` aceita só `supplier_id`, `customer_id`, `bank_account_id`, `cnpj`, `amount`; qualquer outra chave levanta `ValueError`. Faz `flush`, nunca `commit`.
  - `app.memory.retrieve(session, kind, query, *, filters=None, k=5) -> list[Hit]`, ordenada por `score` decrescente.
  - `app.memory.FILTER_COLUMNS: frozenset[str]`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_memory.py`:

```python
import pytest

from app.db import MATCH_DECISION


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
```

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_memory.py -q
```

Esperado: FAIL com `AttributeError: module 'app.memory' has no attribute 'remember'`.

- [ ] **Step 3: Implementar**

Acrescentar aos imports de `app/memory.py`:

```python
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import MemoryItem, audit
```

Acrescentar no fim de `app/memory.py`:

```python
FILTER_COLUMNS = frozenset(
    {"supplier_id", "customer_id", "bank_account_id", "cnpj", "amount"}
)
RRF_K = 60  # constante classica da Reciprocal Rank Fusion


@dataclass(slots=True)
class Hit:
    item: MemoryItem
    score: float
    via: set[str] = field(default_factory=set)


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


def retrieve(
    session: Session,
    kind: str,
    query: str,
    *,
    filters: dict | None = None,
    k: int = 5,
) -> list[Hit]:
    """Busca hibrida: filtro estruturado -> full-text + vetor -> fusao RRF."""
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
        audit(session, "memory", kind, "MEMORY_DEGRADED", kind=kind, reason="embedding_indisponivel")
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
        return []

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
    return hits
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 36 passed. Se `test_retrieve_acha_container_pelo_full_text` falhar porque o vetor fake colocou outro item à frente, aumentar o número de memórias de ruído no teste não resolve o problema de verdade: conferir que o canal léxico está realmente filtrando por `@@` (só o item com o container deve entrar na CTE `lexical`, ficando com `rank = 1` e o melhor score).

- [ ] **Step 5: Commit**

```bash
git add app/memory.py tests/test_memory.py
git commit -m "feat(memory): remember e retrieve com busca hibrida e fusao RRF

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Adaptadores LangChain (ponte para a Spec 2)

Camada fina: o agente da Spec 2 consome `retrieve()` sem reescrita.

**Files:**
- Modify: `app/memory.py`
- Test: `tests/test_memory.py` (acrescentar)

**Interfaces:**
- Consumes: `retrieve`, `Hit` (Task 4).
- Produces: `app.memory.RecebaRetriever(session, kind, *, filters=None, k=5)` — subclasse de `langchain_core.retrievers.BaseRetriever` com os campos `session`, `kind`, `filters`, `k`; e `app.memory.retriever_tool(session, kind, *, filters=None, k=5, name=None, description=None) -> BaseTool`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_memory.py`:

```python
def test_retriever_devolve_documents_com_metadata(session, company):
    alvo = _lembrar(
        session,
        "documento com container MSCU1234567",
        item_metadata={"invoice_number": "44120"},
        source_entity="document",
        source_id=9,
        supplier_id=company["supplier"].id,
    )
    retriever = memory.RecebaRetriever(session=session, kind=db.EXTRACTION_EXAMPLE, k=3)

    docs = retriever.invoke("MSCU1234567")
    assert docs[0].page_content == alvo.content
    assert docs[0].metadata["memory_id"] == alvo.id
    assert docs[0].metadata["kind"] == db.EXTRACTION_EXAMPLE
    assert docs[0].metadata["source"] == "document:9"
    assert docs[0].metadata["invoice_number"] == "44120"
    assert "lexical" in docs[0].metadata["via"]


def test_retriever_tool_e_chamavel_pelo_agente(session):
    _lembrar(session, "documento com container MSCU1234567")
    tool = memory.retriever_tool(session, db.EXTRACTION_EXAMPLE, k=2)

    assert tool.name == "buscar_extraction_example"
    assert "MSCU1234567" in tool.invoke({"query": "MSCU1234567"})
```

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_memory.py -q
```

Esperado: FAIL com `AttributeError: module 'app.memory' has no attribute 'RecebaRetriever'`.

- [ ] **Step 3: Implementar**

Acrescentar aos imports de `app/memory.py`:

```python
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
```

Acrescentar no fim de `app/memory.py`:

```python
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
        return [
            Document(
                page_content=hit.item.content,
                metadata={
                    **(hit.item.item_metadata or {}),
                    "memory_id": hit.item.id,
                    "kind": hit.item.kind,
                    "score": hit.score,
                    "via": sorted(hit.via),
                    "source": f"{hit.item.source_entity}:{hit.item.source_id}",
                },
            )
            for hit in hits
        ]


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
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 38 passed.

- [ ] **Step 5: Commit**

```bash
git add app/memory.py tests/test_memory.py
git commit -m "feat(memory): retriever e tool LangChain sobre retrieve()

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `PATCH /documents/{id}/extracted` grava `EXTRACTION_EXAMPLE`

**Files:**
- Modify: `app/main.py` (acrescentar endpoint depois de `process_document`, que termina na linha 201)
- Test: `tests/test_api.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.remember` (Task 4); `app.db.EXTRACTION_EXAMPLE` (Task 2).
- Produces: endpoint `PATCH /documents/{document_id}/extracted` (perfil `WRITE`), corpo `ExtractionCorrectionIn` com todos os campos opcionais (`supplier_name`, `supplier_cnpj`, `invoice_number`, `amount`, `due_date`, `bank_code`, `bank_agency`, `bank_account`, `purchase_order`, `bill_of_lading`, `container`, `cost_center`, `observation`); resposta `{"id", "status", "corrected_fields", "memory_id"}`.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar em `tests/test_api.py`:

```python
def test_patch_extracted_corrige_campos_e_grava_memoria(client, session):
    upload = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("nota.txt", NOTA.encode(), "text/csv")},
    )
    document_id = upload.json()["id"]
    client.post(f"/documents/{document_id}/process", headers=HEADERS)

    response = client.patch(
        f"/documents/{document_id}/extracted",
        headers=HEADERS,
        json={"invoice_number": "44121", "amount": "7600.00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "EXTRACTED"
    assert sorted(body["corrected_fields"]) == ["amount", "invoice_number"]
    assert body["memory_id"] is not None

    detalhe = client.get(f"/documents/{document_id}", headers=HEADERS).json()
    assert detalhe["extracted"]["invoice_number"] == "44121"
    assert detalhe["extracted"]["amount"] == "7600.00"

    auditoria = client.get(f"/audit/document/{document_id}", headers=HEADERS).json()
    tipos = [evento["event_type"] for evento in auditoria]
    assert "EXTRACTION_CORRECTED" in tipos


def test_patch_extracted_exige_documento_processado(client):
    upload = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("nota.txt", NOTA.encode(), "text/csv")},
    )
    response = client.patch(
        f"/documents/{upload.json()['id']}/extracted",
        headers=HEADERS,
        json={"invoice_number": "44121"},
    )
    assert response.status_code == 409


def test_patch_extracted_nega_viewer(client):
    upload = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("nota.txt", NOTA.encode(), "text/csv")},
    )
    document_id = upload.json()["id"]
    client.post(f"/documents/{document_id}/process", headers=HEADERS)
    response = client.patch(
        f"/documents/{document_id}/extracted", headers=VIEWER, json={"invoice_number": "x"}
    )
    assert response.status_code == 403
```

Conferir antes se a fixture `client` já registra a chave `viewer-key` usada por `VIEWER`; se não, acrescentar `monkeypatch.setitem(API_KEYS, "viewer-key", "VIEWER")` na fixture.

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_api.py -q -k patch_extracted
```

Esperado: FAIL com status 405 (método não permitido), porque a rota ainda não existe.

- [ ] **Step 3: Implementar**

Em `app/main.py`, acrescentar aos imports:

```python
from app import bank_import, matching, memory, reconcile
from app.db import (
    ...,
    EXTRACTION_EXAMPLE,
    ...,
)
```

Acrescentar depois de `process_document` (que termina na linha 201):

```python
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

    corrections = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not corrections:
        raise HTTPException(422, "Nenhum campo informado")

    before = extracted.structured_payload or {}
    for field_name, value in corrections.items():
        setattr(extracted, field_name, value)
    extracted.structured_payload = {**before, **jsonable_encoder(corrections)}
    extracted.model_name = f"{extracted.model_name or 'desconhecido'}+humano"

    supplier = session.scalar(
        select(Supplier).where(Supplier.cnpj == digits(extracted.supplier_cnpj))
    ) if extracted.supplier_cnpj else None

    text = read_text(os.path.join(STORAGE_DIR, document.storage_key))
    item = memory.remember(
        session,
        EXTRACTION_EXAMPLE,
        text[:4000],
        item_metadata=extracted.structured_payload,
        source_entity="document",
        source_id=document_id,
        supplier_id=supplier.id if supplier else None,
        cnpj=digits(extracted.supplier_cnpj) or None,
        amount=extracted.amount,
    )

    # Campos mudaram: a validacao anterior nao vale mais.
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
    )
    session.commit()
    return {
        "id": document_id,
        "status": document.status,
        "corrected_fields": sorted(corrections),
        "memory_id": item.id,
    }
```

Acrescentar aos imports do topo de `app/main.py`:

```python
from fastapi.encoders import ENCODERS_BY_TYPE, jsonable_encoder
from app.db import Supplier  # junto dos outros nomes importados de app.db
from app.parser import digits
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 41 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat(api): PATCH de correcao de extracao grava memoria procedural

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Few-shot na extração

**Files:**
- Modify: `app/extract.py` (`PROMPT` nas linhas 106-109, `extract_with_llm` nas linhas 112-138)
- Modify: `app/main.py` (`process_document`, linha 167 e o `audit` da linha 189)
- Test: `tests/test_mvp.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.retrieve` (Task 4); `app.db.EXTRACTION_EXAMPLE` (Task 2).
- Produces: `app.extract.extract_with_llm(text, session=None) -> tuple[FinancialDocument, str, list[int]]` — **a assinatura muda**: passa a receber uma `session` opcional e a devolver uma terceira posição com os ids das memórias usadas (lista vazia quando não houve nenhuma). `app.extract.build_few_shot(hits) -> str`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_mvp.py`:

```python
from app import extract, memory
from app.db import EXTRACTION_EXAMPLE


def test_extract_sem_api_key_nao_usa_memoria(session):
    memory.remember(session, EXTRACTION_EXAMPLE, "exemplo qualquer", cnpj="11222333000181")
    session.commit()

    documento, modelo, memory_ids = extract.extract_with_llm(
        "CNPJ 11.222.333/0001-81 Valor Total 7.500,00", session=session
    )
    assert modelo == "offline-regex"
    assert memory_ids == [], "sem LLM nao ha prompt onde injetar exemplo"
    assert documento.supplier_cnpj == "11222333000181"


def test_extract_com_llm_injeta_exemplo_do_mesmo_cnpj(session, monkeypatch):
    item = memory.remember(
        session,
        EXTRACTION_EXAMPLE,
        "NOTA ACME layout antigo invoice 44120",
        item_metadata={"invoice_number": "44120", "supplier_name": "ACME LTDA"},
        cnpj="11222333000181",
    )
    session.commit()

    capturado = {}

    class FakeBlock:
        type = "tool_use"
        input = {"document_type": "INVOICE", "invoice_number": "44121"}

    class FakeMessages:
        def create(self, **kwargs):
            capturado.update(kwargs)
            return type("R", (), {"content": [FakeBlock()]})()

    class FakeAnthropic:
        messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-de-teste")
    monkeypatch.setattr(extract, "_anthropic_client", lambda: FakeAnthropic())

    documento, modelo, memory_ids = extract.extract_with_llm(
        "NOTA FISCAL CNPJ 11.222.333/0001-81 invoice 44121", session=session
    )

    assert memory_ids == [item.id]
    assert documento.invoice_number == "44121"
    enviado = capturado["messages"][0]["content"]
    assert "44120" in enviado, "o exemplo corrigido deve entrar no prompt"
    assert "EXEMPLOS" in enviado


def test_extract_com_llm_sem_memoria_mantem_prompt_simples(session, monkeypatch):
    class FakeBlock:
        type = "tool_use"
        input = {"document_type": "INVOICE"}

    capturado = {}

    class FakeMessages:
        def create(self, **kwargs):
            capturado.update(kwargs)
            return type("R", (), {"content": [FakeBlock()]})()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-de-teste")
    monkeypatch.setattr(
        extract, "_anthropic_client", lambda: type("A", (), {"messages": FakeMessages()})()
    )

    _, _, memory_ids = extract.extract_with_llm("NOTA FISCAL sem historico", session=session)
    assert memory_ids == []
    assert "EXEMPLOS" not in capturado["messages"][0]["content"]
```

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_mvp.py -q -k extract
```

Esperado: FAIL com `TypeError: extract_with_llm() got an unexpected keyword argument 'session'`.

- [ ] **Step 3: Implementar**

Em `app/extract.py`, acrescentar aos imports:

```python
from app.db import EXTRACTION_EXAMPLE
```

Substituir `PROMPT` e `extract_with_llm` (linhas 106-138) por:

```python
PROMPT = (
    "Extraia os campos financeiros do documento abaixo. Nao calcule, nao infira valores "
    "que nao estejam escritos e use null quando o campo nao existir no texto.\n\n"
)
FEW_SHOT_LIMIT = 3
FEW_SHOT_CHARS = 1500


def build_few_shot(hits) -> str:
    """Exemplos corrigidos por humano, do mesmo fornecedor quando possivel."""
    if not hits:
        return ""
    blocos = []
    for hit in hits:
        fields = json.dumps(hit.item.item_metadata or {}, ensure_ascii=False, default=str)
        blocos.append(f"DOCUMENTO:\n{hit.item.content[:FEW_SHOT_CHARS]}\nCAMPOS CORRETOS:\n{fields}")
    return (
        "EXEMPLOS de documentos deste fornecedor ja corrigidos por um humano. "
        "Use o layout como referencia, nunca copie os valores:\n\n"
        + "\n\n---\n\n".join(blocos)
        + "\n\n---\n\n"
    )


def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic()


def extract_with_llm(text: str, session=None) -> tuple[FinancialDocument, str, list[int]]:
    """Structured output via tool schema. Sem API key, cai no parser deterministico."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return extract_offline(text), "offline-regex", []

    # Regex primeiro so para achar o CNPJ: custa zero e dirige a busca na memoria.
    hits = []
    if session is not None:
        from app import memory

        preliminar = extract_offline(text)
        hits = memory.retrieve(
            session,
            EXTRACTION_EXAMPLE,
            text,
            filters={"cnpj": preliminar.supplier_cnpj},
            k=FEW_SHOT_LIMIT,
        )

    schema = FinancialDocument.model_json_schema()
    response = _anthropic_client().messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[{
            "name": "emit_document",
            "description": "Campos extraidos do documento financeiro",
            "input_schema": {
                "type": "object",
                "properties": schema["properties"],
                "$defs": schema.get("$defs", {}),
            },
        }],
        tool_choice={"type": "tool", "name": "emit_document"},
        messages=[{"role": "user", "content": PROMPT + build_few_shot(hits) + text[:120_000]}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return FinancialDocument(**block.input), MODEL, [hit.item.id for hit in hits]
    raise RuntimeError(f"LLM nao retornou tool_use: {json.dumps(response.model_dump())[:400]}")
```

Em `app/main.py`, na linha 167, trocar:

```python
        data, model_name = extract_with_llm(text)
```

por:

```python
        data, model_name, memory_ids = extract_with_llm(text, session=session)
```

E no `audit` da linha 189, trocar:

```python
    audit(session, "document", document_id, "DOCUMENT_EXTRACTED", model=model_name)
```

por:

```python
    audit(session, "document", document_id, "DOCUMENT_EXTRACTED",
          model=model_name, memory_ids=memory_ids)
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 44 passed. Se algum teste existente chamar `extract_with_llm` esperando duas posições, atualizar a chamada para desempacotar três.

- [ ] **Step 5: Commit**

```bash
git add app/extract.py app/main.py tests/test_mvp.py
git commit -m "feat(extract): few-shot com exemplos corrigidos do mesmo fornecedor

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Embeddings em lote no import de extrato

**Files:**
- Modify: `app/bank_import.py` (`import_csv`, linhas 42-103)
- Test: `tests/test_mvp.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.embed` (Task 3); `app.db.BankTransaction.embedding` (Task 2).
- Produces: `import_csv` passa a preencher `BankTransaction.embedding`; a chave `"embedded"` entra no dict de retorno, com a contagem de transações que receberam vetor.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_mvp.py`:

```python
from app import bank_import

CSV = (
    "date;amount;description\n"
    "09/10/2026;-7500,00;PAGAMENTO ACME INVOICE 44120\n"
    "09/10/2026;-1200,00;PAGAMENTO BETA INVOICE 99999\n"
)


def test_import_gera_embedding_em_lote(session, company):
    resultado = bank_import.import_csv(session, company["account"].id, CSV.encode())
    assert resultado["imported"] == 2
    assert resultado["embedded"] == 2

    for transaction_id in resultado["ids"]:
        transaction = session.get(db.BankTransaction, transaction_id)
        assert len(transaction.embedding) == EMBEDDING_DIM


def test_import_nao_quebra_quando_embedding_falha(session, company, monkeypatch):
    monkeypatch.setattr("app.memory.embed", lambda texts: None)

    resultado = bank_import.import_csv(session, company["account"].id, CSV.encode())
    assert resultado["imported"] == 2
    assert resultado["embedded"] == 0
    assert session.get(db.BankTransaction, resultado["ids"][0]).embedding is None
```

Conferir que o topo de `tests/test_mvp.py` já importa `db` e `EMBEDDING_DIM`; se não, acrescentar `from app import db` e `from app.config import EMBEDDING_DIM`.

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_mvp.py -q -k embedding
```

Esperado: FAIL com `KeyError: 'embedded'`.

- [ ] **Step 3: Implementar**

Em `app/bank_import.py`, acrescentar aos imports:

```python
from app import memory
```

Trocar o fim de `import_csv` (linhas 100-103), de:

```python
        created.append(transaction.id)

    session.commit()
    return {"imported": len(created), "duplicated": duplicated, "ids": created}
```

para:

```python
        created.append(transaction.id)
        pending.append(transaction)

    embedded = _embed_pending(session, pending)
    session.commit()
    return {
        "imported": len(created),
        "duplicated": duplicated,
        "embedded": embedded,
        "ids": created,
    }
```

Na linha 52, trocar `created, duplicated = [], 0` por:

```python
    created, duplicated, pending = [], 0, []
```

E acrescentar antes de `import_csv`:

```python
def _embed_pending(session: Session, transactions: list[BankTransaction]) -> int:
    """Embeddings em lote: a Voyage aceita varios textos de uma vez, o matching nao espera API."""
    if not transactions:
        return 0
    vectors = memory.embed([t.description_normalized or "" for t in transactions])
    if vectors is None:
        audit(session, "bank_account", transactions[0].bank_account_id, "MEMORY_DEGRADED",
              reason="embedding_indisponivel", transactions=len(transactions))
        return 0
    for transaction, vector in zip(transactions, vectors):
        transaction.embedding = vector
    return len(transactions)
```

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 46 passed.

- [ ] **Step 5: Commit**

```bash
git add app/bank_import.py tests/test_mvp.py
git commit -m "feat(bank_import): embeddings em lote no import de extrato

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Gravar `MATCH_DECISION` na aprovação e na rejeição

**Files:**
- Modify: `app/main.py` (`_approve_group` nas linhas 520-548, `reject_matching` nas linhas 394-410)
- Test: `tests/test_api.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.remember` (Task 4); `app.db.MATCH_DECISION` (Task 2).
- Produces: função auxiliar `app.main._remember_match(session, group, decision, *, actor, reason=None) -> MemoryItem | None`. O `item_metadata` gravado tem as chaves `decision` (`"APPROVED"` ou `"REJECTED"`), `entity_type`, `entity_id`, `group_type`, `amount`, `reason`; as colunas `supplier_id`, `customer_id`, `bank_account_id` e `amount` são preenchidas. Retorna `None` quando o grupo não tem item de título.

- [ ] **Step 1: Escrever o teste que falha**

Acrescentar em `tests/test_api.py`:

```python
from app import memory
from app.db import MATCH_DECISION


def test_aprovacao_de_match_grava_memoria_episodica(client, session, company, make_payable,
                                                    make_transaction):
    payable = make_payable("44120", "7500.00")
    transaction = make_transaction("7500.00", "PAGAMENTO ACME NF 44120 DIVERGENTE")
    client.post(f"/matching/{transaction.id}/run", headers=HEADERS)
    grupo = client.get(f"/matching/{transaction.id}/candidates", headers=HEADERS)
    assert grupo.status_code == 200

    resposta = client.post(f"/matching/{transaction.id}/approve", headers=HEADERS)
    assert resposta.status_code == 200

    hits = memory.retrieve(session, MATCH_DECISION, "PAGAMENTO ACME NF 44120")
    assert hits, "aprovacao humana deve virar memoria"
    metadados = hits[0].item.item_metadata
    assert metadados["decision"] == "APPROVED"
    assert metadados["entity_id"] == payable.id
    assert hits[0].item.supplier_id == company["supplier"].id


def test_rejeicao_de_match_grava_memoria_negativa(client, session, make_payable,
                                                  make_transaction):
    make_payable("44120", "7500.00")
    transaction = make_transaction("7500.00", "PAGAMENTO ACME NF 44120 DIVERGENTE")
    client.post(f"/matching/{transaction.id}/run", headers=HEADERS)

    resposta = client.post(
        f"/matching/{transaction.id}/reject",
        headers=HEADERS,
        params={"reason": "fornecedor errado"},
    )
    assert resposta.status_code == 200

    hits = memory.retrieve(session, MATCH_DECISION, "PAGAMENTO ACME NF 44120")
    assert hits[0].item.item_metadata["decision"] == "REJECTED"
    assert hits[0].item.item_metadata["reason"] == "fornecedor errado"
```

Se o fluxo de `run` não gerar grupo para essa transação (por cair em `AUTO_MATCHED`), ajustar a descrição ou o valor da transação para que ela caia em `REVIEW_REQUIRED` e gere grupo, conferindo com os casos já existentes em `tests/test_api.py`.

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_api.py -q -k memoria_episodica
```

Esperado: FAIL no `assert hits`, porque nenhuma memória é gravada.

- [ ] **Step 3: Implementar**

Em `app/main.py`, acrescentar depois de `_recalc`:

```python
def _remember_match(
    session: Session,
    group: ReconciliationGroup,
    decision: str,
    *,
    actor: Actor,
    reason: str | None = None,
):
    """Decisao humana de match vira memoria episodica, com a parte envolvida como chave."""
    transaction = session.get(BankTransaction, group.bank_transaction_id)
    item = session.scalar(
        select(ReconciliationGroupItem).where(
            ReconciliationGroupItem.reconciliation_group_id == group.id,
            ReconciliationGroupItem.entity_type.in_(("PAYABLE", "RECEIVABLE")),
        )
    )
    if transaction is None or item is None:
        return None

    if item.entity_type == "PAYABLE":
        title = session.get(Payable, item.entity_id)
        supplier_id = title.supplier_id if title else None
        customer_id = None
    else:
        title = session.get(Receivable, item.entity_id)
        supplier_id = None
        customer_id = title.customer_id if title else None

    return memory.remember(
        session,
        MATCH_DECISION,
        transaction.description_normalized or transaction.description_raw or "",
        item_metadata={
            "decision": decision,
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "group_type": group.group_type,
            "amount": str(transaction.amount),
            "reason": reason,
            "actor": actor.key,
        },
        source_entity="reconciliation_group",
        source_id=group.id,
        supplier_id=supplier_id,
        customer_id=customer_id,
        bank_account_id=transaction.bank_account_id,
        amount=Decimal(transaction.amount),
    )
```

Em `_approve_group`, acrescentar logo antes do `audit` (linha 545):

```python
    _remember_match(session, group, "APPROVED", actor=actor)
```

Em `reject_matching`, acrescentar logo antes do `audit` (linha 407):

```python
    _remember_match(session, group, "REJECTED", actor=actor, reason=reason)
```

Acrescentar `MATCH_DECISION` aos nomes importados de `app.db` no topo do arquivo.

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 48 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat(api): aprovacao e rejeicao de match viram memoria episodica

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Peso `history` no matching, com teto

A tarefa mais sensível do plano: o teto é o que mantém o princípio do produto.

**Files:**
- Modify: `app/config.py` (`WEIGHTS`)
- Modify: `app/matching.py` (`Scored` nas linhas 37-55, `candidates` nas linhas 103-129, `evaluate` nas linhas 155-255, `run` na linha 268)
- Modify: `.env.example`
- Test: `tests/test_mvp.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.retrieve` (Task 4); o formato de `item_metadata` de `MATCH_DECISION` (Task 9); `app.config.HISTORY_HITS` (Task 1).
- Produces: `app.matching.Scored.history: int` (default 0), `app.matching.Scored.effective: int` (property, `min(100, max(0, score + history))`), e `app.matching.apply_history(session, transaction, pool) -> None`, que muta o `pool` no lugar. `config.WEIGHTS` ganha a chave `"history"` com valor 10.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_mvp.py`:

```python
from app import matching
from app.db import MATCH_DECISION


def _memoria_de_match(session, company, payable, decision, descricao):
    memory.remember(
        session,
        MATCH_DECISION,
        descricao,
        item_metadata={
            "decision": decision,
            "entity_type": "PAYABLE",
            "entity_id": payable.id,
        },
        supplier_id=company["supplier"].id,
        bank_account_id=company["account"].id,
    )
    session.commit()


def test_history_sozinho_nao_gera_auto_match(session, company, make_payable, make_transaction):
    """Teto da memoria: sem sinal forte proprio, history nao libera baixa automatica."""
    payable = make_payable("44120", "7500.00")
    transaction = make_transaction("9999.00", "TRANSFERENCIA ACME SEM REFERENCIA")
    _memoria_de_match(session, company, payable, "APPROVED",
                      "TRANSFERENCIA ACME SEM REFERENCIA")

    resultado, pool = matching.evaluate(session, transaction)
    assert resultado.status != matching.AUTO_MATCHED
    assert any(c.history > 0 for c in pool), "history deve ter sido aplicado"


def test_history_promove_unmatched_para_revisao(session, company, make_payable,
                                                make_transaction):
    payable = make_payable("44120", "500.00")
    transaction = make_transaction("777.00", "TRANSFERENCIA ACME MENSAL")
    sem_memoria, _ = matching.evaluate(session, transaction)

    _memoria_de_match(session, company, payable, "APPROVED", "TRANSFERENCIA ACME MENSAL")
    com_memoria, pool = matching.evaluate(session, transaction)

    assert sem_memoria.confidence < com_memoria.confidence
    if sem_memoria.status == matching.UNMATCHED:
        assert com_memoria.status in {matching.UNMATCHED, matching.REVIEW_REQUIRED}


def test_rejeicao_passada_rebaixa_candidato(session, company, make_payable, make_transaction):
    payable = make_payable("44120", "7500.00")
    transaction = make_transaction("7500.00", "PAGAMENTO ACME MENSAL")
    _memoria_de_match(session, company, payable, "REJECTED", "PAGAMENTO ACME MENSAL")

    _, pool = matching.evaluate(session, transaction)
    alvo = next(c for c in pool if c.id == payable.id)
    assert alvo.history < 0
    assert "history_rejected" in alvo.reasons
    assert alvo.effective < alvo.score


def test_history_desempata_candidatos(session, company, make_payable, make_transaction):
    primeiro = make_payable("44120", "7500.00")
    make_payable("44121", "7500.00")
    transaction = make_transaction("7500.00", "PAGAMENTO ACME MENSAL")
    _memoria_de_match(session, company, primeiro, "APPROVED", "PAGAMENTO ACME MENSAL")

    _, pool = matching.evaluate(session, transaction)
    assert pool[0].id == primeiro.id, "memoria positiva deve ordenar o candidato na frente"


def test_matching_funciona_sem_nenhuma_memoria(session, make_payable, make_transaction):
    """Regressao: pipeline inteiro continua igual quando a memoria esta vazia."""
    make_payable("44120", "7500.00")
    transaction = make_transaction("7500.00", "PAGAMENTO NF 44120")
    resultado, pool = matching.evaluate(session, transaction)
    assert resultado.status == matching.AUTO_MATCHED
    assert all(c.history == 0 for c in pool)
```

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest tests/test_mvp.py -q -k history
```

Esperado: FAIL com `AttributeError: 'Scored' object has no attribute 'history'`.

- [ ] **Step 3: Implementar**

Em `app/config.py`, trocar o default de `MATCH_WEIGHTS` (linhas 26-29) por:

```python
WEIGHTS: dict[str, int] = json.loads(
    os.getenv(
        "MATCH_WEIGHTS",
        '{"invoice":40,"cnpj":25,"amount":20,"party":10,"date":5,"bl":15,"container":15,'
        '"history":10}',
    )
)
```

Em `.env.example`, acrescentar:

```
MATCH_WEIGHTS={"invoice":40,"cnpj":25,"amount":20,"party":10,"date":5,"bl":15,"container":15,"history":10}
```

Em `app/matching.py`, acrescentar aos imports:

```python
from app import config, memory
from app.db import (
    ...,
    MATCH_DECISION,
    ...,
)
```

Na dataclass `Scored` (linhas 37-55), acrescentar o campo e a property:

```python
@dataclass(slots=True)
class Scored:
    entity_type: str
    entity: object
    score: int  # score base: so sinais do documento. O teto da memoria mora aqui.
    reasons: list[str] = field(default_factory=list)
    history: int = 0  # ajuste vindo da memoria; nunca entra no score base

    @property
    def id(self) -> int:
        return self.entity.id

    @property
    def amount_open(self) -> Decimal:
        return Decimal(self.entity.amount_open)

    @property
    def strong(self) -> bool:
        return any(r in STRONG_SIGNALS for r in self.reasons)

    @property
    def effective(self) -> int:
        """Score com a memoria. Usado para ordenar, desempatar e promover revisao."""
        return min(100, max(0, self.score + self.history))
```

Acrescentar antes de `candidates`:

```python
def apply_history(session: Session, transaction: BankTransaction, pool: list[Scored]) -> None:
    """Aplica memoria episodica ao pool. Muta no lugar; nunca mexe no score base."""
    weight = config.WEIGHTS.get("history", 0)
    if not pool or not weight:
        return
    query = transaction.description_normalized or transaction.description_raw or ""
    if not query:
        return

    hits = memory.retrieve(
        session,
        MATCH_DECISION,
        query,
        filters={"bank_account_id": transaction.bank_account_id},
        k=config.HISTORY_HITS,
    )
    if not hits:
        return

    # A memoria vale pela parte envolvida, nao pelo titulo: titulo pago nao reaparece.
    by_party: dict[tuple[str, int], str] = {}
    for hit in hits:
        decision = (hit.item.item_metadata or {}).get("decision")
        if decision not in {"APPROVED", "REJECTED"}:
            continue
        for column, value in (("supplier", hit.item.supplier_id),
                              ("customer", hit.item.customer_id)):
            if value is not None:
                by_party.setdefault((column, value), decision)

    for candidate in pool:
        party = _party(session, candidate.entity)
        if party is None:
            continue
        key = ("supplier" if isinstance(candidate.entity, Payable) else "customer", party.id)
        decision = by_party.get(key)
        if decision == "APPROVED":
            candidate.history += weight
            candidate.reasons.append("history")
        elif decision == "REJECTED":
            candidate.history -= weight
            candidate.reasons.append("history_rejected")
```

Em `candidates`, trocar o fim da função (linhas 122-129) por:

```python
    entity_type = "PAYABLE" if model is Payable else "RECEIVABLE"
    scored = []
    for row in rows:
        value, reasons = score(transaction, row, _party(session, row), tokens)
        if value > 0:
            scored.append(Scored(entity_type, row, value, reasons))
    apply_history(session, transaction, scored)
    scored.sort(key=lambda s: (-s.effective, s.amount_open))
    return scored
```

Em `evaluate`, aplicar exatamente estas quatro trocas, mantendo todo o resto igual:

1. Linha 167, trocar `tied = [c for c in pool if c.score == best.score]` por:

```python
    tied = [c for c in pool if c.effective == best.effective]
```

2. Linha 186, **não mexer**: `if best.score >= config.AUTO_MATCH_MIN and "amount" in best.reasons:` continua usando `score`, que é o score base. É aqui que o teto vive — acrescentar o comentário acima da linha:

```python
    # 2) score alto com valor batendo. Teto da memoria: score base, sem history.
```

3. Linha 253, trocar `if best.score >= config.REVIEW_MIN:` por:

```python
    # History pode promover UNMATCHED -> REVIEW_REQUIRED, nunca liberar baixa.
    if best.effective >= config.REVIEW_MIN:
```

4. Nas linhas 180, 189, 227, 248, 254 e 255, onde aparece `confidence=best.score`, trocar por `confidence=best.effective`. E na linha 240, `min(item.score for item in group)` passa a `min(item.effective for item in group)`.

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 53 passed. Se `test_history_sozinho_nao_gera_auto_match` falhar com `AUTO_MATCHED`, conferir que a troca 2 foi respeitada: a condição de auto-match **não** pode usar `effective`.

- [ ] **Step 5: Confirmar o teto também pelo caminho 1 (exact match)**

```bash
.venv/bin/python -m pytest tests/test_mvp.py -q -k "exact or auto"
```

Esperado: PASS. O caminho de exact match decide por `reasons` e `amount`, não por score, então `history` não o afeta: `"history"` não está em `STRONG_SIGNALS`. Confirmar lendo a linha 34 de `app/matching.py`.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/matching.py .env.example tests/test_mvp.py
git commit -m "feat(matching): peso history com teto no score base

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: `DIVERGENCE_CASE` e `similar_cases` nas divergências

**Files:**
- Modify: `app/reconcile.py` (`investigate`, linhas 24-84)
- Modify: `app/main.py` (`approve_reconciliation`, linhas 634-650)
- Test: `tests/test_mvp.py` (acrescentar)

**Interfaces:**
- Consumes: `app.memory.retrieve`, `app.memory.remember` (Task 4); `app.db.DIVERGENCE_CASE` (Task 2).
- Produces: cada dict de `investigate()` dos tipos `TRANSACTION_NOT_MAPPED` e `GROUP_DIFFERENCE` ganha a chave `similar_cases: list[dict]`, cada item com `memory_id`, `content`, `resolution` e `score`. `approve_reconciliation` grava um `DIVERGENCE_CASE` por finding quando a conciliação tinha findings, e devolve `{"id", "status", "memory_ids"}`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar em `tests/test_mvp.py`:

```python
from app import reconcile
from app.db import DIVERGENCE_CASE


def test_finding_traz_casos_parecidos(session, company, make_transaction):
    memory.remember(
        session,
        DIVERGENCE_CASE,
        "TARIFA BANCARIA MENSAL nao mapeada",
        item_metadata={"resolution": "lancada como despesa bancaria"},
        bank_account_id=company["account"].id,
    )
    session.commit()
    make_transaction("45.00", "TARIFA BANCARIA MENSAL")

    findings = reconcile.investigate(session, company["account"].id, TODAY)
    nao_mapeada = [f for f in findings if f["type"] == "TRANSACTION_NOT_MAPPED"]
    assert nao_mapeada
    similares = nao_mapeada[0]["similar_cases"]
    assert similares and similares[0]["resolution"] == "lancada como despesa bancaria"


def test_finding_sem_historico_traz_lista_vazia(session, company, make_transaction):
    make_transaction("45.00", "TARIFA BANCARIA MENSAL")
    findings = reconcile.investigate(session, company["account"].id, TODAY)
    nao_mapeada = [f for f in findings if f["type"] == "TRANSACTION_NOT_MAPPED"]
    assert nao_mapeada[0]["similar_cases"] == []
```

E em `tests/test_api.py`:

```python
from app.db import DIVERGENCE_CASE


def test_aprovacao_de_conciliacao_com_findings_grava_memoria(client, session, company,
                                                             make_transaction):
    make_transaction("45.00", "TARIFA BANCARIA MENSAL")
    client.post(
        "/bank-accounts/balances",
        headers=HEADERS,
        json={
            "bank_account_id": company["account"].id,
            "balance_date": TODAY.isoformat(),
            "opening_balance": "0.00",
            "closing_balance": "-45.00",
        },
    )
    execucao = client.post(
        "/reconciliations/run",
        headers=HEADERS,
        json={
            "bank_account_id": company["account"].id,
            "reconciliation_date": TODAY.isoformat(),
        },
    )
    assert execucao.status_code == 200
    reconciliation_id = execucao.json()["id"]

    aprovacao = client.post(
        f"/reconciliations/{reconciliation_id}/approve", headers=HEADERS
    )
    assert aprovacao.status_code == 200
    assert aprovacao.json()["memory_ids"], "findings resolvidos devem virar memoria"

    hits = memory.retrieve(session, DIVERGENCE_CASE, "TARIFA BANCARIA MENSAL")
    assert hits
```

Conferir o corpo que `/bank-accounts/balances` e `/reconciliations/run` esperam nos testes já existentes e ajustar as chaves do JSON se divergirem.

- [ ] **Step 2: Rodar para ver falhar**

```bash
.venv/bin/python -m pytest -q -k "casos_parecidos or conciliacao_com_findings"
```

Esperado: FAIL com `KeyError: 'similar_cases'`.

- [ ] **Step 3: Implementar**

Em `app/reconcile.py`, acrescentar aos imports:

```python
from app import memory
from app.db import DIVERGENCE_CASE
```

Acrescentar antes de `investigate`:

```python
SIMILAR_CASES = 3


def _similar_cases(session: Session, bank_account_id: int, query: str) -> list[dict]:
    """Evidencia para o humano. Nao altera nenhuma decisao de conciliacao."""
    # ponytail: uma busca por finding; se a lista crescer, batelar numa query so.
    if not query:
        return []
    hits = memory.retrieve(
        session,
        DIVERGENCE_CASE,
        query,
        filters={"bank_account_id": bank_account_id},
        k=SIMILAR_CASES,
    )
    return [
        {
            "memory_id": hit.item.id,
            "content": hit.item.content[:300],
            "resolution": (hit.item.item_metadata or {}).get("resolution"),
            "score": round(hit.score, 6),
        }
        for hit in hits
    ]
```

Dentro de `investigate`, no finding `TRANSACTION_NOT_MAPPED` (linhas 36-41), acrescentar a chave:

```python
            findings.append({
                "type": "TRANSACTION_NOT_MAPPED",
                "bank_transaction_id": transaction.id,
                "amount": str(transaction.amount),
                "description": transaction.description_raw,
                "similar_cases": _similar_cases(
                    session, bank_account_id, transaction.description_normalized or ""
                ),
            })
```

E no finding `GROUP_DIFFERENCE` (linhas 68-72):

```python
        findings.append({
            "type": "GROUP_DIFFERENCE",
            "group_id": group.id,
            "difference": str(group.difference),
            "similar_cases": _similar_cases(
                session, bank_account_id, f"diferenca de grupo {group.difference}"
            ),
        })
```

Em `app/main.py`, substituir o corpo de `approve_reconciliation` (linhas 634-650) por:

```python
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

    # Divergencia que o humano aceitou vira memoria: a proxima investigacao ja comeca com pista.
    memory_ids = []
    for finding in (record.findings or {}).get("items", []):
        content = finding.get("description") or f"{finding.get('type')} {finding}"
        item = memory.remember(
            session,
            DIVERGENCE_CASE,
            content[:2000],
            item_metadata={
                "finding": finding,
                "resolution": f"aprovada na conciliacao {record.id} por {actor.key}",
            },
            source_entity="daily_reconciliation",
            source_id=record.id,
            bank_account_id=record.bank_account_id,
        )
        memory_ids.append(item.id)

    audit(session, "daily_reconciliation", record.id, "HUMAN_APPROVED",
          actor_type="USER", actor_id=actor.key, memory_ids=memory_ids)
    session.commit()
    return {"id": record.id, "status": record.status, "memory_ids": memory_ids}
```

Acrescentar `DIVERGENCE_CASE` aos nomes importados de `app.db` no topo de `app/main.py`. Conferir o nome real do campo de conta em `DailyReconciliation` (linhas 243-256 de `app/db.py`) e usar esse nome em `bank_account_id=`.

- [ ] **Step 4: Rodar os testes**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 56 passed.

- [ ] **Step 5: Commit**

```bash
git add app/reconcile.py app/main.py tests/test_mvp.py tests/test_api.py
git commit -m "feat(reconcile): casos parecidos nos findings e memoria de divergencia

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Documentar no README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: tudo das tasks 1-11.
- Produces: nada em código.

- [ ] **Step 1: Atualizar a seção "Rodar"**

Substituir o bloco de comandos da seção `## Rodar` (linhas 11-17) por:

```bash
docker compose up -d db                       # Postgres 17 + pgvector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # ajuste API_KEYS, DATABASE_URL e as chaves de IA
.venv/bin/python -m pytest -q   # inclui os 10 casos obrigatorios do MVP
.venv/bin/uvicorn app.main:app --reload
```

E substituir o parágrafo das linhas 19-20 por:

```markdown
Sem `ANTHROPIC_API_KEY` a extração cai no parser determinístico (regex + observation parser)
e sem `VOYAGE_API_KEY` os embeddings viram vetores fake determinísticos, então o pipeline
inteiro roda offline — inclusive os testes. O Postgres com pgvector, porém, é obrigatório:
sem ele a suíte é pulada com uma mensagem explicando como subir o container.
```

- [ ] **Step 2: Acrescentar a seção de memória**

Acrescentar depois da seção `## Fluxo`, antes de `## Módulos`:

```markdown
## Memória e RAG híbrido

Memória só nasce de decisão humana e **nunca** produz baixa automática sozinha.

| memória | tipo | nasce de | é usada em |
|---|---|---|---|
| `EXTRACTION_EXAMPLE` | procedural | `PATCH /documents/{id}/extracted` | few-shot no prompt de extração |
| `MATCH_DECISION` | episódica | aprovação/rejeição de match | peso `history` no score |
| `DIVERGENCE_CASE` | episódica | aprovação de conciliação com findings | `similar_cases` nos findings |

A busca é híbrida numa query só: filtro estruturado (CNPJ, fornecedor, conta), full-text
`portuguese` e vetor cosine, fundidos por RRF. Se o provedor de embeddings cair, a busca
degrada para estruturado + full-text e grava `MEMORY_DEGRADED` no audit trail.

**O teto do `history`:** a condição de `AUTO_MATCHED` usa sempre o score base, sem memória.
O `history` só ordena candidatos, desempata e pode promover `UNMATCHED` para
`REVIEW_REQUIRED`. Regra garantida pelo teste `test_history_sozinho_nao_gera_auto_match`.

`app/memory.py` expõe também `RecebaRetriever` e `retriever_tool()`, os adaptadores LangChain
que o agente da Spec 2 (orquestração com LangGraph) vai consumir sem reescrever a busca.
```

- [ ] **Step 2b: Acrescentar a linha na tabela de módulos**

Na tabela da seção `## Módulos`, acrescentar depois da linha de `app/parser.py`:

```markdown
| `app/memory.py` | memória + RAG híbrido (RRF); adaptadores LangChain para a Spec 2 |
```

E atualizar a linha de `app/extract.py` para:

```markdown
| `app/extract.py` | PDF/texto → `FinancialDocument`; LLM com structured output e few-shot da memória, fallback regex |
```

- [ ] **Step 3: Conferir que a suíte segue verde**

```bash
.venv/bin/python -m pytest -q
```

Esperado: 56 passed.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: memoria e RAG hibrido no README

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Cobertura da spec

| requisito da spec | task |
|---|---|
| tabela `memory_items` com tsvector + vector + índices | 2 |
| `bank_transactions.embedding` | 2 |
| `embed()` com Voyage e fake offline | 3 |
| `remember()` | 4 |
| `retrieve()` híbrido com RRF e degradação + `MEMORY_DEGRADED` | 4 |
| `RecebaRetriever` e `retriever_tool` | 5 |
| `PATCH /documents/{id}/extracted` + `EXTRACTION_CORRECTED` | 6 |
| few-shot na extração + `memory_ids` no audit | 7 |
| embeddings em lote no import | 8 |
| `MATCH_DECISION` em approve/reject/group approve | 9 |
| peso `history` com teto | 10 |
| `DIVERGENCE_CASE` + `similar_cases` | 11 |
| docker-compose, env, extensão, conftest Postgres | 1 |
| 6 casos de teste obrigatórios da spec | 6, 4, 10, 10, 11, 4 |
| 20 testes existentes seguem passando | 1 (e verificado em todas) |
| README e `.env.example` | 1, 10, 12 |
