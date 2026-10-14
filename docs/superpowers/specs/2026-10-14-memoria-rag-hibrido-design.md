# Spec 1 — Memória e RAG híbrido

Data: 2026-10-14 · Status: aprovado · Próxima spec: agente LangGraph (Spec 2)

## Objetivo

Dar ao RECEBA memória construída a partir de decisões humanas, recuperada por busca híbrida
(estruturada + full-text + vetorial), para:

1. **melhorar a extração** — exemplos corrigidos do mesmo fornecedor entram como few-shot;
2. **ajudar o matching** — decisões passadas parecidas viram um peso `history` com teto;
3. **acelerar a investigação de divergências** — casos parecidos já resolvidos aparecem como evidência.

É a fundação da Spec 2: o agente LangGraph vai consumir `retrieve()` como ferramenta.

| memória | tipo | origem |
|---|---|---|
| `EXTRACTION_EXAMPLE` | procedural | correção humana de campos extraídos |
| `MATCH_DECISION` | episódica | aprovação/rejeição humana de match |
| `DIVERGENCE_CASE` | episódica | aprovação de reconciliação que tinha findings |

## Princípio que não muda

"IA interpreta. Código valida. Regras decidem. Humano resolve exceções."
A memória **nunca** produz baixa automática sozinha. Toda memória usada é registrada no audit trail.

## Dados

Postgres + extensão `pgvector`. Tabela nova `memory_items`:

| coluna | tipo | papel |
|---|---|---|
| `id` | serial | |
| `kind` | text | `EXTRACTION_EXAMPLE` \| `MATCH_DECISION` \| `DIVERGENCE_CASE` |
| `supplier_id` | int null | filtro estruturado |
| `customer_id` | int null | filtro estruturado |
| `bank_account_id` | int null | filtro estruturado |
| `cnpj` | text null | filtro estruturado (só dígitos) |
| `amount` | numeric null | filtro estruturado |
| `content` | text | texto buscável |
| `content_tsv` | tsvector (gerado, `portuguese`) | busca léxica, índice GIN |
| `embedding` | vector(1024) null | busca semântica, índice HNSW cosine |
| `metadata` | jsonb | payload (campos corrigidos, decisão, resolução) |
| `source_entity`, `source_id` | text, int | rastreabilidade |
| `created_at` | timestamp | |

`bank_transactions` ganha `embedding vector(1024) null`, preenchido no import.

## Módulo `app/memory.py`

**Núcleo (SQL puro, sem framework):**

- `embed(texts: list[str]) -> list[list[float]] | None` — Voyage via `langchain-voyageai`
  (`VoyageAIEmbeddings`, modelo por env `EMBEDDING_MODEL`). Sem `VOYAGE_API_KEY`: embedding fake
  determinístico (hash → vetor normalizado). Erro de API: retorna `None`.
- `remember(session, kind, content, *, metadata, source_entity, source_id, **keys) -> MemoryItem`
- `retrieve(session, kind, query, *, filters: dict, k=5) -> list[Hit]` — uma query:
  1. filtro estruturado por `kind` + `filters`;
  2. top `N` (env `RETRIEVE_CANDIDATES`, padrão 20) por `ts_rank` e top `N` por distância cosine;
  3. fusão **RRF**: `score = Σ 1 / (60 + rank)`;
  4. retorna `Hit(item, score, via={"lexical","semantic"})`.
  Se `embed()` retornar `None`, roda só estruturado + léxico e registra `MEMORY_DEGRADED` no audit.

**Adaptadores LangChain (finos, para a Spec 2):**

- `RecebaRetriever(BaseRetriever)` sobre `retrieve()`, devolvendo `Document`s com `metadata` e `source`.
- `retriever_tool(kind)` — `retriever.as_tool(...)` pronto para o agente.

## Escrita de memória (sempre a partir de sinal humano)

| gatilho | memória gravada |
|---|---|
| novo `PATCH /documents/{id}/extracted` (perfil de escrita) | `EXTRACTION_EXAMPLE`: content = texto do documento (truncado), metadata = campos corrigidos, `cnpj`/`supplier_id` |
| `POST /matching/{tx}/approve` | `MATCH_DECISION` aprovada: content = descrição da transação, metadata = entidade casada + motivos |
| `POST /matching/{tx}/reject` | `MATCH_DECISION` rejeitada |
| `POST /reconciliation-groups/{id}/approve` | `MATCH_DECISION` aprovada (lote) |
| `POST /reconciliations/{id}/approve` com findings | `DIVERGENCE_CASE`: content = findings, metadata = resolução |

O `PATCH` atualiza `extracted_documents`, guarda o payload anterior e o novo no audit
(`EXTRACTION_CORRECTED`) e exige revalidação do documento.

## Leitura de memória

### Extração (`app/extract.py`)

1. `extract_offline(text)` primeiro, para pegar o CNPJ (custo zero).
2. `retrieve("EXTRACTION_EXAMPLE", text, filters={"cnpj": cnpj} se houver, k=3)`.
3. Exemplos (texto truncado + campos corretos) entram no prompt como few-shot.
4. `extract_with_llm` passa a retornar também os ids de memória usados → `audit(..., memory_ids=[...])`.

Sem `ANTHROPIC_API_KEY`: comportamento atual (só regex, sem memória).

### Matching (`app/matching.py`)

- Uma `retrieve("MATCH_DECISION", tx.description, filters={"bank_account_id": ...})` por transação,
  usando o embedding já salvo na transação.
- Hit aprovado cuja entidade é o candidato → `+WEIGHTS["history"]` (padrão 10), reason `history`.
  Hit rejeitado → `-WEIGHTS["history"]`, reason `history_rejected`.
- **Teto:** a condição de `AUTO_MATCHED` usa o **score sem history**. History só ordena,
  desempata e pode promover `UNMATCHED → REVIEW_REQUIRED`.

### Divergências (`app/reconcile.py`)

Cada finding de `investigate()` ganha `similar_cases`: até 3 `DIVERGENCE_CASE` parecidos com
sua resolução. É só evidência e não altera `reconciled`.

### Import de extrato (`app/bank_import.py`)

Embeddings das descrições gerados em lote e gravados em `bank_transactions.embedding`.
Falha de embedding não bloqueia o import (coluna fica null).

## Configuração (env)

`VOYAGE_API_KEY`, `EMBEDDING_MODEL` (padrão `voyage-3.5`), `EMBEDDING_DIM` (1024),
`RETRIEVE_CANDIDATES` (20), `MATCH_WEIGHTS` ganha `"history":10`.

## Infra

- `docker-compose.yml` com `pgvector/pgvector:pg17`.
- `DATABASE_URL` padrão → Postgres local. SQLite deixa de ser suportado.
- `db.py` executa `CREATE EXTENSION IF NOT EXISTS vector` antes de `create_all`.
- Sem Alembic: não há banco de produção. Comentário `ponytail:` marca a entrada no primeiro deploy real.
- Dependências novas: `pgvector`, `langchain-core`, `langchain-voyageai`.

## Testes

`conftest.py` usa `TEST_DATABASE_URL` (Postgres) com rollback por teste; sem Postgres
disponível, os testes são pulados com mensagem clara. Embedding fake mantém tudo offline.

Casos novos:

1. `PATCH` de correção → `EXTRACTION_EXAMPLE` recuperado na extração seguinte do mesmo CNPJ.
2. Busca híbrida acha BL/container via full-text mesmo quando o vetor não acha.
3. History sozinho **não** gera `AUTO_MATCHED`.
4. Rejeição passada rebaixa o candidato.
5. Finding de conciliação traz `similar_cases`.
6. Voyage indisponível → `retrieve()` degrada, sem exceção, e registra `MEMORY_DEGRADED`.

Os 20 testes existentes continuam passando sobre Postgres.

## Fora do escopo

Ver [backlog-fora-de-escopo.md](../../backlog-fora-de-escopo.md).
