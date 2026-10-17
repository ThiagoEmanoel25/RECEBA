# RECEBA — validação e conciliação financeira

Espinha vertical do MVP descrito em `IMPLEMENTACAO_SISTEMA_CONCILIACAO_FINANCEIRA_SEM_N8N.md`:
documento entra, vira dado estruturado, passa por regras determinísticas, vira título,
cruza com o extrato bancário e termina em baixa automática ou revisão humana — tudo com
audit trail.

**IA interpreta. Código valida. Regras decidem. Humano resolve exceções.**
O único ponto probabilístico é a extração (`app/extract.py`). Daí para frente nada é LLM.

## Rodar

```bash
docker compose up -d db                       # Postgres 17 + pgvector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # ajuste API_KEYS, DATABASE_URL e as chaves de IA
.venv/bin/python -m pytest -q   # inclui os 10 casos obrigatórios do MVP
.venv/bin/uvicorn app.main:app --reload
```

Sem `ANTHROPIC_API_KEY` a extração cai no parser determinístico (regex + observation parser)
e sem `VOYAGE_API_KEY` os embeddings viram vetores fake determinísticos, então o pipeline
inteiro roda offline — inclusive os testes. O Postgres com pgvector, porém, é obrigatório:
sem ele a suíte é pulada com uma mensagem explicando como subir o container.

Toda chamada exige `X-API-Key`. Perfis: `ADMIN`, `ANALYST` (escrita), `FINANCIAL_APPROVER`
(aprovação), `VIEWER` (leitura).

## Fluxo

```
POST /documents                      upload + dedupe por content_hash
POST /documents/{id}/process         texto -> Structured Output (Pydantic)
POST /documents/{id}/validate        Validation Engine -> APPROVED | REVIEW_REQUIRED | BLOCKED
                                     APPROVED cria o título em payables
POST /bank-transactions/import       CSV idempotente + matching automático
POST /matching/{tx}/run              score -> SINGLE | BATCH | PARTIAL | ADJUSTMENT | REVIEW
POST /reconciliation-groups/{id}/... human in the loop: add/remove item, approve
POST /reconciliations/run            conciliação diária + investigação de divergências
GET  /metrics                        auto_match_rate, batch, parcial, revisão, não conciliado
GET  /audit/{entity_type}/{id}       trilha completa, append-only
```

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
O `history` só ordena candidatos para revisão e pode promover `UNMATCHED` para
`REVIEW_REQUIRED`. As seleções que geram baixa automática, inclusive lote, usam uma ordem
total independente da memória.

`app/memory.py` expõe também `RecebaRetriever` e `retriever_tool()`, os adaptadores LangChain
que o agente da Spec 2 (orquestração com LangGraph) poderá consumir sem reescrever a busca.

## Módulos

| arquivo | papel |
|---|---|
| `app/parser.py` | normalização e Observation Parser (regex, sem IA) |
| `app/extract.py` | PDF/texto → `FinancialDocument`; LLM com structured output e few-shot da memória, fallback regex |
| `app/validation.py` | 9 regras determinísticas; conta divergente e duplicidade **bloqueiam** |
| `app/matching.py` | score configurável, exact match, parcial, tarifa/juros/multa/desconto, lote |
| `app/reconcile.py` | saldo esperado × saldo real, findings para investigação |
| `app/bank_import.py` | CSV com idempotência (`external_id` ou `data+valor+descrição`) |
| `app/memory.py` | memória + RAG híbrido (RRF); adaptadores LangChain para a Spec 2 |
| `app/db.py` | schema da seção 32 + `audit()` |
| `app/config.py` | pesos, faixas, tolerâncias e limites — tudo por env |

## Decisões de ajuste automático

| situação | resultado |
|---|---|
| referência forte única + valor exato | `AUTO_MATCHED`, baixa total |
| referência forte + sobra ≤ `FEE_MAX` | `MATCH_WITH_ADJUSTMENT` (FEE/INTEREST/PENALTY) |
| referência forte + valor menor | `PARTIALLY_PAID`, nunca baixa total |
| uma combinação de títulos fecha o valor | `MATCHED_BATCH` |
| duas ou mais combinações fecham | `REVIEW_REQUIRED` — o sistema não escolhe |
| nenhum candidato ≥ `REVIEW_MIN` | `UNMATCHED` |

## O que ficou de fora (de propósito)

- **Celery/Redis**: o pipeline roda síncrono. Entram quando um documento começar a estourar
  timeout de request ou quando a conciliação diária precisar de agendamento (Celery Beat).
- **Alembic**: `create_all` no startup. Trocar por migrations antes do primeiro deploy.
- **S3/R2**: arquivo vai para `STORAGE_DIR` local. Trocar o `open()` em `app/main.py` por
  um client quando houver mais de uma instância.
- **OCR**: PDF escaneado sai sem texto e cai no fallback. Plugar tesseract no primeiro caso real.
- **Conectores Gmail/Drive/ERP, OFX, API bancária, frontend Next.js, OpenTelemetry**:
  sprint 10 do spec. A API já tem os pontos de entrada (`source` em `/documents`,
  `/bank-transactions/import`).
