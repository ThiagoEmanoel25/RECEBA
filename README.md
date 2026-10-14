# RECEBA — validação e conciliação financeira

Espinha vertical do MVP descrito em `IMPLEMENTACAO_SISTEMA_CONCILIACAO_FINANCEIRA_SEM_N8N.md`:
documento entra, vira dado estruturado, passa por regras determinísticas, vira título,
cruza com o extrato bancário e termina em baixa automática ou revisão humana — tudo com
audit trail.

**IA interpreta. Código valida. Regras decidem. Humano resolve exceções.**
O único ponto probabilístico é a extração (`app/extract.py`). Daí para frente nada é LLM.

## Rodar

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # ajuste API_KEYS e DATABASE_URL
.venv/bin/python -m pytest -q   # 20 testes, inclui os 10 casos obrigatórios do MVP
.venv/bin/uvicorn app.main:app --reload
```

Sem `ANTHROPIC_API_KEY` a extração cai no parser determinístico (regex + observation parser),
então o pipeline inteiro roda offline — inclusive os testes.

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

## Módulos

| arquivo | papel |
|---|---|
| `app/parser.py` | normalização e Observation Parser (regex, sem IA) |
| `app/extract.py` | PDF/texto → `FinancialDocument`; LLM com structured output, fallback regex |
| `app/validation.py` | 9 regras determinísticas; conta divergente e duplicidade **bloqueiam** |
| `app/matching.py` | score configurável, exact match, parcial, tarifa/juros/multa/desconto, lote |
| `app/reconcile.py` | saldo esperado × saldo real, findings para investigação |
| `app/bank_import.py` | CSV com idempotência (`external_id` ou `data+valor+descrição`) |
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
