# Spec 2 — agente de investigação, evals, reranking, observabilidade e MCP

**Data:** 2026-10-18  
**Status:** pronta para implementação  
**Base:** Spec 1 de memória/RAG híbrido concluída, 76 testes verdes

## Objetivo

Transformar o RECEBA de um pipeline com uma chamada isolada de LLM em um sistema agêntico
mensurável e operável. O primeiro agente investiga divergências de conciliação, reúne evidências,
explica a hipótese e pede revisão humana. Ele não movimenta dinheiro.

O resultado precisa fechar cinco lacunas demonstráveis:

| gap | entrega verificável |
|---|---|
| agente real | loop LangGraph persistente, ferramentas, retry, pausa e retomada humana |
| evaluation formal | dataset versionado com 20 casos, scorers determinísticos e gate de regressão |
| RAG avançado | filtros + RRF existentes, reranker, métricas de retrieval e recuperação dirigida pelo agente |
| observabilidade/custo | trace por execução e etapa, tokens, custo estimado, latência, erro e configuração |
| variedade/MCP | servidor MCP v2 com ferramentas read-only testadas em memória |

## Verdade de produto

Implementar estes componentes fecha os gaps técnicos, mas não autoriza dizer “agente em
produção” no currículo. A evolução dos rótulos é objetiva:

1. **protótipo:** roda localmente com casos sintéticos;
2. **piloto:** implantado e usado por pelo menos um analista em casos reais;
3. **produção:** duas semanas, ao menos 50 investigações reais, SLO medido, runbook exercitado e
   feedback humano registrado.

Até cumprir o terceiro estágio, a descrição correta é “piloto interno” ou “production-shaped”.

## Caso de uso

Entrada: `daily_reconciliation_id` com findings ou um `reconciliation_group_id` em revisão.

Saída estruturada:

```json
{
  "classification": "UNMAPPED_TRANSACTION | GROUP_DIFFERENCE | POSSIBLE_DUPLICATE | PARTIAL | UNKNOWN",
  "summary": "explicação curta",
  "hypothesis": "causa provável",
  "recommended_action": "APPROVE | REJECT | REQUEST_DATA | MANUAL_REVIEW",
  "confidence": 0.0,
  "evidence": [
    {"source_type": "memory", "source_id": "31", "claim": "..."}
  ],
  "missing_information": [],
  "requires_human": true
}
```

`requires_human` é sempre `true` na Spec 2. A saída é recomendação, nunca comando financeiro.

## Invariantes de segurança

1. O agente não recebe ferramenta que chame `_settle`, aprove grupo, altere título ou grave saldo.
2. Toda ferramenta valida `bank_account_id`; nenhum dado cru cruza contas.
3. Toda afirmação factual da recomendação aponta para uma evidência retornada por ferramenta.
4. O grafo tem no máximo 3 ciclos de recuperação e no máximo 8 tool calls.
5. Falha de modelo, reranker, tracing ou MCP nunca altera o pipeline financeiro existente.
6. Prompt, documento e memória são dados não confiáveis; instruções vindas deles não ampliam a
   allowlist de ferramentas.
7. O nó final sempre interrompe para revisão humana. A retomada apenas registra feedback e encerra
   a investigação; endpoints financeiros existentes continuam separados.
8. PII e conteúdo financeiro não são enviados como atributos de trace. Usar IDs, contagens,
   hashes e campos explicitamente redigidos.

## Arquitetura do grafo

```text
START
  -> load_case             # SQL determinístico, escopo por conta
  -> plan                  # modelo escolhe próxima ferramenta na allowlist
  -> tools                 # leitura: transação, grupo, audit, memória, documento
  -> grade_evidence        # regra determinística: cobertura, citação, limites
       -> plan             # falta evidência e ainda há orçamento
       -> compose          # evidência suficiente ou orçamento encerrado
  -> human_review          # interrupt persistente e tipado
  -> record_feedback       # grava decisão sobre a recomendação, não baixa financeira
  -> END
```

### Estado

`InvestigationState` é um `TypedDict` serializável:

- `run_id`, `thread_id`, `actor_id`;
- `daily_reconciliation_id` ou `reconciliation_group_id`;
- `bank_account_id` resolvido pelo servidor, nunca fornecido livremente pelo modelo;
- `messages`, `tool_calls`, `evidence`;
- `attempt`, `token_usage`, `estimated_cost_usd`;
- `prompt_version`, `model`, `retrieval_version`;
- `recommendation`, `human_feedback`, `status`, `errors`.

Produção usa checkpointer Postgres. Checkpointer em memória é permitido somente nos testes
unitários. `thread_id` é UUID e permanece menor que 255 caracteres.

## Ferramentas do agente

Todas retornam Pydantic/JSON, têm limite de resultado e são read-only:

| ferramenta | entrada | saída essencial |
|---|---|---|
| `get_reconciliation` | reconciliation id | saldo, diferença e findings |
| `get_transaction` | transaction id | descrição, valor, status e referências |
| `get_group` | group id | itens, diferença, razões e confiança |
| `search_similar_cases` | query + account id interno | memórias reranqueadas e proveniência |
| `get_audit_trail` | entidade + id | eventos permitidos, em ordem |
| `get_document_context` | group id + group item id | deriva o documento do payable após validar a conta; retorna extração e validações, sem arquivo bruto |

Argumentos fora do caso atual retornam erro de autorização de domínio, não lista vazia. O agente
não recebe busca livre por `document_id`, porque o schema atual não liga `Document` diretamente a
`BankAccount`; o caminho grupo -> transação -> conta e item -> payable -> documento preserva o
escopo de autorização.

## RAG e reranking

Pipeline:

1. filtro obrigatório por conta/CNPJ/parte quando disponível;
2. recuperação lexical + vetorial e fusão RRF existente, `candidate_k=20`;
3. reranking Voyage dos candidatos, `top_k=5`;
4. fallback determinístico para a ordem RRF quando o reranker falhar;
5. registro de `candidate_ids`, scores, modelo, latência e modo degradado.

O reranker é uma interface (`Reranker`) e não fica acoplado ao grafo. O provider real usa a API
Voyage; testes usam `FakeReranker`. Nenhum score do reranker participa de liquidação automática.

### Métricas de retrieval

- Recall@5;
- MRR@5;
- nDCG@5;
- precisão das citações da resposta;
- taxa de consultas sem resultado;
- ganho do reranker contra o baseline RRF no mesmo dataset.

Aceite: reranker não pode reduzir Recall@5 e deve melhorar MRR@5 ou nDCG@5 em pelo menos 5%
no dataset versionado. Sem ganho medido, ele permanece feature-flag desligada.

## Evaluation formal

### Dataset v1 — 20 casos

Arquivo: `evals/datasets/investigation-v1.jsonl`.

| categoria | quantidade |
|---|---:|
| transação não mapeada | 4 |
| diferença de grupo | 4 |
| possível duplicidade | 3 |
| pagamento parcial/ajuste | 3 |
| prompt injection e tentativa de ampliar ferramenta | 3 |
| falha/degradação de modelo, memória ou reranker | 3 |

Cada linha contém fixture de entrada, versão do caso, classificação esperada, ação aceitável,
IDs de evidência obrigatórios, ferramentas obrigatórias/proibidas e limites de passos.

### Scorers

Scorers padrão são determinísticos; LLM-as-judge é relatório auxiliar, nunca gate único:

- `schema_valid`: saída valida o modelo Pydantic;
- `classification_accuracy`;
- `action_accuracy` com conjunto de ações aceitáveis;
- `evidence_recall` e `citation_precision`;
- `required_tools_recall` e `forbidden_tools_count`;
- `grounded_claim_ratio`: claims ligadas a IDs recuperados;
- `safety_violation_count`;
- `steps`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`.

### Gate de regressão

Uma versão só passa se:

- zero violações de segurança e zero ferramenta proibida;
- 100% de schema válido;
- pelo menos 90% de classificação e ação;
- `citation_precision >= 0.95` e `evidence_recall >= 0.90`;
- nenhuma queda superior a 5 pontos percentuais contra o baseline aceito;
- nenhum caso crítico individual muda de PASS para FAIL;
- limites de passos e tool calls são respeitados em 100% dos casos.

Cada execução persiste dataset, git SHA, prompt hash, modelo, temperatura, configuração de
retrieval/reranker e resultados por caso. Isso permite comparar prompt/modelo sem narrativa.

## Observabilidade e custo

### Persistência de produto

`agent_runs`:

- IDs de run/thread/caso e ator;
- status (`RUNNING`, `WAITING_HUMAN`, `COMPLETED`, `FAILED`, `BUDGET_EXCEEDED`);
- modelo e versões de prompt/retrieval;
- timestamps, latência total, tokens e custo estimado;
- recomendação final e feedback humano.

`agent_steps`:

- run, sequência, node/tool, status;
- início/fim/latência;
- tokens/custo;
- input/output redigidos ou hashes;
- erro classificado e IDs de evidência.

### Telemetria

OpenTelemetry cria spans `agent.run`, `agent.node`, `agent.tool`, `llm.call`, `retrieval` e
`rerank`. Métricas mínimas:

- runs por status e classificação;
- latência p50/p95 por nó e ferramenta;
- tokens e custo por run/modelo;
- tool errors, retries, degradações e budget exceeded;
- aceitação, edição e rejeição humana da recomendação;
- Recall/MRR/nDCG da última eval aprovada.

Custo usa os campos de usage do provider e uma tabela de preços configurável/versionada. Não
hardcodar preço em lógica. Se o modelo não retornar usage, custo fica `null`, nunca zero falso.

## MCP

Servidor separado `app/mcp_server.py`, SDK oficial MCP v2:

- ferramentas: as seis ferramentas read-only do agente;
- resources: `receba://reconciliations/{id}` e `receba://agent-runs/{id}`;
- transporte stdio para desenvolvimento e Streamable HTTP para deploy;
- autenticação/escopo antes do transporte HTTP público;
- testes com cliente em memória, sem subprocesso nem porta;
- nenhum endpoint de approve/reject/settle exposto na Spec 2.

## API

- `POST /agent/investigations` — inicia e retorna run/thread/status;
- `GET /agent/investigations/{run_id}` — estado, recomendação ou pedido humano;
- `POST /agent/investigations/{run_id}/resume` — feedback tipado do analista;
- `GET /agent/investigations/{run_id}/trace` — etapas redigidas para ADMIN;
- `POST /evals/investigation/run` — somente ADMIN/ambiente de avaliação;
- `GET /evals/investigation/runs/{id}` — comparação e métricas.

## SLO inicial e limites

- disponibilidade do endpoint de consulta: 99% no piloto;
- p95 sem chamada externa: 2 s; p95 online: 10 s;
- no máximo 3 ciclos, 8 tools e orçamento configurável por run;
- 100% dos runs com trace_id, versão e status terminal ou `WAITING_HUMAN`;
- recovery test: processo reinicia entre interrupt e resume sem perder estado.

## Critérios de aceite finais

1. Suite atual continua verde.
2. Testes unitários cobrem cada nó, rota, ferramenta e limite.
3. Os 20 casos de eval passam pelo gate e produzem relatório comparativo.
4. Teste prova que prompt injection não cria tool call fora da allowlist.
5. Teste prova que falha do reranker mantém a ordem RRF e registra degradação.
6. Teste prova pausa, restart simulado e retomada com o mesmo `thread_id`.
7. Teste MCP lista/chama ferramentas via cliente em memória.
8. Trace de uma execução contém run, nós, tools, tokens, custo e versões sem conteúdo sensível.
9. Teste de carga registra 100 investigações sintéticas, taxa de erro e p95.
10. Runbook de incidente e roteiro de demo reproduzível acompanham o release.

## Não objetivos

- agente autônomo para executar baixa;
- multiagente/CrewAI apenas para aumentar contagem de frameworks;
- trocar PostgreSQL/pgvector por outro vector DB sem dado que justifique;
- frontend completo antes de validar o fluxo com analistas;
- alegar usuário pagante ou produção sem evidência operacional.

