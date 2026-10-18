# Plano de implementação — Spec 2

**Spec:** `docs/superpowers/specs/2026-10-18-agente-producao-evals-rag-observabilidade-design.md`

## Ordem de execução

Cada task termina com testes focais e suíte completa. Mudança de prompt/modelo não pode ser
aceita antes de existir o runner de eval da Task 3.

### Task 1 — contratos, fixtures e ferramentas read-only

**Arquivos:** `app/agent_tools.py`, `tests/test_agent_tools.py`

- Criar modelos Pydantic para entradas, evidências e `InvestigationRecommendation`.
- Implementar as seis ferramentas com escopo obrigatório por conta e limites de resultado.
- Reusar `memory.retrieve`; não duplicar SQL de RAG.
- Testar happy path, 404, tentativa cross-account, resultado vazio e serialização.
- Aceite: nenhuma função importa `_settle` nem chama `session.commit()`.

### Task 2 — persistência e observabilidade de domínio

**Arquivos:** `app/db.py`, `app/agent_observability.py`, `tests/test_agent_observability.py`

- Adicionar `AgentRun`, `AgentStep`, `EvalRun` e `EvalCaseResult`.
- Criar context manager/decorator de step que mede monotonic time e sempre fecha status.
- Registrar usage e custo por tabela configurável com `pricing_version`.
- Redigir payloads; armazenar hash por padrão.
- Testar erro, custo desconhecido (`null`), ordenação dos steps e ausência de texto sensível.

### Task 3 — dataset v1, scorers e runner offline

**Arquivos:** `evals/datasets/investigation-v1.jsonl`, `app/evals.py`, `tests/test_evals.py`

- Escrever os 20 casos antes do grafo real.
- Implementar `AgentAdapter` para rodar fake determinístico e, depois, o grafo real.
- Implementar todos os scorers e comparação contra baseline JSON.
- CLI: `.venv/bin/python -m app.evals run --dataset investigation-v1`.
- Exit code não zero quando o gate falhar.
- Testar cada scorer com mutação que prove que ele falha.

### Task 4 — grafo LangGraph com orçamento e allowlist

**Arquivos:** `app/agent.py`, `tests/test_agent.py`, `requirements.txt`, `.env.example`

- Fixar versões resolvidas de LangGraph e checkpointer Postgres.
- Implementar `InvestigationState` e nós `load_case`, `plan`, `tools`, `grade_evidence`,
  `compose`, `human_review`, `record_feedback`.
- Injetar `ModelGateway`; testes usam fake roteirizável, sem rede.
- Aplicar limites de 3 ciclos, 8 tools, tokens e custo.
- Rejeitar ferramenta fora da allowlist antes de executar.
- Rodar os 20 casos; criar primeiro baseline aceito.

### Task 5 — human-in-the-loop persistente

**Arquivos:** `app/agent.py`, `app/agent_runtime.py`, `tests/test_agent_runtime.py`

- Checkpointer em memória só em unit tests; Postgres no runtime.
- Interrupt tipado com recomendação, evidências e campos editáveis.
- Resume registra `accepted`, `edited` ou `rejected`, nota e ator.
- Teste reinicia o runtime entre pausa e retomada.
- Provar idempotência: repetir resume não duplica feedback ou efeito.

### Task 6 — API do agente

**Arquivos:** `app/main.py`, `tests/test_agent_api.py`

- Implementar start/get/resume/trace com RBAC.
- `ANALYST` inicia e revisa; trace completo somente `ADMIN`.
- Resposta nunca inclui prompt cru, chave ou payload bancário integral.
- Testar concorrência otimista: dois resumes, um vence e outro recebe 409.

### Task 7 — reranker e eval de retrieval

**Arquivos:** `app/rerank.py`, `app/memory.py`, `app/config.py`, `tests/test_rerank.py`,
`evals/datasets/retrieval-v1.jsonl`

- Interface `Reranker`, `VoyageReranker` e `FakeReranker`.
- Recuperar 20 por RRF, reranquear e devolver 5.
- Preservar distância cosine separada do score RRF/relevance.
- Em falha: ordem RRF, audit/step degradado, zero bloqueio.
- Calcular Recall@5, MRR@5 e nDCG@5 com e sem reranker.
- Feature flag só liga se o ganho da spec for comprovado.

### Task 8 — OpenTelemetry, métricas e orçamento

**Arquivos:** `app/telemetry.py`, `app/agent_observability.py`, `docker-compose.yml`,
`tests/test_telemetry.py`

- Adicionar API/SDK OTel e exportador OTLP configurável; no-op sem configuração.
- Spans hierárquicos para run/node/tool/LLM/retrieval/rerank.
- Métricas de latência, erro, retry, tokens, custo e feedback.
- Testar com exporter em memória; atributos proibidos não podem aparecer.
- Criar painel/consulta SQL mínima para volume, p95, custo e aceitação humana.

### Task 9 — servidor MCP read-only

**Arquivos:** `app/mcp_server.py`, `tests/test_mcp_server.py`, `requirements.txt`

- Fixar SDK MCP v2 resolvido.
- Registrar tools/resources da spec chamando `app.agent_tools`, sem duplicar regra.
- Testar listagem, schemas, chamada, erro cross-account e conteúdo estruturado via cliente
  em memória.
- Documentar stdio local e Streamable HTTP autenticado; não usar SSE novo.

### Task 10 — reliability, segurança e carga

**Arquivos:** `tests/test_agent_security.py`, `tests/test_agent_recovery.py`,
`scripts/load_agent.py`, `docs/runbooks/agent-investigation.md`

- Prompt injection: documento/memória tenta ordenar approve/settle e falha.
- Tool loop, payload enorme, timeout, provider 429/500, DB rollback e restart.
- Carga reproduzível com 100 runs sintéticos; relatório contém throughput, p50/p95 e erros.
- Runbook: degradar provider, desligar feature flag, localizar run/trace, retomar checkpoint,
  reprocessar e abrir incidente.

### Task 11 — prova operacional e material de entrevista

**Arquivos:** `README.md`, `docs/case-study-agente-investigacao.md`

- Roteiro de demo de 5 minutos com um caso bem-sucedido e um caso degradado.
- Tabela automática: runs, casos, taxa de sucesso, aceitação humana, p95 e custo médio.
- Checklist de piloto: usuário, período, volume, incidentes e feedback.
- Separar claramente números sintéticos, piloto e produção.
- Só promover o rótulo para “produção” quando os critérios da spec forem atendidos.

## Gates por gap

| depois da task | gap demonstrável |
|---|---|
| 3 | evaluation formal |
| 4–6 | agente com loop e human-in-the-loop |
| 7 | RAG com reranking medido |
| 8 | observabilidade, custo e SLO |
| 9 | MCP |
| 10–11 | operação production-shaped e caminho honesto para produção real |

## Comandos finais esperados

```bash
docker compose up -d db
.venv/bin/python -m pytest -q
.venv/bin/python -m app.evals run --dataset investigation-v1 --compare baseline
.venv/bin/python scripts/load_agent.py --runs 100 --offline
.venv/bin/python -m app.mcp_server --transport stdio
```

## Referências técnicas verificadas em 2026-10-18

- LangGraph persistence e checkpointers: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph interrupts/HITL: https://docs.langchain.com/oss/python/langgraph/interrupts
- Voyage rerankers: https://docs.voyageai.com/docs/reranker
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/instrumentation/
- MCP Python SDK v2: https://py.sdk.modelcontextprotocol.io/

