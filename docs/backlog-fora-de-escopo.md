# Backlog — fora do escopo da Spec 1 (memória e RAG híbrido)

Itens adiados de propósito. Cada um diz **quando** passa a valer a pena.

| item | por que ficou de fora | entra quando |
|---|---|---|
| **Agente LangGraph** (Spec 2) | precisava da memória e das ferramentas prontas primeiro | spec pronta em `docs/superpowers/specs/2026-10-18-agente-producao-evals-rag-observabilidade-design.md`; implementar pelo plano correspondente |
| **Reranker** (Voyage/Cohere rerank) | RRF basta para começar | métricas mostrarem recuperação ruim no top-k |
| **Esquecimento / expiração de memória** | volume inicial pequeno | memórias velhas começarem a atrapalhar (ex.: fornecedor mudou layout ou conta) |
| **UI de curadoria de memória** | sem usuário pedindo | analistas precisarem apagar ou editar memórias erradas |
| **Chunking de documentos longos** | entrada hoje é nota/boleto, curta | aparecerem contratos ou documentos de várias páginas |
| **Memória multi-tenant** | uma empresa só | mais de uma empresa no mesmo banco |
| **Alembic (migrações)** | não há banco de produção | primeiro deploy real com dados |
| **OCR** (já marcado em `app/extract.py`) | PDFs atuais têm texto | primeiro PDF escaneado real |

## Ledger técnico consolidado após a Spec 1

Os itens abaixo passaram pela revisão final da branch. Nenhum bloqueia o merge da Spec 1;
`P1` deve entrar no próximo ciclo, `P2` quando o fluxo afetado ganhar uso real e `P3` é
principalmente cobertura, manutenção ou performance preventiva.

| # | prioridade | item adiado | gatilho para executar |
|---:|:---:|---|---|
| 1 | P1 | Criar backfill para `BankTransaction.embedding` nulo após queda da Voyage; reimportar hoje só conta a linha como duplicada. | antes de usar similaridade das transações como sinal de produção |
| 2 | P1 | Trocar o `plainto_tsquery` estrito do `history`: descrições com NF/NSU/data variável quase nunca têm todos os termos em comum. | medir baixa taxa de aplicação de `history`; avaliar `websearch_to_tsquery` e limiar cosine |
| 3 | P1 | Definir limiar de distância do canal semântico; hoje ele sempre devolve vizinhos, mesmo irrelevantes. | antes de permitir que hit exclusivamente semântico influencie matching |
| 4 | P2 | Dedupe/supersede de `EXTRACTION_EXAMPLE` por documento corrigido repetidamente. | primeira ocorrência de correções sucessivas do mesmo documento |
| 5 | P2 | Permitir limpar explicitamente um campo extraído; `exclude_none=True` hoje trata `null` como ausência. | analista precisar remover dado incorreto, não apenas substituí-lo |
| 6 | P2 | Rejeitar uma segunda rejeição do mesmo grupo para não criar memória negativa duplicada. | endurecimento dos endpoints de decisão humana |
| 7 | P2 | Validar a existência de `entity_id` de PAYABLE/RECEIVABLE em `add_group_item`. | antes de expor edição de grupos a uma UI |
| 8 | P2 | Separar erro de configuração/import do cliente Voyage de indisponibilidade do provedor; hoje ambos degradam silenciosamente. | observabilidade de deploy/produção |
| 9 | P2 | Marcar de forma durável `MemoryItem.embedding IS NULL` para backfill; `remember()` aceita a degradação sem fila de reparo. | junto do job de backfill |
| 10 | P2 | Tornar a resolução de `DIVERGENCE_CASE` específica por finding; aprovação hoje grava uma mensagem genérica. | quando o analista puder informar a ação tomada |
| 11 | P2 | Batelar a busca de `similar_cases`; a implementação faz uma consulta por finding. | findings numerosos ou latência perceptível na conciliação diária |
| 12 | P2 | Isolar findings globais por conta: `GROUP_DIFFERENCE` e títulos parciais ainda vêm de consultas sem escopo de `bank_account_id`. | operação com mais de uma conta bancária ativa |
| 13 | P2 | Tratar payload legado cujo `structured_payload.bank_account` seja string; o merge atual espera dict. | somente se houver dados criados pelo commit intermediário nunca liberado |
| 14 | P2 | Evitar que CNPJ inválido corrigido vire string vazia na coluna; normalizar para `NULL` ou rejeitar. | validação mais estrita do PATCH de extração |
| 15 | P2 | Definir política de cópia/serialização do `RecebaRetriever`, que carrega uma `Session` viva e não suporta deepcopy. | integração efetiva com LangGraph na Spec 2 |
| 16 | P3 | Substituir o atributo ad hoc `test_client.bank_account_id` por fixture explícita. | próxima refatoração das fixtures da API |
| 17 | P3 | Provar em teste que `content_tsv` usa configuração `portuguese`, não apenas que foi preenchido. | ao mexer no schema/full-text |
| 18 | P3 | Avaliar índice em `MemoryItem.amount`; hoje a coluna pode ser filtro estruturado sem índice. | quando buscas por valor aparecerem em traces lentos |
| 19 | P3 | Avaliar índice vetorial em `BankTransaction.embedding`. | quando houver consulta cosine sobre transações |
| 20 | P3 | Adicionar teste do caminho de sucesso real do cliente Voyage. | CI com credencial/fixture do SDK estável |
| 21 | P3 | Cobrir contratos de borda de `retrieve`/`remember`: filtro inválido, ausência de commit implícito e ordenação degradada. | próxima rodada de testes do módulo de memória |
| 22 | P3 | Reduzir warning duplicado na queda da Voyage (`embed_query` e `retrieve`). | ajuste de observabilidade |
| 23 | P3 | Ampliar testes do retriever para colisão de metadata, `item_metadata=None` e ordenação de `via`. | antes da Spec 2 consumir metadata como protocolo |
| 24 | P3 | Cobrir falha não SQL na recuperação few-shot e os ramos de memória de PARTIAL/ADJUSTMENT/título ausente. | próxima rodada de cobertura de regressão |
| 25 | P3 | Endurecer e limitar o prompt completo: variantes de caixa dos marcadores, teste da neutralização de metadata e teto total incluindo exemplos. | ao aumentar número/tamanho dos few-shots |
