# Backlog — fora do escopo da Spec 1 (memória e RAG híbrido)

Itens adiados de propósito. Cada um diz **quando** passa a valer a pena.

| item | por que ficou de fora | entra quando |
|---|---|---|
| **Agente LangGraph** (Spec 2) | precisa da memória e das ferramentas prontas primeiro | Spec 1 entregue; candidatos: investigação de divergências e re-extração com autocorreção |
| **Reranker** (Voyage/Cohere rerank) | RRF basta para começar | métricas mostrarem recuperação ruim no top-k |
| **Esquecimento / expiração de memória** | volume inicial pequeno | memórias velhas começarem a atrapalhar (ex.: fornecedor mudou layout ou conta) |
| **UI de curadoria de memória** | sem usuário pedindo | analistas precisarem apagar ou editar memórias erradas |
| **Chunking de documentos longos** | entrada hoje é nota/boleto, curta | aparecerem contratos ou documentos de várias páginas |
| **Memória multi-tenant** | uma empresa só | mais de uma empresa no mesmo banco |
| **Alembic (migrações)** | não há banco de produção | primeiro deploy real com dados |
| **OCR** (já marcado em `app/extract.py`) | PDFs atuais têm texto | primeiro PDF escaneado real |
