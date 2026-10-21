"""Contratos de seguranca das ferramentas read-only do agente."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from app import agent_tools, db, memory
from tests.conftest import TODAY


@pytest.fixture()
def investigation_case(session, company, make_transaction):
    transaction = make_transaction("100.00", "PAGAMENTO NF 123")
    document = db.Document(
        source="UPLOAD",
        document_type="INVOICE",
        filename="nf-123.pdf",
        storage_key="private/raw-file.pdf",
        status="VALIDATED",
        content_hash="a" * 64,
        raw_metadata={"secret": "nao expor"},
    )
    session.add(document)
    session.flush()
    payable = db.Payable(
        supplier_id=company["supplier"].id,
        document_id=document.id,
        invoice_number="123",
        amount_original=Decimal("100.00"),
        amount_open=Decimal("100.00"),
    )
    group = db.ReconciliationGroup(
        bank_transaction_id=transaction.id,
        group_type="SINGLE",
        total_transaction_amount=Decimal("100.00"),
        total_allocated_amount=Decimal("100.00"),
        difference=Decimal("0.00"),
        status="REVIEW_REQUIRED",
        confidence_score=85,
        match_reasons={"reasons": ["invoice_exact"]},
    )
    reconciliation = db.DailyReconciliation(
        bank_account_id=company["account"].id,
        reconciliation_date=TODAY,
        opening_balance=Decimal("1000.00"),
        total_credits=Decimal("0.00"),
        total_debits=Decimal("100.00"),
        expected_balance=Decimal("900.00"),
        real_balance=Decimal("899.00"),
        difference=Decimal("1.00"),
        status="INVESTIGATION_REQUIRED",
        findings={"items": [{"type": "GROUP_DIFFERENCE", "group_id": None}]},
    )
    session.add_all([payable, group, reconciliation])
    session.flush()
    reconciliation.findings = {
        "items": [{"type": "GROUP_DIFFERENCE", "group_id": group.id}]
    }
    item = db.ReconciliationGroupItem(
        reconciliation_group_id=group.id,
        entity_type="PAYABLE",
        entity_id=payable.id,
        amount_allocated=Decimal("100.00"),
    )
    extracted = db.ExtractedDocument(
        document_id=document.id,
        supplier_name="ACME LTDA",
        supplier_cnpj="11222333000181",
        invoice_number="123",
        amount=Decimal("100.00"),
        currency="BRL",
        structured_payload={"invoice_number": "123"},
        model_name="fixture",
    )
    validation = db.ValidationResult(
        document_id=document.id,
        rule_code="AMOUNT_POSITIVE",
        status="PASS",
        actual_value="100.00",
    )
    event = db.AuditEvent(
        entity_type="reconciliation_group",
        entity_id=str(group.id),
        event_type="CREATED",
        event_metadata={"safe": True},
    )
    session.add_all([item, extracted, validation, event])
    session.commit()
    context = agent_tools.resolve_investigation_scope(
        session, daily_reconciliation_id=reconciliation.id
    )
    return {
        "context": context,
        "transaction": transaction,
        "document": document,
        "payable": payable,
        "group": group,
        "item": item,
        "reconciliation": reconciliation,
    }


def test_ferramentas_leem_caso_e_serializam(session, investigation_case):
    case = investigation_case
    context = case["context"]

    reconciliation = agent_tools.get_reconciliation(
        session, case["reconciliation"].id, context
    )
    transaction = agent_tools.get_transaction(session, case["transaction"].id, context)
    group = agent_tools.get_group(session, case["group"].id, context)
    audit = agent_tools.get_audit_trail(
        session, "reconciliation_group", case["group"].id, context
    )
    document = agent_tools.get_document_context(
        session, case["group"].id, case["item"].id, context
    )

    assert reconciliation.findings[0]["type"] == "GROUP_DIFFERENCE"
    assert transaction.description == "PAGAMENTO NF 123"
    assert group.reasons == ["invoice_exact"]
    assert group.items[0].entity_id == case["payable"].id
    assert audit[0].event_type == "CREATED"
    assert document.extracted.invoice_number == "123"
    assert document.validations[0].rule_code == "AMOUNT_POSITIVE"
    payload = document.model_dump_json()
    assert "private/raw-file.pdf" not in payload
    assert "secret" not in payload


def test_resolver_rejeita_finding_de_outra_conta(session, investigation_case):
    other = db.BankAccount(company="OUTRA EMPRESA")
    session.add(other)
    session.flush()
    transaction = db.BankTransaction(
        bank_account_id=other.id,
        external_id="OTHER-1",
        transaction_date=TODAY,
        amount=Decimal("10.00"),
        direction="DEBIT",
    )
    session.add(transaction)
    session.flush()
    investigation_case["reconciliation"].findings = {
        "items": [{"type": "TRANSACTION_NOT_MAPPED", "bank_transaction_id": transaction.id}]
    }
    session.commit()

    with pytest.raises(agent_tools.ToolScopeError):
        agent_tools.resolve_investigation_scope(
            session,
            daily_reconciliation_id=investigation_case["reconciliation"].id,
        )


def test_recurso_da_mesma_conta_mas_fora_do_caso_e_bloqueado(
    session, investigation_case, make_transaction
):
    transaction = make_transaction("50.00", "OUTRO CASO")
    other_group = db.ReconciliationGroup(
        bank_transaction_id=transaction.id,
        group_type="SINGLE",
        total_transaction_amount=Decimal("50.00"),
        total_allocated_amount=Decimal("0.00"),
        difference=Decimal("50.00"),
        status="REVIEW_REQUIRED",
        confidence_score=50,
    )
    session.add(other_group)
    session.commit()

    with pytest.raises(agent_tools.ToolScopeError):
        agent_tools.get_transaction(
            session, transaction.id, investigation_case["context"]
        )
    with pytest.raises(agent_tools.ToolScopeError):
        agent_tools.get_group(session, other_group.id, investigation_case["context"])


def test_contexto_de_grupo_congela_apenas_recursos_alcancaveis(session, investigation_case):
    context = agent_tools.resolve_investigation_scope(
        session, reconciliation_group_id=investigation_case["group"].id
    )
    assert context.allowed_group_ids == {investigation_case["group"].id}
    assert context.allowed_transaction_ids == {investigation_case["transaction"].id}
    assert context.allowed_document_ids == {investigation_case["document"].id}


def test_recurso_inexistente_nao_vira_lista_vazia(session, investigation_case):
    with pytest.raises(agent_tools.ToolNotFoundError):
        agent_tools.get_transaction(session, 999999, investigation_case["context"])


def test_busca_de_memoria_forca_filtro_da_conta(
    session, investigation_case, monkeypatch
):
    captured = {}
    item = db.MemoryItem(
        kind=db.DIVERGENCE_CASE,
        content="diferenca causada por tarifa",
        item_metadata={"resolution": "classificar tarifa"},
        bank_account_id=investigation_case["context"].bank_account_id,
    )
    session.add(item)
    session.flush()

    def fake_retrieve(session_arg, kind, query, *, filters, k, audit_degradation):
        captured.update(
            kind=kind,
            query=query,
            filters=filters,
            k=k,
            audit_degradation=audit_degradation,
        )
        return memory.RetrievalOutcome(
            hits=[memory.Hit(item=item, score=0.9, via={"semantic"})]
        )

    monkeypatch.setattr(agent_tools.memory, "retrieve_outcome", fake_retrieve)
    result = agent_tools.search_similar_cases(
        session, " tarifa bancaria ", investigation_case["context"], limit=3
    )

    assert captured == {
        "kind": db.DIVERGENCE_CASE,
        "query": "tarifa bancaria",
        "filters": {
            "bank_account_id": investigation_case["context"].bank_account_id
        },
        "k": 3,
        "audit_degradation": False,
    }
    assert result.cases[0].resolution == "classificar tarifa"
    assert result.cases[0].via == ["semantic"]
    assert result.degraded is False


def test_busca_sem_resultado_e_limites(session, investigation_case, monkeypatch):
    monkeypatch.setattr(
        agent_tools.memory,
        "retrieve_outcome",
        lambda *args, **kwargs: memory.RetrievalOutcome(hits=[]),
    )
    assert agent_tools.search_similar_cases(
        session, "sem precedente", investigation_case["context"]
    ).cases == []
    with pytest.raises(agent_tools.InvalidToolInputError):
        agent_tools.search_similar_cases(session, " ", investigation_case["context"])
    with pytest.raises(agent_tools.InvalidToolInputError):
        agent_tools.get_audit_trail(
            session, "supplier", 1, investigation_case["context"]
        )


def test_busca_degradada_nao_escreve_audit_na_sessao_read_only(
    session, investigation_case, monkeypatch
):
    monkeypatch.setattr(agent_tools.memory, "embed_query", lambda query: None)
    before = session.query(db.AuditEvent).count()

    result = agent_tools.search_similar_cases(
        session, "sem embedding", investigation_case["context"]
    )

    assert result.degraded is True
    assert session.query(db.AuditEvent).count() == before


def test_contrato_da_recomendacao_exige_humano_e_evidencia_tipadas():
    recommendation = agent_tools.InvestigationRecommendation(
        classification="GROUP_DIFFERENCE",
        summary="Grupo nao fecha",
        hypothesis="Possivel tarifa bancaria",
        recommended_action="MANUAL_REVIEW",
        confidence=0.8,
        evidence=[
            agent_tools.Evidence(source_type="group", source_id="10", claim="Diferenca 1.00")
        ],
        claims=[
            agent_tools.GroundedClaim(
                text="Diferenca confirmada", evidence_ids=["10"]
            )
        ],
    )
    assert recommendation.requires_human is True
    assert '"classification":"GROUP_DIFFERENCE"' in recommendation.model_dump_json()


def test_payload_nao_confiavel_e_limitado(session, investigation_case):
    document_id = investigation_case["document"].id
    extracted = session.query(db.ExtractedDocument).filter_by(
        document_id=document_id
    ).one()
    extracted.observation = "x" * (agent_tools.MAX_FIELD_CHARS + 100)
    extracted.structured_payload = {
        "nested": {"a": {"b": {"c": {"secret": "profundo"}}}},
        "many": list(range(agent_tools.MAX_RESULT_ITEMS + 20)),
    }
    session.commit()

    result = agent_tools.get_document_context(
        session,
        investigation_case["group"].id,
        investigation_case["item"].id,
        investigation_case["context"],
    )
    assert result.extracted.observation.endswith("...[truncated]")
    assert len(result.extracted.structured_payload["many"]) == agent_tools.MAX_RESULT_ITEMS
    assert "profundo" not in result.model_dump_json()


def test_modulo_nao_importa_liquidacao_nem_commita():
    source = open(agent_tools.__file__, encoding="utf-8").read()
    assert "_settle" not in source
    assert ".commit(" not in source
