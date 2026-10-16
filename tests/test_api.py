"""Fluxo ponta a ponta pela API: upload -> extracao -> validacao -> matching -> auditoria."""
import datetime as dt
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import db, memory
from app.db import MATCH_DECISION
from app.main import app
from tests.conftest import TODAY

HEADERS = {"X-API-Key": "dev-key"}
VIEWER = {"X-API-Key": "viewer-key"}

NOTA = (
    "NOTA FISCAL\nFornecedor ACME LTDA\nCNPJ 11.222.333/0001-81\n"
    "Invoice 44120\nPedido 771\nBanco 341 Agencia 4321 Conta 98765\n"
    "Valor Total 7.500,00\nVencimento 15/10/2026\n"
)


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
    monkeypatch.setitem(app.dependency_overrides, db.get_session, lambda: iter([]))

    def override():
        with Local() as session:
            yield session

    app.dependency_overrides[db.get_session] = override
    from app.config import API_KEYS

    API_KEYS["viewer-key"] = "VIEWER"
    with Local() as session:
        supplier = db.Supplier(name="ACME LTDA", cnpj="11222333000181")
        account = db.BankAccount(company="EMPRESA", bank_code="001", agency="1", account="1")
        session.add_all([supplier, account])
        session.flush()
        session.add_all([
            db.SupplierBankAccount(supplier_id=supplier.id, bank_code="341", agency="4321",
                                   account="98765", active=True),
            db.PurchaseOrder(number="771", supplier_id=supplier.id),
        ])
        session.commit()
    with TestClient(app) as test_client:
        test_client.bank_account_id = account.id  # ids vem de sequence, nao sao fixos
        yield test_client
    app.dependency_overrides.clear()
    API_KEYS.pop("viewer-key", None)


def test_sem_api_key_recusa(client):
    assert client.get("/documents").status_code == 422
    assert client.get("/documents", headers={"X-API-Key": "nope"}).status_code == 401


def test_viewer_nao_escreve(client):
    response = client.post(
        "/documents", headers=VIEWER,
        files={"file": ("nf.txt", NOTA.encode(), "text/csv")},
    )
    assert response.status_code == 403


def test_mime_nao_aceito(client):
    response = client.post(
        "/documents", headers=HEADERS,
        files={"file": ("x.exe", b"MZ", "application/x-msdownload")},
    )
    assert response.status_code == 415


def test_fluxo_completo(client):
    upload = client.post(
        "/documents", headers=HEADERS, files={"file": ("nf.txt", NOTA.encode(), "text/csv")}
    )
    assert upload.status_code == 201, upload.text
    document_id = upload.json()["id"]
    assert upload.json()["duplicate"] is False

    again = client.post(
        "/documents", headers=HEADERS, files={"file": ("nf.txt", NOTA.encode(), "text/csv")}
    )
    assert again.json()["duplicate"] is True  # idempotencia por content_hash

    extracted = client.post(f"/documents/{document_id}/process", headers=HEADERS).json()
    assert extracted["extracted"]["invoice_number"] == "44120"
    assert extracted["extracted"]["amount"] == "7500.00"

    validated = client.post(f"/documents/{document_id}/validate", headers=HEADERS).json()
    assert validated["status"] == "APPROVED", validated["checks"]
    assert validated["payable_id"]

    csv_bytes = (
        "date;amount;description\n09/10/2026;-7500,00;Pgto Forn. ACME Inv-44120\n"
    ).encode()
    imported = client.post(
        f"/bank-transactions/import?bank_account_id={client.bank_account_id}", headers=HEADERS,
        files={"file": ("extrato.csv", csv_bytes, "text/csv")},
    ).json()
    assert imported["imported"] == 1
    assert imported["matching"] == {"AUTO_MATCHED": 1}

    transaction_id = imported["ids"][0]
    detail = client.get(f"/bank-transactions/{transaction_id}", headers=HEADERS).json()
    assert detail["status"] == "AUTO_MATCHED"
    group = client.get(f"/reconciliation-groups/{detail['group_id']}", headers=HEADERS).json()
    assert group["difference"] == "0.00" and group["items"][0]["entity_type"] == "PAYABLE"

    client.post(
        "/bank-accounts/balances", headers=HEADERS,
        json={"bank_account_id": client.bank_account_id, "balance_date": str(TODAY),
              "opening_balance": "10000.00", "closing_balance": "2500.00"},
    )
    reconciliation = client.post(
        "/reconciliations/run", headers=HEADERS,
        json={"bank_account_id": client.bank_account_id, "reconciliation_date": str(TODAY)},
    ).json()
    assert reconciliation["status"] == "RECONCILED", reconciliation["findings"]

    trail = [e["event"] for e in client.get(f"/audit/document/{document_id}",
                                            headers=HEADERS).json()]
    assert trail[:3] == ["DOCUMENT_STORED", "DUPLICATE_UPLOAD_IGNORED", "DOCUMENT_EXTRACTED"]
    assert "VALIDATION_APPROVED" in trail

    metrics = client.get("/metrics", headers=HEADERS).json()
    assert metrics["auto_match_rate"] == 1.0


def test_aprovacao_exige_perfil_aprovador(client):
    imported = client.post(
        f"/bank-transactions/import?bank_account_id={client.bank_account_id}&run_matching=false",
        headers=HEADERS,
        files={"file": ("e.csv",
                        b"date;amount;description\n09/10/2026;-50,00;PGTO SEM REFERENCIA\n",
                        "text/csv")},
    ).json()
    response = client.post(f"/matching/{imported['ids'][0]}/run", headers=HEADERS).json()
    assert response["status"] == "UNMATCHED"


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
    corrigido = [e for e in auditoria if e["event"] == "EXTRACTION_CORRECTED"]
    assert len(corrigido) == 1
    memoria = session.get(db.MemoryItem, body["memory_id"])
    assert memoria.kind == db.EXTRACTION_EXAMPLE
    assert memoria.cnpj == "11222333000181"
    assert corrigido[0]["metadata"]["memory_id"] == body["memory_id"]
    assert sorted(corrigido[0]["metadata"]["fields"]) == ["amount", "invoice_number"]


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


def _processado(client):
    upload = client.post(
        "/documents",
        headers=HEADERS,
        files={"file": ("nota.txt", NOTA.encode(), "text/csv")},
    )
    document_id = upload.json()["id"]
    client.post(f"/documents/{document_id}/process", headers=HEADERS)
    return document_id


def test_patch_extracted_parcial_preserva_campos_omitidos(client, session):
    document_id = _processado(client)
    response = client.patch(
        f"/documents/{document_id}/extracted", headers=HEADERS, json={"amount": "7600.00"}
    )
    assert response.status_code == 200
    linha = session.query(db.ExtractedDocument).filter_by(document_id=document_id).one()
    session.refresh(linha)
    assert linha.invoice_number == "44120"
    assert linha.amount == Decimal("7600.00")
    detalhe = client.get(f"/documents/{document_id}", headers=HEADERS).json()
    assert detalhe["extracted"]["invoice_number"] == "44120"
    assert linha.structured_payload["invoice_number"] == "44120"


def test_patch_extracted_corpo_vazio_ou_nulo_422(client):
    document_id = _processado(client)
    for corpo in ({}, {"amount": None}):
        response = client.patch(
            f"/documents/{document_id}/extracted", headers=HEADERS, json=corpo
        )
        assert response.status_code == 422


def test_patch_extracted_banco_mantem_formato_aninhado(client, session):
    document_id = _processado(client)
    # O extrator deterministico nao preenche banco; semeia o formato aninhado do LLM.
    linha = session.query(db.ExtractedDocument).filter_by(document_id=document_id).one()
    linha.structured_payload = {
        **linha.structured_payload,
        "bank_account": {"bank_code": "341", "agency": "4321", "account": "98765"},
    }
    session.commit()
    antes = client.get(f"/documents/{document_id}", headers=HEADERS).json()["extracted"]
    assert isinstance(antes["bank_account"], dict)
    response = client.patch(
        f"/documents/{document_id}/extracted", headers=HEADERS, json={"bank_account": "12345-6"}
    )
    assert response.status_code == 200
    extracted = client.get(f"/documents/{document_id}", headers=HEADERS).json()["extracted"]
    assert extracted["bank_account"]["account"] == "12345-6"
    assert extracted["bank_account"]["bank_code"] == antes["bank_account"]["bank_code"]
    assert extracted["bank_account"]["agency"] == antes["bank_account"]["agency"]
    memoria = session.get(db.MemoryItem, response.json()["memory_id"])
    assert memoria.item_metadata["bank_account"] == extracted["bank_account"]
    assert "bank_code" not in extracted and "bank_agency" not in extracted


DESC_MATCH = "PAGAMENTO ACME NF 44120 DIVERGENTE"


def _gera_grupo(client, make_payable, make_transaction):
    payable = make_payable("44120", "7500.00")
    transaction = make_transaction("7500.00", DESC_MATCH)
    # auto_settle=false deixa o grupo aberto para decisao humana
    run = client.post(f"/matching/{transaction.id}/run?auto_settle=false",
                      headers=HEADERS).json()
    return payable, transaction, run


def test_aprovacao_de_match_grava_memoria_episodica(client, session, company, make_payable,
                                                    make_transaction):
    payable, transaction, run = _gera_grupo(client, make_payable, make_transaction)
    assert run["group_id"], run

    resposta = client.post(f"/matching/{transaction.id}/approve", headers=HEADERS)
    assert resposta.status_code == 200, resposta.text

    hits = memory.retrieve(session, MATCH_DECISION, DESC_MATCH,
                           filters={"supplier_id": company["supplier"].id})
    assert hits, "aprovacao humana deve virar memoria"
    item = hits[0].item
    assert item.item_metadata == {
        "decision": "APPROVED",
        "entity_type": "PAYABLE",
        "entity_id": payable.id,
        "group_type": item.item_metadata["group_type"],
        "amount": "7500.00",
        "reason": None,
        "actor": "dev-***",  # Actor.key ja vem mascarado
    }
    assert item.supplier_id == company["supplier"].id
    assert item.customer_id is None
    assert item.bank_account_id == company["account"].id
    assert item.amount == Decimal("7500.00")


def test_rejeicao_de_match_grava_memoria_negativa(client, session, company, make_payable,
                                                  make_transaction):
    payable, transaction, run = _gera_grupo(client, make_payable, make_transaction)
    assert run["group_id"], run

    resposta = client.post(
        f"/matching/{transaction.id}/reject",
        headers=HEADERS,
        params={"reason": "fornecedor errado"},
    )
    assert resposta.status_code == 200, resposta.text

    hits = memory.retrieve(session, MATCH_DECISION, DESC_MATCH)
    assert hits
    item = hits[0].item
    assert item.item_metadata["decision"] == "REJECTED"
    assert item.item_metadata["reason"] == "fornecedor errado"
    assert item.item_metadata["entity_id"] == payable.id
    assert item.supplier_id == company["supplier"].id


def test_aprovacao_de_recebivel_grava_customer_id(client, session, company, make_transaction):
    receivable = db.Receivable(customer_id=company["customer"].id, invoice_number="9001",
                               amount_original=Decimal("1200.00"),
                               amount_open=Decimal("1200.00"), due_date=TODAY)
    session.add(receivable)
    session.commit()
    transaction = make_transaction("1200.00", "RECEBIMENTO BETA NF 9001", direction="CREDIT")
    run = client.post(f"/matching/{transaction.id}/run?auto_settle=false",
                      headers=HEADERS).json()
    assert run["group_id"], run

    assert client.post(f"/matching/{transaction.id}/approve", headers=HEADERS).status_code == 200

    hits = memory.retrieve(session, MATCH_DECISION, "RECEBIMENTO BETA NF 9001",
                           filters={"customer_id": company["customer"].id})
    assert hits
    item = hits[0].item
    assert item.customer_id == company["customer"].id
    assert item.supplier_id is None
    assert item.item_metadata["entity_type"] == "RECEIVABLE"
    assert item.item_metadata["entity_id"] == receivable.id
    assert item.amount == Decimal("1200.00")


def test_lote_grava_uma_memoria_por_titulo_com_valor_proprio(client, session, company,
                                                             make_payable, make_transaction):
    outro = db.Supplier(name="OUTRO FORNECEDOR", cnpj="99888777000166")
    session.add(outro)
    session.commit()
    p1 = make_payable("111", "3000.00")
    p2 = db.Payable(supplier_id=outro.id, invoice_number="222",
                    amount_original=Decimal("4500.00"), amount_open=Decimal("4500.00"),
                    due_date=TODAY)
    session.add(p2)
    session.commit()
    transaction = make_transaction("7500.00", "PAGAMENTO LOTE FORNECEDORES")
    run = client.post(f"/matching/{transaction.id}/run?auto_settle=false",
                      headers=HEADERS).json()
    assert run["group_type"] == "BATCH_PAYMENT", run

    assert client.post(f"/matching/{transaction.id}/approve", headers=HEADERS).status_code == 200

    itens = session.scalars(
        select(db.MemoryItem).where(db.MemoryItem.kind == MATCH_DECISION,
                                    db.MemoryItem.source_id == run["group_id"])
    ).all()
    por_titulo = {i.item_metadata["entity_id"]: i for i in itens}
    assert set(por_titulo) == {p1.id, p2.id}
    assert por_titulo[p1.id].supplier_id == company["supplier"].id
    assert por_titulo[p1.id].amount == Decimal("3000.00")
    assert por_titulo[p1.id].item_metadata["amount"] == "3000.00"
    assert por_titulo[p2.id].supplier_id == outro.id
    assert por_titulo[p2.id].amount == Decimal("4500.00")
    assert por_titulo[p2.id].item_metadata["amount"] == "4500.00"
