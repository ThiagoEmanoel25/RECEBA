"""Fluxo ponta a ponta pela API: upload -> extracao -> validacao -> matching -> auditoria."""
import datetime as dt
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import db
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
def client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    db.Base.metadata.create_all(engine)
    Local = sessionmaker(engine, expire_on_commit=False, future=True)
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
        session.add_all([
            db.Supplier(name="ACME LTDA", cnpj="11222333000181"),
            db.BankAccount(company="EMPRESA", bank_code="001", agency="1", account="1"),
        ])
        session.flush()
        session.add_all([
            db.SupplierBankAccount(supplier_id=1, bank_code="341", agency="4321",
                                   account="98765", active=True),
            db.PurchaseOrder(number="771", supplier_id=1),
        ])
        session.commit()
    with TestClient(app) as test_client:
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
        "/bank-transactions/import?bank_account_id=1", headers=HEADERS,
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
        json={"bank_account_id": 1, "balance_date": str(TODAY),
              "opening_balance": "10000.00", "closing_balance": "2500.00"},
    )
    reconciliation = client.post(
        "/reconciliations/run", headers=HEADERS,
        json={"bank_account_id": 1, "reconciliation_date": str(TODAY)},
    ).json()
    assert reconciliation["status"] == "RECONCILED", reconciliation["findings"]

    trail = [e["event"] for e in client.get(f"/audit/document/{document_id}",
                                            headers=HEADERS).json()]
    assert trail[:3] == ["DOCUMENT_STORED", "DUPLICATE_UPLOAD_IGNORED", "DOCUMENT_EXTRACTED"]
    assert "VALIDATION_APPROVED" in trail

    metrics = client.get("/metrics", headers=HEADERS).json()
    assert metrics["auto_match_rate"] == 1.0


def test_aprovacao_exige_perfil_aprovador(client):
    client.post(
        "/bank-transactions/import?bank_account_id=1&run_matching=false", headers=HEADERS,
        files={"file": ("e.csv",
                        b"date;amount;description\n09/10/2026;-50,00;PGTO SEM REFERENCIA\n",
                        "text/csv")},
    )
    response = client.post("/matching/1/run", headers=HEADERS).json()
    assert response["status"] == "UNMATCHED"
