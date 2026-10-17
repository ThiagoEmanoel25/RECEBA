import datetime as dt
import os
import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import db

warnings.filterwarnings("ignore", message=".*does \\*not\\* support Decimal.*")

TODAY = dt.date(2026, 10, 9)
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://receba:receba@localhost:5433/receba_test"
)
SKIP_MESSAGE = (
    f"Postgres com pgvector indisponivel em {TEST_DATABASE_URL}. "
    "Suba com: docker compose up -d db"
)


@pytest.fixture(scope="session")
def engine():
    """Schema criado uma vez por sessao de teste; cada teste roda em transacao propria."""
    database = make_url(TEST_DATABASE_URL).database or ""
    if not database.endswith("_test"):
        raise RuntimeError(
            f"TEST_DATABASE_URL deve apontar para um banco terminado em '_test', nao {database!r}"
        )
    eng = create_engine(TEST_DATABASE_URL, future=True)
    try:
        with eng.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except SQLAlchemyError as exc:
        eng.dispose()
        pytest.skip(f"{SKIP_MESSAGE} ({exc.__class__.__name__})")
    db.Base.metadata.drop_all(eng)
    db.Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def connection(engine):
    """Transacao externa com rollback no fim: os commits do app viram savepoints."""
    conn = engine.connect()
    transaction = conn.begin()
    yield conn
    transaction.rollback()
    conn.close()


@pytest.fixture()
def session(connection):
    with Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    ) as s:
        yield s


@pytest.fixture()
def company(session):
    """Fornecedor ACME + conta bancaria cadastrada + conta da empresa no banco."""
    supplier = db.Supplier(name="ACME LTDA", cnpj="11222333000181")
    customer = db.Customer(name="BETA COMERCIO", cnpj="11222333000181")
    account = db.BankAccount(company="MINHA EMPRESA", bank_code="001", agency="1234",
                             account="567890")
    session.add_all([supplier, customer, account])
    session.flush()
    session.add(
        db.SupplierBankAccount(supplier_id=supplier.id, bank_code="341", agency="4321",
                               account="98765", active=True)
    )
    session.commit()
    return {"supplier": supplier, "customer": customer, "account": account}


@pytest.fixture()
def make_payable(session, company):
    def _make(invoice, amount, due_offset=0, **kwargs):
        payable = db.Payable(
            supplier_id=company["supplier"].id,
            invoice_number=invoice,
            amount_original=Decimal(amount),
            amount_open=Decimal(amount),
            due_date=TODAY + dt.timedelta(days=due_offset),
            **kwargs,
        )
        session.add(payable)
        session.commit()
        return payable

    return _make


@pytest.fixture()
def make_transaction(session, company):
    counter = iter(range(1, 999))

    def _make(amount, description, direction="DEBIT", date=TODAY, **kwargs):
        from app.parser import normalize, parse_observation

        tokens = parse_observation(description)
        transaction = db.BankTransaction(
            bank_account_id=company["account"].id,
            external_id=f"TX{next(counter)}",
            transaction_date=date,
            amount=Decimal(amount),
            direction=direction,
            description_raw=description,
            description_normalized=normalize(description),
            document_number=tokens.get("invoice"),
            reference=tokens.get("bl") or tokens.get("container"),
            **kwargs,
        )
        session.add(transaction)
        session.commit()
        return transaction

    return _make


@pytest.fixture(autouse=True)
def _sem_chaves_de_ia(monkeypatch):
    """Suite roda offline: embedding fake e extracao por regex, nunca a rede."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
