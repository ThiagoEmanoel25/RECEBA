import datetime as dt
import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import db

warnings.filterwarnings("ignore", message=".*does \\*not\\* support Decimal.*")

TODAY = dt.date(2026, 10, 9)


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    db.Base.metadata.create_all(engine)
    with sessionmaker(engine, expire_on_commit=False, future=True)() as s:
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
