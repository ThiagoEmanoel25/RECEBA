"""Importacao de extrato CSV com idempotencia (spec secoes 12 e 35)."""
from __future__ import annotations

import csv
import datetime as dt
import io
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import BankBalance, BankTransaction, audit
from app.parser import normalize, parse_observation

# Colunas aceitas: date, amount, description e opcionalmente external_id,
# direction, document_number, reference, currency.
REQUIRED = {"date", "amount", "description"}


def _decimal(raw: str) -> Decimal:
    text = (raw or "").strip().replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Valor invalido: {raw!r}") from exc


def _date(raw: str) -> dt.date:
    text = (raw or "").strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Data invalida: {raw!r}")


def import_csv(session: Session, bank_account_id: int, content: bytes) -> dict:
    text = content.decode("utf-8-sig")
    sample = text[:2048]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    headers = {(h or "").strip().lower() for h in (reader.fieldnames or [])}
    missing = REQUIRED - headers
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    created, duplicated = [], 0
    for line, raw_row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw_row.items()}
        if not any(row.values()):
            continue
        try:
            amount = _decimal(row["amount"])
            date = _date(row["date"])
        except ValueError as exc:
            raise ValueError(f"Linha {line}: {exc}") from exc

        direction = (row.get("direction") or "").upper()
        if direction not in {"CREDIT", "DEBIT"}:
            direction = "DEBIT" if amount < 0 else "CREDIT"
        description = row["description"]
        tokens = parse_observation(description)
        # Fallback de idempotencia quando o banco nao manda id (spec secao 35).
        external_id = row.get("external_id") or (
            f"{date.isoformat()}|{abs(amount)}|{direction}|{normalize(description)[:80]}"
        )

        if session.scalar(
            select(BankTransaction).where(
                BankTransaction.bank_account_id == bank_account_id,
                BankTransaction.external_id == external_id,
            )
        ):
            duplicated += 1
            continue

        transaction = BankTransaction(
            bank_account_id=bank_account_id,
            external_id=external_id,
            transaction_date=date,
            amount=abs(amount),
            direction=direction,
            description_raw=description,
            description_normalized=normalize(description),
            document_number=row.get("document_number") or tokens.get("invoice"),
            reference=row.get("reference") or tokens.get("bl") or tokens.get("container"),
            currency=(row.get("currency") or "BRL").upper(),
            status="IMPORTED",
            raw_payload=row,
        )
        session.add(transaction)
        session.flush()
        audit(session, "bank_transaction", transaction.id, "TRANSACTION_IMPORTED",
              external_id=external_id)
        created.append(transaction.id)

    session.commit()
    return {"imported": len(created), "duplicated": duplicated, "ids": created}


def set_balance(
    session: Session,
    bank_account_id: int,
    day: dt.date,
    opening: Decimal,
    closing: Decimal,
) -> BankBalance:
    balance = session.scalar(
        select(BankBalance).where(
            BankBalance.bank_account_id == bank_account_id, BankBalance.balance_date == day
        )
    ) or BankBalance(bank_account_id=bank_account_id, balance_date=day)
    balance.opening_balance = opening
    balance.closing_balance = closing
    session.add(balance)
    session.commit()
    return balance
