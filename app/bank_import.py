"""Importacao de extrato CSV com idempotencia (spec secoes 12 e 35)."""
from __future__ import annotations

import csv
import datetime as dt
import io
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import memory
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


def _embed_pending(session: Session, transactions: list[BankTransaction]) -> int:
    """Embeddings em lote: a Voyage aceita varios textos de uma vez, o matching nao espera API."""
    if not transactions:
        return 0
    vectors = memory.embed([t.description_normalized or "" for t in transactions])
    # Lote parcial conta como falha total: nada e gravado pela metade, o proximo import refaz.
    if vectors is None or len(vectors) != len(transactions):
        audit(session, "bank_account", transactions[0].bank_account_id, "MEMORY_DEGRADED",
              reason="embedding_indisponivel" if vectors is None else "lote_incompleto",
              transactions=len(transactions),
              vectors=None if vectors is None else len(vectors))
        return 0
    for transaction, vector in zip(transactions, vectors):
        transaction.embedding = vector
    return len(transactions)


def import_csv(session: Session, bank_account_id: int, content: bytes) -> dict:
    text = content.decode("utf-8-sig")
    sample = text[:2048]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    headers = {(h or "").strip().lower() for h in (reader.fieldnames or [])}
    missing = REQUIRED - headers
    if missing:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(missing)}")

    created, duplicated, pending = [], 0, []
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
        pending.append(transaction)

    embedded = _embed_pending(session, pending)
    session.commit()
    return {
        "imported": len(created),
        "duplicated": duplicated,
        "embedded": embedded,
        "ids": created,
    }


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
