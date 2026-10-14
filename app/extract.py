"""Document Processor (spec secoes 8 e 9). A IA para aqui: a saida e dado, nao decisao."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, field_validator

from app.parser import digits, normalize, parse_observation

MODEL = os.getenv("EXTRACTION_MODEL", "claude-opus-5")


class BankAccountOut(BaseModel):
    bank_code: str | None = None
    agency: str | None = None
    account: str | None = None


class FinancialDocument(BaseModel):
    document_type: str = "UNKNOWN"
    supplier_name: str | None = None
    supplier_cnpj: str | None = None
    invoice_number: str | None = None
    amount: Decimal | None = None
    currency: str = "BRL"
    due_date: dt.date | None = None
    bank_account: BankAccountOut | None = None
    purchase_order: str | None = None
    bill_of_lading: str | None = None
    container: str | None = None
    cost_center: str | None = None
    observation: str | None = None

    @field_validator("supplier_cnpj", mode="before")
    @classmethod
    def _only_digits(cls, value):
        return digits(value) or None

    @field_validator("amount", mode="before")
    @classmethod
    def _money(cls, value):
        if value in (None, ""):
            return None
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value))
        text = str(value).strip().replace(" ", "")
        text = re.sub(r"[^\d,.\-]", "", text)
        if "," in text and "." in text:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", ".")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None


def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        # ponytail: sem OCR. Se o PDF vier escaneado o texto sai vazio e cai no fallback
        # deterministico; plugar tesseract/pytesseract quando aparecer o primeiro caso real.
        return text
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", errors="replace")


_RULES = [
    ("supplier_cnpj", re.compile(r"(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})")),
    ("amount", re.compile(r"(?:VALOR|TOTAL|AMOUNT)\D{0,12}([\d.,]{4,})", re.I)),
    ("due_date", re.compile(r"(?:VENC\w*|DUE\s*DATE)\D{0,12}(\d{2}[/-]\d{2}[/-]\d{2,4})", re.I)),
]


def extract_offline(text: str) -> FinancialDocument:
    """Fallback sem LLM: regex + observation parser. Mantem o pipeline rodando offline."""
    tokens = parse_observation(text)
    data: dict = {
        "invoice_number": tokens.get("invoice"),
        "purchase_order": tokens.get("purchase_order"),
        "bill_of_lading": tokens.get("bl"),
        "container": tokens.get("container"),
        "supplier_name": tokens.get("supplier_hint"),
        "observation": normalize(text)[:500] or None,
    }
    for field, pattern in _RULES:
        match = pattern.search(text)
        if match:
            data[field] = match.group(1)
    if isinstance(data.get("due_date"), str):
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
            try:
                data["due_date"] = dt.datetime.strptime(data["due_date"], fmt).date()
                break
            except ValueError:
                data["due_date"] = None
    return FinancialDocument(**{k: v for k, v in data.items() if v is not None})


PROMPT = (
    "Extraia os campos financeiros do documento abaixo. Nao calcule, nao infira valores "
    "que nao estejam escritos e use null quando o campo nao existir no texto.\n\n"
)


def extract_with_llm(text: str) -> tuple[FinancialDocument, str]:
    """Structured output via tool schema. Sem API key, cai no parser deterministico."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return extract_offline(text), "offline-regex"

    from anthropic import Anthropic

    schema = FinancialDocument.model_json_schema()
    response = Anthropic().messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[{
            "name": "emit_document",
            "description": "Campos extraidos do documento financeiro",
            "input_schema": {
                "type": "object",
                "properties": schema["properties"],
                "$defs": schema.get("$defs", {}),
            },
        }],
        tool_choice={"type": "tool", "name": "emit_document"},
        messages=[{"role": "user", "content": PROMPT + text[:120_000]}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return FinancialDocument(**block.input), MODEL
    raise RuntimeError(f"LLM nao retornou tool_use: {json.dumps(response.model_dump())[:400]}")
