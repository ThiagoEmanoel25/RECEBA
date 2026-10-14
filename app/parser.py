"""Normalizacao e Observation Parser (spec secoes 13 e 14). 100% deterministico."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^A-Z0-9]+")
_SPACES = re.compile(r"\s+")

# Ordem importa: container antes de BL para nao deixar BL comer o codigo do container.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("container", re.compile(r"\b([A-Z]{4}\d{7})\b")),
    ("invoice", re.compile(r"\b(?:INV|INVOICE|NF|NFE|FAT|FATURA|DUPL)\s*N?\s*0*(\d{3,})\b")),
    ("purchase_order", re.compile(r"\b(?:PED|PEDIDO|PO|OC)\s*N?\s*0*(\d{2,})\b")),
    ("bl", re.compile(r"\b(?:BL|BILL OF LADING|CONHECIMENTO)\s*N?\s*([A-Z0-9]{5,})\b")),
    ("cnpj", re.compile(r"\b(\d{14})\b")),
]

FEE_HINTS = {
    "FEE": ("TARIFA", "TAR", "TXA", "TAXA", "IOF"),
    "INTEREST": ("JUROS", "MORA"),
    "PENALTY": ("MULTA",),
    "DISCOUNT": ("DESCONTO", "DESC", "ABATIMENTO"),
}


def normalize(text: str | None) -> str:
    """`Pgto Forn. ABC Inv-93882` -> `PGTO FORN ABC INV 93882`."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).upper()
    return _SPACES.sub(" ", _PUNCT.sub(" ", folded)).strip()


def digits(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def ref_key(value: str | None) -> str:
    """Chave canonica de referencia: so alfanumerico, sem zeros a esquerda."""
    key = _PUNCT.sub("", normalize(value))
    stripped = key.lstrip("0")
    return stripped or key


def parse_observation(text: str | None) -> dict[str, str]:
    """Extrai tokens da observacao bancaria. Vazio = parser nao achou nada."""
    normalized = normalize(text)
    tokens: dict[str, str] = {}
    for name, pattern in PATTERNS:
        match = pattern.search(normalized)
        if match:
            tokens[name] = match.group(1)
    # supplier_hint: primeira palavra "de nome" que nao seja palavra de sistema
    words = [w for w in normalized.split() if len(w) >= 3 and not w.isdigit()]
    noise = {"PGTO", "PAGTO", "PAGAMENTO", "FORN", "FORNECEDOR", "CLI", "CLIENTE",
             "INV", "INVOICE", "NFE", "FAT", "FATURA", "DUPL", "PED", "PEDIDO",
             "CONT", "CONTAINER", "TED", "DOC", "PIX", "TRANSF", "REF"}
    for word in words:
        if word not in noise and word not in tokens.values():
            tokens["supplier_hint"] = word
            break
    return tokens


def classify_adjustment(text: str | None) -> str | None:
    """Retorna FEE / INTEREST / PENALTY / DISCOUNT quando a observacao indica."""
    normalized = normalize(text)
    words = set(normalized.split())
    for kind, hints in FEE_HINTS.items():
        if any(hint in words for hint in hints):
            return kind
    return None


def same_ref(a: str | None, b: str | None) -> bool:
    """Compara referencias: 'INV-093882' == '93882'. Exige 3+ digitos no numero."""
    if not a or not b:
        return False
    if ref_key(a) == ref_key(b):
        return True
    da, db = digits(a).lstrip("0"), digits(b).lstrip("0")
    return bool(da) and len(da) >= 3 and da == db
