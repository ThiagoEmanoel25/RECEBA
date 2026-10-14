"""Parametros do sistema. Tudo que o spec chama de "configuravel" vive aqui."""
import json
import os
from decimal import Decimal

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./receba.db")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")

# chave -> perfil (ADMIN / ANALYST / FINANCIAL_APPROVER / VIEWER)
API_KEYS: dict[str, str] = json.loads(os.getenv("API_KEYS", '{"dev-key":"ADMIN"}'))

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
ALLOWED_MIME = {
    "application/pdf",
    "application/xml",
    "text/xml",
    "text/csv",
    "image/png",
    "image/jpeg",
}

# ---- matching -------------------------------------------------------------
AMOUNT_TOLERANCE = Decimal(os.getenv("AMOUNT_TOLERANCE", "0.01"))
FEE_MAX = Decimal(os.getenv("FEE_MAX", "150.00"))  # tarifa/juros/multa aceita sem revisao
WEIGHTS: dict[str, int] = json.loads(
    os.getenv(
        "MATCH_WEIGHTS",
        '{"invoice":40,"cnpj":25,"amount":20,"party":10,"date":5,"bl":15,"container":15}',
    )
)
AUTO_MATCH_MIN = int(os.getenv("AUTO_MATCH_MIN", 90))
REVIEW_MIN = int(os.getenv("REVIEW_MIN", 70))
DATE_WINDOW_DAYS = int(os.getenv("DATE_WINDOW_DAYS", 30))
DUE_DATE_TOLERANCE_DAYS = int(os.getenv("DUE_DATE_TOLERANCE_DAYS", 10))

# ---- batch ----------------------------------------------------------------
BATCH_MAX_CANDIDATES = int(os.getenv("BATCH_MAX_CANDIDATES", 15))
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", 4))
