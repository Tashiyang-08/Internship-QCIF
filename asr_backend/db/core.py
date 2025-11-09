# asr_backend/db/core.py
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# -------------------------------------------------------------------
# SQLite file at: <project-root>/asr_backend/data/transcriptions.db
# You can override with:
#   ASR_DB_URL      -> full SQLAlchemy URL, e.g. sqlite:///C:/path/to/file.db
#   ASR_DB_FILENAME -> just the filename inside asr_backend/data (default: transcriptions.db)
# -------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../asr_backend
_DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(_DATA_DIR, exist_ok=True)

_DB_FILENAME = os.environ.get("ASR_DB_FILENAME", "transcriptions.db")
_DB_PATH = os.path.join(_DATA_DIR, _DB_FILENAME)

# Normalize backslashes for SQLAlchemy URLs on Windows
_DB_PATH_URL = _DB_PATH.replace("\\", "/")
DATABASE_URL = os.environ.get("ASR_DB_URL", f"sqlite:///{_DB_PATH_URL}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

__all__ = ["engine", "SessionLocal", "Base", "DATABASE_URL"]
