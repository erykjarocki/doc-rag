"""Structured logging for doc-rag.

Provides a consistent logging setup across all modules. By default uses
human-readable format for terminals, but supports JSON output via the
DOC_RAG_LOG_FORMAT=json environment variable.

Usage:
    from src.log import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", extra={"file": "test.pdf", "chunks": 42})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Per-request correlation ID, set by middleware
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_LOG_FORMAT = os.getenv("DOC_RAG_LOG_FORMAT", "text")
_LOG_LEVEL = os.getenv("DOC_RAG_LOG_LEVEL", "INFO").upper()

_configured = False


def _configure_root():
    """Set up root logger once per process."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("src")
    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if _LOG_FORMAT == "json" else _TextFormatter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the 'src' hierarchy.

    Args:
        name: Module name, typically __name__.

    Returns:
        Configured logging.Logger instance.
    """
    _configure_root()
    # Normalize: src.retriever -> src.retriever (already under src namespace)
    if not name.startswith("src"):
        name = f"src.{name}"
    return logging.getLogger(name)


class _TextFormatter(logging.Formatter):
    """Human-readable format with colors for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # cyan
        logging.INFO: "\033[32m",     # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",    # red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        reset = self.RESET if color else ""
        level = record.levelname.ljust(8)
        module = record.name.replace("src.", "")
        req_id = request_id_var.get("")
        req_tag = f" [{req_id[:8]}]" if req_id else ""

        msg = record.getMessage()

        # Include extra fields if present
        extras = _extract_extras(record)
        if extras:
            extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
            msg = f"{msg}  {extra_str}"

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return f"{color}{level}{reset} {module}{req_tag}  {msg}"


class _JsonFormatter(logging.Formatter):
    """JSON-lines format for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        req_id = request_id_var.get("")
        if req_id:
            log_entry["request_id"] = req_id

        extras = _extract_extras(record)
        if extras:
            log_entry["data"] = extras

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def _extract_extras(record: logging.LogRecord) -> dict[str, Any]:
    """Pull out non-standard LogRecord fields as extras.

    Standard LogRecord fields (name, msg, levelno, etc.) are excluded.
    """
    standard = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName", "pathname",
        "filename", "funcName",
    }
    return {
        k: v for k, v in record.__dict__.items()
        if k not in standard and not k.startswith("_")
    }


def generate_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:12]
