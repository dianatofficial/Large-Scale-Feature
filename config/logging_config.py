import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

class StructuredJsonFormatter(logging.Formatter):
    """Production JSON formatter for structured logging in distributed systems."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "process_id": record.process,
            "thread_name": record.threadName,
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def configure_logger(name: str = "vectorscale", level: str = "INFO", log_format: str = "json") -> logging.Logger:
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        if log_format.lower() == "json":
            handler.setFormatter(StructuredJsonFormatter())
        else:
            fmt_str = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
            standard_formatter = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(standard_formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
