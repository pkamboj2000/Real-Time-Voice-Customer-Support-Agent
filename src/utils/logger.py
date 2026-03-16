"""
Structured logging setup using structlog.

Every module should grab its own logger via get_logger(__name__).
Logs are written as JSON in production and pretty-printed in dev.
"""

import logging
import sys

import structlog
from configs.settings import settings


def _configure_structlog():
    """
    One-time structlog configuration. Called at import time so every
    module that does `from src.utils.logger import get_logger` gets a
    properly configured logger without any extra ceremony.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_production:
        # JSON lines for prod — easy to ship to datadog / elk / whatever
        renderer = structlog.processors.JSONRenderer()
    else:
        # colored console output for local dev
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # also configure the stdlib root logger so third-party libs play nice
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=shared_processors + [renderer],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# run once on import
_configure_structlog()


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)
