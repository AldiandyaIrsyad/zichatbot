"""Structured JSON logging configuration using ``structlog``.

Configures two sinks:
    1. ``stdout`` — human-readable JSON lines (via ``structlog.JSONRenderer``).
    2. TCP socket — JSON lines shipped to Vector (``observability/vector.yaml``)
       which forwards to Loki for centralised log aggregation.

The :class:`CorrelationIdMiddleware` in ``app/shared/middleware.py`` binds a
``request_id`` to ``structlog.contextvars`` per request; the
``merge_contextvars`` processor here injects it into every log line so a
single request's logs can be filtered in Loki/Grafana.
"""

import logging
import logging.handlers
import sys

import structlog

from app.shared.config import get_logger_settings


class JSONTCPHandler(logging.handlers.SocketHandler):
    """Sends JSON log records over TCP to Vector.

    Overrides :meth:`makePickle` to emit a single ``\\n``-terminated UTF-8
    JSON line — the format Vector's ``tcp`` source expects — instead of the
    default pickled record, which Vector cannot parse.
    """

    def makePickle(self, record: logging.LogRecord) -> bytes:
        return (self.format(record) + '\n').encode('utf-8')


def setup_logging() -> None:
    """Configure structured JSON logging: the root logger (stdout + TCP
    handler) at the configured level, plus the ``structlog`` processor chain
    (contextvars merge → level filter → logger/level stamping → ISO timestamp
    → JSON render). Called once at import in ``app/main.py``.
    """
    settings = get_logger_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Add TCP handler for Vector → Loki pipeline.
    root_logger = logging.getLogger()
    tcp_handler = JSONTCPHandler(settings.vector_host, settings.vector_port)
    tcp_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(tcp_handler)

    structlog.configure(
        processors=[
            # Inject request_id from CorrelationIdMiddleware's contextvars bind.
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
