import logging


def _log_level(verbose_logs: bool) -> int:
    return logging.DEBUG if verbose_logs else logging.ERROR


def configure_logging(verbose_logs: bool) -> None:
    level = _log_level(verbose_logs)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

