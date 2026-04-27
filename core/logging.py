from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level.upper())
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
