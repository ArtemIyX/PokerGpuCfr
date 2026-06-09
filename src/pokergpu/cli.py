import logging

from .app import create_app


def main() -> int:
    settings = create_app()
    logger = logging.getLogger(__name__)
    logger.info("PokerGPU initialized")
    print(f"PokerGPU ready on device={settings.device}")
    return 0
