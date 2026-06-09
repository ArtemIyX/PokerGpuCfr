from .config import Settings, load_settings
from .logging_utils import configure_logging


def create_app(settings: Settings | None = None) -> Settings:
    app_settings = settings or load_settings()
    configure_logging(app_settings.log_level)
    return app_settings
