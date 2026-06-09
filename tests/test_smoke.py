from pathlib import Path

from pokergpu.app import create_app
from pokergpu.config import Settings


def test_create_app_returns_settings() -> None:
    settings = Settings(
        project_root=Path.cwd(),
        data_dir=Path.cwd() / "data",
        artifact_dir=Path.cwd() / "artifacts",
    )

    result = create_app(settings)

    assert result is settings
