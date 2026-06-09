from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifact_dir: Path
    log_level: str = "INFO"
    device: str = "auto"


def load_settings() -> Settings:
    project_root = Path(os.getenv("POKERGPU_PROJECT_ROOT", Path.cwd())).resolve()
    data_dir = Path(os.getenv("POKERGPU_DATA_DIR", project_root / "data")).resolve()
    artifact_dir = Path(
        os.getenv("POKERGPU_ARTIFACT_DIR", project_root / "artifacts")
    ).resolve()
    log_level = os.getenv("POKERGPU_LOG_LEVEL", "INFO").upper()
    device = os.getenv("POKERGPU_DEVICE", "auto").lower()
    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        log_level=log_level,
        device=device,
    )
