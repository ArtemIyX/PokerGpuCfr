import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    artifact_dir: Path
    log_level: str = "INFO"
    device: str = "auto"
    max_depth: int = 1
    max_nodes: int = 256
    min_reach_prob: float = 0.0


def load_settings() -> Settings:
    project_root = Path(os.getenv("POKERGPU_PROJECT_ROOT", Path.cwd())).resolve()
    data_dir = Path(os.getenv("POKERGPU_DATA_DIR", project_root / "data")).resolve()
    artifact_dir = Path(
        os.getenv("POKERGPU_ARTIFACT_DIR", project_root / "artifacts")
    ).resolve()
    log_level = os.getenv("POKERGPU_LOG_LEVEL", "INFO").upper()
    device = os.getenv("POKERGPU_DEVICE", "auto").lower()
    max_depth = int(os.getenv("POKERGPU_MAX_DEPTH", "1"))
    max_nodes = int(os.getenv("POKERGPU_MAX_NODES", "256"))
    min_reach_prob = float(os.getenv("POKERGPU_MIN_REACH_PROB", "0.0"))
    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        log_level=log_level,
        device=device,
        max_depth=max_depth,
        max_nodes=max_nodes,
        min_reach_prob=min_reach_prob,
    )
