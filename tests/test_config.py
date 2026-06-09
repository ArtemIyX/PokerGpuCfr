from pathlib import Path

from pokergpu.config import load_settings


def test_load_settings_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("POKERGPU_PROJECT_ROOT", str(Path("C:/tmp/pokergpu-root")))
    monkeypatch.setenv("POKERGPU_DATA_DIR", str(Path("C:/tmp/pokergpu-data")))
    monkeypatch.setenv("POKERGPU_ARTIFACT_DIR", str(Path("C:/tmp/pokergpu-artifacts")))
    monkeypatch.setenv("POKERGPU_LOG_LEVEL", "debug")
    monkeypatch.setenv("POKERGPU_DEVICE", "cpu")

    settings = load_settings()

    assert settings.project_root == Path("C:/tmp/pokergpu-root").resolve()
    assert settings.data_dir == Path("C:/tmp/pokergpu-data").resolve()
    assert settings.artifact_dir == Path("C:/tmp/pokergpu-artifacts").resolve()
    assert settings.log_level == "DEBUG"
    assert settings.device == "cpu"
