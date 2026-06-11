import sys
from pathlib import Path

import numpy as np
import pytest

from pokergpu import cli
from pokergpu.value_network.dataset import (
    DatasetManifestEntry,
    FeatureNormalizer,
    LabelNormalizer,
    ValueDatasetSample,
    save_dataset_manifest,
    save_feature_normalizer,
    save_label_normalizer,
    save_value_sample,
)


def test_kuhn_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pokergpu", "kuhn", "50", "--variant", "cfr_plus"],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "variant=cfr_plus" in captured.out
    assert "iterations=50" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out


def test_leduc_cli_prints_strategy_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pokergpu", "leduc", "20", "--variant", "dcfr"],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "variant=dcfr" in captured.out
    assert "iterations=20" in captured.out
    assert "avg_value_p0=" in captured.out
    assert "root_bet_J=" in captured.out
    assert "root_bet_Q=" in captured.out
    assert "root_bet_K=" in captured.out


def test_compare_toy_cli_writes_csv_with_progress(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "toy.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pokergpu",
            "compare-toy",
            "--output",
            str(output_path),
            "--kuhn",
            "1",
            "--leduc",
            "1",
            "--variants",
            "vanilla",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "progress [" in captured.out
    assert f"output={output_path}" in captured.out
    assert output_path.exists()
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_dataset_sanity_report_prints_stats(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.json"
    sample = ValueDatasetSample(
        sample_id="spot-1",
        features=np.asarray([1.0, 3.0], dtype=np.float32),
        label=np.asarray([[2.0, -2.0]], dtype=np.float32),
        metadata={},
    )
    save_value_sample(sample, dataset_dir / "train/spot-1.npz")
    save_dataset_manifest(
        [
            DatasetManifestEntry(
                sample_id="spot-1",
                split="train",
                path="train/spot-1.npz",
                feature_count=2,
                label_shape=(1, 2),
            )
        ],
        manifest_path,
    )
    save_feature_normalizer(
        FeatureNormalizer(
            mean=np.asarray([1.0, 3.0], dtype=np.float32),
            std=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
        dataset_dir / "normalizer.json",
    )
    save_label_normalizer(
        LabelNormalizer(
            mean=np.asarray([0.0, 0.0], dtype=np.float32),
            std=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
        dataset_dir / "label_normalizer.json",
    )

    exit_code = cli.main.__globals__["_print_dataset_sanity_report"](
        cli.main.__globals__["_DatasetSanityReportArgs"](
            manifest_path=manifest_path,
            dataset_dir=dataset_dir,
            feature_normalizer_path=dataset_dir / "normalizer.json",
            label_normalizer_path=dataset_dir / "label_normalizer.json",
            split_key="sample_id",
        )
    )
    captured = capsys.readouterr()

    assert exit_code is None
    assert "pokergpu: dataset-sanity-report" in captured.out
    assert "samples=1" in captured.out
    assert "feature_counts=2" in captured.out
