import logging
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable

import numpy as np

from .abstraction.hands import PlayerRangeVectors, RangeVector
from .app import create_app
from .benchmarks import run_benchmark
from .cfr import (
    CFRVariant,
    KuhnCard,
    LeducRank,
    average_strategy_root_bet_probability,
    average_strategy_root_bet_probability_leduc,
    expected_game_value_for_average_strategy,
    expected_game_value_for_average_strategy_leduc,
    run_toy_game_comparison,
    train_kuhn_cfr,
    train_leduc_cfr,
)
from .core.betting import (
    BettingRoundState,
    BlindStructure,
    PlayerBet,
    PlayerIndex,
    PlayerStack,
    Pot,
    chips,
)
from .core.board import Board
from .core.state import GameState, PlayerState
from .eval import (
    EvalDeviceConfig,
    LeafEvaluator,
    LeafFeatureBatch,
    LeafValueBatch,
    make_leaf_evaluator,
)
from .runtime import PostflopResolveSpec, resolve_postflop_hu
from .value_network import (
    DatasetManifestEntry,
    DatasetSplitRule,
    FeatureNormalizer,
    LabelNormalizer,
    ValueDatasetSample,
    ValueFeatureSpec,
    ValueTargetKind,
    build_value_label,
    export_dataset_sample,
    fit_feature_normalizer,
    fit_label_normalizer,
    load_value_sample,
    save_dataset_manifest,
    save_feature_normalizer,
    save_label_normalizer,
    save_value_sample,
    scalar_ev_target,
)
from .value_network.train import TrainingConfig, train_baseline


def main() -> int:
    settings = create_app()
    logger = logging.getLogger(__name__)
    print("pokergpu: cli start")
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        result = run_benchmark("noop", lambda: None)
        print(
            "benchmark="
            f"{result.name} iterations={result.iterations} "
            f"seconds={result.total_seconds:.6f} "
            f"per_iter={result.seconds_per_iteration:.9f}"
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "compare-toy":
        output_path, kuhn_iterations, leduc_iterations, variants = _parse_compare_args(
            sys.argv[2:],
            default_output=settings.artifact_dir / "toy_game_comparison.csv",
        )
        run_toy_game_comparison(
            output_path=output_path,
            kuhn_iterations=kuhn_iterations,
            leduc_iterations=leduc_iterations,
            variants=variants,
            progress_callback=_print_progress,
        )
        print(f"output={output_path}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "kuhn":
        iterations, variant = _parse_solver_args(sys.argv[2:], default_iterations=2000)
        store = train_kuhn_cfr(iterations, variant=variant)
        print(f"variant={variant.value}")
        print(f"iterations={iterations}")
        print(
            "avg_value_p0="
            f"{expected_game_value_for_average_strategy(store):.12f}"
        )
        print(
            "root_bet_J="
            f"{average_strategy_root_bet_probability(store, KuhnCard.JACK):.12f}"
        )
        print(
            "root_bet_Q="
            f"{average_strategy_root_bet_probability(store, KuhnCard.QUEEN):.12f}"
        )
        print(
            "root_bet_K="
            f"{average_strategy_root_bet_probability(store, KuhnCard.KING):.12f}"
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "leduc":
        iterations, variant = _parse_solver_args(sys.argv[2:], default_iterations=800)
        store = train_leduc_cfr(iterations, variant=variant)
        jack_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.JACK)
        queen_bet = average_strategy_root_bet_probability_leduc(
            store,
            LeducRank.QUEEN,
        )
        king_bet = average_strategy_root_bet_probability_leduc(store, LeducRank.KING)
        print(f"variant={variant.value}")
        print(f"iterations={iterations}")
        print(
            "avg_value_p0="
            f"{expected_game_value_for_average_strategy_leduc(store):.12f}"
        )
        print("root_bet_J=" f"{jack_bet:.12f}")
        print("root_bet_Q=" f"{queen_bet:.12f}")
        print("root_bet_K=" f"{king_bet:.12f}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "postflop-resolve":
        print("pokergpu: postflop resolve")
        demo_spec = _demo_postflop_spec(Random())
        demo_evaluator = _DemoBiasedLeafEvaluator(
            make_leaf_evaluator(EvalDeviceConfig(mode="cuda"))
        )
        resolve_result = resolve_postflop_hu(
            demo_spec,
            evaluator=demo_evaluator,
        )
        print(f"board={demo_spec.state.board}")
        print(f"root_infoset_id={resolve_result.root_infoset_id}")
        print(f"root_actions={','.join(resolve_result.root_actions)}")
        print(
            "root_strategy="
            + ",".join(
                f"{float(value):.6f}" for value in resolve_result.root_strategy
            )
        )
        print(f"iterations={resolve_result.iterations}")
        print(f"node_count={resolve_result.node_count}")
        print(f"leaf_count={resolve_result.leaf_count}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "generate-value-data":
        value_data_args = _parse_value_data_args(sys.argv[2:])
        _generate_value_data(value_data_args)
        print(f"manifest={value_data_args.manifest_path}")
        print(f"dataset_dir={value_data_args.output_dir}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "export-solver-labels":
        solver_label_args = _parse_solver_label_args(sys.argv[2:])
        started_at = time.monotonic()
        device_name = _runtime_device_name()
        print("pokergpu: export-solver-labels start")
        print(f"device={device_name}")
        print(
            "args="
            + json.dumps(
                {
                    "output_dir": str(solver_label_args.output_dir),
                    "manifest_path": str(solver_label_args.manifest_path),
                    "sample_count": solver_label_args.sample_count,
                    "player_count": solver_label_args.player_count,
                    "validation_modulo": solver_label_args.validation_modulo,
                    "validation_remainder": solver_label_args.validation_remainder,
                    "solve_time_sec": solver_label_args.solve_time_sec,
                    "max_depth": solver_label_args.max_depth,
                    "max_nodes": solver_label_args.max_nodes,
                },
                sort_keys=True,
            )
        )
        _export_solver_labels(solver_label_args, progress_callback=_print_progress)
        elapsed = time.monotonic() - started_at
        print(f"manifest={solver_label_args.manifest_path}")
        print(f"dataset_dir={solver_label_args.output_dir}")
        print(f"elapsed_seconds={elapsed:.3f}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "train-value-network":
        train_args = _parse_value_train_args(sys.argv[2:])
        started_at = time.monotonic()
        device_name = _runtime_device_name()
        print("pokergpu: train-value-network start")
        print(f"device={device_name}")
        print(
            "args="
            + json.dumps(
                {
                    "manifest_path": str(train_args.manifest_path),
                    "dataset_dir": str(train_args.dataset_dir),
                    "output_dir": str(train_args.output_dir),
                    "player_count": train_args.feature_spec.player_count,
                    "feature_history": train_args.feature_spec.max_history_length,
                    "epochs": train_args.training_config.epochs,
                    "batch_size": train_args.training_config.batch_size,
                    "learning_rate": train_args.training_config.learning_rate,
                    "validation_modulo": train_args.training_config.validation_modulo,
                    "validation_remainder": train_args.training_config.validation_remainder,
                },
                sort_keys=True,
            )
        )
        training_result = train_baseline(
            manifest_path=train_args.manifest_path,
            dataset_dir=train_args.dataset_dir,
            output_dir=train_args.output_dir,
            feature_spec=train_args.feature_spec,
            target_kind=ValueTargetKind.SCALAR_EV,
            config=train_args.training_config,
            normalizer=train_args.normalizer,
            progress_callback=_print_progress,
        )
        print(f"train_loss={training_result.train_loss:.6f}")
        print(f"val_loss={training_result.val_loss:.6f}")
        print(f"checkpoint={train_args.output_dir / 'best_checkpoint.pt'}")
        if training_result.predictions.size > 0:
            print(
                "preview="
                + ",".join(
                    f"{float(value):.6f}"
                    for value in training_result.predictions[0]
                )
            )
        print(f"elapsed_seconds={time.monotonic() - started_at:.3f}")
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "export-curated-solver-labels":
        curated_args = _parse_curated_solver_label_args(sys.argv[2:])
        started_at = time.monotonic()
        print("pokergpu: export-curated-solver-labels start")
        print(f"device={_runtime_device_name()}")
        print(
            "args="
            + json.dumps(
                {
                    "output_dir": str(curated_args.output_dir),
                    "manifest_path": str(curated_args.manifest_path),
                    "limit": curated_args.limit,
                },
                sort_keys=True,
            )
        )
        _export_curated_solver_labels(curated_args, progress_callback=_print_progress)
        print(f"elapsed_seconds={time.monotonic() - started_at:.3f}")
        print(f"manifest={curated_args.manifest_path}")
        print(f"dataset_dir={curated_args.output_dir}")
        return 0
    logger.info("PokerGPU initialized")
    print(f"PokerGPU ready on device={settings.device}")
    return 0


@dataclass(frozen=True, slots=True)
class _ValueDataArgs:
    output_dir: Path
    manifest_path: Path
    sample_count: int
    player_count: int
    feature_history: int
    validation_modulo: int
    validation_remainder: int

    @property
    def feature_spec(self) -> ValueFeatureSpec:
        return ValueFeatureSpec(
            player_count=self.player_count,
            max_history_length=self.feature_history,
        )


@dataclass(frozen=True, slots=True)
class _ValueTrainArgs:
    manifest_path: Path
    dataset_dir: Path
    output_dir: Path
    feature_spec: ValueFeatureSpec
    training_config: TrainingConfig
    normalizer: FeatureNormalizer | None


@dataclass(frozen=True, slots=True)
class _SolverLabelArgs:
    output_dir: Path
    manifest_path: Path
    sample_count: int
    player_count: int
    validation_modulo: int
    validation_remainder: int
    solve_time_sec: float
    max_depth: int
    max_nodes: int


@dataclass(frozen=True, slots=True)
class _CuratedSolverLabelArgs:
    output_dir: Path
    manifest_path: Path
    limit: int


def _parse_value_data_args(args: list[str]) -> _ValueDataArgs:
    output_dir = Path("artifacts/value_data")
    manifest_path = output_dir / "manifest.json"
    sample_count = 32
    player_count = 2
    feature_history = 8
    validation_modulo = 10
    validation_remainder = 0
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--output-dir" and index + 1 < len(args):
            output_dir = Path(args[index + 1]).resolve()
            manifest_path = output_dir / "manifest.json"
            index += 2
            continue
        if option == "--manifest" and index + 1 < len(args):
            manifest_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--samples" and index + 1 < len(args):
            sample_count = int(args[index + 1])
            index += 2
            continue
        if option == "--players" and index + 1 < len(args):
            player_count = int(args[index + 1])
            index += 2
            continue
        if option == "--history" and index + 1 < len(args):
            feature_history = int(args[index + 1])
            index += 2
            continue
        if option == "--val-mod" and index + 1 < len(args):
            validation_modulo = int(args[index + 1])
            index += 2
            continue
        if option == "--val-rem" and index + 1 < len(args):
            validation_remainder = int(args[index + 1])
            index += 2
            continue
        raise ValueError(f"invalid generate-value-data arguments: {args!r}")
    return _ValueDataArgs(
        output_dir=output_dir,
        manifest_path=manifest_path,
        sample_count=sample_count,
        player_count=player_count,
        feature_history=feature_history,
        validation_modulo=validation_modulo,
        validation_remainder=validation_remainder,
    )


def _parse_value_train_args(args: list[str]) -> _ValueTrainArgs:
    manifest_path = Path("artifacts/value_data/manifest.json").resolve()
    dataset_dir = Path("artifacts/value_data").resolve()
    output_dir = Path("artifacts/value_run").resolve()
    player_count = 2
    feature_history = 8
    epochs = 3
    batch_size = 8
    learning_rate = 1e-3
    normalizer: FeatureNormalizer | None = None
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--manifest" and index + 1 < len(args):
            manifest_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--dataset-dir" and index + 1 < len(args):
            dataset_dir = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--output-dir" and index + 1 < len(args):
            output_dir = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--players" and index + 1 < len(args):
            player_count = int(args[index + 1])
            index += 2
            continue
        if option == "--history" and index + 1 < len(args):
            feature_history = int(args[index + 1])
            index += 2
            continue
        if option == "--epochs" and index + 1 < len(args):
            epochs = int(args[index + 1])
            index += 2
            continue
        if option == "--batch-size" and index + 1 < len(args):
            batch_size = int(args[index + 1])
            index += 2
            continue
        if option == "--lr" and index + 1 < len(args):
            learning_rate = float(args[index + 1])
            index += 2
            continue
        if option == "--normalizer" and index + 1 < len(args):
            normalizer = FeatureNormalizer.from_json(
                json.loads(Path(args[index + 1]).read_text(encoding="utf-8"))
            )
            index += 2
            continue
        raise ValueError(f"invalid train-value-network arguments: {args!r}")
    return _ValueTrainArgs(
        manifest_path=manifest_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        feature_spec=ValueFeatureSpec(
            player_count=player_count,
            max_history_length=feature_history,
        ),
        training_config=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ),
        normalizer=normalizer,
    )


def _parse_solver_label_args(args: list[str]) -> _SolverLabelArgs:
    output_dir = Path("artifacts/solver_labels").resolve()
    manifest_path = output_dir / "manifest.json"
    sample_count = 8
    player_count = 2
    validation_modulo = 10
    validation_remainder = 0
    solve_time_sec = 0.25
    max_depth = 3
    max_nodes = 128
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--output-dir" and index + 1 < len(args):
            output_dir = Path(args[index + 1]).resolve()
            manifest_path = output_dir / "manifest.json"
            index += 2
            continue
        if option == "--manifest" and index + 1 < len(args):
            manifest_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--samples" and index + 1 < len(args):
            sample_count = int(args[index + 1])
            index += 2
            continue
        if option == "--players" and index + 1 < len(args):
            player_count = int(args[index + 1])
            index += 2
            continue
        if option == "--val-mod" and index + 1 < len(args):
            validation_modulo = int(args[index + 1])
            index += 2
            continue
        if option == "--val-rem" and index + 1 < len(args):
            validation_remainder = int(args[index + 1])
            index += 2
            continue
        if option == "--time" and index + 1 < len(args):
            solve_time_sec = float(args[index + 1])
            index += 2
            continue
        if option == "--depth" and index + 1 < len(args):
            max_depth = int(args[index + 1])
            index += 2
            continue
        if option == "--nodes" and index + 1 < len(args):
            max_nodes = int(args[index + 1])
            index += 2
            continue
        raise ValueError(f"invalid export-solver-labels arguments: {args!r}")
    return _SolverLabelArgs(
        output_dir=output_dir,
        manifest_path=manifest_path,
        sample_count=sample_count,
        player_count=player_count,
        validation_modulo=validation_modulo,
        validation_remainder=validation_remainder,
        solve_time_sec=solve_time_sec,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def _parse_curated_solver_label_args(args: list[str]) -> _CuratedSolverLabelArgs:
    output_dir = Path("artifacts/curated_solver_labels").resolve()
    manifest_path = output_dir / "manifest.json"
    limit = 9
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--output-dir" and index + 1 < len(args):
            output_dir = Path(args[index + 1]).resolve()
            manifest_path = output_dir / "manifest.json"
            index += 2
            continue
        if option == "--manifest" and index + 1 < len(args):
            manifest_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--limit" and index + 1 < len(args):
            limit = int(args[index + 1])
            index += 2
            continue
        raise ValueError(f"invalid export-curated-solver-labels arguments: {args!r}")
    return _CuratedSolverLabelArgs(
        output_dir=output_dir,
        manifest_path=manifest_path,
        limit=limit,
    )


def _generate_value_data(args: _ValueDataArgs) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(1337)
    feature_spec = args.feature_spec
    target = scalar_ev_target(args.player_count)
    entries: list[DatasetManifestEntry] = []
    samples: list[ValueDatasetSample] = []
    normalizer_probe = np.zeros(
        1 + 1 + 5 + args.player_count + (args.player_count * 1326) + args.player_count + args.feature_history,
        dtype=np.float32,
    )
    for index in range(args.sample_count):
        features = normalizer_probe.copy()
        features[0] = np.float32(rng.randrange(4))
        features[1] = np.float32(100.0 + 10.0 * rng.randrange(1, 8))
        features[2:7] = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        label_values = np.asarray(
            [rng.uniform(-1.0, 1.0) for _ in range(args.player_count)],
            dtype=np.float32,
        )
        sample = ValueDatasetSample(
            sample_id=f"spot-{index:05d}",
            features=features,
            label=build_value_label(label_values, target).values,
            metadata={
                "street": "synthetic",
                "index": index,
            },
        )
        split_rule = DatasetSplitRule(
            validation_modulo=args.validation_modulo,
            validation_remainder=args.validation_remainder,
        )
        split = split_rule.split(sample.sample_id)
        relative_path = f"{split}/{sample.sample_id}.npz"
        samples.append(sample)
        entries.append(
            DatasetManifestEntry(
                sample_id=sample.sample_id,
                split=split,
                path=relative_path,
                feature_count=sample.features.shape[0],
                label_shape=(sample.label.shape[0], sample.label.shape[1]),
            )
        )
        save_value_sample(sample, args.output_dir / relative_path)
    normalizer = fit_feature_normalizer(samples)
    label_normalizer = fit_label_normalizer(samples)
    save_dataset_manifest(entries, args.manifest_path)
    save_feature_normalizer(normalizer, args.output_dir / "normalizer.json")
    save_label_normalizer(label_normalizer, args.output_dir / "label_normalizer.json")


def _export_solver_labels(
    args: _SolverLabelArgs,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    from .runtime.postflop import PostflopResolveSpec, resolve_postflop_hu

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[DatasetManifestEntry] = []
    samples: list[ValueDatasetSample] = []
    split_rule = DatasetSplitRule(
        validation_modulo=args.validation_modulo,
        validation_remainder=args.validation_remainder,
    )
    feature_spec = ValueFeatureSpec(player_count=args.player_count, max_history_length=8)
    target = scalar_ev_target(args.player_count)
    for index in range(args.sample_count):
        spec = _demo_postflop_spec(Random(9000 + index))
        spec = PostflopResolveSpec(
            state=spec.state,
            range_p0=spec.range_p0,
            range_p1=spec.range_p1,
            time_budget_sec=args.solve_time_sec,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            min_reach_prob=spec.min_reach_prob,
            cache_state=spec.cache_state,
        )
        result = resolve_postflop_hu(spec)
        entry = export_dataset_sample(
            sample_id=f"solver-{index:05d}",
            state=spec.state,
            ranges=PlayerRangeVectors.from_values((spec.range_p0, spec.range_p1)),
            label=build_value_label(
                [result.root_ev_player0, result.root_ev_player1],
                target,
            ),
            feature_spec=feature_spec,
            output_dir=args.output_dir,
            split_rule=split_rule,
            metadata={
                "solver": "postflop_hu",
                "iterations": result.iterations,
                "elapsed_seconds": result.elapsed_seconds,
                "root_infoset_id": result.root_infoset_id,
                "root_actions": result.root_actions,
            },
        )
        entries.append(entry)
        loaded = load_value_sample(args.output_dir / entry.path)
        samples.append(loaded)
        if progress_callback is not None:
            progress_callback(index + 1, args.sample_count, "export labels")
    normalizer = fit_feature_normalizer(samples)
    label_normalizer = fit_label_normalizer(samples)
    save_dataset_manifest(entries, args.manifest_path)
    save_feature_normalizer(normalizer, args.output_dir / "normalizer.json")
    save_label_normalizer(label_normalizer, args.output_dir / "label_normalizer.json")


def _export_curated_solver_labels(
    args: _CuratedSolverLabelArgs,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    from .value_network.dataset import curated_solver_spots
    from .runtime.postflop import PostflopResolveSpec, resolve_postflop_hu

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spots = curated_solver_spots()[: args.limit]
    entries: list[DatasetManifestEntry] = []
    samples: list[ValueDatasetSample] = []
    split_rule = DatasetSplitRule()
    feature_spec = ValueFeatureSpec(player_count=2, max_history_length=8)
    target = scalar_ev_target(2)
    for index, spot in enumerate(spots):
        spec = PostflopResolveSpec(
            state=spot.state,
            range_p0=RangeVector.uniform(),
            range_p1=RangeVector.uniform(),
            time_budget_sec=0.25,
            max_depth=3,
            max_nodes=128,
        )
        result = resolve_postflop_hu(spec)
        entry = export_dataset_sample(
            sample_id=f"curated-{index:05d}",
            state=spot.state,
            ranges=PlayerRangeVectors.from_values((spec.range_p0, spec.range_p1)),
            label=build_value_label(
                [result.root_ev_player0, result.root_ev_player1],
                target,
            ),
            feature_spec=feature_spec,
            output_dir=args.output_dir,
            split_rule=split_rule,
            metadata={
                "solver": "postflop_hu",
                "board_texture": spot.board_texture,
                "pot_bucket": spot.pot_bucket,
                "stack_bucket": spot.stack_bucket,
                "action_line": spot.action_line,
            },
        )
        entries.append(entry)
        samples.append(load_value_sample(args.output_dir / entry.path))
        if progress_callback is not None:
            progress_callback(index + 1, len(spots), "curated export")
    normalizer = fit_feature_normalizer(samples)
    label_normalizer = fit_label_normalizer(samples)
    save_dataset_manifest(entries, args.manifest_path)
    save_feature_normalizer(normalizer, args.output_dir / "normalizer.json")
    save_label_normalizer(label_normalizer, args.output_dir / "label_normalizer.json")


def _demo_postflop_spec(rng: Random | None = None) -> PostflopResolveSpec:
    rng = rng or Random()
    boards = ("AhKdTc", "QsJh9d", "Ac7c2d", "KhQh8s")
    board = Board.from_str(boards[rng.randrange(len(boards))])
    pot = chips(200 + 100 * rng.randrange(1, 4))
    stack_p0 = chips(1500 + 100 * rng.randrange(0, 6))
    stack_p1 = chips(1500 + 100 * rng.randrange(0, 6))
    state = GameState(
        board=board,
        players=(
            PlayerState(player=PlayerIndex(0)),
            PlayerState(player=PlayerIndex(1)),
        ),
        betting_round=BettingRoundState(
            pot=Pot(amount=pot),
            stacks=(
                PlayerStack(player=PlayerIndex(0), stack=stack_p0),
                PlayerStack(player=PlayerIndex(1), stack=stack_p1),
            ),
            bets=(
                PlayerBet(player=PlayerIndex(0), committed=chips(0)),
                PlayerBet(player=PlayerIndex(1), committed=chips(0)),
            ),
            blinds=BlindStructure(small_blind=chips(50), big_blind=chips(100)),
            to_act=PlayerIndex(0),
        ),
        dealer=PlayerIndex(0),
    )
    return PostflopResolveSpec(
        state=state,
        range_p0=RangeVector.uniform(),
        range_p1=RangeVector.uniform(),
        time_budget_sec=0.5,
        max_depth=3,
        max_nodes=128,
    )


class _DemoBiasedLeafEvaluator(LeafEvaluator):
    def __init__(self, base: LeafEvaluator) -> None:
        self._base = base

    def evaluate(self, batch: LeafFeatureBatch) -> LeafValueBatch:
        values = self._base.evaluate(batch)
        bias = np.asarray(batch.node_indices, dtype=np.float32) * np.float32(0.01)
        return LeafValueBatch(
            ev_player0=values.ev_player0 + bias,
            ev_player1=values.ev_player1 - bias,
        )


def _parse_solver_args(
    args: list[str],
    default_iterations: int,
) -> tuple[int, CFRVariant]:
    iterations = default_iterations
    variant = CFRVariant.VANILLA
    index = 0
    if index < len(args) and not args[index].startswith("--"):
        iterations = int(args[index])
        index += 1
    while index < len(args):
        if args[index] != "--variant" or index + 1 >= len(args):
            raise ValueError(f"invalid solver arguments: {args!r}")
        variant = CFRVariant(args[index + 1])
        index += 2
    return iterations, variant


def _parse_compare_args(
    args: list[str],
    default_output: Path,
) -> tuple[Path, tuple[int, ...], tuple[int, ...], tuple[CFRVariant, ...]]:
    output_path = default_output
    kuhn_iterations: tuple[int, ...] = (100, 500, 1000, 2000)
    leduc_iterations: tuple[int, ...] = (100, 300, 800)
    variants: tuple[CFRVariant, ...] = (
        CFRVariant.VANILLA,
        CFRVariant.CFR_PLUS,
        CFRVariant.DCFR,
    )
    index = 0
    while index < len(args):
        option = args[index]
        if option == "--output" and index + 1 < len(args):
            output_path = Path(args[index + 1]).resolve()
            index += 2
            continue
        if option == "--kuhn" and index + 1 < len(args):
            kuhn_iterations = _parse_iteration_list(args[index + 1])
            index += 2
            continue
        if option == "--leduc" and index + 1 < len(args):
            leduc_iterations = _parse_iteration_list(args[index + 1])
            index += 2
            continue
        if option == "--variants" and index + 1 < len(args):
            variants = tuple(CFRVariant(value) for value in args[index + 1].split(","))
            index += 2
            continue
        raise ValueError(f"invalid compare arguments: {args!r}")
    return output_path, kuhn_iterations, leduc_iterations, variants


def _parse_iteration_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value.split(",") if item)
    if not items:
        raise ValueError("iteration list must not be empty")
    return items


def _print_progress(current: int, total: int, label: str) -> None:
    width = 24
    filled = width if total == 0 else int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rprogress [{bar}] {current}/{total} {label}",
        end="" if current < total else "\n",
        flush=True,
    )


def _runtime_device_name() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
