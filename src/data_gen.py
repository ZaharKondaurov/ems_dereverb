"""Data generation utilities (notebooks/sandbox.ipynb)."""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import choice, uniform
from typing import Any, Optional

import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

WALLS_KEYWORDS = [
    "hard_surface",
    "ceramic_tiles",
    "plasterboard",
    "wooden_lining",
    "glass_3mm",
]
FLOOR_KEYWORDS = ["linoleum_on_concrete", "carpet_cotton"]
CEILING_KEYWORDS = [
    "ceiling_plasterboard",
    "ceiling_fissured_tile",
    "ceiling_metal_panel",
]
DEFAULT_ROOM_RELS = [[1, 3], [1, 3], [2, 3], [1, 1]]

RIR_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "output_dir": "data/rirs48_small",
        "room_square": (15.0, 30.0),
        "room_height": (2.5, 3.0),
        "rt60": (0.4, 0.5),
        "count": 200,
    },
    "medium": {
        "output_dir": "data/rirs48_medium",
        "room_square": (30.0, 80.0),
        "room_height": (2.5, 3.0),
        "rt60": (0.5, 0.7),
        "count": 400,
    },
    "large": {
        "output_dir": "data/rirs48_large",
        "room_square": (80.0, 120.0),
        "room_height": (2.5, 3.0),
        "rt60": (0.70, 0.85),
        "count": 160,
    },
    "super_large": {
        "output_dir": "data/rirs48_super_large",
        "room_square": (80.0, 120.0),
        "room_height": (2.5, 3.0),
        "rt60": (0.85, 1.0),
        "count": 80,
    },
}

META_HEADERS = [
    "filepath",
    "width",
    "length",
    "height",
    "rt60",
    "source_coords",
    "mic_coords",
    "wall_material",
    "floor_material",
    "ceil_material",
]


@dataclass
class RIRMetadata:
    filepath: str
    width: float
    length: float
    height: float
    rt60: float
    source_coords: list[float]
    mic_coords: list[float]
    wall_material: str
    floor_material: str
    ceil_material: str

    def as_row(self) -> list[Any]:
        return [
            self.filepath,
            f"{self.width:.4f}",
            f"{self.length:.4f}",
            f"{self.height:.4f}",
            f"{self.rt60:.4f}",
            [round(x, 4) for x in self.source_coords],
            [round(x, 4) for x in self.mic_coords],
            self.wall_material,
            self.floor_material,
            self.ceil_material,
        ]


def split_dir_recursively(
    src_dir: Path,
    train_dir: Path,
    test_dir: Path,
    *,
    partition: float = 0.7,
    seed: Optional[int] = None,
) -> None:
    """Recursively split files in src_dir into train_dir and test_dir."""
    split_done = False
    for item in src_dir.iterdir():
        if item.is_dir():
            train_sub = train_dir / item.name
            test_sub = test_dir / item.name
            train_sub.mkdir(parents=True, exist_ok=True)
            test_sub.mkdir(parents=True, exist_ok=True)
            split_dir_recursively(
                item, train_sub, test_sub, partition=partition, seed=seed
            )
        elif item.is_file() and not split_done:
            files = [p for p in src_dir.iterdir() if p.is_file()]
            rng = random.Random(seed)
            rng.shuffle(files)
            n_train = int(partition * len(files))
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            for i, file in enumerate(files):
                dest = train_dir / file.name if i < n_train else test_dir / file.name
                shutil.copy(str(file), str(dest))
            split_done = True


def split_demand_noise(
    *,
    src_dir: str | Path,
    train_dir: str | Path,
    val_test_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    train_fraction: float = 0.7,
    val_fraction: float = 0.5,
    seed: int = 1984,
) -> None:
    """Split DEMAND noise: train / val / test (notebooks/sandbox.ipynb cell 5)."""
    src_dir = Path(src_dir)
    train_dir = Path(train_dir)
    val_test_dir = Path(val_test_dir)
    val_dir = Path(val_dir)
    test_dir = Path(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_test_dir.mkdir(parents=True, exist_ok=True)
    split_dir_recursively(
        src_dir, train_dir, val_test_dir, partition=train_fraction, seed=seed
    )

    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    split_dir_recursively(
        val_test_dir, val_dir, test_dir, partition=val_fraction, seed=seed + 1
    )
    print(f"Split DEMAND: {src_dir} -> {train_dir}, {val_dir}, {test_dir}")


def load_stimulus(
    stimulus_path: str | Path,
    sample_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    signal, sr = torchaudio.load(stimulus_path)
    if sr != sample_rate:
        signal = Resample(sr, sample_rate)(signal)
    signal_16k = Resample(sample_rate, 16_000)(signal)
    return signal, signal_16k


def simulate_rir_shoebox(
    signal: torch.Tensor,
    *,
    sample_rate: int,
    rt60_range: tuple[float, float],
    room_square: tuple[float, float],
    room_rels: list[list[int]],
    room_height: tuple[float, float],
) -> tuple[np.ndarray, RIRMetadata]:
    square = uniform(*room_square)
    rel = choice(room_rels)
    length = float(np.sqrt(square * (rel[0] / rel[1])))
    width = square / length
    height = uniform(*room_height)
    rt60 = uniform(*rt60_range)
    room_dim = [length, width, height]

    _, max_order = pra.inverse_sabine(rt60, room_dim)

    wall_mat = choice(WALLS_KEYWORDS)
    ceil_mat = choice(CEILING_KEYWORDS)
    floor_mat = choice(FLOOR_KEYWORDS)

    wall = pra.Material(wall_mat)
    ceil = pra.Material(ceil_mat)
    floor = pra.Material(floor_mat)
    material = {
        "east": wall,
        "west": wall,
        "north": wall,
        "south": wall,
        "ceiling": ceil,
        "floor": floor,
    }

    room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=material,
        max_order=max_order,
        use_rand_ism=True,
        max_rand_disp=0.05,
        ray_tracing=False,
    )

    source_locs = [uniform(1.0, length - 1), uniform(1.0, width - 1), uniform(1.0, 2.0)]
    mic_locs = np.array([x * 0.98 for x in source_locs])[:, None]

    room.add_source(source_locs, signal=signal.squeeze(), delay=0.5)
    room.add_microphone_array(mic_locs)
    room.compute_rir()
    room.simulate()

    meta = RIRMetadata(
        filepath="",
        width=width,
        length=length,
        height=height,
        rt60=rt60,
        source_coords=source_locs,
        mic_coords=mic_locs.squeeze(-1).tolist(),
        wall_material=wall_mat,
        floor_material=floor_mat,
        ceil_material=ceil_mat,
    )
    return room.rir[0][0], meta


def generate_rir_dataset(
    *,
    output_dir: str | Path,
    stimulus_path: str | Path,
    sample_rate: int = 48_000,
    count: int,
    rt60_range: tuple[float, float],
    room_square: tuple[float, float],
    room_height: tuple[float, float],
    room_rels: Optional[list[list[int]]] = None,
    seed: int = 1984,
) -> Path:
    """Generate RIR wav files and meta.csv into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    signal, _ = load_stimulus(stimulus_path, sample_rate)
    room_rels = room_rels or DEFAULT_ROOM_RELS

    metadata: list[RIRMetadata] = []
    for i in tqdm(range(count), desc=f"RIR {output_dir.name}"):
        rir, meta = simulate_rir_shoebox(
            signal,
            sample_rate=sample_rate,
            rt60_range=rt60_range,
            room_square=room_square,
            room_rels=room_rels,
            room_height=room_height,
        )
        filepath = output_dir / f"rir_{i}.wav"
        torchaudio.save(
            str(filepath),
            src=torch.from_numpy(rir[None, :]),
            sample_rate=sample_rate,
            format="wav",
        )
        meta.filepath = str(filepath)
        metadata.append(meta)

    meta_path = output_dir / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(META_HEADERS)
        writer.writerows(meta.as_row() for meta in metadata)

    print(f"Generated {count} RIRs -> {output_dir} ({meta_path})")
    return output_dir


def split_rir_dataset(
    rir_dir: str | Path,
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.5,
    seed: int = 1984,
) -> dict[str, Path]:
    """Split RIR dir into train/val/test and copy meta.csv to each split."""
    rir_dir = Path(rir_dir)
    train_dir = Path(f"{rir_dir}_train")
    val_test_dir = Path(f"{rir_dir}_val_test")
    val_dir = Path(f"{rir_dir}_val")
    test_dir = Path(f"{rir_dir}_test")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_test_dir.mkdir(parents=True, exist_ok=True)
    split_dir_recursively(
        rir_dir, train_dir, val_test_dir, partition=train_fraction, seed=seed
    )

    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    split_dir_recursively(
        val_test_dir, val_dir, test_dir, partition=val_fraction, seed=seed + 1
    )

    meta_src = rir_dir / "meta.csv"
    if meta_src.is_file():
        for dest_dir in (train_dir, val_dir, test_dir):
            shutil.copy(str(meta_src), str(dest_dir / "meta.csv"))

    splits = {"train": train_dir, "val": val_dir, "test": test_dir}
    print(f"Split RIRs: {rir_dir} -> {train_dir.name}, {val_dir.name}, {test_dir.name}")
    return splits


def generate_and_split_rir_preset(
    preset_name: str,
    *,
    stimulus_path: str | Path,
    sample_rate: int = 48_000,
    seed: int = 1984,
    train_fraction: float = 0.7,
    val_fraction: float = 0.5,
    count: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Path]:
    preset = RIR_PRESETS[preset_name]
    rir_dir = generate_rir_dataset(
        output_dir=output_dir or preset["output_dir"],
        stimulus_path=stimulus_path,
        sample_rate=sample_rate,
        count=count if count is not None else preset["count"],
        rt60_range=tuple(preset["rt60"]),
        room_square=tuple(preset["room_square"]),
        room_height=tuple(preset["room_height"]),
        seed=seed,
    )
    return split_rir_dataset(
        rir_dir,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
