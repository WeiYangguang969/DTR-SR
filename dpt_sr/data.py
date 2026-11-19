"""Pinned public datasets and byte-verified reproduction of upstream LOO files."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S3_PIN = "2a81540ae18615d88ef88227b0c066e5b74781e5"
S3_BASE = f"https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/{S3_PIN}/data/"
DATASETS = {
    "beauty": ("s3_beauty", "Beauty.txt", S3_BASE + "Beauty.txt"),
    "sports": ("s3_sports", "Sports_and_Outdoors.txt", S3_BASE + "Sports_and_Outdoors.txt"),
    "toys": ("s3_toys", "Toys_and_Games.txt", S3_BASE + "Toys_and_Games.txt"),
    "ml1m": ("ml_1m", "ml-1m.zip", "https://files.grouplens.org/datasets/movielens/ml-1m.zip"),
    "ml20m": ("ml_20m", "ml-20m.zip", "https://files.grouplens.org/datasets/movielens/ml-20m.zip"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest() -> dict:
    return json.loads((ROOT / "data/manifest.json").read_text())


def verify_file(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}\nexpected={expected}\nactual={actual}")


def download(url: str, path: Path, expected: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify_file(path, expected)
        print(f"RAW verified {path}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"DOWNLOAD {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "PACE-SR-reproduction/1.0"})
            with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as out:
                total = int(source.headers.get("Content-Length", 0))
                done, last = 0, time.monotonic()
                for block in iter(lambda: source.read(1 << 20), b""):
                    out.write(block)
                    done += len(block)
                    if time.monotonic() - last > 5:
                        print(f"DOWNLOAD {done / 2**20:.1f} / {total / 2**20:.1f} MiB", flush=True)
                        last = time.monotonic()
            verify_file(partial, expected)
            partial.replace(path)
            return
        except (OSError, ValueError) as e:
            if attempt == 2:
                raise RuntimeError(f"Download failed. Manually save {url} as {path}: {e}") from e
            print(f"Retry {attempt + 1}: {e}", flush=True)


def leave_one_out(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equivalent to pinned LastNSplitter(n=1), preserving original row order.

    Ties go to the last row. Cold users/items and seen targets are filtered
    after removing one last event per user; removed targets are not reinserted.
    """
    # Ascending rank + size is needed: descending rank(method=first) picks the
    # first tied event instead of the LAST one used by the upstream splitter.
    ascending = frame.groupby("user_id")["datetime"].rank(method="first", ascending=True)
    sizes = frame.groupby("user_id")["item_id"].transform("size")
    mask = (sizes - ascending + 1) == 1
    train, held = frame.loc[~mask].copy(), frame.loc[mask].copy()
    held = held[held.user_id.isin(train.user_id.unique()) & held.item_id.isin(train.item_id.unique())]
    pairs = pd.MultiIndex.from_frame(train[["user_id", "item_id"]])
    held_pairs = pd.MultiIndex.from_frame(held[["user_id", "item_id"]])
    held = held.loc[~held_pairs.isin(pairs)].copy()
    return train.reset_index(drop=True), held.reset_index(drop=True)


def raw_frame(dataset: str, path: Path) -> pd.DataFrame:
    if dataset in ("beauty", "sports", "toys"):
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                values = list(map(int, line.split()))
                rows.extend((values[0], item, t) for t, item in enumerate(values[1:], 1))
        frame = pd.DataFrame(rows, columns=["user_id", "item_id", "datetime"])
        frame["weight"] = 1
    else:
        with zipfile.ZipFile(path) as z:
            if dataset == "ml1m":
                with z.open("ml-1m/ratings.dat") as f:
                    frame = pd.read_csv(f, sep="::", engine="python",
                                        names=["userId", "movieId", "rating", "timestamp"])
                frame = frame[frame.rating >= 1.0].copy()
            else:
                with z.open("ml-20m/ratings.csv") as f:
                    frame = pd.read_csv(f)
                frame = frame[frame.rating >= 0.0].copy()
        frame = frame.rename(columns={"userId": "user_id", "movieId": "item_id", "timestamp": "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"], unit="s")
        frame["weight"] = 1
    # RecTools Interactions mutates these columns before the upstream CSV save.
    frame["weight"] = frame["weight"].astype(float)
    frame["datetime"] = frame["datetime"].astype("datetime64[ns]")
    return frame


def processed_paths(dataset: str, data_dir: Path) -> tuple[Path, Path]:
    folder = Path(data_dir) / DATASETS[dataset][0] / "leave_one_out"
    return folder / "train.csv", folder / "holdout.csv"


def verify_processed(dataset: str, data_dir: Path) -> dict:
    result = {}
    m = manifest()["processed"]
    for path in processed_paths(dataset, data_dir):
        key = f"{DATASETS[dataset][0]}/leave_one_out/{path.name}"
        verify_file(path, m[key]["sha256"])
        result[path.name] = {**m[key], "verified": True}
    return result


def prepare(dataset: str, data_dir: Path, raw_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir)
    paths = processed_paths(dataset, data_dir)
    if all(p.exists() for p in paths):
        result = verify_processed(dataset, data_dir)
        print(f"DATA verified {dataset}", flush=True)
        return result
    _, filename, url = DATASETS[dataset]
    raw = Path(raw_dir) if raw_dir else data_dir / "raw"
    path = raw / filename
    download(url, path, manifest()["public_raw"][filename])
    train, held = leave_one_out(raw_frame(dataset, path))
    for frame, output in zip((train, held), paths):
        output.parent.mkdir(parents=True, exist_ok=True)
        # Explicit LF makes Windows files byte-identical to the Linux manifest.
        frame.to_csv(output, index=False, lineterminator="\n")
    result = verify_processed(dataset, data_dir)
    print(f"DATA {dataset}: train={len(train)} holdout={len(held)}; both SHA-256 verified", flush=True)
    return result


def read_interactions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"user_id": "int64", "item_id": "int64"}, parse_dates=["datetime"])
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def load_split(dataset: str, data_dir: Path, split: str):
    verified = verify_processed(dataset, data_dir)
    train_path, test_path = processed_paths(dataset, data_dir)
    train = read_interactions(train_path)
    if split == "valid":
        train, held = leave_one_out(train)
    elif split == "test":
        held = read_interactions(test_path)
    else:
        raise ValueError("split must be valid or test")
    if not len(held):
        raise ValueError("No eligible holdout users")
    return train, held, verified
