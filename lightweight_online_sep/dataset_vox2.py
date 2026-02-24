import csv
import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MixPair:
    partition: str
    source1_subset: str
    source1_speaker: str
    source1_utt: str
    source1_db: float
    source2_subset: str
    source2_speaker: str
    source2_utt: str
    source2_db: float
    duration: float


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_vox_path(root: str, subset: str, speaker: str, utt: str) -> str:
    return os.path.join(
        os.path.expanduser(root),
        "audio_clean",
        str(subset),
        str(speaker),
        f"{str(utt)}.wav",
    )


def _read_mixture_csv(csv_path: str) -> List[MixPair]:
    rows: List[MixPair] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_id, row in enumerate(reader, start=1):
            if len(row) < 10:
                continue
            try:
                rows.append(
                    MixPair(
                        partition=str(row[0]).strip().lower(),
                        source1_subset=str(row[1]).strip(),
                        source1_speaker=str(row[2]).strip(),
                        source1_utt=str(row[3]).strip(),
                        source1_db=_safe_float(row[4], 0.0),
                        source2_subset=str(row[5]).strip(),
                        source2_speaker=str(row[6]).strip(),
                        source2_utt=str(row[7]).strip(),
                        source2_db=_safe_float(row[8], 0.0),
                        duration=max(_safe_float(row[9], 0.0), 0.0),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Invalid csv row at line {line_id}: {row}") from exc
    if not rows:
        raise ValueError(f"No valid samples found in csv: {csv_path}")
    return rows


def _split_rows(rows: List[MixPair], split: str, valid_ratio: float, seed: int) -> List[MixPair]:
    split = str(split).lower()
    train_rows = [r for r in rows if r.partition == "train"]
    valid_rows = [r for r in rows if r.partition in {"val", "valid", "validation", "dev"}]
    test_rows = [r for r in rows if r.partition == "test"]

    if split == "test":
        if not test_rows:
            raise ValueError("CSV does not contain test partition")
        return test_rows

    if not train_rows:
        raise ValueError("CSV does not contain train partition")

    if not valid_rows:
        rng = random.Random(seed)
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        valid_count = max(1, int(round(len(train_rows) * valid_ratio)))
        valid_idx = set(idx[:valid_count])
        valid_rows = [train_rows[i] for i in idx[:valid_count]]
        train_rows = [train_rows[i] for i in idx if i not in valid_idx]

    if split == "train":
        return train_rows
    if split in {"valid", "val"}:
        return valid_rows
    raise ValueError(f"Unknown split: {split}")


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    ratio = float(dst_sr) / float(src_sr)
    new_len = max(1, int(round(len(audio) * ratio)))
    old_x = np.arange(len(audio), dtype=np.float32)
    new_x = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def _load_mono_segment(path: str, start: int, frames: int, target_sr: int) -> np.ndarray:
    with sf.SoundFile(path) as f:
        src_sr = int(f.samplerate)
        start = max(0, int(start))
        if start > len(f):
            return np.zeros(frames, dtype=np.float32)
        f.seek(start)
        audio = f.read(frames=frames, dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if src_sr != target_sr:
        audio = _resample_linear(audio, src_sr, target_sr)

    if len(audio) < frames:
        audio = np.pad(audio, (0, frames - len(audio)))
    elif len(audio) > frames:
        audio = audio[:frames]
    return audio


def _apply_relative_db(source1: np.ndarray, source2: np.ndarray, source1_db: float, source2_db: float) -> np.ndarray:
    rel_db = float(source2_db) - float(source1_db)
    p1 = float(np.mean(np.square(source1, dtype=np.float64)))
    p2 = float(np.mean(np.square(source2, dtype=np.float64)))
    if p1 <= 1e-12 or p2 <= 1e-12:
        return source2
    scalar = (10.0 ** (rel_db / 20.0)) * np.sqrt(p1 / p2)
    return source2 * float(scalar)


class VoxCeleb2MixDataset(Dataset):
    """Online dynamic 2-speaker mixture dataset from VoxCeleb2 2mix csv."""

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = "train",
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        valid_ratio: float = 0.1,
        seed: int = 42,
        train_random_offset: bool = True,
    ):
        self.csv_path = os.path.expanduser(csv_path)
        self.data_root = os.path.expanduser(data_root)
        self.sample_rate = int(sample_rate)
        self.segment_samples = max(1, int(round(float(segment_seconds) * self.sample_rate)))
        self.split = str(split).lower()
        self.train_random_offset = bool(train_random_offset)

        rows = _read_mixture_csv(self.csv_path)
        self.rows = _split_rows(rows, self.split, valid_ratio=valid_ratio, seed=seed)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _get_start_sample(self, duration_sec: float, index: int) -> int:
        duration_samples = max(self.segment_samples, int(round(float(duration_sec) * self.sample_rate)))
        max_start = max(0, duration_samples - self.segment_samples)

        if self.split == "train" and self.train_random_offset and max_start > 0:
            return self.rng.randint(0, max_start)
        if max_start <= 0:
            return 0
        # deterministic center crop for validation/test
        return max_start // 2

    def __getitem__(self, index: int):
        row = self.rows[index]
        start = self._get_start_sample(row.duration, index)

        s1_path = _build_vox_path(
            self.data_root,
            row.source1_subset,
            row.source1_speaker,
            row.source1_utt,
        )
        s2_path = _build_vox_path(
            self.data_root,
            row.source2_subset,
            row.source2_speaker,
            row.source2_utt,
        )

        try:
            source1 = _load_mono_segment(s1_path, start, self.segment_samples, self.sample_rate)
            source2 = _load_mono_segment(s2_path, start, self.segment_samples, self.sample_rate)
        except Exception:
            source1 = np.zeros(self.segment_samples, dtype=np.float32)
            source2 = np.zeros(self.segment_samples, dtype=np.float32)

        source2 = _apply_relative_db(source1, source2, row.source1_db, row.source2_db)
        mix = source1 + source2

        peak = max(
            float(np.max(np.abs(source1))),
            float(np.max(np.abs(source2))),
            float(np.max(np.abs(mix))),
            1e-7,
        )
        if peak > 1.0:
            scale = 0.98 / peak
            source1 = source1 * scale
            source2 = source2 * scale
            mix = mix * scale

        return {
            "mix": torch.from_numpy(mix),
            "sources": torch.stack([
                torch.from_numpy(source1),
                torch.from_numpy(source2),
            ], dim=0),
            "source1_path": s1_path,
            "source2_path": s2_path,
        }
