#!/usr/bin/env python3
"""Render VoxCeleb2 2mix rows into binaural stereo mixtures using front small-angle HRIR only."""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


@dataclass
class Vox2Row:
    partition: str
    s1_subset: str
    s1_spk: str
    s1_utt: str
    s1_db: float
    s2_subset: str
    s2_spk: str
    s2_utt: str
    s2_db: float
    duration: float


@dataclass
class HrirBank:
    hrir: np.ndarray  # [taps, dirs, ch]
    azimuth_deg: np.ndarray  # [dirs]
    elevation_deg: np.ndarray  # [dirs]
    channel_names: List[str]
    left_idx: int
    right_idx: int
    sample_rate: int
    mat_file: str


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mat_value_to_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.reshape(-1).tolist())
        if value.size == 1:
            return _mat_value_to_str(value.item())
    return str(value)


def _mat_cellstr_to_list(value) -> List[str]:
    arr = np.asarray(value, dtype=object).reshape(-1)
    out: List[str] = []
    for item in arr:
        text = _mat_value_to_str(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_style(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_base_and_side(name: str) -> Tuple[str, Optional[str]]:
    norm = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    m = re.match(r"^(.*)_([lr])$", norm)
    if m:
        return m.group(1), m.group(2)
    return norm, None


def _pick_lr_pair(channel_names: List[str], prefer_style: str) -> Tuple[int, int]:
    pairs: List[Tuple[str, int, int]] = []
    index_by_base = {}

    for i, name in enumerate(channel_names):
        base, side = _extract_base_and_side(name)
        if side is None:
            continue
        item = index_by_base.setdefault(base, {})
        item[side] = i

    for base, sides in index_by_base.items():
        if "l" in sides and "r" in sides:
            pairs.append((base, sides["l"], sides["r"]))

    if not pairs:
        if len(channel_names) >= 2:
            return 0, 1
        raise ValueError(f"Cannot find L/R channels in HRIR names: {channel_names}")

    style = _normalize_style(prefer_style)
    if style:
        filtered = [p for p in pairs if style in _normalize_style(p[0])]
        if filtered:
            pairs = filtered

    _, l_idx, r_idx = sorted(pairs, key=lambda x: x[0])[0]
    return l_idx, r_idx


def load_hrir_bank(mat_path: Path, prefer_style: str) -> HrirBank:
    if loadmat is None:
        raise RuntimeError("scipy is required to read .mat HRIR files. Install it by `pip install scipy`.")

    d = loadmat(str(mat_path))
    if "M_data" not in d or "M_directions" not in d:
        raise ValueError(f"{mat_path} is missing M_data or M_directions")

    hrir = np.asarray(d["M_data"], dtype=np.float32)
    directions = np.asarray(d["M_directions"], dtype=np.float32)
    channel_names = _mat_cellstr_to_list(d.get("c_channel_names", []))
    srate = int(round(float(np.asarray(d.get("srate", [[48000]])).squeeze())))

    if directions.ndim != 2:
        raise ValueError(f"M_directions should be 2D, got shape {directions.shape}")
    if directions.shape[0] == 2:
        azimuth = directions[0, :]
        elevation = directions[1, :]
    elif directions.shape[1] == 2:
        azimuth = directions[:, 0]
        elevation = directions[:, 1]
    else:
        raise ValueError(f"Unexpected M_directions shape: {directions.shape}")

    if hrir.ndim != 3:
        raise ValueError(f"M_data should be 3D, got shape {hrir.shape}")
    if hrir.shape[1] != azimuth.shape[0] and hrir.shape[2] == azimuth.shape[0]:
        hrir = np.transpose(hrir, (0, 2, 1))
    if hrir.shape[1] != azimuth.shape[0]:
        raise ValueError(f"HRIR direction dim mismatch: M_data {hrir.shape}, M_directions {directions.shape}")

    if not channel_names:
        channel_names = [f"ch{i}" for i in range(hrir.shape[2])]
    l_idx, r_idx = _pick_lr_pair(channel_names, prefer_style=prefer_style)

    return HrirBank(
        hrir=hrir,
        azimuth_deg=azimuth.astype(np.float32),
        elevation_deg=elevation.astype(np.float32),
        channel_names=channel_names,
        left_idx=l_idx,
        right_idx=r_idx,
        sample_rate=srate,
        mat_file=str(mat_path),
    )


def iter_vox2_rows(csv_path: Path) -> Iterable[Vox2Row]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            yield Vox2Row(
                partition=row[0].strip().lower(),
                s1_subset=row[1].strip(),
                s1_spk=row[2].strip(),
                s1_utt=row[3].strip(),
                s1_db=_safe_float(row[4], 0.0),
                s2_subset=row[5].strip(),
                s2_spk=row[6].strip(),
                s2_utt=row[7].strip(),
                s2_db=_safe_float(row[8], 0.0),
                duration=max(_safe_float(row[9], 0.0), 0.0),
            )


def build_vox_path(root: Path, subset: str, speaker: str, utt: str) -> Path:
    return root / "audio_clean" / subset / speaker / f"{utt}.wav"


def load_mono_audio(path: Path, target_sr: int, target_len: int) -> np.ndarray:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim != 2:
        raise ValueError(f"Expected [C, T], got {tuple(wav.shape)} for {path}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, int(sr), int(target_sr))
    audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if target_len > 0:
        if len(audio) >= target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))
    return audio


def resample_1d(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return signal.astype(np.float32, copy=False)
    src = torch.from_numpy(signal.astype(np.float32, copy=False))[None, :]
    out = torchaudio.functional.resample(src, int(src_sr), int(dst_sr))
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def fft_convolve_1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = int(len(x) + len(h) - 1)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    n_fft = 1 << int(math.ceil(math.log2(max(1, n))))
    y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n]
    return y.astype(np.float32, copy=False)


def angle_distance_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def wrap_azimuth_deg(az: float) -> float:
    return ((float(az) + 180.0) % 360.0) - 180.0


def _front_candidate_indices(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    target_elevation: float,
    front_abs_min: float,
    front_abs_max: float,
) -> Tuple[np.ndarray, float]:
    unique_elev = np.unique(elevation_deg)
    chosen_elev = float(unique_elev[np.argmin(np.abs(unique_elev - target_elevation))])

    front_abs_min = max(0.0, float(front_abs_min))
    front_abs_max = max(front_abs_min, float(front_abs_max))

    az_wrapped = np.array([wrap_azimuth_deg(a) for a in azimuth_deg], dtype=np.float32)
    on_elev = np.isclose(elevation_deg, chosen_elev)
    in_front = np.logical_and(np.abs(az_wrapped) >= front_abs_min, np.abs(az_wrapped) <= front_abs_max)
    idx = np.where(np.logical_and(on_elev, in_front))[0]

    # Fallback if exact target elevation has no valid directions.
    if len(idx) < 2:
        idx = np.where(in_front)[0]
    return idx, chosen_elev


def _valid_front_pairs(
    candidates: np.ndarray,
    azimuth_deg: np.ndarray,
    min_sep_deg: float,
    max_sep_deg: float,
    require_opposite_sides: bool,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    min_sep = max(0.0, float(min_sep_deg))
    max_sep = max(min_sep, float(max_sep_deg))

    az_wrapped = np.array([wrap_azimuth_deg(a) for a in azimuth_deg], dtype=np.float32)
    cands = list(map(int, candidates))
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d1 = cands[i]
            d2 = cands[j]
            a1 = float(az_wrapped[d1])
            a2 = float(az_wrapped[d2])

            sep = angle_distance_deg(a1, a2)
            if sep < min_sep or sep > max_sep:
                continue

            if bool(require_opposite_sides) and (a1 * a2 > 0.0):
                continue

            pairs.append((d1, d2))
    return pairs


def pick_direction_pair_front_small(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    target_elevation: float,
    front_abs_min: float,
    front_abs_max: float,
    min_sep_deg: float,
    max_sep_deg: float,
    require_opposite_sides: bool,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float]:
    candidates, chosen_elev = _front_candidate_indices(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        target_elevation=target_elevation,
        front_abs_min=front_abs_min,
        front_abs_max=front_abs_max,
    )
    if len(candidates) < 2:
        raise RuntimeError(
            "Not enough front-angle HRIR directions. "
            f"Need >=2, got {len(candidates)} (front_abs=[{front_abs_min}, {front_abs_max}] deg)."
        )

    pairs = _valid_front_pairs(
        candidates=candidates,
        azimuth_deg=azimuth_deg,
        min_sep_deg=min_sep_deg,
        max_sep_deg=max_sep_deg,
        require_opposite_sides=require_opposite_sides,
    )
    if not pairs:
        raise RuntimeError(
            "No valid front small-angle pair found. "
            f"front_abs=[{front_abs_min}, {front_abs_max}], sep=[{min_sep_deg}, {max_sep_deg}], "
            f"require_opposite_sides={bool(require_opposite_sides)}"
        )

    d1, d2 = pairs[int(rng.integers(0, len(pairs)))]
    if bool(rng.integers(0, 2)):
        d1, d2 = d2, d1
    return d1, d2, float(azimuth_deg[d1]), float(azimuth_deg[d2]), chosen_elev


def apply_relative_db(source1: np.ndarray, source2: np.ndarray, s1_db: float, s2_db: float) -> np.ndarray:
    rel_db = float(s2_db) - float(s1_db)
    p1 = float(np.mean(np.square(source1, dtype=np.float64)))
    p2 = float(np.mean(np.square(source2, dtype=np.float64)))
    if p1 <= 1e-12 or p2 <= 1e-12:
        return source2
    scalar = (10.0 ** (rel_db / 20.0)) * math.sqrt(p1 / p2)
    return source2 * float(scalar)


def azel_to_vector(az_deg: float, el_deg: float) -> List[float]:
    az = math.radians(float(az_deg))
    el = math.radians(float(el_deg))
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    v = np.array([x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return [float(v[0]), float(v[1]), float(v[2])]


def vector_to_str(v: List[float]) -> str:
    return f"[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]"


def parse_partitions(text: str) -> set:
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    return set(items) if items else {"train", "val", "valid", "validation", "test"}


def safe_utt(utt: str) -> str:
    return utt.replace("/", "_").replace("\\", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Render VoxCeleb2 2mix CSV into binaural stereo mixes using front small-angle HRIR only."
    )
    parser.add_argument(
        "--mix-csv",
        type=Path,
        default=Path("/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/dataset/mixture_data_list_2mix.csv"),
        help="Path to VoxCeleb2 2mix CSV (headerless, 10 cols).",
    )
    parser.add_argument(
        "--vox-root",
        type=Path,
        default=Path("/home/qysun/Neuro-SS/dataset/VoxCeleb2-mix"),
        help="Root containing audio_clean/{subset}/{spk}/{utt}.wav",
    )
    parser.add_argument(
        "--hrir-mat",
        type=Path,
        default=Path("/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/hrir/HRIRs_MAT/KEMAR-TrEPind_MultiCh.mat"),
        help="HRIR .mat file under HRIRs_MAT.",
    )
    parser.add_argument(
        "--ear-style",
        type=str,
        default="InEar",
        help="Preferred channel style when HRIR has multiple L/R pairs (e.g., InEar, Entr, Concha, BTE_fr).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/qysun/Neuro-SS/dataset/VoxCeleb2-mix-hrir-front-small"),
        help="Output root for generated stereo mixtures.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/qysun/Neuro-SS/dataset/VoxCeleb2-mix-hrir-front-small/mix_sources_vox2_hrir_front_small.csv"),
        help="Output CSV (sound-bubble/demo format).",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Output sample rate.")
    parser.add_argument("--target-elev", type=float, default=0.0, help="Target elevation in degrees.")
    parser.add_argument("--front-az-abs-min", type=float, default=0.0, help="Min |azimuth| in front region (deg).")
    parser.add_argument("--front-az-abs-max", type=float, default=30.0, help="Max |azimuth| in front region (deg).")
    parser.add_argument("--min-az-sep", type=float, default=5.0, help="Min separation between 2 speakers (deg).")
    parser.add_argument("--max-az-sep", type=float, default=30.0, help="Max separation between 2 speakers (deg).")
    parser.add_argument(
        "--require-opposite-sides",
        type=int,
        default=1,
        help="If 1, enforce one source on left and one on right in front region.",
    )
    parser.add_argument("--partitions", type=str, default="train,val,test", help="Comma-separated partitions to process.")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to process (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.mix_csv.exists():
        raise FileNotFoundError(args.mix_csv)
    if not args.hrir_mat.exists():
        raise FileNotFoundError(args.hrir_mat)

    wanted_parts = parse_partitions(args.partitions)
    rng = np.random.default_rng(int(args.seed))

    hrir_bank = load_hrir_bank(args.hrir_mat, prefer_style=args.ear_style)
    print(
        f"[HRIR] file={hrir_bank.mat_file}, channels={hrir_bank.channel_names}, "
        f"selected=({hrir_bank.left_idx}, {hrir_bank.right_idx})"
    )

    hrir_l = hrir_bank.hrir[:, :, hrir_bank.left_idx]
    hrir_r = hrir_bank.hrir[:, :, hrir_bank.right_idx]
    if int(hrir_bank.sample_rate) != int(args.sr):
        print(f"[HRIR] resampling HRIR {hrir_bank.sample_rate} -> {args.sr}")
        resampled_l: List[np.ndarray] = []
        resampled_r: List[np.ndarray] = []
        max_taps = 0
        for i in range(hrir_l.shape[1]):
            l = resample_1d(hrir_l[:, i], hrir_bank.sample_rate, args.sr)
            r = resample_1d(hrir_r[:, i], hrir_bank.sample_rate, args.sr)
            resampled_l.append(l)
            resampled_r.append(r)
            max_taps = max(max_taps, len(l), len(r))

        new_l = np.zeros((max_taps, hrir_l.shape[1]), dtype=np.float32)
        new_r = np.zeros((max_taps, hrir_r.shape[1]), dtype=np.float32)
        for i in range(hrir_l.shape[1]):
            new_l[: len(resampled_l[i]), i] = resampled_l[i]
            new_r[: len(resampled_r[i]), i] = resampled_r[i]
        hrir_l, hrir_r = new_l, new_r

    # Print a quick availability summary for chosen front constraints.
    candidates, chosen_elev = _front_candidate_indices(
        azimuth_deg=hrir_bank.azimuth_deg,
        elevation_deg=hrir_bank.elevation_deg,
        target_elevation=args.target_elev,
        front_abs_min=args.front_az_abs_min,
        front_abs_max=args.front_az_abs_max,
    )
    pairs = _valid_front_pairs(
        candidates=candidates,
        azimuth_deg=hrir_bank.azimuth_deg,
        min_sep_deg=args.min_az_sep,
        max_sep_deg=args.max_az_sep,
        require_opposite_sides=bool(args.require_opposite_sides),
    )
    print(
        f"[Front HRIR] elev={chosen_elev:.2f} deg, candidates={len(candidates)}, valid_pairs={len(pairs)} | "
        f"front_abs=[{args.front_az_abs_min:.1f},{args.front_az_abs_max:.1f}], "
        f"sep=[{args.min_az_sep:.1f},{args.max_az_sep:.1f}], "
        f"opp_sides={bool(args.require_opposite_sides)}"
    )
    if len(pairs) == 0:
        raise RuntimeError("No valid front small-angle HRIR pairs under current constraints.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "partition",
        "mix",
        "source1",
        "source1_start",
        "source1_vector",
        "source2",
        "source2_start",
        "source2_vector",
        "duration",
        "source1_db",
        "source2_db",
        "az1_deg",
        "az2_deg",
        "elev_deg",
        "hrir_file",
        "hrir_left_channel",
        "hrir_right_channel",
    ]

    processed = 0
    skipped = 0
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for row_idx, row in enumerate(iter_vox2_rows(args.mix_csv)):
            if row.partition not in wanted_parts:
                continue
            if args.max_rows > 0 and processed >= args.max_rows:
                break

            s1_path = build_vox_path(args.vox_root, row.s1_subset, row.s1_spk, row.s1_utt)
            s2_path = build_vox_path(args.vox_root, row.s2_subset, row.s2_spk, row.s2_utt)
            if not s1_path.exists() or not s2_path.exists():
                skipped += 1
                continue

            try:
                target_len = int(round(row.duration * args.sr)) if row.duration > 0 else 0
                source1 = load_mono_audio(s1_path, target_sr=args.sr, target_len=target_len)
                source2 = load_mono_audio(s2_path, target_sr=args.sr, target_len=target_len)
                source2 = apply_relative_db(source1, source2, row.s1_db, row.s2_db)

                d1, d2, az1, az2, elev = pick_direction_pair_front_small(
                    azimuth_deg=hrir_bank.azimuth_deg,
                    elevation_deg=hrir_bank.elevation_deg,
                    target_elevation=args.target_elev,
                    front_abs_min=args.front_az_abs_min,
                    front_abs_max=args.front_az_abs_max,
                    min_sep_deg=args.min_az_sep,
                    max_sep_deg=args.max_az_sep,
                    require_opposite_sides=bool(args.require_opposite_sides),
                    rng=rng,
                )

                s1_l = fft_convolve_1d(source1, hrir_l[:, d1])
                s1_r = fft_convolve_1d(source1, hrir_r[:, d1])
                s2_l = fft_convolve_1d(source2, hrir_l[:, d2])
                s2_r = fft_convolve_1d(source2, hrir_r[:, d2])
                mix_l = s1_l + s2_l
                mix_r = s1_r + s2_r
                mix = np.stack([mix_l, mix_r], axis=0).astype(np.float32, copy=False)

                peak = float(np.max(np.abs(mix))) if mix.size else 0.0
                if peak > 1.0:
                    mix *= 0.98 / peak

                utt1 = safe_utt(row.s1_utt)
                utt2 = safe_utt(row.s2_utt)
                out_dir = args.out_root / row.partition
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{row_idx:08d}_{row.s1_spk}_{utt1}__{row.s2_spk}_{utt2}_mix_stereo.wav"
                out_path = out_dir / out_name

                torchaudio.save(str(out_path), torch.from_numpy(mix), args.sr)

                vec1 = azel_to_vector(az1, elev)
                vec2 = azel_to_vector(az2, elev)
                writer.writerow(
                    {
                        "partition": row.partition,
                        "mix": str(out_path),
                        "source1": str(s1_path),
                        "source1_start": 0,
                        "source1_vector": vector_to_str(vec1),
                        "source2": str(s2_path),
                        "source2_start": 0,
                        "source2_vector": vector_to_str(vec2),
                        "duration": f"{row.duration:.3f}",
                        "source1_db": f"{row.s1_db:.3f}",
                        "source2_db": f"{row.s2_db:.3f}",
                        "az1_deg": f"{az1:.2f}",
                        "az2_deg": f"{az2:.2f}",
                        "elev_deg": f"{elev:.2f}",
                        "hrir_file": hrir_bank.mat_file,
                        "hrir_left_channel": hrir_bank.channel_names[hrir_bank.left_idx],
                        "hrir_right_channel": hrir_bank.channel_names[hrir_bank.right_idx],
                    }
                )

                processed += 1
                if processed % 100 == 0:
                    print(f"[progress] processed={processed}, skipped={skipped}")
            except Exception as exc:
                skipped += 1
                print(f"[warn] row {row_idx} failed: {exc}")

    print(f"[done] processed={processed}, skipped={skipped}")
    print(f"[done] out_root={args.out_root}")
    print(f"[done] out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
