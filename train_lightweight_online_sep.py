import argparse
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from lightweight_online_sep.dataset_vox2 import VoxCeleb2MixDataset
from lightweight_online_sep.lightweight_sep_model import LightweightCausalSeparator
from lightweight_online_sep.pit_loss import pit_si_snr_loss


DEFAULT_CSV = "/home/qysun/Neuro-SS/baseline/multi-channel/demo/dataset/mixture_data_list_2mix.csv"
DEFAULT_DATA_ROOT = "/home/qysun/Neuro-SS/dataset/VoxCeleb2-mix"
DEFAULT_SAVE_DIR = "/home/qysun/Neuro-SS/baseline/multi-channel/demo/checkpoints"
MODEL_SCALE_PRESETS = {
    # Legacy GRU baselines (kept for backward compatibility).
    "legacy_small": {
        "architecture": "gru",
        "n_fft": 256,
        "hop_length": 128,
        "win_length": 256,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.0,
        "post_ffn_blocks": 1,
        "ffn_expansion": 2,
        "mask_hidden_size": 128,
        "mask_head_layers": 1,
        "time_encoder_dim": 64,
        "time_kernel_size": 256,
        "fusion_hidden_size": 256,
        "lstm_hidden_size": 192,
        "lstm_num_layers": 2,
        "bottleneck_size": 128,
        "tcn_hidden_size": 256,
        "tcn_kernel_size": 3,
        "tcn_blocks": 6,
        "tcn_repeats": 2,
    },
    "legacy_base": {
        "architecture": "gru",
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.1,
        "post_ffn_blocks": 2,
        "ffn_expansion": 4,
        "mask_hidden_size": 256,
        "mask_head_layers": 2,
        "time_encoder_dim": 128,
        "time_kernel_size": 512,
        "fusion_hidden_size": 384,
        "lstm_hidden_size": 384,
        "lstm_num_layers": 3,
        "bottleneck_size": 256,
        "tcn_hidden_size": 512,
        "tcn_kernel_size": 3,
        "tcn_blocks": 8,
        "tcn_repeats": 3,
    },
    # Optional TCN presets for ablation.
    "tcn_small": {
        "architecture": "tcn",
        "n_fft": 256,
        "hop_length": 128,
        "win_length": 256,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.0,
        "post_ffn_blocks": 1,
        "ffn_expansion": 2,
        "mask_hidden_size": 128,
        "mask_head_layers": 2,
        "time_encoder_dim": 64,
        "time_kernel_size": 256,
        "fusion_hidden_size": 256,
        "lstm_hidden_size": 192,
        "lstm_num_layers": 2,
        "bottleneck_size": 128,
        "tcn_hidden_size": 256,
        "tcn_kernel_size": 3,
        "tcn_blocks": 6,
        "tcn_repeats": 2,
    },
    "tcn_base": {
        "architecture": "tcn",
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.05,
        "post_ffn_blocks": 2,
        "ffn_expansion": 4,
        "mask_hidden_size": 256,
        "mask_head_layers": 2,
        "time_encoder_dim": 128,
        "time_kernel_size": 512,
        "fusion_hidden_size": 384,
        "lstm_hidden_size": 384,
        "lstm_num_layers": 3,
        "bottleneck_size": 256,
        "tcn_hidden_size": 512,
        "tcn_kernel_size": 3,
        "tcn_blocks": 8,
        "tcn_repeats": 3,
    },
    # Recommended hybrid LSTM presets (time + frequency).
    "small": {
        "architecture": "lstm_hybrid",
        "n_fft": 256,
        "hop_length": 128,
        "win_length": 256,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.05,
        "post_ffn_blocks": 1,
        "ffn_expansion": 2,
        "mask_hidden_size": 128,
        "mask_head_layers": 2,
        "time_encoder_dim": 64,
        "time_kernel_size": 256,
        "fusion_hidden_size": 256,
        "lstm_hidden_size": 192,
        "lstm_num_layers": 2,
        "bottleneck_size": 128,
        "tcn_hidden_size": 256,
        "tcn_kernel_size": 3,
        "tcn_blocks": 6,
        "tcn_repeats": 2,
    },
    "base": {
        "architecture": "lstm_hybrid",
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "hidden_size": 192,
        "num_layers": 3,
        "dropout": 0.08,
        "post_ffn_blocks": 2,
        "ffn_expansion": 4,
        "mask_hidden_size": 256,
        "mask_head_layers": 2,
        "time_encoder_dim": 128,
        "time_kernel_size": 512,
        "fusion_hidden_size": 384,
        "lstm_hidden_size": 384,
        "lstm_num_layers": 3,
        "bottleneck_size": 256,
        "tcn_hidden_size": 512,
        "tcn_kernel_size": 3,
        "tcn_blocks": 8,
        "tcn_repeats": 3,
    },
    "large": {
        "architecture": "lstm_hybrid",
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "hidden_size": 256,
        "num_layers": 4,
        "dropout": 0.1,
        "post_ffn_blocks": 3,
        "ffn_expansion": 4,
        "mask_hidden_size": 384,
        "mask_head_layers": 2,
        "time_encoder_dim": 192,
        "time_kernel_size": 512,
        "fusion_hidden_size": 512,
        "lstm_hidden_size": 512,
        "lstm_num_layers": 4,
        "bottleneck_size": 384,
        "tcn_hidden_size": 768,
        "tcn_kernel_size": 3,
        "tcn_blocks": 8,
        "tcn_repeats": 4,
    },
    "xlarge": {
        "architecture": "lstm_hybrid",
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "hidden_size": 320,
        "num_layers": 4,
        "dropout": 0.1,
        "post_ffn_blocks": 4,
        "ffn_expansion": 4,
        "mask_hidden_size": 512,
        "mask_head_layers": 3,
        "time_encoder_dim": 256,
        "time_kernel_size": 1024,
        "fusion_hidden_size": 768,
        "lstm_hidden_size": 768,
        "lstm_num_layers": 4,
        "bottleneck_size": 512,
        "tcn_hidden_size": 1024,
        "tcn_kernel_size": 3,
        "tcn_blocks": 10,
        "tcn_repeats": 4,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train lightweight online speech separation model")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to VoxCeleb2 2mix csv")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root of VoxCeleb2-mix")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Checkpoint directory")

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-valid-steps", type=int, default=0, help="0 means full validation")

    parser.add_argument(
        "--model-scale",
        choices=sorted(MODEL_SCALE_PRESETS.keys()),
        default="base",
        help="Model preset. small/base/large/xlarge use hybrid LSTM by default.",
    )
    parser.add_argument(
        "--architecture",
        choices=["gru", "tcn", "lstm_hybrid", "lstm"],
        default=None,
        help="Override preset architecture",
    )
    parser.add_argument("--n-fft", type=int, default=None, help="Override preset n_fft")
    parser.add_argument("--hop-length", type=int, default=None, help="Override preset hop_length")
    parser.add_argument("--win-length", type=int, default=None, help="Override preset win_length")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override preset hidden_size")
    parser.add_argument("--num-layers", type=int, default=None, help="Override preset num_layers")
    parser.add_argument("--dropout", type=float, default=None, help="Override preset dropout")
    parser.add_argument("--post-ffn-blocks", type=int, default=None, help="Override preset post_ffn_blocks")
    parser.add_argument("--ffn-expansion", type=int, default=None, help="Override preset ffn_expansion")
    parser.add_argument("--mask-hidden-size", type=int, default=None, help="Override preset mask_hidden_size")
    parser.add_argument("--mask-head-layers", type=int, default=None, help="Override preset mask_head_layers")
    parser.add_argument("--time-encoder-dim", type=int, default=None, help="Override preset time_encoder_dim")
    parser.add_argument("--time-kernel-size", type=int, default=None, help="Override preset time_kernel_size")
    parser.add_argument("--fusion-hidden-size", type=int, default=None, help="Override preset fusion_hidden_size")
    parser.add_argument("--lstm-hidden-size", type=int, default=None, help="Override preset lstm_hidden_size")
    parser.add_argument("--lstm-num-layers", type=int, default=None, help="Override preset lstm_num_layers")
    parser.add_argument("--bottleneck-size", type=int, default=None, help="Override preset bottleneck_size (TCN)")
    parser.add_argument("--tcn-hidden-size", type=int, default=None, help="Override preset tcn_hidden_size")
    parser.add_argument("--tcn-kernel-size", type=int, default=None, help="Override preset tcn_kernel_size")
    parser.add_argument("--tcn-blocks", type=int, default=None, help="Override preset tcn_blocks")
    parser.add_argument("--tcn-repeats", type=int, default=None, help="Override preset tcn_repeats")

    parser.add_argument("--amp", type=int, default=1, help="Use AMP on CUDA")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=-1,
        help="Used by torchrun for DDP local rank",
    )
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        help="DDP backend: nccl (GPU) or gloo (CPU)",
    )
    return parser.parse_args()


def is_main_process(rank: int) -> bool:
    return rank == 0


def resolve_model_hparams(args: argparse.Namespace) -> dict:
    if args.model_scale not in MODEL_SCALE_PRESETS:
        raise ValueError(f"Unknown model scale: {args.model_scale}")

    hparams = dict(MODEL_SCALE_PRESETS[args.model_scale])
    manual = {
        "architecture": args.architecture,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "post_ffn_blocks": args.post_ffn_blocks,
        "ffn_expansion": args.ffn_expansion,
        "mask_hidden_size": args.mask_hidden_size,
        "mask_head_layers": args.mask_head_layers,
        "time_encoder_dim": args.time_encoder_dim,
        "time_kernel_size": args.time_kernel_size,
        "fusion_hidden_size": args.fusion_hidden_size,
        "lstm_hidden_size": args.lstm_hidden_size,
        "lstm_num_layers": args.lstm_num_layers,
        "bottleneck_size": args.bottleneck_size,
        "tcn_hidden_size": args.tcn_hidden_size,
        "tcn_kernel_size": args.tcn_kernel_size,
        "tcn_blocks": args.tcn_blocks,
        "tcn_repeats": args.tcn_repeats,
    }
    for key, value in manual.items():
        if value is not None:
            hparams[key] = value

    arch = str(hparams.get("architecture", "")).lower()
    if arch == "lstm":
        arch = "lstm_hybrid"
    hparams["architecture"] = arch

    if arch not in {"gru", "tcn", "lstm_hybrid"}:
        raise ValueError(f"architecture must be one of gru/tcn/lstm_hybrid, got: {hparams.get('architecture')}")
    if int(hparams["hop_length"]) > int(hparams["win_length"]):
        raise ValueError("hop_length must be <= win_length")
    if int(hparams["n_fft"]) < int(hparams["win_length"]):
        raise ValueError("n_fft should be >= win_length")
    if arch == "tcn":
        if int(hparams["tcn_blocks"]) < 1:
            raise ValueError("tcn_blocks must be >= 1")
        if int(hparams["tcn_repeats"]) < 1:
            raise ValueError("tcn_repeats must be >= 1")
        if int(hparams["tcn_kernel_size"]) < 2:
            raise ValueError("tcn_kernel_size must be >= 2")
    if arch == "lstm_hybrid":
        if int(hparams["lstm_num_layers"]) < 1:
            raise ValueError("lstm_num_layers must be >= 1")
        if int(hparams["lstm_hidden_size"]) < 1:
            raise ValueError("lstm_hidden_size must be >= 1")
        if int(hparams["time_encoder_dim"]) < 1:
            raise ValueError("time_encoder_dim must be >= 1")
        if int(hparams["time_kernel_size"]) < 1:
            raise ValueError("time_kernel_size must be >= 1")
        if int(hparams["fusion_hidden_size"]) < 1:
            raise ValueError("fusion_hidden_size must be >= 1")
    return hparams


def init_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        rank = 0
        local_rank = 0
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available.")
            if device.index is not None:
                torch.cuda.set_device(device.index)
        return False, rank, 1, local_rank, device

    # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank if args.local_rank >= 0 else 0)))
    backend = args.dist_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return True, rank, world_size, local_rank, device


def make_dataloaders(
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
):
    train_set = VoxCeleb2MixDataset(
        csv_path=args.csv,
        data_root=args.data_root,
        split="train",
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        train_random_offset=True,
    )
    valid_set = VoxCeleb2MixDataset(
        csv_path=args.csv,
        data_root=args.data_root,
        split="valid",
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        train_random_offset=False,
    )

    train_sampler = None
    valid_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        valid_sampler = DistributedSampler(
            valid_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        sampler=train_sampler,
    )
    valid_workers = max(0, args.num_workers // 2)
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=valid_workers,
        pin_memory=pin,
        persistent_workers=valid_workers > 0,
        drop_last=False,
        sampler=valid_sampler,
    )
    return train_loader, valid_loader, train_sampler, valid_sampler


def _maybe_autocast(device: torch.device, enabled: bool, amp_dtype: str):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: str,
    clip_grad: float,
    max_steps: int,
    train: bool,
    distributed: bool,
):
    model.train(mode=train)
    total_loss = 0.0
    total_steps = 0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    for step, batch in enumerate(loader, start=1):
        mix = batch["mix"].to(device=device, dtype=torch.float32, non_blocking=True)
        refs = batch["sources"].to(device=device, dtype=torch.float32, non_blocking=True)

        if train:
            if optimizer is None:
                raise ValueError("optimizer must be provided when train=True")
            optimizer.zero_grad(set_to_none=True)

        with grad_ctx:
            with _maybe_autocast(device, amp_enabled, amp_dtype):
                est = model(mix)
                t = min(est.shape[-1], refs.shape[-1])
                loss = pit_si_snr_loss(est[..., :t], refs[..., :t])

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_steps += 1

        if max_steps > 0 and step >= max_steps:
            break

    if distributed:
        stats = torch.tensor(
            [total_loss, float(total_steps)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_steps = int(stats[1].item())

    if total_steps == 0:
        return float("inf")
    return total_loss / total_steps


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        elif k.startswith("_orig_mod."):
            cleaned[k[10:]] = v
        else:
            cleaned[k] = v
    return cleaned


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank, device = init_distributed(args)
    set_seed(args.seed + rank)
    model_hparams = resolve_model_hparams(args)

    if is_main_process(rank):
        os.makedirs(args.save_dir, exist_ok=True)

    model = LightweightCausalSeparator(
        num_speakers=2,
        n_fft=model_hparams["n_fft"],
        hop_length=model_hparams["hop_length"],
        win_length=model_hparams["win_length"],
        architecture=model_hparams["architecture"],
        hidden_size=model_hparams["hidden_size"],
        num_layers=model_hparams["num_layers"],
        dropout=model_hparams["dropout"],
        post_ffn_blocks=model_hparams["post_ffn_blocks"],
        ffn_expansion=model_hparams["ffn_expansion"],
        mask_hidden_size=model_hparams["mask_hidden_size"],
        mask_head_layers=model_hparams["mask_head_layers"],
        time_encoder_dim=model_hparams["time_encoder_dim"],
        time_kernel_size=model_hparams["time_kernel_size"],
        fusion_hidden_size=model_hparams["fusion_hidden_size"],
        lstm_hidden_size=model_hparams["lstm_hidden_size"],
        lstm_num_layers=model_hparams["lstm_num_layers"],
        bottleneck_size=model_hparams["bottleneck_size"],
        tcn_hidden_size=model_hparams["tcn_hidden_size"],
        tcn_kernel_size=model_hparams["tcn_kernel_size"],
        tcn_blocks=model_hparams["tcn_blocks"],
        tcn_repeats=model_hparams["tcn_repeats"],
    ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    model_for_io = model.module if isinstance(model, DDP) else model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(args.amp)))

    start_epoch = 0
    best_valid = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model_for_io.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_valid = float(ckpt.get("best_valid_loss", best_valid))
        if is_main_process(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_loader, valid_loader, train_sampler, valid_sampler = make_dataloaders(
        args=args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    if is_main_process(rank):
        print(
            f"Distributed: {distributed} | world_size={world_size} | rank={rank} | "
            f"device={device}"
        )
        print(f"Model scale: {args.model_scale}")
        print(f"Resolved model hparams: {model_hparams}")
        print(
            f"Train batches per rank: {len(train_loader)}, "
            f"Valid batches per rank: {len(valid_loader)}"
        )
        print(f"Model params: {model_for_io.model_size_million():.3f} M")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if valid_sampler is not None:
            valid_sampler.set_epoch(epoch)

        tic = time.time()

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=bool(args.amp),
            amp_dtype=args.amp_dtype,
            clip_grad=args.clip_grad,
            max_steps=args.max_train_steps,
            train=True,
            distributed=distributed,
        )
        valid_loss = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            device=device,
            scaler=scaler,
            amp_enabled=bool(args.amp),
            amp_dtype=args.amp_dtype,
            clip_grad=0.0,
            max_steps=args.max_valid_steps,
            train=False,
            distributed=distributed,
        )

        epoch_sec = time.time() - tic
        if is_main_process(rank):
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
                f"train={train_loss:.4f} valid={valid_loss:.4f} | {epoch_sec:.1f}s"
            )

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model_for_io.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "best_valid_loss": float(min(best_valid, valid_loss)),
                "model_config": model_for_io.export_config(),
                "model_scale": args.model_scale,
                "resolved_model_hparams": model_hparams,
                "train_args": vars(args),
            }

            latest_path = os.path.join(args.save_dir, "lightweight_sep_latest.pt")
            torch.save(ckpt, latest_path)

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_path = os.path.join(args.save_dir, "lightweight_sep_best.pt")
                torch.save(ckpt, best_path)
                print(f"Saved new best checkpoint: {best_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
