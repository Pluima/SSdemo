import argparse
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import yaml

from models.TFNet import TFNetSeparator
from models.TFNet_streaming import TFNetStreamer


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args = SimpleNamespace()
    for k, v in config.items():
        setattr(args, k, v)
    return args


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        elif k.startswith("_orig_mod."):
            cleaned[k[10:]] = v
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)


def build_fixed_vec(normalize_vec: bool = True) -> torch.Tensor:
    # x: right(+), y: front(+), z unchanged (0)
    left_front = torch.tensor([-1.0, 1.0, 0.0])
    right_front = torch.tensor([1.0, 1.0, 0.0])
    if normalize_vec:
        left_front = left_front / torch.linalg.norm(left_front)
        right_front = right_front / torch.linalg.norm(right_front)
    return torch.cat([left_front, right_front], dim=0)  # (6,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TFNet streaming inference (mic)")
    parser.add_argument(
        "--config",
        default="/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/config/config_tfnet.yaml",
        help="Path to TFNet config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/checkpoints/soundBubble_tfnet_updated_best.pt",
        help="Path to TFNet checkpoint",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    parser.add_argument("--emit-ms", type=float, default=40.0)
    parser.add_argument("--left-ms", type=float, default=200.0)
    parser.add_argument("--right-ms", type=float, default=200.0)
    parser.add_argument("--normalize-input", action="store_true", default=True)
    parser.add_argument("--no-normalize-input", action="store_false", dest="normalize_input")
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--normalize-vec", action="store_true", default=True)
    parser.add_argument("--no-normalize-vec", action="store_false", dest="normalize_vec")
    parser.add_argument("--mic-channels", type=int, default=1)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument(
        "--input-mono",
        type=int,
        default=None,
        help="Override config: 1 for mono TFNet input, 0 for stereo",
    )
    parser.add_argument(
        "--stereo-loss",
        type=int,
        default=None,
        help="Override config stereo_loss (0/1)",
    )
    return parser.parse_args()


def prepare_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, int, int]:
    cfg = load_config(args.config)
    cfg.network_audio = dict(cfg.network_audio)
    if args.input_mono is not None:
        cfg.input_mono = int(args.input_mono)
    if args.stereo_loss is not None:
        cfg.stereo_loss = int(args.stereo_loss)
    if cfg.input_mono:
        cfg.network_audio["input_nc"] = 2
        cfg.network_audio["output_nc"] = 2

    model = TFNetSeparator(cfg)
    load_checkpoint(model, args.checkpoint, args.device)
    model.to(args.device).eval()
    return model, int(cfg.input_mono), int(cfg.stereo_loss)


def stream_inference() -> None:
    args = parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is required for microphone streaming. Install with: pip install sounddevice"
        ) from exc

    model, input_mono, stereo_loss = prepare_model(args)
    input_channels = 1 if input_mono else 2
    streamer = TFNetStreamer(
        model=model,
        sr=args.sr,
        emit_ms=args.emit_ms,
        left_ms=args.left_ms,
        right_ms=args.right_ms,
        device=args.device,
        input_channels=input_channels,
        normalize_input=args.normalize_input,
        target_lufs=args.target_lufs,
        return_stereo=bool(stereo_loss),
    )
    vec_feature = build_fixed_vec(normalize_vec=args.normalize_vec)

    emit_len = streamer.emit_len

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        if frames != emit_len:
            outdata[:] = 0
            return
        if indata.ndim == 2 and indata.shape[1] > 1:
            if input_mono:
                x = indata.mean(axis=1)  # downmix
            else:
                x = indata[:, :2].T  # (2, emit_len)
        else:
            x = indata[:, 0] if indata.ndim == 2 else indata
            if not input_mono:
                x = torch.as_tensor(x).unsqueeze(0).repeat(2, 1).numpy()
        y = streamer.process(x, vec_feature)
        if y.dim() == 3:
            y = y.mean(dim=1)
        y = y.detach().cpu().numpy().T  # (emit_len, 2)
        outdata[:] = y

    with sd.Stream(
        samplerate=args.sr,
        blocksize=emit_len,
        dtype="float32",
        channels=(args.mic_channels, 2),
        device=(args.input_device, args.output_device),
        callback=callback,
    ):
        print("Streaming... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    stream_inference()
