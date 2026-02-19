from contextlib import nullcontext
from itertools import islice
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import yaml
import yamlargparse

from dataset.dataset import SimDataLoader
from loss import get_loss_function
from models.model_wrapper import get_model


parser = yamlargparse.ArgumentParser("Fine-tune Settings")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--config', default='/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/config/config_tfnetcasual.yaml')
parser.add_argument('--pretrained_checkpoint', type=str, default='/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/checkpoints/stream_best.pt', help='Path to pretrained model checkpoint for initialization')
parser.add_argument('--train_from_last_checkpoint', type=int, default=0, help='Resume finetune from finetune checkpoint')
parser.add_argument('--finetune_csv', type=str, default='mixture_data_list_2mix.csv', help='CSV filename under dataset.path')
parser.add_argument('--n_gpu', type=int, default=8)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--gpu_list', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('--dataset_size', type=int, default=0)
parser.add_argument('--amp', type=int, default=None)
parser.add_argument('--amp_dtype', type=str, default=None)
parser.add_argument('--val_interval', type=int, default=1, help='Validate every N epochs')
parser.add_argument('--early_stop_patience', type=int, default=8, help='Stop if no val improvement for N validations')
parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='Minimum val improvement to reset patience')
parser.add_argument('--lr_scheduler_patience', type=int, default=2, help='ReduceLROnPlateau patience')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='ReduceLROnPlateau decay factor')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum LR for scheduler')
parser.add_argument('--train_last_n_layers', type=int, default=0, help='Only train last N output layer (upconvlayer8~8-N+1). 0 stands for ALL')
parser.add_argument('--aishell_augment', type=int, default=1, help='Enable AISHELL-specific augmentation for train split')
parser.add_argument('--aishell_aug_gain_db_min', type=float, default=-4.0, help='Global gain min (dB) for AISHELL augmentation')
parser.add_argument('--aishell_aug_gain_db_max', type=float, default=4.0, help='Global gain max (dB) for AISHELL augmentation')
parser.add_argument('--aishell_aug_noise_prob', type=float, default=0.6, help='Probability of adding noise to mix')
parser.add_argument('--aishell_aug_snr_db_min', type=float, default=18.0, help='Noise SNR min (dB)')
parser.add_argument('--aishell_aug_snr_db_max', type=float, default=35.0, help='Noise SNR max (dB)')

args = parser.parse_args()


def _resolve_amp_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower().strip()
    if s in ['bf16', 'bfloat16']:
        return torch.bfloat16
    if s in ['fp16', 'float16', 'half']:
        return torch.float16
    return torch.bfloat16


def _use_causal_streaming_loss(cfg_args) -> bool:
    return bool(cfg_args.network_audio.get('streaming_train', False))


def _resolve_online_csv_path(cfg_args) -> str:
    dataset_path = os.path.expanduser(str(cfg_args.dataset.get('path', '')))
    dataset_name = os.path.expanduser(str(cfg_args.dataset.get('name', '')))
    return dataset_name if os.path.isabs(dataset_name) else os.path.join(dataset_path, dataset_name)


def _is_voxceleb2_mixture_csv(csv_path: str) -> bool:
    if not csv_path or not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                cols = [c.strip() for c in line.split(',')]
                if len(cols) < 10:
                    return False
                first = cols[0].lower()
                second = cols[1].lower()
                return first in {'train', 'val', 'test'} and second in {'train', 'test', 'val', 'dev'}
    except Exception:
        return False
    return False


def _auto_disable_vector_cue_for_vox(cfg_args):
    csv_path = _resolve_online_csv_path(cfg_args)
    if _is_voxceleb2_mixture_csv(csv_path):
        if int(getattr(cfg_args, 'vector_cue', 0)) != 0:
            print(f"[Dataset] Detected VoxCeleb2 mixture CSV: {csv_path}")
            print("[Dataset] Direction vectors unavailable for VoxCeleb2, forcing vector_cue=0.")
        cfg_args.vector_cue = 0


def _crop_streaming_loss(targets: torch.Tensor, preds: torch.Tensor, left_len: int):
    if left_len <= 0:
        return targets, preds
    T = min(targets.shape[-1], preds.shape[-1])
    if T <= left_len:
        return targets[..., :T], preds[..., :T]
    return targets[..., left_len:T], preds[..., left_len:T]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def load_model_weights(model, checkpoint_path, device):
    if not checkpoint_path:
        print('No pretrained checkpoint provided; training from current initialization.')
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint_path}')

    print(f'Loading pretrained weights from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    cleaned = {}
    for k, v in state_dict.items():
        if "mask_net.audionet_convlayer1"  in k :
            continue
        if k.startswith('module.'):
            cleaned[k[7:]] = v
        elif k.startswith('_orig_mod.'):
            cleaned[k[10:]] = v
        else:
            cleaned[k] = v

    target_model = _unwrap_model(model)
    if hasattr(target_model, '_orig_mod'):
        target_model = target_model._orig_mod

    missing, unexpected = target_model.load_state_dict(cleaned, strict=False)
    print(f'Loaded pretrained weights. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}')


def load_resume_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print('No finetune resume checkpoint found; start from epoch 0.')
        return 0

    print(f'Resuming finetune from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.') :
            cleaned[k[7:]] = v
        elif k.startswith('_orig_mod.'):
            cleaned[k[10:]] = v
        else:
            cleaned[k] = v

    target_model = _unwrap_model(model)
    if hasattr(target_model, '_orig_mod'):
        target_model = target_model._orig_mod
    target_model.load_state_dict(cleaned, strict=False)

    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return int(checkpoint.get('epoch', 0)) + 1


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    target_model = _unwrap_model(model)
    if hasattr(target_model, '_orig_mod'):
        model_state_dict = target_model._orig_mod.state_dict()
    else:
        model_state_dict = target_model.state_dict()

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        },
        checkpoint_path,
    )


def freeze_encoder_only_train_decoder_output(model: torch.nn.Module, train_last_n_layers: int = 0, use_direction: bool = True):
    """
    For TFNet_causal: freeze encoder, only train decoder + final output layer + DirVecNet.
    Decoder/output correspond to mask_net.audionet_upconvlayer1..8; DirVecNet is the direction-vector conditioning MLP.
    """
    core_model = _unwrap_model(model)
    if hasattr(core_model, '_orig_mod'):
        core_model = core_model._orig_mod

    if not hasattr(core_model, 'mask_net'):
        raise ValueError('Expected model to have mask_net (TFNet-like architecture).')

    dirvec_prefix = 'mask_net.DirVecNet'
    if train_last_n_layers:
        upconv_prefixes = tuple(f'mask_net.audionet_upconvlayer{8-i+1}' for i in range(1, train_last_n_layers+1))
        trainable_prefixes = ('mask_net.audionet_convlayer1',)+ upconv_prefixes
        trainable_desc = ', '.join(f'mask_net.audionet_upconvlayer{8-i+1}' for i in range(1, train_last_n_layers+1))
    else:
        upconv_prefixes = tuple(f'mask_net.audionet_upconvlayer{i}' for i in range(1, 9))
        if use_direction:
            trainable_prefixes = (dirvec_prefix,) + upconv_prefixes
            trainable_desc = 'mask_net.DirVecNet, mask_net.audionet_upconvlayer1..8'
        else:
            trainable_prefixes = upconv_prefixes
            trainable_desc = 'mask_net.audionet_upconvlayer1..8'

    total_params = 0
    trainable_params = 0
    for name, param in core_model.named_parameters():
        total_params += param.numel()
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f'Freeze strategy applied: trainable params {trainable_params}/{total_params}')
    print(f'Trainable modules: {trainable_desc}')


def validate(model, dataloader, loss_fn, device, cfg_args, amp_enabled=False, amp_dtype=torch.float16):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for sample in dataloader:
            mix = sample['mix'].to(device)
            source1 = sample['source1'].to(device, dtype=torch.float)
            source2 = sample['source2'].to(device, dtype=torch.float)
            references = torch.stack([source1, source2], dim=1)

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                if cfg_args.vector_cue:
                    vector1 = sample['source1_vector'].to(device)
                    vector2 = sample['source2_vector'].to(device)
                    concat_vec = torch.cat([vector1, vector2], dim=1)
                    outputs = model(mix, concat_vec) if cfg_args.sstask else model(mix, vector1)
                else:
                    outputs = model(mix)

            if 'moss' in cfg_args.network_audio['backbone']:
                outputs = torch.stack(outputs, dim=2).transpose(1, 2)

            outputs = outputs.float()
            targets = references if cfg_args.sstask else source1
            if _use_causal_streaming_loss(cfg_args):
                model_delay_samples = cfg_args.network_audio.get('stft_hop', 0) * 2
                crop = max(
                    cfg_args.network_audio.get('streaming_left_len', 0),
                    cfg_args.network_audio.get('stft_frame', 0),
                ) + model_delay_samples
                targets, outputs = _crop_streaming_loss(targets, outputs, crop)

            loss = loss_fn(targets, outputs)
            batch_size = mix.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    return total_loss / max(1, num_samples)


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, cfg_args, scaler=None, amp_enabled=False, amp_dtype=torch.float16):
    model.train()
    total_loss = 0.0
    num_samples = 0

    if cfg_args.accu_grad:
        world_size = cfg_args.n_gpu if cfg_args.distributed else 1
        accumulation_steps = cfg_args.effec_batch_size // (cfg_args.dataloader['batch_size'] * world_size)
        accumulation_steps = max(1, accumulation_steps)
    else:
        accumulation_steps = 1

    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None

    if cfg_args.dataset_size and cfg_args.dataset_size > 0:
        max_steps = min(cfg_args.dataset_size, dataloader_len) if dataloader_len is not None else cfg_args.dataset_size
        data_iter = islice(dataloader, max_steps)
        tqdm_total = max_steps
    else:
        max_steps = dataloader_len
        data_iter = dataloader
        tqdm_total = dataloader_len

    optimizer.zero_grad(set_to_none=True)

    with tqdm(data_iter, total=tqdm_total) as td:
        for idx, sample in enumerate(td):
            mix = sample['mix'].to(device)
            source1 = sample['source1'].to(device, dtype=torch.float)
            source2 = sample['source2'].to(device, dtype=torch.float)
            references = torch.stack([source1, source2], dim=1)

            is_last_step = (max_steps is not None) and ((idx + 1) == max_steps)
            is_update_step = ((idx + 1) % accumulation_steps == 0) or is_last_step

            if cfg_args.distributed and not is_update_step:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    if cfg_args.vector_cue:
                        vector1 = sample['source1_vector'].to(device)
                        vector2 = sample['source2_vector'].to(device)
                        concat_vec = torch.cat([vector1, vector2], dim=1)
                        outputs = model(mix, concat_vec) if cfg_args.sstask else model(mix, vector1)
                    else:
                        outputs = model(mix)

                    if 'moss' in cfg_args.network_audio['backbone']:
                        outputs = torch.stack(outputs, dim=2).transpose(1, 2)

                outputs = outputs.float()
                targets = references if cfg_args.sstask else source1
                if _use_causal_streaming_loss(cfg_args):
                    model_delay_samples = cfg_args.network_audio.get('stft_hop', 0) * 2
                    crop = max(
                        cfg_args.network_audio.get('streaming_left_len', 0),
                        cfg_args.network_audio.get('stft_frame', 0),
                    ) + model_delay_samples
                    targets, outputs = _crop_streaming_loss(targets, outputs, crop)

                loss = loss_fn(targets, outputs)
                loss = loss / accumulation_steps

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if is_update_step:
                if hasattr(cfg_args, 'clip_grad_norm') and cfg_args.clip_grad_norm > 0:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg_args.clip_grad_norm,
                    )

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = mix.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size
            num_samples += batch_size
            td.set_postfix(loss=loss.item() * accumulation_steps)

    return total_loss / max(1, num_samples)


if __name__ == '__main__':
    print('Loading config file...')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

    if getattr(args, 'amp', None) is None:
        args.amp = 1
    if getattr(args, 'amp_dtype', None) is None:
        args.amp_dtype = 'bf16'

    args.dataset['type'] = 'online'
    args.dataset['name'] = args.finetune_csv
    is_aishell_csv = 'aishell' in str(args.dataset['name']).lower()
    if not is_aishell_csv:
        args.aishell_augment = 0
    _auto_disable_vector_cue_for_vox(args)

    ft_ckpt_name = f"soundBubble_{args.network_audio['backbone']}_finetune_aishell.pt"
    args.checkpoint_path = os.path.join(args.checkpoint_dir, ft_ckpt_name)

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.n_gpu > 1 and torch.cuda.device_count() > 1:
        args.distributed = True
        if args.local_rank == -1:
            print('Warning: Running in DataParallel mode. For DDP, use torchrun.')
            args.distributed = False
        else:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f'cuda:{args.local_rank}')
            print(f'Initialized DDP on rank {args.local_rank}')
    else:
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for ph in logger.handlers:
        logger.removeHandler(ph)

    if not args.distributed or args.local_rank == 0:
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler(f"logs/finetune_{args.network_audio['backbone']}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    model = get_model(args).to(device)

    load_model_weights(model, args.pretrained_checkpoint, device)
    freeze_encoder_only_train_decoder_output(
        model,
        train_last_n_layers=int(args.train_last_n_layers),
        use_direction=bool(args.vector_cue),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable parameters found after freezing.')

    optimizer = optim.AdamW(trainable_params, lr=float(args.finetune_learning_rate), weight_decay=0.01)
    dataloader = SimDataLoader(args)

    start_epoch = 0
    if args.train_from_last_checkpoint:
        start_epoch = load_resume_checkpoint(model, optimizer, args.checkpoint_path, device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    elif args.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_fn = get_loss_function(args.loss_type) if args.sstask else get_loss_function('sisdr')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.distributed or args.local_rank == 0:
        print(f'Random seed set to {args.seed}')
        print(f"Finetune CSV: {os.path.join(args.dataset['path'], args.dataset['name'])}")
        print(f"Checkpoint path: {args.checkpoint_path}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    no_improve_count = 0

    amp_enabled = bool(getattr(args, 'amp', 0)) and torch.cuda.is_available() and (device.type == 'cuda')
    amp_dtype = _resolve_amp_dtype(getattr(args, 'amp_dtype', 'bf16'))
    if amp_enabled and amp_dtype == torch.bfloat16:
        if hasattr(torch.cuda, 'is_bf16_supported') and (not torch.cuda.is_bf16_supported()):
            if not args.distributed or args.local_rank == 0:
                print('[AMP] bf16 not supported on this GPU/PyTorch build, falling back to fp16.')
            amp_dtype = torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_decay_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.min_lr,
    )

    if not args.distributed or args.local_rank == 0:
        print(f"[AMP] enabled={amp_enabled}, dtype={str(amp_dtype).replace('torch.', '')}, grad_scaler={scaler.is_enabled()}")
        logger.info(
            f"Finetune settings | val_interval={args.val_interval}, early_stop_patience={args.early_stop_patience}, "
            f"early_stop_min_delta={args.early_stop_min_delta}, init_lr={float(args.finetune_learning_rate)}, "
            f"train_last_n_layers={int(args.train_last_n_layers)}, aishell_augment={bool(args.aishell_augment)}"
        )

    start_time = time.time()
    should_stop = False

    for epoch in range(start_epoch, args.max_epoch//10):
        if args.distributed:
            dataloader['train'].sampler.set_epoch(epoch)

        if not args.distributed or args.local_rank == 0:
            print(f'\nEpoch {epoch + 1}/{args.max_epoch//10}')

        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            dataloader['train'],
            loss_fn,
            optimizer,
            device,
            args,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        train_time = time.time() - epoch_start

        if not args.distributed or args.local_rank == 0:
            print(f'Train Loss: {train_loss:.4f}')
            logger.info(
                f"Epoch {epoch + 1}/{args.max_epoch//10} | train_loss={train_loss:.6f} | "
                f"train_time={train_time:.2f}s | lr={optimizer.param_groups[0]['lr']:.6e}"
            )

        if (epoch + 1) % max(1, args.val_interval) == 0:
            val_start = time.time()
            val_loss = validate(
                model,
                dataloader['valid'],
                loss_fn,
                device,
                args,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            val_time = time.time() - val_start
            scheduler.step(val_loss)

            if not args.distributed or args.local_rank == 0:
                print(f'Validation Loss: {val_loss:.4f}')
                save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, args.checkpoint_path)
                print(f'Checkpoint saved to {args.checkpoint_path}')
                logger.info(
                    f"Epoch {epoch + 1}/{args.max_epoch} | val_loss={val_loss:.6f} | val_time={val_time:.2f}s | "
                    f"epoch_time={time.time() - epoch_start:.2f}s | total_time={time.time() - start_time:.2f}s | "
                    f"gap(val-train)={val_loss - train_loss:.6f}"
                )

                if val_loss < (best_val_loss - args.early_stop_min_delta):
                    best_val_loss = val_loss
                    no_improve_count = 0
                    best_path = args.checkpoint_path.replace('.pt', '_best.pt')
                    save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, best_path)
                    print(f'Best model saved to {best_path}')
                    logger.info(f'Best model updated at epoch {epoch + 1}, best_val={best_val_loss:.6f}')
                else:
                    no_improve_count += 1
                    logger.info(
                        f"No validation improvement: {no_improve_count}/{args.early_stop_patience} "
                        f"(best={best_val_loss:.6f}, current={val_loss:.6f})"
                    )
                    if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                        should_stop = True
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}.")

        if args.distributed:
            stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            break

    if args.distributed:
        dist.destroy_process_group()

    if not args.distributed or args.local_rank == 0:
        print('Finetuning completed!')
