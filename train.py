from tkinter import EW
from models.model_wrapper import get_model
from dataset.dataset import SimDataLoader
from loss import get_loss_function, get_loss, Loss
import yamlargparse
import os,yaml
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import logging
from contextlib import nullcontext
from itertools import islice
from torch.cuda.amp import autocast, GradScaler

parser = yamlargparse.ArgumentParser("Settings")
parser.add_argument('--seed', type=int,default = 1234)
parser.add_argument('--config', help='config file path', default="/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/config/config_tfnetcasual.yaml")
parser.add_argument('--train_from_last_checkpoint', type=int, help='whether to train from a checkpoint, includes model weight, optimizer settings',default = 0)
parser.add_argument('--evaluate_only',  type=int, default=0, help='Only perform evaluation')
parser.add_argument('--n_gpu', type=int, default=8, help='number of GPUs to use (0 for CPU)')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (-1 for DataParallel)')
parser.add_argument('--gpu_list', type=str, default='0,1,2,3,4,5,6,7', help='list of GPUs to use')
parser.add_argument(
    '--dataset_size',
    type=int,
    default=0,
    help='limit number of training iterations (batches) per epoch for debugging; 0 means full epoch',
)
# AMP settings (let YAML config fill defaults when these are None)
parser.add_argument('--amp', type=int, default=None, help='Enable AMP autocast (1/0). None -> read from config or default to 1.')
parser.add_argument('--amp_dtype', type=str, default=None, help='AMP dtype: bf16 or fp16. None -> read from config or default to bf16.')


device = torch.device(f'cuda:0')  # Primary GPU
args = parser.parse_args()

print("Loading config file...")
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# 将config中的值合并到args
for key, value in config.items():
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)
args.checkpoint_path = os.path.join(args.checkpoint_dir, "soundBubble_" + args.network_audio['backbone'] + "_updated" + ".pt")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

# -------------------------
# AMP defaults & utilities
# -------------------------
if getattr(args, "amp", None) is None:
    args.amp = 1
if getattr(args, "amp_dtype", None) is None:
    args.amp_dtype = "bf16"

def _resolve_amp_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower().strip()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16", "half"]:
        return torch.float16
    # fallback
    return torch.bfloat16

def _use_causal_streaming_loss(args) -> bool:
    return (
        bool(args.network_audio.get("streaming_train", False))
    )


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


_auto_disable_vector_cue_for_vox(args)

def _crop_streaming_loss(targets: torch.Tensor, preds: torch.Tensor, left_len: int):
    if left_len <= 0:
        return targets, preds
    T = min(targets.shape[-1], preds.shape[-1])
    
    if T <= left_len:
        return targets[..., :T], preds[..., :T]
    return targets[..., left_len:T], preds[..., left_len:T]

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint with DataParallel and torch.compile compatibility"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle DataParallel state_dict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}

        # Remove 'module.' prefix if present (from DataParallel/DDP)
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            elif k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v  # remove '_orig_mod.' prefix
            else:
                new_state_dict[k] = v

        # Handle torch.compile compatibility
        if hasattr(model, '_orig_mod'):
            # If model is compiled, load into the original module
            model._orig_mod.load_state_dict(new_state_dict)
        else:
            # Regular model loading
            model.load_state_dict(new_state_dict)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")
        return start_epoch
    else:
        print("Checkpoint not found, starting training from scratch")
        return 0


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    """Save checkpoint with DataParallel and torch.compile compatibility"""
    # Get state_dict, removing DataParallel/DDP wrapper and torch.compile wrapper if present
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        # For DDP/DataParallel, get the underlying module
        underlying_model = model.module
    else:
        underlying_model = model

    # If the model is compiled, get the original module's state_dict
    if hasattr(underlying_model, '_orig_mod'):
        model_state_dict = underlying_model._orig_mod.state_dict()
    else:
        model_state_dict = underlying_model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)




def validate(model, dataloader, loss_fn, device, amp_enabled: bool = False, amp_dtype: torch.dtype = torch.float16):
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for sample in dataloader:
            # Move data to device
            # mix[B,2, max_length*16000]
            mix = sample['mix'].to(device)
            source1 = sample['source1'].to(device, dtype = torch.float)
            source2 = sample['source2'].to(device, dtype = torch.float)
            
            # Create reference signals tensor [batch, num_speakers, time]
            references = torch.stack([source1, source2], dim=1)

            # Forward pass
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                if args.vector_cue:
                    vector1 = sample['source1_vector'].to(device)
                    vector2 = sample['source2_vector'].to(device)
                    concat_vec = torch.cat([vector1,vector2],dim=1)
                    #[B,6]
                    if args.sstask:
                        outputs = model(mix,concat_vec)
                    else:
                        outputs = model(mix,vector1)
                else:
                    outputs = model(mix)
            # Compute loss
            if "moss" in args.network_audio['backbone']:
                    outputs = torch.stack(outputs, dim=2).transpose(1,2)

            # Compute loss in fp32 for stability
            outputs = outputs.float()
            targets = references if args.sstask else source1
            if _use_causal_streaming_loss(args):
                model_delay_samples = args.network_audio.get("stft_hop", 0) * 2
                crop = max(args.network_audio.get("streaming_left_len", 0), args.network_audio.get("stft_frame", 0)) + model_delay_samples
                targets, outputs = _crop_streaming_loss(targets, outputs, crop)
            loss = loss_fn(targets, outputs)

            # Update statistics
            batch_size = mix.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

    avg_loss = total_loss / num_samples
    if not args.distributed or args.local_rank == 0:
        logger.info(f"Epoch:{epoch+1} Validation Loss: {avg_loss:.4f}")
    return avg_loss

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler: GradScaler = None, amp_enabled: bool = False, amp_dtype: torch.dtype = torch.float16):
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    if args.accu_grad:
        world_size = args.n_gpu if args.distributed else 1
        accumulation_steps = args.effec_batch_size // (args.dataloader['batch_size'] * world_size)
        if accumulation_steps < 1:
            accumulation_steps = 1
    else:
        accumulation_steps = 1

    # Optionally limit number of iterations (batches) for debugging.
    # NOTE: tqdm objects are not sliceable; use islice on the underlying iterator instead.
    try:
        dataloader_len = len(dataloader)
    except TypeError:
        dataloader_len = None

    if args.dataset_size and args.dataset_size > 0:
        max_steps = min(args.dataset_size, dataloader_len) if dataloader_len is not None else args.dataset_size
        data_iter = islice(dataloader, max_steps)
        tqdm_total = max_steps
    else:
        max_steps = dataloader_len
        data_iter = dataloader
        tqdm_total = dataloader_len

    with tqdm(data_iter, total=tqdm_total) as td:
        for idx, sample in enumerate(td):
            # Move data to device

            mix = sample['mix'].to(device)
            source1 = sample['source1'].to(device, dtype = torch.float)
            source2 = sample['source2'].to(device, dtype = torch.float)

            # Create reference signals tensor [batch, num_speakers, time]
            references = torch.stack([source1, source2], dim=1)

            # Determine if we should sync gradients (only on update steps)
            # If we truncated the epoch, treat the last processed batch as an update step too.
            is_last_step = (max_steps is not None) and ((idx + 1) == max_steps)
            is_update_step = ((idx + 1) % accumulation_steps == 0) or is_last_step
            
            if args.distributed and not is_update_step:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                # Forward pass
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    if args.vector_cue:
                        vector1 = sample['source1_vector'].to(device)
                        vector2 = sample['source2_vector'].to(device)
                        concat_vec = torch.cat([vector1,vector2],dim=1)
                        if args.sstask:
                            outputs = model(mix,concat_vec)# [batch, num_speakers, time]
                        else:
                            outputs = model(mix,vector1)
                    else:
                        outputs = model(mix)
                        if "moss" in args.network_audio['backbone']:
                            outputs = torch.stack(outputs, dim=2).transpose(1,2)
                # Compute loss
                outputs = outputs.float()
                targets = references if args.sstask else source1
                if _use_causal_streaming_loss(args):
                    model_delay_samples = args.network_audio.get("stft_hop", 0) * 2
                    crop = max(args.network_audio.get("streaming_left_len", 0), args.network_audio.get("stft_frame", 0)) + model_delay_samples
                    targets, outputs = _crop_streaming_loss(targets, outputs, crop)
                loss = loss_fn(targets, outputs)
                
                # Normalize loss for accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            if is_update_step:

                # Gradient clipping
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update statistics (multiply back by accumulation_steps to log correct loss)
            batch_size = mix.size(0)
            total_loss += loss.item() * accumulation_steps * batch_size
            num_samples += batch_size
            
            # Update progress bar
            td.set_postfix(loss=loss.item() * accumulation_steps)
    avg_loss = total_loss / num_samples
    if not args.distributed or args.local_rank == 0:
        logger.info(f"Epoch:{epoch+1}     loss:{avg_loss}")
    return avg_loss


# Main training loop
if __name__ == "__main__":
    # Check for DDP environment variables (set by torchrun)
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    # Set up distributed training
    if args.n_gpu > 1 and torch.cuda.device_count() > 1:
        args.distributed = True
        if args.local_rank == -1:
             # Fallback to DataParallel if local_rank is not set (e.g. running without torchrun)
             print("Warning: Running in DataParallel mode. For DDP, use torchrun.")
             args.distributed = False
        else:
             dist.init_process_group(backend='nccl')
             torch.cuda.set_device(args.local_rank)
             device = torch.device(f'cuda:{args.local_rank}')
             print(f"Initialized DDP on rank {args.local_rank}")
    else:
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # loss_fn = loss_fn.to(device)
    
    # Setup logging only on rank 0
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for ph in logger.handlers:
        logger.removeHandler(ph)
        
    if not args.distributed or (args.distributed and args.local_rank == 0):
        # add FileHandler to log file
        formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(f"logs/simple_{args.network_audio['backbone']}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)
        
        # Add console handler as well
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter_file)
        logger.addHandler(ch)

    model = get_model(args)
    model = model.to(device)

    # Compile model first (before checkpoint loading and DDP setup)
    # model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.init_learning_rate,weight_decay=0.01)
    # Pass distributed flag to dataloader
    dataloader = SimDataLoader(args)
    start_epoch = 0
    if args.train_from_last_checkpoint and args.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path , device)
        if not args.distributed or args.local_rank == 0:
            print(f"Dataset sizes - Train: {len(dataloader['train'])}, Valid: {len(dataloader['valid'])}")

    # Setup DDP/DataParallel after checkpoint loading
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    elif args.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    
    # Create loss function
    if args.sstask:
        loss_fn = get_loss_function(args.loss_type)
        # loss_fn = get_loss(args.speaker_no)
        # loss_fn = Loss
    else:
        loss_fn = get_loss_function("sisdr")
        # Create optimizer (after model is on device)
    

    # Load checkpoint if specified (after model and optimizer are created and on correct device)
   
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not args.distributed or args.local_rank == 0:
        print(f"Random seed set to {args.seed}")

    # Create checkpoint directory
    if not args.distributed or args.local_rank == 0:
        os.makedirs(os.path.dirname(args.checkpoint_dir), exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    # AMP runtime settings (created after device is set)
    amp_enabled = bool(getattr(args, "amp", 0)) and torch.cuda.is_available() and (device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(getattr(args, "amp_dtype", "bf16"))
    if amp_enabled and amp_dtype == torch.bfloat16:
        if hasattr(torch.cuda, "is_bf16_supported") and (not torch.cuda.is_bf16_supported()):
            if not args.distributed or args.local_rank == 0:
                print("[AMP] bf16 not supported on this GPU/PyTorch build, falling back to fp16.")
            amp_dtype = torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    if not args.distributed or args.local_rank == 0:
        print(f"[AMP] enabled={amp_enabled}, dtype={str(amp_dtype).replace('torch.', '')}, grad_scaler={scaler.is_enabled()}")

    for epoch in range(start_epoch, args.max_epoch):
        if args.distributed:
             dataloader['train'].sampler.set_epoch(epoch)
             
        if not args.distributed or args.local_rank == 0:
            print(f"\nEpoch {epoch+1}/{args.max_epoch}")

        # Train one epoch
        train_loss = train_one_epoch(model, dataloader['train'], loss_fn, optimizer, device, scaler=scaler, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        
        if not args.distributed or args.local_rank == 0:
            print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if (epoch+1) % 10 == 0:
            val_loss = validate(model, dataloader['valid'], loss_fn, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
            
            if not args.distributed or args.local_rank == 0:
                print(f"Validation Loss: {val_loss:.4f}")

                # Save checkpoint
                checkpoint_path = args.checkpoint_path
                save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = checkpoint_path.replace('.pt', '_best.pt')
                    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_checkpoint_path)
                    print(f"Best model saved to {best_checkpoint_path}")

        # Learning rate scheduling (optional)
        # You can add learning rate scheduler here if needed

    if args.distributed:
        dist.destroy_process_group()

    if not args.distributed or args.local_rank == 0:
        print("Training completed!")
