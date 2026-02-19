from models.model_wrapper import get_model
import yamlargparse
import os,yaml
import torch
import pandas as pd
import librosa,soundfile,torchaudio
from dataset.dataset import SimDataLoader
import numpy as np
import torch.nn.functional as F
import ast
from loss import get_loss_function, get_loss, Loss
parser = yamlargparse.ArgumentParser("Settings")
parser.add_argument('--seed', type=int,default = 1234)
parser.add_argument('--config', help='config file path', default="/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/config/config_tfnetcasual.yaml")
parser.add_argument('--checkpoint_dir', type=str, help='the name of the log')
parser.add_argument('--train_from_last_checkpoint', type=int, help='whether to train from a checkpoint, includes model weight, optimizer settings',default = 0)
parser.add_argument('--evaluate_only',  type=int, default=0, help='Only perform evaluation')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use (0 for CPU)')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (-1 for DataParallel)')
parser.add_argument('--gpu_list', type=str, default='1', help='list of GPUs to use')



args = parser.parse_args()

print("Loading config file...")
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# 将config中的值合并到args
for key, value in config.items():
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)


def _normalize_direction_vector(vec, eps=1e-8):
    """Normalize direction vectors to unit length for relative direction."""
    if vec is None:
        return vec
    vec = np.asarray(vec, dtype=np.float32)
    if vec.size == 0:
        return vec
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm
model = get_model(args)
def save_audio(tensor,output_dir = "./",sample_rate=16000):
    tensor = tensor.detach().cpu()
    batch_size, num_sources, time_steps = tensor.shape
    for b in range(batch_size):
        for s in range(num_sources):
            source_wav = tensor[b,s,:]
            max_val = torch.abs(source_wav).max()
            if max_val > 0:
                source_wav = source_wav / max_val * 0.9  # 缩放到 [-0.9, 0.9] 留一点余量
            
            # 4. 调整维度以符合 torchaudio 要求: (Channels, Time)
            # 因为分离出的通常是单声道，我们unsqueeze增加一个维度
            source_wav = source_wav.unsqueeze(0) 
            
            # 5. 保存
            filename = f"{output_dir}/batch{b}_{s+1}.wav"
            torchaudio.save(filename, source_wav, sample_rate)
            print(f"Saved: {filename}")
def load_checkpoint(model, checkpoint_path):
    """Load checkpoint with DataParallel compatibility"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle DataParallel state_dict
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}

        # Remove 'module.' prefix if present (from DataParallel)
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)


        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")
        return model
    else:
        print("Checkpoint not found, starting training from scratch")
        return 0
def save_origins(csv_path,target,res):
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        if target in row['mix']:
            test_audio, _ = soundfile.read(row['mix'])
            soundfile.write(f"./saved_audio/{target[:-4]}.wav",test_audio,16000)
            source1, _ = soundfile.read(row['source1'],start=row['source1_start'], frames=16000*120)
            # source1 = source1[18*16000:24*16000]
            soundfile.write(f"./saved_audio/{target[:-4]}_source1.wav",source1,16000)
            source2, _ = soundfile.read(row['source2'],start=row['source2_start'], frames=16000*120)
            # source2 = source2[18*16000:24*16000]
            soundfile.write(f"./saved_audio/{target[:-4]}_source2.wav",source2,16000)
            if source1.ndim == 2:
                source1 = np.mean(source1, axis=1)
            if source2.ndim == 2:
                source2 = np.mean(source2, axis=1)
            source1 = torch.tensor(source1, dtype=torch.float32).unsqueeze(0)
            source2 = torch.tensor(source2, dtype=torch.float32).unsqueeze(0)
            targets = torch.stack([source1,source2],dim=1)

            loss = loss_fn(targets,res)
            print(f"loss: {loss.item()}"   )

model = load_checkpoint(model,f"./checkpoints/soundBubble_{args.network_audio['backbone']}_updated_best.pt")
model.eval()
device = torch.device("cuda:0")
loss_fn = get_loss_function("sisdr")
# args.dataset['type'] = 'pickle'
# args.dataset['path'] = "/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/dataset/"
# dataloader = SimDataLoader(args)
# train_loader = dataloader['train']
# from tqdm import tqdm
# with tqdm(train_loader) as td:
#     for idx,sample in enumerate(td):
#         if idx < 30:
#             continue
#         print(sample['mix_path'])
#         exit(0)
BASE_DIR = "/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_1000_updated/"
META_DIR = BASE_DIR.replace("/scenes_cafe_CH/","/metadata/")
CSV = "/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/dataset/mix_sources_updated.csv"
#train:['/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S04985000005_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S00355000049_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S05350000091_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S04720000235_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S01072000092_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S01072000238_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S04630000195_mix_CH0.wav', '/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S03281000170_mix_CH0.wav']
#valid:['/home/qysun/Neuro-SS/dataset/clarity/clarity_data/aad/scenes_cafe_notsofar/S05542000073_mix_CH0.wav',xs
filename = "S00042000225_mix_CH0.wav"
test_audio,_ = soundfile.read(BASE_DIR + filename)
if args.input_mono:
    test_audio = torch.mean(torch.tensor(test_audio, dtype=torch.float32),axis=1).unsqueeze(0)
else:
    test_audio = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0).transpose(1,2)
df = pd.read_csv(CSV)
for idx, row in df.iterrows():
    if filename in row['mix']:
        test_vector1 = ast.literal_eval(row['source1_vector'])
        test_vector2 = ast.literal_eval(row['source2_vector'])
        test_vector1 = _normalize_direction_vector(test_vector1)
        test_vector2 = _normalize_direction_vector(test_vector2)
        break
test_vector1 =  torch.tensor(test_vector1, dtype=torch.float32).unsqueeze(0)
test_vector2 =  torch.tensor(test_vector2, dtype=torch.float32).unsqueeze(0)
test_vector = torch.cat([test_vector1,test_vector2],dim=1)
print(test_vector)

print(test_audio.shape,test_audio.device)
args.streaming_train=False
res = model(test_audio,test_vector)
# res = torch.stack(res,dim=2).transpose(1,2)
save_audio(res,"/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/saved_audio")
save_origins(CSV,filename,res)