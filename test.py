import librosa
import os
import numpy as np
from scipy.io import loadmat, wavfile
from scipy.io.wavfile import write as wavwrite
import pickle
from models.model import av_Mossformer_ConvTasnet
from models.dprnn import DPRNN
from dataset.dataset import SimDataLoader
from loss import get_loss_function
import yamlargparse
import os,yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import logging


parser = yamlargparse.ArgumentParser("Settings")
parser.add_argument('--seed', type=int,default = 1234)
parser.add_argument('--config', help='config file path', default="/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/config/dprnn_config.yaml")
parser.add_argument('--checkpoint_dir', type=str, help='the name of the log')
parser.add_argument('--train_from_last_checkpoint', type=int, help='whether to train from a checkpoint, includes model weight, optimizer settings',default = 0)
parser.add_argument('--evaluate_only',  type=int, default=0, help='Only perform evaluation')
parser.add_argument('--n_gpu', type=int, default=4, help='number of GPUs to use (0 for CPU)')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (-1 for DataParallel)')




args = parser.parse_args()


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


# 将config中的值合并到args
for key, value in config.items():
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)


# model = av_Mossformer_ConvTasnet(args).to('cuda:4')
model = DPRNN(args).to('cuda:0')

dataloader = SimDataLoader(args)['train']

# Create directory for saving audio files
save_dir = './saved_audio'
os.makedirs(save_dir, exist_ok=True)

# Create loss function
loss_fn = get_loss_function('sisdr')
optimizer = optim.Adam(model.parameters(), lr=args.init_learning_rate)
# Create optimizer
# a= torch.randn(4,96000).to('cuda:0')
reference = torch.randn(4,2,96000).to('cuda:0')
a = torch.mean(reference,dim=1)
print(a.shape)
for i in range(500):
    output = model(a)
    optimizer.zero_grad()
    loss = loss_fn(reference[:,0,:],output)
    loss.backward()
    optimizer.step()
    print(loss.item())
# with tqdm(dataloader) as td:
#     for batch_idx, sample in enumerate(td):
#         # print(sample)
#         mix = sample['mix']
#         source1 = sample['source1']
#         source2 = sample['source2']
        # print(sample['source1_vector'])
        # with open('/home/qysun/Neuro-SS/baseline/multi-channel/sound-bubble/dataset/data/train.pkl','rb') as f:
        #     data=pickle.load(f)
        # print(sample['source1_vector'].shape,sample['source1_vector'].transpose(0,1).shape)
        # print(sample['source1_vector'],sample['source1_vector'].transpose(0,1))

        # Save audio files for each batch
        # batch_size = mix.shape[0]
        # for i in range(batch_size):
        #     # Convert to numpy and ensure proper shape
        #     mix_audio = mix[i].cpu().numpy().squeeze()
        #     source1_audio = source1[i].cpu().numpy().squeeze()
        #     source2_audio = source2[i].cpu().numpy().squeeze()

        #     # Save WAV files
        #     global_idx = batch_idx * batch_size + i
        #     wavwrite(os.path.join(save_dir, f'mix_{global_idx:04d}.wav'), 16000, mix_audio.astype(np.float32))
        #     wavwrite(os.path.join(save_dir, f'source1_{global_idx:04d}.wav'), 16000, source1_audio.astype(np.float32))
        #     wavwrite(os.path.join(save_dir, f'source2_{global_idx:04d}.wav'), 16000, source2_audio.astype(np.float32))
        #     print(sample['mix_path'][i])
        #     break
        # # # Only process first few batches for testing
        # if batch_idx >= 1:  # Save only first 5 batches (20 samples) for testing
        #     break
        


# print(loss)
# print(reference.shape)