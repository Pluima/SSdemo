import os
import logging
import pickle
import numpy as np
import pandas as pd
import ast
import soundfile as sf
try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

ONLINE_REQUIRED_COLUMNS = {
    'mix',
    'source1',
    'source1_start',
    'source1_vector',
    'source2',
    'source2_start',
    'source2_vector',
}


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _is_valid_path(value):
    return isinstance(value, str) and len(value) > 0 and value.lower() != 'nan'


def _build_voxceleb_source_path(root, subset, speaker, utt):
    return os.path.join(
        os.path.expanduser(str(root)),
        "audio_clean",
        str(subset),
        str(speaker),
        f"{str(utt)}.wav",
    )


def _convert_voxceleb2_csv(df_raw, dataset_root):
    """
    Convert ClearerVoice/USEV VoxCeleb2 2mix csv:
    [partition,s1_subset,s1_spk,s1_utt,s1_db,s2_subset,s2_spk,s2_utt,s2_db,duration]
    into the unified online format used by this project.
    """
    if df_raw.shape[1] < 10:
        raise ValueError(f"VoxCeleb2 csv expects >=10 columns, got {df_raw.shape[1]}")

    converted = pd.DataFrame()
    converted['partition'] = df_raw.iloc[:, 0].astype(str).str.lower().str.strip()
    converted['source1'] = df_raw.apply(
        lambda r: _build_voxceleb_source_path(dataset_root, r.iloc[1], r.iloc[2], r.iloc[3]),
        axis=1,
    )
    converted['source2'] = df_raw.apply(
        lambda r: _build_voxceleb_source_path(dataset_root, r.iloc[5], r.iloc[6], r.iloc[7]),
        axis=1,
    )
    converted['source1_start'] = 0
    converted['source2_start'] = 0
    converted['source1_vector'] = [[0.0, 0.0, 0.0] for _ in range(len(converted))]
    converted['source2_vector'] = [[0.0, 0.0, 0.0] for _ in range(len(converted))]
    converted['source1_db'] = pd.to_numeric(df_raw.iloc[:, 4], errors='coerce').fillna(0.0)
    converted['source2_db'] = pd.to_numeric(df_raw.iloc[:, 8], errors='coerce').fillna(0.0)
    converted['duration'] = pd.to_numeric(df_raw.iloc[:, 9], errors='coerce').fillna(0.0)
    converted['mix'] = ''
    return converted


def _load_online_dataframe(csv_path, dataset_root):
    # Try normal headered csv first.
    df = pd.read_csv(csv_path)
    if ONLINE_REQUIRED_COLUMNS.issubset(set(df.columns)):
        return df, 'soundbubble'

    # Fallback: headerless format (e.g., VoxCeleb2 mixture_data_list_2mix.csv).
    df_raw = pd.read_csv(csv_path, header=None)
    if df_raw.shape[1] >= 10:
        first_col_values = set(df_raw.iloc[:, 0].astype(str).str.lower().str.strip().unique().tolist())
        if {'train', 'val', 'test'} & first_col_values:
            return _convert_voxceleb2_csv(df_raw, dataset_root), 'voxceleb2'

    missing = sorted(list(ONLINE_REQUIRED_COLUMNS - set(df.columns)))
    raise ValueError(
        f"Unsupported CSV format: {csv_path}. Missing required columns: {missing}"
    )

def _to_time_first(audio):
    """Ensure audio is (T, C) for multi-channel inputs."""
    if audio is None or audio.ndim != 2:
        return audio
    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        return audio.transpose(1, 0)
    return audio

def _downmix_to_2ch(audio, pair=(0, 4)):
    """Downmix multi-channel audio to stereo."""
    audio = _to_time_first(audio)
    if audio is None or audio.ndim != 2:
        return audio
    channels = audio.shape[1]
    if channels == 2:
        return audio
    if channels >= 8:
        left = np.mean(audio[:, 0:4], axis=1, keepdims=True)
        right = np.mean(audio[:, 4:8], axis=1, keepdims=True)
        return np.concatenate([left, right], axis=1)
    left_idx, right_idx = pair
    left_idx = int(left_idx)
    right_idx = int(right_idx)
    if left_idx < 0 or right_idx < 0 or left_idx >= channels or right_idx >= channels:
        left_idx, right_idx = 0, min(1, channels - 1)
    return audio[:, [left_idx, right_idx]]
    return audio[:, :2]

def _resample_audio_linear(audio, orig_sr, target_sr):
    """Fallback linear resampling for mono or multi-channel audio."""
    if audio is None or orig_sr == target_sr:
        return audio
    ratio = float(target_sr) / float(orig_sr)
    if audio.ndim == 1:
        new_len = int(round(len(audio) * ratio))
        if new_len <= 0:
            return audio[:0]
        x_old = np.arange(len(audio), dtype=np.float32)
        x_new = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
        return np.interp(x_new, x_old, audio).astype(audio.dtype)
    # audio shape: (T, C)
    new_len = int(round(audio.shape[0] * ratio))
    if new_len <= 0:
        return audio[:0]
    x_old = np.arange(audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, audio.shape[0] - 1, new_len, dtype=np.float32)
    resampled = np.empty((new_len, audio.shape[1]), dtype=audio.dtype)
    for ch in range(audio.shape[1]):
        resampled[:, ch] = np.interp(x_new, x_old, audio[:, ch])
    return resampled

def _resample_audio(audio, orig_sr, target_sr):
    """Resample audio to target_sr if needed."""
    if audio is None or orig_sr == target_sr:
        return audio
    if resample_poly is None:
        return _resample_audio_linear(audio, orig_sr, target_sr)
    # resample_poly supports 1D and ND with axis; audio is (T,) or (T, C)
    return resample_poly(audio, target_sr, orig_sr, axis=0)

def load_audio(file_path, sr=16000, start_sample=0, duration_seconds=120):
    """Load audio file using soundfile with optional start position (in target sr)."""
    try:
        if duration_seconds is None:
            duration_seconds = 0
        duration_samples = int(sr * duration_seconds)
        with sf.SoundFile(file_path) as f:
            file_sr = f.samplerate
            # Convert start position from target sr to file sr
            if sr > 0:
                start_in_file = int(round(start_sample * float(file_sr) / float(sr)))
            else:
                start_in_file = int(start_sample)
            if start_in_file < 0:
                start_in_file = 0
            f.seek(start_in_file)
            frames = int(round(duration_seconds * float(file_sr))) if duration_seconds > 0 else -1
            audio = f.read(frames=frames, dtype='float32', always_2d=False)

        # Resample if needed
        if file_sr != sr:
            audio = _resample_audio(audio, file_sr, sr)

        # Pad/trim to the expected length in target sr
        if duration_samples > 0:
            if len(audio) < duration_samples:
                padding = duration_samples - len(audio)
                if audio.ndim > 1:
                    audio = np.pad(audio, ((0, padding), (0, 0)))
                else:
                    audio = np.pad(audio, (0, padding))
            elif len(audio) > duration_samples:
                audio = audio[:duration_samples]

        return audio
    except Exception as e:
        # logging.error(f"Error loading {file_path}: {e}")
        return None

def normalize_audio_loudness(audio, target_lufs=-23.0):
    """Normalize audio loudness to target LUFS level using RMS-based approach"""
    if audio is None or len(audio) == 0:
        return audio

    # Calculate RMS of the audio
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return audio

    # Convert RMS to LUFS-like scale
    current_lufs = 20 * np.log10(rms)

    # Calculate gain needed to reach target loudness
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply gain
    normalized_audio = audio * gain_linear

    # Prevent clipping by checking peak after normalization
    max_val = np.max(np.abs(normalized_audio))
    if max_val > 1.0:
        # If clipping would occur, reduce gain to prevent it
        clipping_gain = 0.95 / max_val  # Leave some headroom
        normalized_audio = normalized_audio * clipping_gain

    return normalized_audio

def _calc_loudness_gain(audio, target_lufs=-23.0, eps=1e-8):
    """Return linear gain to match target loudness; 1.0 if invalid/zero."""
    if audio is None or len(audio) == 0:
        return 1.0
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < eps:
        return 1.0
    current_lufs = 20.0 * np.log10(rms + eps)
    return 10 ** ((target_lufs - current_lufs) / 20.0)

def normalize_audio_loudness_group(audios, target_lufs=-23.0, ref_audio=None):
    """
    Normalize a list of audio arrays with the same gain to preserve mix-source consistency.
    """
    if not audios:
        return audios
    ref = ref_audio
    if ref is None:
        for a in audios:
            if a is not None and len(a) > 0:
                ref = a
                break
    if ref is None:
        return audios
    gain = _calc_loudness_gain(ref, target_lufs=target_lufs)
    scaled = []
    for a in audios:
        if a is None:
            scaled.append(a)
        else:
            scaled.append(a * gain)
    # Prevent clipping with a shared post-gain
    max_val = 0.0
    for a in scaled:
        if a is None or len(a) == 0:
            continue
        max_val = max(max_val, float(np.max(np.abs(a))))
    if max_val > 1.0:
        clip_gain = 0.95 / max_val
        scaled = [a * clip_gain if a is not None else a for a in scaled]
    return scaled

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

class OnlineDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.input_mono = args.input_mono
        self.stereo_loss = args.stereo_loss
        self.stereo_pair = getattr(args, "stereo_pair", [0, 4])
        dataset_path = os.path.expanduser(str(args.dataset['path']))
        dataset_name = os.path.expanduser(str(args.dataset['name']))
        csv_path = dataset_name if os.path.isabs(dataset_name) else os.path.join(dataset_path, dataset_name)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file {csv_path} not found")

        voxceleb_root = os.path.expanduser(str(args.dataset.get('voxceleb_root', dataset_path)))
        df, csv_format = _load_online_dataframe(csv_path, voxceleb_root)
        self.csv_format = csv_format
        
        if self.csv_format == 'voxceleb2':
            part = df['partition'].astype(str).str.lower().str.strip()
            train = df[part == 'train']
            val = df[part.isin(['val', 'valid', 'validation'])]
            test = df[part == 'test']
            # Keep compatibility if val split is absent.
            if len(val) == 0 and len(train) >= 10:
                train, val = train_test_split(train, test_size=0.1, random_state=42)
        else:
            # Consistent split
            if len(df) >= 10:
                train_val, test = train_test_split(df, test_size=0.1, random_state=42)
                train, val = train_test_split(train_val, test_size=0.111, random_state=42)
            else:
                train, val, test = df, pd.DataFrame(), pd.DataFrame()
            
        if mode == 'train':
            self.data = train.reset_index(drop=True)
        elif mode == 'valid':
            self.data = val.reset_index(drop=True)
        elif mode == 'test':
            self.data = test.reset_index(drop=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Target segment length
        self.segment_length = 16000 * args.max_length
        # For fixed-length 2-minute data default to 120s; for VoxCeleb2 use per-row duration.
        if 'duration' in self.data.columns:
            durations = self.data['duration'].apply(_safe_float).to_numpy(dtype=np.float32)
            durations = np.maximum(durations, 0.0)
            sample_lengths = np.maximum(
                1,
                (durations * 16000.0 // float(self.segment_length)).astype(np.int64),
            )
        else:
            # Assume 2 minutes audio for existing sound-bubble CSV format.
            self.original_length = 16000 * 120
            segments = max(1, self.original_length // self.segment_length)
            sample_lengths = np.full(len(self.data), int(segments), dtype=np.int64)
        self.sample_segment_counts = sample_lengths
        self.sample_segment_cumsum = np.cumsum(self.sample_segment_counts) if len(self.sample_segment_counts) > 0 else np.array([], dtype=np.int64)

        # AISHELL-specific augmentation (enabled only for train split).
        csv_name = str(args.dataset.get('name', '')).lower()
        self.enable_aishell_aug = bool(getattr(args, 'aishell_augment', 0)) and (mode == 'train') and ('aishell' in csv_name)
        self.aug_gain_db_min = float(getattr(args, 'aishell_aug_gain_db_min', -4.0))
        self.aug_gain_db_max = float(getattr(args, 'aishell_aug_gain_db_max', 4.0))
        self.aug_noise_prob = float(getattr(args, 'aishell_aug_noise_prob', 0.6))
        self.aug_snr_db_min = float(getattr(args, 'aishell_aug_snr_db_min', 18.0))
        self.aug_snr_db_max = float(getattr(args, 'aishell_aug_snr_db_max', 35.0))

    def _random_db_gain(self):
        if self.aug_gain_db_max <= self.aug_gain_db_min:
            db = self.aug_gain_db_min
        else:
            db = np.random.uniform(self.aug_gain_db_min, self.aug_gain_db_max)
        return float(10.0 ** (db / 20.0))

    def _add_noise_to_mix(self, mix):
        if mix is None:
            return mix
        if np.random.rand() > self.aug_noise_prob:
            return mix

        mix_power = float(np.mean(np.square(mix, dtype=np.float64)))
        if mix_power < 1e-12:
            return mix

        if self.aug_snr_db_max <= self.aug_snr_db_min:
            snr_db = self.aug_snr_db_min
        else:
            snr_db = np.random.uniform(self.aug_snr_db_min, self.aug_snr_db_max)

        noise_power = mix_power / (10.0 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 1e-12)))
        noise = np.random.normal(0.0, noise_std, size=mix.shape).astype(mix.dtype, copy=False)
        return mix + noise

    def _apply_aishell_augment(self, source1, source2, mix):
        if not self.enable_aishell_aug:
            return source1, source2, mix

        gain = self._random_db_gain()
        source1 = source1 * gain
        source2 = source2 * gain
        mix = mix * gain
        mix = self._add_noise_to_mix(mix)

        max_val = max(
            float(np.max(np.abs(source1))) if source1 is not None and len(source1) > 0 else 0.0,
            float(np.max(np.abs(source2))) if source2 is not None and len(source2) > 0 else 0.0,
            float(np.max(np.abs(mix))) if mix is not None and len(mix) > 0 else 0.0,
        )
        if max_val > 1.0:
            clip_gain = 0.95 / max_val
            source1 = source1 * clip_gain
            source2 = source2 * clip_gain
            mix = mix * clip_gain
        return source1, source2, mix

    def __len__(self):
        if len(self.sample_segment_cumsum) == 0:
            return 0
        return int(self.sample_segment_cumsum[-1])

    def __getitem__(self, index):
        sample_idx = int(np.searchsorted(self.sample_segment_cumsum, index, side='right'))
        segment_start = 0 if sample_idx == 0 else int(self.sample_segment_cumsum[sample_idx - 1])
        segment_idx = int(index - segment_start)
        
        row = self.data.iloc[sample_idx]
        
        # Calculate start and end positions relative to the 2-minute clip
        start_pos_in_clip = segment_idx * self.segment_length
        
        # Audio loading parameters
        source1_start = _safe_int(row.get('source1_start', 0), 0) + start_pos_in_clip
        source2_start = _safe_int(row.get('source2_start', 0), 0) + start_pos_in_clip
        
        # Duration to load (just one segment)
        load_duration = self.args.max_length

        try:
            source1_path = str(row['source1'])
            source2_path = str(row['source2'])
            # Load sources directly for the needed segment
            source1 = load_audio(source1_path, start_sample=source1_start, duration_seconds=load_duration)
            source2 = load_audio(source2_path, start_sample=source2_start, duration_seconds=load_duration)
            
            if source1 is None or source2 is None:
                # Handle error (return zeros or raise)
                source1 = np.zeros(self.segment_length)
                source2 = np.zeros(self.segment_length)

            # For VoxCeleb2 csv, preserve relative loudness ratio encoded by db columns.
            if self.csv_format == 'voxceleb2' and source1 is not None and source2 is not None:
                db1 = _safe_float(row.get('source1_db', 0.0), 0.0)
                db2 = _safe_float(row.get('source2_db', 0.0), 0.0)
                rel_db = db2 - db1
                p1 = float(np.mean(np.square(source1, dtype=np.float64)))
                p2 = float(np.mean(np.square(source2, dtype=np.float64)))
                if p1 > 1e-12 and p2 > 1e-12:
                    scalar = (10.0 ** (rel_db / 20.0)) * np.sqrt(p1 / p2)
                    source2 = source2 * scalar
                
            # Load mix or create it
            mix_path = str(row.get('mix', '')) if not pd.isna(row.get('mix', '')) else ''
            if _is_valid_path(mix_path) and os.path.exists(mix_path):
                if "_CH/" in mix_path:
                    mix_start = 16000 * 3 + start_pos_in_clip
                else:
                    mix_start = start_pos_in_clip
                mix = load_audio(mix_path, start_sample=mix_start, duration_seconds=load_duration)
            else:
                mix = source1 + source2
            
            # mix = source1 + source2
            # Normalize loudness with a shared gain to keep mix-source consistency
            source1, source2, mix = normalize_audio_loudness_group(
                [source1, source2, mix],
                target_lufs=-23.0,
                ref_audio=mix,
            )
            source1, source2, mix = self._apply_aishell_augment(source1, source2, mix)
            
            # Convert to tensors
            mix = torch.tensor(mix, dtype=torch.float32)
            
            # Handle Mono/Multi-channel
            if self.input_mono:
                if mix.ndim > 1:
                    mix = torch.mean(mix, axis=1)
            elif mix.ndim > 1:
                mix = mix.transpose(0, 1)


            # Process sources
            if source1.ndim == 2:
                if not self.input_mono and self.stereo_loss:
                    source1 = _downmix_to_2ch(source1, self.stereo_pair).transpose()
                else:
                    source1 = np.mean(source1, axis=1)
            if source2.ndim == 2:
                if not self.input_mono and self.stereo_loss:
                    source2 = _downmix_to_2ch(source2, self.stereo_pair).transpose()
                else:
                    source2 = np.mean(source2, axis=1)

            # Vectors
            s1_vec_raw = row.get('source1_vector', [0.0, 0.0, 0.0])
            s2_vec_raw = row.get('source2_vector', [0.0, 0.0, 0.0])
            source1_vector = ast.literal_eval(s1_vec_raw) if isinstance(s1_vec_raw, str) and s1_vec_raw.strip().startswith('[') else s1_vec_raw
            source2_vector = ast.literal_eval(s2_vec_raw) if isinstance(s2_vec_raw, str) and s2_vec_raw.strip().startswith('[') else s2_vec_raw
            source1_vector = _normalize_direction_vector(source1_vector)
            source2_vector = _normalize_direction_vector(source2_vector)

            return {
                'source1': torch.tensor(source1, dtype=torch.float32),
                'source2': torch.tensor(source2, dtype=torch.float32),
                'mix': mix,
                'mix_path': mix_path,
                'source1_path': source1_path,
                'source2_path': source2_path,
                'length': self.segment_length,
                'original_sample_idx': sample_idx,
                'segment_idx': segment_idx,
            'source1_vector': torch.tensor(source1_vector, dtype=torch.float32),
            'source2_vector': torch.tensor(source2_vector, dtype=torch.float32)
            }
            
        except Exception as e:
            logging.error(f"Error processing sample {index}: {e}")
            # Return dummy data to prevent crash
            dummy_audio = torch.zeros(self.segment_length)
            return {
                'source1': dummy_audio,
                'source2': dummy_audio,
                'mix': dummy_audio if self.input_mono else torch.zeros(2, self.segment_length),
                'mix_path': '',
                'source1_path': '',
                'source2_path': '',
                'length': self.segment_length,
                'original_sample_idx': sample_idx,
                'segment_idx': segment_idx,
            'source1_vector': torch.zeros(3),
            'source2_vector': torch.zeros(3)
            }

class SimDataset(Dataset):
    def __init__(self,args,mode='train'):
        print(f"loading {mode} data from {args.dataset['path']}")
        with open(os.path.join(args.dataset['path'],"data/"+mode+".pkl"),'rb') as f:
            data = pickle.load(f)
        self.data=data
        self.input_mono = args.input_mono
        self.stereo_pair = getattr(args, "stereo_pair", [0, 4])
        
        # 原始数据长度（120秒 * 16000Hz = 1920000）
        self.original_length = 1920000
        # 目标片段长度（6秒 * 16000Hz = 96000）
        self.segment_length = 16000*args.max_length
        # 每个原始样本可以切分的片段数量
        self.segments_per_sample = self.original_length // self.segment_length
        # print(np.asarray(data[0]['source1']).shape)
        # exit(0)
        # self.mix=data['mix']
        # self.source1=data['source1']
        # self.source2=data['source2']

    def __len__(self):
        # 返回总的片段数量
        return len(self.data) * self.segments_per_sample

    def __getitem__(self, index):
        # 计算对应的原始样本索引和片段索引
        sample_idx = index // self.segments_per_sample
        segment_idx = index % self.segments_per_sample

        # 获取原始数据
        sample = self.data[sample_idx]
        # 计算片段的起始位置
        start_pos = segment_idx * self.segment_length
        end_pos = start_pos + self.segment_length
        mix = sample['mix'][start_pos:end_pos]
        mix = torch.tensor(mix, dtype=torch.float32)
        if self.input_mono:
            mix = torch.mean(mix,axis=1)
            # mix = mix[:,0]
        
        

        if not self.input_mono:
            mix = mix.transpose(0,1)
        source1 = sample['source1']
        source2 = sample['source2']
        if source1.ndim==2:
            if not self.input_mono:
                source1 = _downmix_to_2ch(sample['source1'], self.stereo_pair).transpose()
            else:
                source1=np.mean(sample['source1'],axis=1)
        if source2.ndim==2:
            if not self.input_mono:
                source2 = _downmix_to_2ch(sample['source2'], self.stereo_pair).transpose()
            else:
                source2=np.mean(sample['source2'],axis=1)
        # 切分数据
        #print([type(i) for i in sample['source1_vector']])
        #print([i for i in sample['source1']])
        source1_vector = _normalize_direction_vector(sample['source1_vector'])
        source2_vector = _normalize_direction_vector(sample['source2_vector'])
        segment = {
            'source1': source1[start_pos:end_pos],
            'source2': source2[start_pos:end_pos],
            'mix': mix,
            'mix_path': sample['mix_path'],
            'source1_path': sample['source1_path'],
            'source2_path': sample['source2_path'],
            'length': self.segment_length,
            'original_sample_idx': sample_idx,
            'segment_idx': segment_idx,
            'source1_vector': torch.tensor(source1_vector, dtype=torch.float32),
            'source2_vector': torch.tensor(source2_vector, dtype=torch.float32)
        }

        return segment


def SimDataLoader(args):
    
    # # Determine which dataset class to use
    # dataset_type = getattr(args.dataset, 'type', 'pickle') if hasattr(args, 'dataset') else 'pickle'
    # # Fallback to checking args directly if dataset is a dict
    # if isinstance(args.dataset, dict):
    #     dataset_type = args.dataset.get('type', 'pickle')
    dataset_type = args.dataset['type']
    DatasetClass = OnlineDataset if dataset_type == 'online' else SimDataset

    print(f"Using dataset type: {dataset_type}")

    datasets = {
        'train': DatasetClass(args, mode='train'),
        'valid': DatasetClass(args, mode='valid'),
        'test': DatasetClass(args, mode='test')
    }

    dataLoader = {}
    for ds_name, dataset in datasets.items():
        sampler = None
        if hasattr(args, 'distributed') and args.distributed:
            sampler = DistributedSampler(dataset, shuffle=(ds_name == 'train'))
        
        dataLoader[ds_name] = DataLoader(
            dataset,
            batch_size=args.dataloader['batch_size'],
            num_workers=args.dataloader['num_workers'],
            pin_memory=True,
            persistent_workers=True if args.dataloader['num_workers'] > 0 else False,
            prefetch_factor=4 if args.dataloader['num_workers'] > 0 else None,
            shuffle=(sampler is None and ds_name == 'train'),
            sampler=sampler
        )
    
    return dataLoader
