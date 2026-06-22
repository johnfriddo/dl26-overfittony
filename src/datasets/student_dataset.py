import os
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TVF
import torchaudio.transforms as T

class MultimodalStudentDataset(Dataset):
    def __init__(self, csv_file, frames_dir, hdf5_path, num_frames=1, transform=None,
                 train=True, fps_ratio=4, sample_rate=24000, target_frames=1024):
        
        self.annotations = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.hdf5_path = hdf5_path
        self.num_frames = num_frames
        self.transform = transform
        self.train = train
        self.fps_ratio = fps_ratio
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.h5_file = None
        self._cache = {}  
        
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=128)
        
        if self.train:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self) -> int:
        return len(self.annotations)

    def _frame_files(self, video_id):
        if video_id not in self._cache:
            d = os.path.join(self.frames_dir, video_id)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing {d}")
            files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            files.sort(key=lambda f: int("".join(ch for ch in f if ch.isdigit()) or 0))
            if not files:
                raise FileNotFoundError(f"Empty {d}")
            self._cache[video_id] = (d, files)
        return self._cache[video_id]

    def _to_disk_idx(self, frame_60fps):
        return int((frame_60fps - 1) // self.fps_ratio + 1)

    @staticmethod
    def _randint(a, b):
        return int(torch.randint(int(a), int(b) + 1, (1,)).item())

    def _select_indices(self, lo, hi, n_avail):
        lo = max(1, min(lo, n_avail))
        hi = max(lo, min(hi, n_avail))
        if self.num_frames == 1:
            return [self._randint(lo, hi)] if self.train else [(lo + hi) // 2]
        
        bounds = np.linspace(lo, hi + 1, self.num_frames + 1)
        idxs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            a, b = int(a), max(int(a), int(b) - 1)
            idxs.append(self._randint(a, b) if self.train else (a + b) // 2)
        return idxs

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        video_id = str(row["video_id"]).strip()
        
        # -- 1. ESTRAZIONE VIDEO CON FALLBACK SILENZIOSO --
        try:
            d, files = self._frame_files(video_id)
            n_avail = len(files)
            
            raw_lo = self._to_disk_idx(int(row["start_frame"]))
            raw_hi = self._to_disk_idx(int(row["stop_frame"]))
            
            if raw_hi > n_avail:
                lo, hi = 1, n_avail
            else:
                lo, hi = max(1, raw_lo), min(raw_hi, n_avail)

            frames = []
            for k in self._select_indices(lo, hi, n_avail):
                k = min(max(k, 1), n_avail)
                img = Image.open(os.path.join(d, files[k - 1])).convert("RGB")
                img = img.resize((224, 224))
                frames.append(TVF.to_tensor(img))
            clip = torch.stack(frames)

            if self.transform:
                clip = self.transform(clip)

            if self.num_frames == 1:
                clip = clip.squeeze(0)
                
        except (FileNotFoundError, KeyError, ValueError, IndexError):
            # Fallback tensor per missing modality: NON stampiamo nulla per non intasare i log di validazione
            clip = torch.zeros((3, 224, 224)) if self.num_frames == 1 else torch.zeros((self.num_frames, 3, 224, 224))

        # -- 2. ESTRAZIONE AUDIO --
        start_sample = int(row['start_sample'])
        stop_sample = int(row['stop_sample'])
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            
        audio_segment = self.h5_file[video_id][start_sample:stop_sample]
        
        if len(audio_segment) == 0:
            audio_segment = np.zeros(self.sample_rate, dtype=np.float32)
            
        waveform = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(0)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)
        
        current_frames = mel_spec_db.shape[2]
        if current_frames < self.target_frames:
            mel_spec_db = F.pad(mel_spec_db, (0, self.target_frames - current_frames))
        elif current_frames > self.target_frames:
            mel_spec_db = mel_spec_db[:, :, :self.target_frames]
        
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        if self.train:
            if torch.rand(1).item() > 0.5: mel_spec_db = self.freq_mask(mel_spec_db)
            if torch.rand(1).item() > 0.5: mel_spec_db = self.time_mask(mel_spec_db)

        verb_label = int(row["verb_class"]) if "verb_class" in row else -1
        noun_label = int(row["noun_class"]) if "noun_class" in row else -1
        audio_label = int(row["class_id"]) if "class_id" in row else -1
        
        return clip, mel_spec_db, torch.tensor(verb_label, dtype=torch.long), torch.tensor(noun_label, dtype=torch.long), torch.tensor(audio_label, dtype=torch.long)