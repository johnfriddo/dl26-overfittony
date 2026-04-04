import os
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio.transforms as T
from torch.utils.data import Dataset
import numpy as np

class EPICSoundsDataset(Dataset):
    """
    Dataset PyTorch per caricare l'audio di EPIC-Sounds da un file HDF5,
    tagliarlo usando le annotazioni CSV e convertirlo in Spettrogrammi di Mel.
    """
    
    def __init__(self, annotations_file: str, hdf5_path: str, sample_rate: int = 24000, target_frames: int = 1024):
        self.annotations = pd.read_csv(annotations_file)
        self.hdf5_path = hdf5_path
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        

        self.h5_file = None
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        
        video_id = row['video_id']
        start_sample = int(row['start_sample'])
        stop_sample = int(row['stop_sample'])
        label = int(row['class_id'])
        
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
            pad_amount = self.target_frames - current_frames
            mel_spec_db = F.pad(mel_spec_db, (0, pad_amount))
        elif current_frames > self.target_frames:
            mel_spec_db = mel_spec_db[:, :, :self.target_frames]
        
        return mel_spec_db, label