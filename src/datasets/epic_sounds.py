import os
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
import torchaudio.transforms as T
from torch.utils.data import Dataset

class EPICSoundsDataset(Dataset):
    """
    Dataset PyTorch per caricare l'audio di EPIC-Sounds da un file HDF5,
    tagliarlo usando le annotazioni CSV e convertirlo in Spettrogrammi di Mel.
    """
    
    def __init__(self, annotations_file: str, hdf5_path: str, sample_rate: int = 24000, target_frames: int = 1024):
        """
        Args:
            annotations_file (str): Percorso al file CSV (es. EPIC_Sounds_train.csv).
            hdf5_path (str): Percorso al file HDF5.
            sample_rate (int): Frequenza di campionamento dell'audio (24kHz di default).
            target_frames (int): Numero fisso di frame temporali per lo spettrogramma.
        """

        self.annotations = pd.read_csv(annotations_file)
        self.hdf5_path = hdf5_path
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        
        # trasformazione da onda grezza a spettrogramma di Mel
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )

    def __len__(self) -> int:
        """Restituisce il numero totale di clip audio nel dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int):
        """Estrae e restituisce un singolo clip audio processato e la sua etichetta."""
        
        # prendiamo la riga corrispondente all'indice richiesto
        row = self.annotations.iloc[idx]
        
        video_id = row['video_id']
        start_sample = int(row['start_sample'])
        stop_sample = int(row['stop_sample'])
        label = int(row['class_id'])
        
        # estraiamo l'audio grezzo dall'HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            audio_segment = f[video_id][start_sample:stop_sample]
        
        # convertiamo l'array numpy in un tensore PyTorch (1, num_samples)
        waveform = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(0)
        
        # applichiamo la trasformazione in spettrogramma
        mel_spec = self.mel_spectrogram(waveform)
        
        # convertiamo la potenza in Decibel
        mel_spec_db = T.AmplitudeToDB()(mel_spec)
        
        # standardizziamo la lunghezza temporale dello spettrogramma
        current_frames = mel_spec_db.shape[2]
        
        if current_frames < self.target_frames:
            # aggiungiamo zero-padding sull'asse temporale (a destra)
            pad_amount = self.target_frames - current_frames
            mel_spec_db = F.pad(mel_spec_db, (0, pad_amount))
        elif current_frames > self.target_frames:
            # tronchiamo i frame in eccesso sull'asse temporale
            mel_spec_db = mel_spec_db[:, :, :self.target_frames]
        
        return mel_spec_db, label