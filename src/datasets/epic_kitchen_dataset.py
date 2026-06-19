import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as F


class EpicKitchensDataset(Dataset):
    """
    Per ogni cartella ci sono i frame del video INTERO ricampionato a 15 fps (456x256), rinumerati da 1.

    Le etichette restituite sono gli ID ufficiali EPIC (verb 0-96, noun 0-299):
    le teste del modello vanno quindi dimensionate a 97 e 300.

    train=True  -> frame CASUALE (anti-overfitting: varieta' temporale tra epoche)
    train=False -> frame centrale/uniforme deterministico (riproducibile in val/test)
    """

    def __init__(self, csv_file, frames_dir, num_frames=1, transform=None,
                 train=True, fps_ratio=4):
        self.annotations = pd.read_csv(csv_file)
        self.frames_dir = frames_dir
        self.num_frames = num_frames
        self.transform = transform
        self.train = train
        self.fps_ratio = fps_ratio
        self._cache = {}  # video_id -> (dir, [file ordinati]); lazy, per-worker

    def __len__(self):
        return len(self.annotations)

    def _frame_files(self, video_id):
        """Lista ordinata dei frame della cartella. Nessuna assunzione sul formato
        del nome: indicizziamo per posizione. Se la cartella manca, fallisce in modo
        RUMOROSO."""
        if video_id not in self._cache:
            d = os.path.join(self.frames_dir, video_id)
            files = [f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            files.sort(key=lambda f: int("".join(ch for ch in f if ch.isdigit()) or 0))
            if not files:
                raise FileNotFoundError(f"Nessun frame trovato in {d}")
            self._cache[video_id] = (d, files)
        return self._cache[video_id]

    def _to_disk_idx(self, frame_60fps):
        # frame 1-indexed a 60 fps -> indice 1-indexed a 15 fps
        return int((frame_60fps - 1) // self.fps_ratio + 1)

    @staticmethod
    def _randint(a, b):
        # intero casuale in [a, b] usando il RNG di torch (seedato per-worker:
        # evita l'augmentation duplicata tra worker tipica di numpy.random)
        return int(torch.randint(int(a), int(b) + 1, (1,)).item())

    def _select_indices(self, lo, hi, n_avail):
        lo = max(1, min(lo, n_avail))
        hi = max(lo, min(hi, n_avail))
        if self.num_frames == 1:
            return [self._randint(lo, hi)] if self.train else [(lo + hi) // 2]
        # num_frames > 1: stile TSN (un frame per segmento)
        bounds = np.linspace(lo, hi + 1, self.num_frames + 1)
        idxs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            a, b = int(a), max(int(a), int(b) - 1)
            idxs.append(self._randint(a, b) if self.train else (a + b) // 2)
        return idxs

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_id = str(row["video_id"])
        verb_label = int(row["verb_class"])
        noun_label = int(row["noun_class"])

        d, files = self._frame_files(video_id)
        n_avail = len(files)
        lo = self._to_disk_idx(int(row["start_frame"]))
        hi = self._to_disk_idx(int(row["stop_frame"]))

        frames = []
        for k in self._select_indices(lo, hi, n_avail):
            k = min(max(k, 1), n_avail)               # clamp di sicurezza
            img = Image.open(os.path.join(d, files[k - 1])).convert("RGB")
            frames.append(F.to_tensor(img))           # -> [0,1], (C,H,W)
        clip = torch.stack(frames)                    # (T, C, H, W)

        if self.transform:
            clip = self.transform(clip)               # crop/flip/jitter/norm coerenti sui T frame

        if self.num_frames == 1:
            clip = clip.squeeze(0)                     # (C, H, W) per la ResNet 2D
        return clip, torch.tensor(verb_label, dtype=torch.long), \
            torch.tensor(noun_label, dtype=torch.long)
