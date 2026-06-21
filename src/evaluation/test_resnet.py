import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, average_precision_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from datasets.epic_kitchen_dataset import EpicKitchensDataset
from models.epic_resnet import EpicResNet

# ==========================================
# CONFIG
# ==========================================
TEST_CSV = 'dataset/EPIC_100_test.csv'
FRAMES_DIR = 'dataset/video'
BATCH_SIZE = 8

# (etichetta, fusion, num_frames). num_frames DEVE combaciare con quello usato in training:
# per late_fc/early l'architettura e' fissata su T; per single-frame e' 1.
# Il checkpoint atteso e' quello salvato dal train: miglior_modello_<fusion>-T<num_frames>.pth
RUNS = [
    ('single-frame',         'late_pool', 1),
    ('late-fusion-pooling',  'late_pool', 8),
    ('late-fusion-fc',       'late_fc',   8),
    ('early-fusion',         'early',     8),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# in test: solo center crop + normalize, e campionamento frame deterministico (train=False)
test_tf = v2.Compose([v2.CenterCrop(224), v2.Normalize(mean=MEAN, std=STD)])

# ==========================================
# METRICHE (identiche al train, per coerenza)
# ==========================================
def calc_topk_acc(preds, labels, k=1):
    _, topk = preds.topk(k, dim=1)
    return topk.eq(labels.view(-1, 1).expand_as(topk)).sum().item() / labels.size(0) * 100

def calc_map(preds, labels):
    preds_np, labels_np = preds.numpy(), labels.numpy()
    aps = [average_precision_score((labels_np == c).astype(int), preds_np[:, c])
           for c in np.unique(labels_np)]
    return float(np.mean(aps)) * 100 if aps else 0.0

def calc_mca(preds, labels):
    return balanced_accuracy_score(labels.numpy(), np.argmax(preds.numpy(), axis=1)) * 100


@torch.no_grad()
def evaluate(fusion, num_frames, ckpt):
    ds = EpicKitchensDataset(TEST_CSV, FRAMES_DIR, num_frames=num_frames,
                             transform=test_tf, train=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    # weights=None: non serve la ResNet ImageNet, carichiamo direttamente i pesi allenati
    model = EpicResNet(97, 300, weights=None, fusion=fusion, num_frames=num_frames).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_vp, all_np, all_vl, all_nl = [], [], [], []
    for clips, vl, nl in tqdm(loader, desc=f"Test {fusion}-T{num_frames}"):
        clips = clips.to(device)
        pv, pn = model(clips)
        all_vp.append(pv.softmax(1).cpu()); all_np.append(pn.softmax(1).cpu())
        all_vl.append(vl); all_nl.append(nl)

    vp, npd = torch.cat(all_vp), torch.cat(all_np)
    vl, nl = torch.cat(all_vl), torch.cat(all_nl)
    ap = (vp.unsqueeze(2) * npd.unsqueeze(1)).view(vp.size(0), -1)
    al = vl * 300 + nl

    return {
        'verb_top1': calc_topk_acc(vp, vl, 1), 'verb_top5': calc_topk_acc(vp, vl, 5),
        'verb_mAP': calc_map(vp, vl), 'verb_mCA': calc_mca(vp, vl),
        'noun_top1': calc_topk_acc(npd, nl, 1), 'noun_top5': calc_topk_acc(npd, nl, 5),
        'noun_mAP': calc_map(npd, nl), 'noun_mCA': calc_mca(npd, nl),
        'action_top1': calc_topk_acc(ap, al, 1), 'action_top5': calc_topk_acc(ap, al, 5),
        'action_mAP': calc_map(ap, al), 'action_mCA': calc_mca(ap, al),
    }


if __name__ == "__main__":
    print(f"Test su {TEST_CSV} | device: {device}")
    rows = []
    for label, fusion, nf in RUNS:
        ckpt = f'miglior_modello_{fusion}-T{nf}.pth'
        if not os.path.exists(ckpt):
            print(f"[skip] checkpoint mancante: {ckpt}")
            continue
        m = evaluate(fusion, nf, ckpt)
        m['esperimento'] = label
        rows.append(m)
        print(f"\n== {label} ==")
        print(f" VERBI  Top1 {m['verb_top1']:.2f} | Top5 {m['verb_top5']:.2f} | mAP {m['verb_mAP']:.2f} | mCA {m['verb_mCA']:.2f}")
        print(f" NOMI   Top1 {m['noun_top1']:.2f} | Top5 {m['noun_top5']:.2f} | mAP {m['noun_mAP']:.2f} | mCA {m['noun_mCA']:.2f}")
        print(f" AZIONI Top1 {m['action_top1']:.2f} | Top5 {m['action_top5']:.2f} | mAP {m['action_mAP']:.2f} | mCA {m['action_mCA']:.2f}")

    if rows:
        df = pd.DataFrame(rows).set_index('esperimento')
        df.round(2).to_csv('risultati_test.csv')
        print("\n==== CONFRONTO FINALE SUL TEST (Top-1) ====")
        print(df[['verb_top1', 'noun_top1', 'action_top1']].round(2).to_string())
        print("\nSalvato: risultati_test.csv (tutte le metriche)")
    else:
        print("\nNessun checkpoint trovato: allena prima i modelli.")
