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
from models.epic_slowfast import EpicSlowFast

# ==========================================
# CONFIG
# ==========================================
TEST_CSV = 'dataset/EPIC_100_test.csv'
FRAMES_DIR = 'dataset/video'
NUM_FRAMES = 32
RUN = f"slowfast_r50-T{NUM_FRAMES}"
CKPT = f'miglior_modello_{RUN}.pth'
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SF_MEAN = [0.45, 0.45, 0.45]
SF_STD = [0.225, 0.225, 0.225]
test_tf = v2.Compose([v2.Resize(256), v2.CenterCrop(224),
                      v2.Normalize(mean=SF_MEAN, std=SF_STD)])

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
def main():
    if not os.path.exists(CKPT):
        print(f"Checkpoint mancante: {CKPT} (allena prima con train_slowfast.py)")
        return

    ds = EpicKitchensDataset(TEST_CSV, FRAMES_DIR, num_frames=NUM_FRAMES,
                             transform=test_tf, train=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = EpicSlowFast(97, 300, weights=None).to(device)   # i pesi Kinetics non servono in test
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    all_vp, all_np, all_vl, all_nl = [], [], [], []
    for clips, vl, nl in tqdm(loader, desc=f"Test {RUN}"):
        clips = clips.to(device)
        pv, pn = model(clips)
        all_vp.append(pv.softmax(1).cpu()); all_np.append(pn.softmax(1).cpu())
        all_vl.append(vl); all_nl.append(nl)

    vp, npd = torch.cat(all_vp), torch.cat(all_np)
    vl, nl = torch.cat(all_vl), torch.cat(all_nl)
    ap = (vp.unsqueeze(2) * npd.unsqueeze(1)).view(vp.size(0), -1)
    al = vl * 300 + nl

    row = {
        'esperimento': RUN,
        'verb_top1': calc_topk_acc(vp, vl, 1), 'verb_top5': calc_topk_acc(vp, vl, 5),
        'verb_mAP': calc_map(vp, vl), 'verb_mCA': calc_mca(vp, vl),
        'noun_top1': calc_topk_acc(npd, nl, 1), 'noun_top5': calc_topk_acc(npd, nl, 5),
        'noun_mAP': calc_map(npd, nl), 'noun_mCA': calc_mca(npd, nl),
        'action_top1': calc_topk_acc(ap, al, 1), 'action_top5': calc_topk_acc(ap, al, 5),
        'action_mAP': calc_map(ap, al), 'action_mCA': calc_mca(ap, al),
    }
    print(f"\n== {RUN} ==")
    print(f" VERBI  Top1 {row['verb_top1']:.2f} | Top5 {row['verb_top5']:.2f} | mAP {row['verb_mAP']:.2f} | mCA {row['verb_mCA']:.2f}")
    print(f" NOMI   Top1 {row['noun_top1']:.2f} | Top5 {row['noun_top5']:.2f} | mAP {row['noun_mAP']:.2f} | mCA {row['noun_mCA']:.2f}")
    print(f" AZIONI Top1 {row['action_top1']:.2f} | Top5 {row['action_top5']:.2f} | mAP {row['action_mAP']:.2f} | mCA {row['action_mCA']:.2f}")

    # accumula nel confronto: parte da risultati_test_all.csv se esiste, altrimenti da
    # risultati_test.csv (i 4 modelli 2D); sostituisce eventuale riga omonima
    base = None
    for f in ('risultati_test_all.csv', 'risultati_test.csv'):
        if os.path.exists(f):
            base = pd.read_csv(f, index_col=0)
            break
    df_new = pd.DataFrame([row]).set_index('esperimento')
    df_all = df_new if base is None else pd.concat([base.drop(index=RUN, errors='ignore'), df_new])
    df_all.round(2).to_csv('risultati_test_all.csv')
    print("\nSalvato: risultati_test_all.csv")


if __name__ == "__main__":
    main()
