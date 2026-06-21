import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.datasets.epic_sounds import EPICSoundsDataset
from src.models.efficientat import EPICEfficientATBaseline

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


VAL_CSV = 'data/epic-sounds-annotations/EPIC_Sounds_validation.csv'
HDF5_PATH = 'data/EPIC_audio.hdf5'
CHECKPOINT = './experiments/checkpoints/best_efficientat.pth' 
NUM_CLASSES = 44
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_topk_acc(probs, targets, k=1):
    topk_preds = np.argsort(probs, axis=1)[:, -k:]
    if k == 1:
        return np.mean(topk_preds[:, 0] == targets) * 100
    else:
        return np.mean([target in topk for target, topk in zip(targets, topk_preds)]) * 100

def calc_map(probs, targets, num_classes):
    targets_onehot = label_binarize(targets, classes=range(num_classes))
    try:
        return average_precision_score(targets_onehot, probs, average='macro') * 100
    except ValueError:
        return 0.0

def calc_mca(preds_top1, targets):
    return balanced_accuracy_score(targets, preds_top1) * 100

@torch.no_grad()
def evaluate():
    print(f"Avvio test su {VAL_CSV} | device: {device}")
    
    ds = EPICSoundsDataset(annotations_file=VAL_CSV, hdf5_path=HDF5_PATH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = EPICEfficientATBaseline(num_classes=NUM_CLASSES).to(device)
    
    if not os.path.exists(CHECKPOINT):
        print(f"\n[ERRORE] Checkpoint non trovato: {CHECKPOINT}")
        return

    print("Caricamento pesi del modello EfficientAT...")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device), strict=False)
    model.eval()

    all_probs = []
    all_targets = []

    for inputs, labels in tqdm(loader, desc="Test EfficientAT Baseline"):
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        all_probs.extend(probs)
        all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds_top1 = np.argmax(all_probs, axis=1)

    top1 = calc_topk_acc(all_probs, all_targets, k=1)
    top5 = calc_topk_acc(all_probs, all_targets, k=5)
    map_score = calc_map(all_probs, all_targets, NUM_CLASSES)
    mca = calc_mca(preds_top1, all_targets)

    print("\n" + "=" * 45)
    print(" RISULTATI FINALI SUL TEST SET (EFFICIENT-AT) ")
    print("=" * 45)
    print(f" Top-1 Accuracy : {top1:.2f}%")
    print(f" Top-5 Accuracy : {top5:.2f}%")
    print(f" mAP            : {map_score:.2f}%")
    print(f" mCA            : {mca:.2f}%")
    print("=" * 45)

    df_results = pd.DataFrame({
        'modello': ['EfficientAT'],
        'top1_acc': [top1],
        'top5_acc': [top5],
        'mAP': [map_score],
        'mCA': [mca]
    })
    df_results.to_csv('risultati_test_efficientat.csv', index=False)
    print("\nRisultati salvati in: risultati_test_efficientat.csv")

if __name__ == "__main__":
    evaluate()