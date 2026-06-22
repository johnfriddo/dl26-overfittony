import os
import sys
import argparse
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
from src.models.ast_student import EPICASTBaseline

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Student AST (distillazione cross-modale)")
    parser.add_argument("--test_csv", type=str,
                        default="data/epic-sounds-annotations/EPIC_Sounds_validation.csv")
    parser.add_argument("--frames_dir", type=str,
                        default="/home/gnfmrc01b01a494o/dataset/video")
    parser.add_argument("--hdf5_path", type=str,
                        default="/home/rsnnng02c19b202w/dl26-overfittony/data/EPIC_audio.hdf5")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/student_distillation/best_student.pth")
    parser.add_argument("--num_classes", type=int, default=44)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_csv", type=str, default="risultati_test_student.csv")
    return parser.parse_args()


def calc_metrics(all_probs, all_targets, num_classes):
    preds_top1 = np.argmax(all_probs, axis=1)
    top1 = np.mean(preds_top1 == all_targets) * 100
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5 = np.mean([t in p for t, p in zip(all_targets, top5_preds)]) * 100
    mca = balanced_accuracy_score(all_targets, preds_top1) * 100
    targets_onehot = label_binarize(all_targets, classes=range(num_classes))
    try:
        mAP = average_precision_score(targets_onehot, all_probs, average='macro') * 100
    except ValueError:
        mAP = 0.0
    return top1, top5, mca, mAP


@torch.no_grad()
def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = EPICSoundsDataset(
        annotations_file=args.test_csv,
        hdf5_path=args.hdf5_path,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = EPICASTBaseline(num_classes=args.num_classes).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"[ERRORE] Checkpoint non trovato: {args.checkpoint}")
        return

    print(f"Caricamento pesi da: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs, all_targets = [], []

    for mel_spec, audio_label in tqdm(loader, desc="Evaluation Student"):
        mel_spec = mel_spec.to(device)
        logits, _ = model(mel_spec)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(audio_label.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    top1, top5, mca, mAP = calc_metrics(all_probs, all_targets, args.num_classes)

    print("\n" + "=" * 45)
    print("   RISULTATI STUDENT (distillazione cross-modale)")
    print("=" * 45)
    print(f" Top-1 Accuracy : {top1:.2f}%")
    print(f" Top-5 Accuracy : {top5:.2f}%")
    print(f" mAP            : {mAP:.2f}%")
    print(f" mCA            : {mca:.2f}%")
    print("=" * 45)

    pd.DataFrame({
        "modello": ["AST_Student_Distillation"],
        "top1_acc": [top1],
        "top5_acc": [top5],
        "mAP": [mAP],
        "mCA": [mca],
    }).to_csv(args.output_csv, index=False)
    print(f"\nRisultati salvati in: {args.output_csv}")


if __name__ == "__main__":
    evaluate()
