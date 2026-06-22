import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import wandb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dataset.student_dataset import MultimodalStudentDataset
from ast_student import EPICASTBaseline
from model_2d_student import EpicResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-Based Knowledge Distillation: Teacher (ResNet50) -> Student (AST)")

    # Path Configuration
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--hdf5_path", type=str, required=True)
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--student_weights", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints_distill")

    # Hyperparameters
    parser.add_argument("--num_classes", type=int, default=44)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_proj", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                         help="Stesso valore usato nel training del teacher (train_2d.py), per coerenza")
    parser.add_argument("--lambda_loss", type=float, default=0.5,
                         help="lambda * task_loss + (1-lambda) * distill_loss. "
                              "lambda=1.0 disattiva la distillazione (solo task), lambda=0.0 disattiva il task (solo distill)")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--distill_loss", type=str, default="cosine", choices=["mse", "cosine"],
                         help="Tipo di distillation loss: 'mse' (MSE su feature normalizzate) o 'cosine' (1 - cosine_similarity)")
    parser.add_argument("--freeze_projector_epochs", type=int, default=0,
                         help="Epoche iniziali di warmup col projector congelato. "
                              "Impostare >= epochs per disattivare la distillazione del tutto.")
    parser.add_argument("--color_jitter", action="store_true",
                         help="Applica ColorJitter ai frame in train (oltre a Normalize). Di default disattivato: "
                              "la distillazione vuole feature del teacher quanto piu' fedeli/stabili possibile, "
                              "l'augmentation fotometrica e' piu' utile quando si allena il teacher stesso (train_2d.py), "
                              "meno chiaro il beneficio quando le immagini servono solo a generare un target congelato.")
    parser.add_argument("--run_name", type=str, default=None, help="Nome del run per wandb, utile per distinguere gli esperimenti")

    return parser.parse_args()


def calc_metrics(all_probs, all_targets, num_classes):
    preds_top1 = np.argmax(all_probs, axis=1)
    top1_acc = np.mean(preds_top1 == all_targets) * 100

    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([target in top5 for target, top5 in zip(all_targets, top5_preds)]) * 100
    mca = balanced_accuracy_score(all_targets, preds_top1) * 100

    targets_onehot = label_binarize(all_targets, classes=range(num_classes))
    try:
        mAP = average_precision_score(targets_onehot, all_probs, average='macro') * 100
    except ValueError:
        mAP = 0.0

    try:
        mAUC = roc_auc_score(targets_onehot, all_probs, average='macro', multi_class='ovr') * 100
    except ValueError:
        mAUC = 0.0

    return top1_acc, top5_acc, mca, mAP, mAUC


def split_decay_params(module):
    """Esclude bias e parametri di LayerNorm dal weight decay."""
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "bias" in name or "layernorm" in name.lower() or "layer_norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def main():
    args = parse_args()

    os.environ["WANDB_MODE"] = "offline"
    run_name = args.run_name or f"fitnets_lambda_{args.lambda_loss}_warmup_{args.freeze_projector_epochs}"
    wandb.init(project="epic-sounds-distillation", name=run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device rilevato: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATALOADERS SETUP
    # -------------------------------------------------------------------------
    print("Inizializzazione dei moduli Dataset...")
    # Normalizzazione confermata identica a quella usata per il teacher in train_2d.py
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform_steps = []
    if args.color_jitter:
        train_transform_steps.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    train_transform_steps.append(v2.Normalize(mean=MEAN, std=STD))
    train_transform = v2.Compose(train_transform_steps)
    val_transform = v2.Compose([v2.Normalize(mean=MEAN, std=STD)])

    train_dataset = MultimodalStudentDataset(csv_file=args.train_csv, frames_dir=args.frames_dir,
                                              hdf5_path=args.hdf5_path, num_frames=1, train=True,
                                              transform=train_transform)
    val_dataset = MultimodalStudentDataset(csv_file=args.val_csv, frames_dir=args.frames_dir,
                                            hdf5_path=args.hdf5_path, num_frames=1, train=False,
                                            transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------------------------------------------------------
    # MODELS CONFIGURATION & WEIGHT LOADING
    # -------------------------------------------------------------------------
    print("Caricamento dei modelli...")
    teacher = EpicResNet(num_verbs=97, num_nouns=300, weights=None, fusion='late_pool', num_frames=1)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location='cpu'))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = EPICASTBaseline(num_classes=args.num_classes)
    if args.student_weights:
        print(f"Caricamento pesi baseline Student da: {args.student_weights}")
        student.load_state_dict(torch.load(args.student_weights, map_location='cpu'))
        # Reset head: il backbone mantiene le feature audio apprese, ma la head
        # viene reinizializzata per non ereditare l'overfitting del baseline.
        nn.init.xavier_uniform_(student.classifier.weight)
        nn.init.zeros_(student.classifier.bias)
    student.to(device)

    projector = nn.Linear(768, 2048).to(device)

    # -------------------------------------------------------------------------
    # OPTIMIZER, CRITERIONS & SCALER (AMP)
    # -------------------------------------------------------------------------
    criterion_task = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    student_decay, student_no_decay = split_decay_params(student)
    optimizer = optim.AdamW([
        {'params': student_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': student_no_decay, 'lr': args.lr, 'weight_decay': 0.0},
        {'params': projector.parameters(), 'lr': args.lr_proj, 'weight_decay': args.weight_decay},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_mAP = 0.0
    patience_counter = 0

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    print("Inizio addestramento Distillation...")
    for epoch in range(args.epochs):
        student.train()
        projector.train()

        # Warmup: il projector resta congelato le prime N epoche, cosi' solo
        # task_loss guida i pochi parametri allenabili dell'AST all'inizio.
        projector_active = epoch >= args.freeze_projector_epochs
        for p in projector.parameters():
            p.requires_grad = projector_active
        if epoch == args.freeze_projector_epochs:
            print(f"--> Epoca {epoch+1}: projector scongelato, distillazione attiva.")

        running_loss = 0.0
        running_task_loss = 0.0
        running_distill_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (clips, mel_spec_db, _, _, audio_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
            clips = clips.to(device, non_blocking=True)
            mel_spec_db = mel_spec_db.to(device, non_blocking=True)
            audio_labels = audio_labels.to(device, non_blocking=True)

            with torch.no_grad():
                _, _, teacher_feats = teacher(clips)

            with torch.amp.autocast('cuda'):
                student_logits, student_token = student(mel_spec_db)
                task_loss = criterion_task(student_logits, audio_labels)

                if projector_active:
                    projected_feats = projector(student_token)
                    if args.distill_loss == "cosine":
                        distill_loss = (1.0 - F.cosine_similarity(projected_feats, teacher_feats, dim=1)).mean()
                    else:
                        distill_loss = F.mse_loss(F.normalize(projected_feats, dim=1), F.normalize(teacher_feats, dim=1))
                    total_loss = (args.lambda_loss * task_loss) + ((1.0 - args.lambda_loss) * distill_loss)
                else:
                    distill_loss = torch.tensor(0.0, device=device)
                    total_loss = task_loss

                total_loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += total_loss.item() * args.gradient_accumulation_steps
            running_task_loss += task_loss.item()
            running_distill_loss += distill_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_task_loss = running_task_loss / len(train_loader)
        avg_distill_loss = running_distill_loss / len(train_loader)

        # -------------------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------------------
        student.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []

        with torch.no_grad():
            for _, mel_spec_db, _, _, audio_labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                mel_spec_db = mel_spec_db.to(device, non_blocking=True)
                audio_labels = audio_labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    student_logits, _ = student(mel_spec_db)
                    loss = criterion_task(student_logits, audio_labels)

                val_loss += loss.item()
                probs = F.softmax(student_logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(audio_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_probs, all_targets = np.array(all_probs), np.array(all_targets)

        top1, top5, mca, mAP, mAUC = calc_metrics(all_probs, all_targets, args.num_classes)

        print(f"\n[EPOCA {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Metrics -> Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | mAP: {mAP:.2f}% | mCA: {mca:.2f}%")
        # Gap esplicito train/val loss: e' la prima cosa da guardare per capire
        # se siamo ancora in overfit o se i fix hanno avuto effetto.
        print(f"Gap Train/Val Loss: {avg_val_loss - avg_train_loss:+.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/task_loss": avg_task_loss,
            "train/distill_loss": avg_distill_loss,
            "val/loss": avg_val_loss,
            "train_val_gap": avg_val_loss - avg_train_loss,
            "val/top1_acc": top1, "val/top5_acc": top5, "val/mCA": mca, "val/mAP": mAP, "val/mAUC": mAUC,
            "lr_student": optimizer.param_groups[0]["lr"], "projector_active": int(projector_active),
        }, step=epoch + 1)

        scheduler.step()

        if mAP > best_mAP:
            best_mAP = mAP
            patience_counter = 0
            torch.save(student.state_dict(), os.path.join(args.checkpoint_dir, "best_distilled_student.pth"))
            torch.save(projector.state_dict(), os.path.join(args.checkpoint_dir, "best_projector.pth"))
            print(f"--> Nuovo miglior modello salvato con mAP: {best_mAP:.2f}%")
        else:
            patience_counter += 1
            print(f"--> Nessun miglioramento del mAP da {patience_counter} epoche.")
            if patience_counter >= args.patience:
                print("EARLY STOPPING ATTIVATO.")
                break

    wandb.finish()


if __name__ == "__main__":
    main()