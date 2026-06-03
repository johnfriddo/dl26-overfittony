import os
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.datasets.epic_sounds import EPICSoundsDataset
from src.models.ast import EPICASTBaseline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
from sklearn.metrics import f1_score
import torchaudio.transforms as T
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Training script per EPIC-Sounds AST Baseline")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path al file CSV di training")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path al file HDF5")
    parser.add_argument("--num_classes", type=int, default=44, help="Numero di classi nel dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche di addestramento")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight Decay per regolarizzazione")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo da utilizzare per l'addestramento (cpu, cuda, mps)")
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints", help="Cartella dove salvare i modelli")
    return parser.parse_args()


def main():
    args = parse_args()
    
    wandb.init(
        project="epic-sounds-ast",
        name=f"baseline_lr{args.lr}_bs{args.batch_size}",
        config=vars(args)
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        

        allowed_bytes = 11000 * 1024 * 1024 
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        

        fraction = allowed_bytes / total_bytes
        torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0))

        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")    

    print(f"Dispositivo in uso: {device}")


    print("Caricamento dataset...")
    dataset = EPICSoundsDataset(annotations_file=args.annotations_file, hdf5_path=args.hdf5_path)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Creazione degli split Train/Val basati su video_id...")

    full_df = pd.read_csv(args.annotations_file)
    

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(full_df, groups=full_df['video_id']))
    
    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]
    

    train_csv_path = "temp_train_annotations.csv"
    val_csv_path = "temp_val_annotations.csv"
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print("Caricamento dataset...")

    train_dataset = EPICSoundsDataset(annotations_file=train_csv_path, hdf5_path=args.hdf5_path)
    val_dataset = EPICSoundsDataset(annotations_file=val_csv_path, hdf5_path=args.hdf5_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    model = EPICASTBaseline(num_classes=args.num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_map = 0.0

    patience = 3  # epoche di tolleranza prima di fermare tutto
    patience_counter = 0

    # SPEC-AUGMENT (Maschere per Frequenza e Tempo)
    freq_masker = T.FrequencyMasking(freq_mask_param=15).to(device)
    time_masker = T.TimeMasking(time_mask_param=35).to(device)

    # ciclo di addestramento
    print(f"Inizio addestramento per {args.epochs} epoche...")
    for epoch in range(args.epochs):
        
        # --- FASE DI TRAINING ---
        model.train()
        running_loss = 0.0
        
        accumulation_steps = 4
        
        optimizer.zero_grad() 
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # applichiamo casualmente le mascherature solo sui dati di training
            inputs = freq_masker(inputs)
            inputs = time_masker(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            real_loss_value = loss.item() * accumulation_steps
            running_loss += real_loss_value
            
            if batch_idx % 10 == 0:
                print(f"   Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss corrente: {real_loss_value:.4f}")
                wandb.log({"train/batch_loss": real_loss_value})
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- FASE DI VALIDAZIONE ---
        model.eval()
        val_loss = 0.0
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # --- CALCOLO METRICHE EPIC-SOUNDS ---
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # 1. Top-1 Accuracy
        preds_top1 = np.argmax(all_probs, axis=1)
        top1_acc = np.mean(preds_top1 == all_targets) * 100
        
        # 2. Top-5 Accuracy
        top5_preds = np.argsort(all_probs, axis=1)[:, -5:] 
        top5_acc = np.mean([target in top5 for target, top5 in zip(all_targets, top5_preds)]) * 100
        
        # 3. mCA (Mean Per-Class Accuracy)
        mca_acc = balanced_accuracy_score(all_targets, preds_top1) * 100
        
        # 4. mAP (Mean Average Precision)
        targets_onehot = label_binarize(all_targets, classes=range(args.num_classes))
        
        try:
            map_score = average_precision_score(targets_onehot, all_probs, average='macro') * 100
        except ValueError:
            map_score = 0.0
        
        # 5. mAUC (Mean Area Under ROC Curve)
        try:
            mauc_score = roc_auc_score(targets_onehot, all_probs, average='macro', multi_class='ovr') * 100
        except ValueError:
            mauc_score = 0.0
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Metrics -> Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}% | mCA: {mca_acc:.2f}% | mAP: {map_score:.2f}% | mAUC: {mauc_score:.2f}%")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/top1_acc": top1_acc,
            "val/top5_acc": top5_acc,
            "val/mCA": mca_acc,
            "val/mAP": map_score,
            "val/mAUC": mauc_score
            
        })

        # --- SALVATAGGIO E EARLY STOPPING ---
        if map_score > best_map:
            best_map = map_score
            patience_counter = 0
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_ast_v2.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Nuovo miglior modello salvato! mAP migliorato a {best_map:.2f}%")
        else:
            patience_counter += 1
            print(f"--> Nessun miglioramento del mAP da {patience_counter} epoche (Miglior mAP: {best_map:.2f}%).")
            
            if patience_counter >= patience:
                print(f"EARLY STOPPING INNESCATO! Il mAP non migliora da {patience} epoche.")
                break
                
    wandb.finish()

    if os.path.exists(train_csv_path): os.remove(train_csv_path)
    if os.path.exists(val_csv_path): os.remove(val_csv_path)

if __name__ == "__main__":
    main()