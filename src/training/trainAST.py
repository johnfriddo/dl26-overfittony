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

    os.makedirs(args.checkpoint_dir, exist_ok=True)


    print("Caricamento dataset...")
    dataset = EPICSoundsDataset(annotations_file=args.annotations_file, hdf5_path=args.hdf5_path)
    
    # split 80% Training e 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    model = EPICASTBaseline(num_classes=args.num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float('inf')

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
        correct = 0
        total = 0
        
        # liste per l'F1-Score
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val F1: {val_f1:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/accuracy": val_accuracy,
            "val/f1_macro": val_f1
        })

        # --- SALVATAGGIO E EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_ast_baseline.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Nuovo miglior modello salvato in {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"--> Nessun miglioramento della Val Loss da {patience_counter} epoche.")
            
            if patience_counter >= patience:
                print(f"EARLY STOPPING INNESCATO! Il modello non migliora da {patience} epoche.")
                break
                
    wandb.finish()

if __name__ == "__main__":
    main()