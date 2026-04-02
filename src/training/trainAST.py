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


def parse_args():
    parser = argparse.ArgumentParser(description="Training script per EPIC-Sounds AST Baseline")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path al file CSV di training")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path al file HDF5")
    parser.add_argument("--num_classes", type=int, default=44, help="Numero di classi nel dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Dimensione del batch")
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche di addestramento")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo da utilizzare per l'addestramento (cpu, cuda, mps)")
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints", help="Cartella dove salvare i modelli")
    return parser.parse_args()

def main():
    args = parse_args()
    
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")    
    '''
    device = torch.device(args.device)
    print(f"Dispositivo in uso: {device}")

    # creazione cartella checkpoints se non esiste
    os.makedirs(args.checkpoint_dir, exist_ok=True)


    print("Caricamento dataset...")
    dataset = EPICSoundsDataset(annotations_file=args.annotations_file, hdf5_path=args.hdf5_path)
    
    # split 80% Training e 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # setup modello, loss e optimizer
    model = EPICASTBaseline(num_classes=args.num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # ciclo di addestramento
    print(f"Inizio addestramento per {args.epochs} epoche...")
    for epoch in range(args.epochs):
        
        # --- FASE DI TRAINING ---
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"   Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss corrente: {loss.item():.4f}")
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- FASE DI VALIDAZIONE ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # --- SALVATAGGIO DEI CHECKPOINT ---
        # salviamo i pesi solo se il modello è migliorato sulla validazione
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_ast_baseline.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Nuovo miglior modello salvato in {checkpoint_path}")

if __name__ == "__main__":
    main()