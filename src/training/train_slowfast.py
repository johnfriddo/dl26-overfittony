import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import wandb

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from datasets.epic_kitchen_dataset import EpicKitchensDataset
from models.epic_slowfast import EpicSlowFast

# ==========================================
# CONFIG
# ==========================================
NUM_FRAMES = 32          # fast pathway; slow = 32/alpha = 8 (alpha=4 nel modello)
RUN = f"slowfast_r50-T{NUM_FRAMES}"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_CSV = 'dataset/EPIC_100_train.csv'
VAL_CSV = 'dataset/EPIC_100_val.csv'
FRAMES_DIR = 'dataset/video'
PESI_SF = 'slowfast_r50_kinetics.pth'

os.environ["WANDB_MODE"] = "offline"
wandb.init(
    project="epic-kitchens-resnet",
    name=RUN,
    config={
        "modello": "slowfast_r50", "num_frames": NUM_FRAMES, "alpha": 4,
        "lr_backbone": 1e-5, "lr_head": 1e-4, "weight_decay": 1e-4,
        "epochs": 25, "patience": 8, "batch_size": 4,
        "dropout": 0.5, "label_smoothing": 0.1,
    },
)
cfg = wandb.config

# ==========================================
# TRASFORMAZIONI (statistiche SlowFast/Kinetics di pytorchvideo)
# ==========================================
SF_MEAN = [0.45, 0.45, 0.45]
SF_STD = [0.225, 0.225, 0.225]
train_transforms = v2.Compose([
    v2.Resize(256),
    v2.RandomCrop(224),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.Normalize(mean=SF_MEAN, std=SF_STD),
])
val_transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Normalize(mean=SF_MEAN, std=SF_STD),
])

train_dataset = EpicKitchensDataset(TRAIN_CSV, FRAMES_DIR, num_frames=cfg.num_frames,
                                    transform=train_transforms, train=True)
val_dataset = EpicKitchensDataset(VAL_CSV, FRAMES_DIR, num_frames=cfg.num_frames,
                                  transform=val_transforms, train=False)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Esperimento: {RUN} | device: {device}")

# ==========================================
# MODELLO, LOSS, OTTIMIZZATORE
# ==========================================
modello = EpicSlowFast(num_verbs=97, num_nouns=300, weights=PESI_SF,
                       dropout=cfg.dropout, alpha=cfg.alpha).to(device)

criterion_verb = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
criterion_noun = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

backbone_params = list(modello.backbone.parameters())
head_params = [p for n, p in modello.named_parameters() if not n.startswith("backbone.")]
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': cfg.lr_backbone},
    {'params': head_params, 'lr': cfg.lr_head},
], weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

best_metric = -1.0
epochs_no_improve = 0

# ==========================================
# METRICHE
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

# ==========================================
# CICLO DI ADDESTRAMENTO
# ==========================================
for epoch in range(cfg.epochs):
    print(f"\n{'='*40}\n EPOCA {epoch+1}/{cfg.epochs}\n{'='*40}")

    modello.train()
    train_loss = 0.0
    for clips, verb_labels, noun_labels in tqdm(train_loader, desc="Training  "):
        clips, verb_labels, noun_labels = clips.to(device), verb_labels.to(device), noun_labels.to(device)
        optimizer.zero_grad()
        pred_verbs, pred_nouns = modello(clips)
        loss = criterion_verb(pred_verbs, verb_labels) + criterion_noun(pred_nouns, noun_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    modello.eval()
    val_loss = 0.0
    all_vp, all_np, all_vl, all_nl = [], [], [], []
    with torch.no_grad():
        for clips, verb_labels, noun_labels in tqdm(val_loader, desc="Validation"):
            clips, verb_labels, noun_labels = clips.to(device), verb_labels.to(device), noun_labels.to(device)
            pred_verbs, pred_nouns = modello(clips)
            val_loss += (criterion_verb(pred_verbs, verb_labels) +
                         criterion_noun(pred_nouns, noun_labels)).item()
            all_vp.append(pred_verbs.softmax(1).cpu()); all_np.append(pred_nouns.softmax(1).cpu())
            all_vl.append(verb_labels.cpu()); all_nl.append(noun_labels.cpu())
    avg_val_loss = val_loss / len(val_loader)

    all_vp, all_np = torch.cat(all_vp), torch.cat(all_np)
    all_vl, all_nl = torch.cat(all_vl), torch.cat(all_nl)

    verb_top1, verb_top5 = calc_topk_acc(all_vp, all_vl, 1), calc_topk_acc(all_vp, all_vl, 5)
    verb_map, verb_mca = calc_map(all_vp, all_vl), calc_mca(all_vp, all_vl)
    noun_top1, noun_top5 = calc_topk_acc(all_np, all_nl, 1), calc_topk_acc(all_np, all_nl, 5)
    noun_map, noun_mca = calc_map(all_np, all_nl), calc_mca(all_np, all_nl)

    action_preds = (all_vp.unsqueeze(2) * all_np.unsqueeze(1)).view(all_vp.size(0), -1)
    action_labels = all_vl * 300 + all_nl
    action_top1 = calc_topk_acc(action_preds, action_labels, 1)
    action_top5 = calc_topk_acc(action_preds, action_labels, 5)
    action_map = calc_map(action_preds, action_labels)
    action_mca = calc_mca(action_preds, action_labels)
    del action_preds

    print(f"\nRISULTATI EPOCA {epoch+1}:")
    print(f" - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f" [VERBI] Top1 {verb_top1:.2f} | Top5 {verb_top5:.2f} | mAP {verb_map:.2f} | mCA {verb_mca:.2f}")
    print(f" [NOMI]  Top1 {noun_top1:.2f} | Top5 {noun_top5:.2f} | mAP {noun_map:.2f} | mCA {noun_mca:.2f}")
    print(f" [AZIONI] Top1 {action_top1:.2f} | Top5 {action_top5:.2f} | mAP {action_map:.2f} | mCA {action_mca:.2f}")

    wandb.log({
        "epoch": epoch + 1, "loss/train": avg_train_loss, "loss/val": avg_val_loss,
        "val/top1_verb": verb_top1, "val/top5_verb": verb_top5, "val/mAP_verb": verb_map, "val/mCA_verb": verb_mca,
        "val/top1_noun": noun_top1, "val/top5_noun": noun_top5, "val/mAP_noun": noun_map, "val/mCA_noun": noun_mca,
        "val/top1_action": action_top1, "val/top5_action": action_top5, "val/mAP_action": action_map, "val/mCA_action": action_mca,
        "lr": optimizer.param_groups[0]["lr"],
    })

    scheduler.step()

    monitor = (verb_top1 + noun_top1) / 2
    if monitor > best_metric:
        print(f"Metrica migliorata ({best_metric:.2f} -> {monitor:.2f}). Salvo il modello.")
        best_metric = monitor
        epochs_no_improve = 0
        torch.save(modello.state_dict(), f'miglior_modello_{RUN}.pth')
    else:
        epochs_no_improve += 1
        print(f"Nessun miglioramento ({epochs_no_improve}/{cfg.patience}).")

    torch.save({'epoch': epoch + 1, 'model_state_dict': modello.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric}, f'checkpoint_{RUN}.pth')

    if epochs_no_improve >= cfg.patience:
        print(f"\nEarly stopping: nessun miglioramento da {cfg.patience} epoche.")
        break

wandb.finish()
print("\nADDESTRAMENTO COMPLETATO")
