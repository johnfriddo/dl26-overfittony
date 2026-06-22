# Cross-Modal Knowledge Distillation (Audio to Vision)

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Group and Project Information
- **Group ID**: Overfittony
- **Project ID**: 5

## Project Description

Cross-Modal Knowledge Distillation pipeline from a video teacher to an audio-only student on EPIC-Kitchens/EPIC-Sounds (44-class sound event classification). The goal is to transfer visual context into a lightweight audio model that operates without video at inference time.

> For all theoretical details, performance analysis, architecture choices, and group contributions, refer to **[docs/REPORT.md](docs/REPORT.md)**.

---

## Technical Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/johnfriddo/dl26-overfittony.git
cd dl26-overfittony

# Linux / cluster
conda env create -f environment.yml

# macOS (Apple Silicon)
conda env create -f environment-mac.yml

conda activate dl-project
pip install wandb scikit-learn h5py
```

### 2. Data

**EPIC-Sounds** (audio): download annotations from the [official repository](https://github.com/epic-kitchens/epic-sounds-annotations) and place them under:
```
data/epic-sounds-annotations/
  EPIC_Sounds_train.csv
  EPIC_Sounds_validation.csv
```
Audio spectrograms (HDF5): place `EPIC_audio.hdf5` under `data/`.

**EPIC-Kitchens** (video): download annotations from the [official repository](https://github.com/epic-kitchens/epic-kitchens-100-annotations). Frames must be pre-extracted at 15 fps and placed under:
```
data/video/<video_id>/frame_000001.jpg ...
```

**Cross-modal alignment** (required for distillation training only): run once to generate the multimodal CSV splits:
```bash
python src/datasets/merge_annotations.py
```
This produces `dataset/multimodal_train.csv` and `dataset/multimodal_val.csv`.

---

### 3. Training

All models were trained on the DMI cluster (gcluster.dmi.unict.it) via SLURM + Apptainer, partition `dl-course-q2`, QoS `gpu-large` (4 CPU, 16G RAM, 11264 MB VRAM) or `gpu-xlarge` for the distillation. Jobs are submitted with `sbatch` and run inside `/shared/sifs/latest.sif`. The commands below can also be run locally with a GPU by replacing the `apptainer run --nv /shared/sifs/latest.sif` prefix with `python`.

#### Baseline Audio — AST (Cross-Entropy)

Submitted on cluster:
```bash
sbatch src/training/run_baseline.sh
```

Local equivalent:
```bash
python -m src.training.trainAST2_cross-entropy \
  --annotations_file data/epic-sounds-annotations/EPIC_Sounds_train.csv \
  --hdf5_path data/EPIC_audio.hdf5 \
  --batch_size 8 --epochs 50 --device cuda --lr 5e-5 --weight_decay 0.05
```

#### Baseline Audio — EfficientAT

Submitted on cluster:
```bash
sbatch src/training/run_efficientat.sh
```

Local equivalent:
```bash
python -m src.training.train_efficientat \
  --annotations_file data/epic-sounds-annotations/EPIC_Sounds_train.csv \
  --hdf5_path data/EPIC_audio.hdf5 \
  --batch_size 8 --epochs 50 --device cuda --lr 5e-5 --weight_decay 0.05
```

#### Teacher — ResNet-50 (and variants)

```bash
python -m src.training.train_resnet
```

#### Teacher — SlowFast-R50

```bash
python -m src.training.train_slowfast
```

#### Student — Distillation (AST + ResNet-50 teacher)

All 4 experiments submitted in sequence on cluster:
```bash
bash src/training/run_experiments.sh
```

Or submit a single experiment directly:
```bash
sbatch src/training/train_student.sh \
  --distill_loss cosine \
  --lambda_loss 0.5 \
  --checkpoint_dir checkpoints/student_distillation/exp2_cosine_l05 \
  --run_name exp2_cosine_l05
```

Local equivalent:
```bash
python src/training/train_student.py \
  --train_csv dataset/multimodal_train.csv \
  --val_csv dataset/multimodal_val.csv \
  --frames_dir data/video \
  --hdf5_path data/EPIC_audio.hdf5 \
  --teacher_weights checkpoints/miglior_modello_late_pool-T8.pth \
  --student_weights experiments/checkpoints/best_ast_v2.pth \
  --batch_size 16 --gradient_accumulation_steps 4 \
  --distill_loss cosine --lambda_loss 0.5 \
  --lr 1e-5 --epochs 30 --patience 6
```

---

### 4. Evaluation

#### Baseline AST
```bash
python -m src.evaluation.test_AST_cross-entropy
```

#### EfficientAT
```bash
python -m src.evaluation.test_EfficientAT
```

#### Teacher ResNet-50 / SlowFast
```bash
python -m src.evaluation.test_resnet
python -m src.evaluation.test_slowfast
```

#### Student (distillation)
```bash
python src/evaluation/test_student.py \
  --test_csv data/epic-sounds-annotations/EPIC_Sounds_validation.csv \
  --hdf5_path data/EPIC_audio.hdf5 \
  --checkpoint checkpoints/student_distillation/best_student.pth
```

#### Benchmark (size + inference time)
```bash
python -m src.evaluation.benchmark_AST
python -m src.evaluation.benchmark_AST_efficientat
```

---

*For individual contributions and AI usage declaration, refer to `docs/REPORT.md`.*
