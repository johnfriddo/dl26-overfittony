#!/bin/bash
#SBATCH --job-name=ast_baseline
#SBATCH --output=experiments/logs/training_%j.out
#SBATCH --error=experiments/logs/training_%j.err

# --- CONFIGURAZIONI GCLUSTER DMI ---
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# --- ALLOCAZIONE GPU E VRAM ---
#SBATCH --gres=shard:11264

echo "Inizio addestramento sul nodo: $HOSTNAME"

cd /home/rsnnng02c19b202w/dl26-overfittony

export APPTAINERENV_WANDB_API_KEY="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"
export APPTAINERENV_WANDB_MODE="offline"
export APPTAINERENV_HDF5_USE_FILE_LOCKING="FALSE"

echo "Installazione librerie aggiuntive..."
singularity exec --nv /shared/sifs/latest.sif pip install --user wandb scikit-learn h5py

echo "Avvio ciclo di training AST..."
singularity exec --nv /shared/sifs/latest.sif python -m src.training.trainAST \
  --annotations_file ./data/epic-sounds-annotations/EPIC_Sounds_train.csv \
  --hdf5_path ./data/EPIC_audio.hdf5 \
  --batch_size 8 \
  --epochs 50 \
  --device cuda \
  --lr 1e-4

echo "Addestramento completato."