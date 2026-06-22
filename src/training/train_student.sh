#!/bin/bash
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/distill-%j.out
#SBATCH --error=logs/distill-%j.err

mkdir -p logs

# Dice a Python di cercare i moduli anche nella cartella corrente (root)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

apptainer run --nv --bind /home /shared/sifs/latest.sif python src/train_student.py \
  --train_csv "dataset/multimodal_train.csv" \
  --val_csv "dataset/multimodal_val.csv" \
  --frames_dir "/home/gnfmrc01b01a494o/dataset/video" \
  --hdf5_path "/home/rsnnng02c19b202w/dl26-overfittony/data/EPIC_audio.hdf5" \
  --teacher_weights "checkpoints/miglior_modello_late_pool-T8.pth" \
  --student_weights "/home/rsnnng02c19b202w/dl26-overfittony/experiments/checkpoints/best_ast_v2.pth" \
  --checkpoint_dir "checkpoints/student_distillation" \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --lambda_loss 0.5 \
  --lr 1e-5 \
  --lr_proj 5e-5 \
  --epochs 30 \
  --patience 6 \
  --freeze_projector_epochs 0 \
  "$@"
