#!/bin/bash
# Lancia i 4 esperimenti di distillazione in sequenza.
# Setup fisso: teacher ResNet-50 late_pool T8, student AST, projector Linear(768->2048).
# Variabili: distill_loss (mse/cosine) e lambda_loss (0.3/0.5/0.7).

sbatch src/training/train_student.sh \
  --distill_loss mse \
  --lambda_loss 0.5 \
  --checkpoint_dir "checkpoints/student_distillation/exp0_mse_l05" \
  --run_name "exp0_mse_l05"

sbatch src/training/train_student.sh \
  --distill_loss mse \
  --lambda_loss 0.3 \
  --checkpoint_dir "checkpoints/student_distillation/exp1a_mse_l03" \
  --run_name "exp1a_mse_l03"

sbatch src/training/train_student.sh \
  --distill_loss mse \
  --lambda_loss 0.7 \
  --checkpoint_dir "checkpoints/student_distillation/exp1b_mse_l07" \
  --run_name "exp1b_mse_l07"

sbatch src/training/train_student.sh \
  --distill_loss cosine \
  --lambda_loss 0.5 \
  --checkpoint_dir "checkpoints/student_distillation/exp2_cosine_l05" \
  --run_name "exp2_cosine_l05"
