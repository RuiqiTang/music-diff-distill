#!/usr/bin/env bash
# 1) preprocess (encode wavs to latents)
python preprocess/preprocess_encodec.py --config configs/config.yaml

# 2) try to finetune teacher (optional)
python teacher/teacher_finetune.py --config configs/config.yaml

# 3) generate teacher samples (adjust teacher_sampling.py as necessary)
python teacher/teacher_sampling.py --config configs/config.yaml --teacher_ckpt checkpoints/teacher_finetuned_latest.pth

# 4) prepare targets (encode teacher-generated wavs -> latents)
python distill/prepare_targets.py --config configs/config.yaml --teacher_gen_dir ./teacher_generated

# 5) progressive distillation
python distill/progressive_distill.py --config configs/config.yaml --teacher_ckpt checkpoints/teacher_finetuned_latest.pth

# 6) eval and decode
python eval/decode_and_eval.py --config configs/config.yaml --student_ckpt checkpoints/student_steps8_final.pth
