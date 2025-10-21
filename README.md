# Music Diffusion Progressive Distillation (latent-space, EnCodec)

## 快速开始（单 GPU，5 小时数据假定）
1. 安装依赖：
   pip install -r requirements.txt
   # 如果 EnCodec 报错，请参照 https://github.com/facebookresearch/encodec 安装

2. 放置数据：
   Put .wav files (pure music) into `./data/wavs/`

3. 预处理（编码 windows -> latents）：
   python preprocess/preprocess_encodec.py --config configs/config.yaml

4. 训练教师（latent diffusion）：
   python train/teacher_train.py --config configs/config.yaml

5. 用教师做确定性采样生成蒸馏目标：
   python train/distill_prepare.py --config configs/config.yaml --teacher_ckpt checkpoints/teacher_latest.pth

6. 进行 progressive distillation：
   python train/progressive_distill.py --config configs/config.yaml --teacher_ckpt checkpoints/teacher_latest.pth

7. 评估并导出样本：
   python eval/decode_and_eval.py --config configs/config.yaml --student_ckpt checkpoints/student_final.pth
