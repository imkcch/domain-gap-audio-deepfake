# Zero-Shot Deepfake Enhancement

**Status:** Private — code for paper under journal review  
**Funding:** NSTC College/University Student Research Project  
**Supervisor:** Prof. Wen-Chieh Fang, MPLL, NDHU
 
---
 
## Overview

Applying DR-DiffuSE (speech enhancement diffusion model) to audio deepfakes in a **zero-shot** manner — no task-specific retraining. The model only learns noise→clean mapping, but we test if it generalizes to fake→real.

This project is a continuation of prior undergraduate research at NDHU's Machine Perception and Learning Lab (MPLL) under Prof. Wen-Chieh Fang.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/train.py` | Train condition generator |
| `src/train_ddpm.py` | Train DDPM (DiffuSE / DiffuSEC) |
| `src/joint_finetune.py` | Train refiner |
| `src/test_ddpm.py` | Inference |
| `src/chunk_inference.py` | Our addition — chunked processing for long audio |

---

## Method
 
We apply the [DR-DiffuSE](https://github.com/judiebig/DR-DiffuSE) framework, a denoising diffusion probabilistic model originally designed for speech enhancement, to audio deepfake enhancement in a zero-shot manner.
 
**The key insight:** both acoustic noise and synthetic artifacts represent deviations from natural speech. A model trained to remove noise learns representations that partially generalize to artifact removal.

---
 
## Environment

**Server:** ASUS ROG RTX 4090 24GB, Intel Xeon W-3335, 256GB RAM  
**OS:** Ubuntu 24.04.3 LTS  
**Docker:** nvcr.io/nvidia/pytorch:23.07-py3
 
```bash
# Create and start container
docker run --gpus all -it --name istft \
  --shm-size=2g \
  -v ~/raproject:/workspace \
  -v lj_volume:/data/lj_volume \
  nvcr.io/nvidia/pytorch:23.07-py3 bash
 
# Access existing container
docker start istft && docker attach istft
 
# Navigate to working directory
cd /workspace/new_diffusion/DR-DiffuSE/src
```
 
### Dataset 
Mounted as read-only volume at /data/lj_volume/voicebank/
 
```
/data/lj_volume/voicebank/
├── clean_trainset_wav/    # 10,480 LJSpeech files
├── noisy_trainset_wav/    # 10,480 WaveFake files
├── clean_testset_wav/     # 2,620 LJSpeech files (held-out)
└── noisy_testset_wav/     # 2,620 WaveFake files (held-out)
```
**Vocoders used:** MelGAN, HiFi-GAN, Parallel WaveGAN, Multi-band MelGAN, WaveGlow

Data is organized in VoiceBank dataset structure as expected by the original DR-DiffuSE repo. No actual VoiceBank data is used — only LJSpeech and WaveFake.
 
---

## Installation
 
```bash
pip install -r requirements.txt
pip install wandb rich pystoi pesq
```

---

## Training
 
# Step 1: Condition generator
python train.py

# Step 2: DDPM variants
python train_ddpm.py --model DiffuSEC --wandb
python train_ddpm.py --model DiffuSEC --c_gen --wandb
python train_ddpm.py --model DiffuSE --c_gen --wandb

# Step 3: Refiner
python joint_finetune.py --fast_sampling --from_base --wandb

---

## Inference (Ablations)
# Baseline
python test_ddpm.py --model DiffuSE
python test_ddpm.py --model DiffuSEC

# Best traditional metrics result
python test_ddpm.py --model DiffuSE --c_gen
python test_ddpm.py --model DiffuSEC --fast_sampling

# Best perceptual quality
python test_ddpm.py --model DiffuSE --c_gen --fast_sampling --c_guidance --refine
python test_ddpm.py --model DiffuSEC --c_gen -- c_guidance

---

## Acknowledgements
 
This work was supported by the **National Science and Technology Council (NSTC)** via the NSTC College/University Student Research Project.
 
We acknowledge the authors of [DR-DiffuSE](https://github.com/judiebig/DR-DiffuSE) (Tai et al., AAAI 2023) for making their framework publicly available.
 
Supervised by **Prof. Wen-Chieh Fang**, Machine Perception and Learning Laboratory (MPLL), National Dong Hwa University, Taiwan.
 
---
 
## Citation
 
> Paper under review. Citation will be updated upon acceptance.
 
---
 
## Related Work
 
- [DR-DiffuSE](https://github.com/judiebig/DR-DiffuSE) — base framework (Tai et al., AAAI 2023)
- [WaveFake Dataset](https://github.com/RUB-SysSec/WaveFake) — Frank & Schönherr, NeurIPS 2021
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) — Ito & Johnson, 2017
- [NISQA](https://github.com/gabrielmittag/NISQA) — perceptual quality metric
