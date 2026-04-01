# zero-shot-deepfake-enhancement

> **Paper:** *Towards Undetectable Audio Deepfakes: A Zero-Shot Transfer Learning Approach Using Speech Enhancement Models*
> **Status:** Under review for journal/conference submission
> **Funding:** National Science and Technology Council (NSTC), Taiwan — NSTC College/University Student Research Project
 
---
 
## Overview
 
This repository contains the code and documentation for our research on zero-shot transfer learning for audio deepfake enhancement. We investigate whether speech enhancement models trained exclusively for noise remduction can remove synthetic artifacts from audio deepfakes into more natural-sounding speech, without any task-specific retraining.

This project is a continuation of prior undergraduate research at NDHU's Machine Perception and Learning Lab (MPLL) under Prof. Wen-Chieh Fang.

---

## Method
 
We apply the [DR-DiffuSE](https://github.com/judiebig/DR-DiffuSE) framework, a denoising diffusion probabilistic model originally designed for speech enhancement, to audio deepfake enhancement in a zero-shot manner.
 
**The key insight:** both acoustic noise and synthetic artifacts represent deviations from natural speech. A model trained to remove noise learns representations that partially generalize to artifact removal.

---
 
## Environment
 
### Computing Server
We run the code on a computing server with ASUS ROG RTX 4090 24GB, Intel Xeon W-3335, 256GB DDR4 ECC.

### Docker Setup
 
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
 
### Dataset Structure (mounted as read-only volume)
 
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
 
### Step 1 — Train Condition Generator (c_gen) model. Navigate to working directory and run: 
```
/python train.py
```
### Step 2 — Train DDPM model. Navigate to working directory and run:
for DiffuSEC:
```
/python train_ddpm.py --model DiffuSEC --wandb
```
for DiffuSEC + condition generator:
```
/python train_ddpm.py --model DiffuSEC --c_gen --wandb
```
for DiffuSE + condition generator:
```
/python train_ddpm.py --model DiffuSE --c_gen --wandb
```
for refiner
```
python joint_finetune.py --fast_sampling --from_base --wandb
```
### Step 3 — Inference
```
/python test_ddpm.py --model DiffuSE --fast_sampling --c_gen --c_guidance --refine
```

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
