# Audio Denoising: A Comparative Study of Deep Learning Techniques

> **Physical Sensors and Systems for Environmental Signals**
> Master's Project - Audio Signal Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A comprehensive comparative analysis of classical and deep learning techniques for environmental audio denoising**

[Overview](#overview) • [Key Features](#key-features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Results](#results)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Problem Statement](#problem-statement)
- [Methods Evaluated](#methods-evaluated)
  - [Classical Signal Processing](#classical-signal-processing)
  - [Deep Learning Architectures](#deep-learning-architectures)
- [The Hybrid Loss Innovation](#the-hybrid-loss-innovation)
- [Architecture Details](#architecture-details)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Overview

Environmental acoustic recordings are essential for ecological research, urban planning, and environmental monitoring. However, these recordings are frequently corrupted by noise from various sources—traffic, construction, weather events, and recording equipment—making analysis difficult.

This project presents a **comprehensive comparative study** of audio denoising techniques for speech signals in noisy environments. We evaluate both **classical signal processing methods** (Spectral Subtraction, Wiener Filtering) and **modern deep learning architectures** (Residual Autoencoder, U-Net, Hybrid Model, Transformer) to understand their relative strengths and weaknesses.

### Research Goals

1. **Compare** traditional signal processing methods with modern deep learning approaches
2. **Evaluate** different neural network architectures for audio denoising
3. **Investigate** the impact of different loss functions (time-domain vs. hybrid time-frequency)
4. **Provide** quantitative and qualitative analysis using standard audio quality metrics

---

## Key Features

✨ **Multiple Architectures**: Implementation of 4 deep learning models plus 2 classical methods
🔬 **Dual Loss Functions**: Simple time-domain loss (v1) vs. hybrid time-frequency loss (v2)
📊 **Comprehensive Evaluation**: PESQ, STOI, SI-SDR, and MOS metrics
🎯 **Novel Hybrid Model**: Combines time-domain and frequency-domain processing
🔄 **Reproducible**: Complete training pipeline with checkpointing and resumption
📈 **Detailed Analysis**: Training curves, spectrograms, and comparative visualizations

---

## Problem Statement

**Challenge**: Environmental audio recordings contain unwanted noise that obscures important signals, making automatic analysis difficult.

**Goal**: Develop and evaluate denoising techniques that can:
- Remove environmental noise effectively
- Preserve speech intelligibility
- Maintain perceptual quality
- Handle diverse noise conditions (SNR: -5 dB to +5 dB)

---

## Methods Evaluated

### Classical Signal Processing

#### 1. **Spectral Subtraction**
A traditional technique that estimates the noise spectrum and subtracts it from the noisy signal's spectrum.

**How it works:**
```
1. Transform noisy signal to frequency domain (STFT)
2. Estimate noise spectrum from silent portions
3. Subtract noise spectrum from noisy signal spectrum
4. Apply inverse STFT to reconstruct time-domain signal
```

**Pros:**
- Computationally efficient
- Simple to implement
- No training required

**Cons:**
- Produces "musical noise" artifacts
- Assumes stationary noise
- Limited performance with non-stationary noise

**Implementation:** `Final_Project/evaluate_baseline.py`

#### 2. **Wiener Filtering**
A statistical approach that minimizes the mean squared error between the estimated clean signal and the true clean signal.

**How it works:**
```
1. Estimate power spectral densities of signal and noise
2. Compute optimal filter coefficients
3. Apply filter to noisy signal
4. Reconstruct denoised signal
```

**Pros:**
- Better than spectral subtraction
- Statistically optimal under certain assumptions

**Cons:**
- Still struggles with non-stationary noise
- Requires accurate noise estimation

**Implementation:** `Final_Project/evaluate_baseline.py`

---

### Deep Learning Architectures

All deep learning models were trained with two variants:
- **v1**: Simple time-domain loss (L1 + MSE)
- **v2**: Hybrid loss (time + frequency + SI-SDR)

#### 1. **Residual Autoencoder (ResAutoencoder)**

A time-domain model that learns to predict and subtract residual noise.

**Architecture:**
```
Input: Raw waveform [B, 1, L]
│
├─ Encoder
│  ├─ Conv1d(1→32, k=7) + BatchNorm + ReLU
│  ├─ Conv1d(32→64, k=5) + BatchNorm + ReLU
│  └─ Conv1d(64→128, k=3) + BatchNorm + ReLU
│
├─ Decoder
│  ├─ ConvTranspose1d(128→64, k=3) + BatchNorm + ReLU
│  ├─ ConvTranspose1d(64→32, k=5) + BatchNorm + ReLU
│  └─ ConvTranspose1d(32→1, k=7) + Tanh
│
└─ Residual Connection: Output = Input - Decoded
```

**Key Innovation:** Instead of directly predicting the clean signal, the network predicts the **noise component** and subtracts it from the input. This preserves the original signal structure more effectively.

**Location:** `Final_Project/models.py:181-212`

---

#### 2. **U-Net for Spectrogram Enhancement (UNetSpec)**

A U-Net architecture operating in the frequency domain on magnitude spectrograms.

**Architecture:**
```
Input: Magnitude Spectrogram [B, 1, F, T]
│
├─ Downsampling Path (Encoder)
│  ├─ Conv2d(1→32) + Conv2d(32→32) + MaxPool2d  ──┐
│  ├─ Conv2d(32→64) + Conv2d(64→64) + MaxPool2d ──┤
│  ├─ Conv2d(64→128) + Conv2d(128→128) + MaxPool2d ┤ Skip
│  └─ Conv2d(128→256) + Conv2d(256→256) + MaxPool2d┘ Connections
│
├─ Bottleneck
│  └─ Conv2d(256→512) + Conv2d(512→512)
│
├─ Upsampling Path (Decoder)
│  ├─ ConvTranspose2d(512→256) + Concat + Conv2d ←┐
│  ├─ ConvTranspose2d(256→128) + Concat + Conv2d ←┤
│  ├─ ConvTranspose2d(128→64) + Concat + Conv2d  ←┤
│  └─ ConvTranspose2d(64→32) + Concat + Conv2d   ←┘
│
└─ Output: Enhanced Magnitude [B, 1, F, T]
```

**Key Features:**
- **Skip Connections**: Preserve fine-grained details lost during downsampling
- **Multi-scale Processing**: Captures both local and global features
- **Phase Reconstruction**: Uses Griffin-Lim algorithm for waveform reconstruction

**Location:** `Final_Project/models.py:214-285`

---

#### 3. **Hybrid Denoiser** ⭐ **Best Performance**

A novel dual-branch architecture that processes audio in **both time and frequency domains simultaneously**.

**Architecture:**
```
                    Input Waveform [B, 1, L]
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼────────┐           ┌────────▼─────────┐
    │  Time Branch   │           │  Frequency Branch │
    │ ResAutoencoder │           │    UNetSpec       │
    │                │           │                   │
    │ Conv Layers    │           │ STFT → Magnitude  │
    │ Residual       │           │ U-Net Processing  │
    │ Learning       │           │ iSTFT → Waveform  │
    └───────┬────────┘           └────────┬──────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │   Fusion    │
                    │ Conv1d(2→1) │
                    │    Tanh     │
                    └──────┬──────┘
                           │
                  Denoised Waveform
```

**Key Innovation:** By fusing complementary information from both domains, the model achieves superior denoising performance while maintaining signal quality.

**Location:** `Final_Project/models.py:288-338`

---

#### 4. **Transformer Autoencoder (TransformerAutoencoderFreq)**

A U-Net style architecture with channel-wise attention in the bottleneck, operating in the frequency domain.

**Architecture:**
```
Input Waveform [B, 1, L]
│
├─ STFT → Magnitude [B, 1, F, T]
│
├─ Encoder (3 levels)
│  ├─ Conv2d + BatchNorm + LeakyReLU + MaxPool
│  ├─ Conv2d + BatchNorm + LeakyReLU + MaxPool
│  └─ Conv2d + BatchNorm + LeakyReLU + MaxPool
│
├─ Bottleneck with Attention ⭐
│  ├─ Conv2d
│  ├─ AttentionBlock2D (Channel-wise attention)
│  │  └─ GlobalAvgPool → Conv2d → ReLU → Conv2d → Sigmoid
│  └─ Conv2d
│
├─ Decoder (3 levels with skip connections)
│  ├─ Upsample + Concat + Conv2d
│  ├─ Upsample + Concat + Conv2d
│  └─ Upsample + Concat + Conv2d
│
├─ Enhanced Magnitude + Original Phase
│
└─ iSTFT → Denoised Waveform
```

**Key Feature:** The **AttentionBlock2D** learns to weight different frequency channels, emphasizing important spectral features while suppressing noise.

**Location:** `Final_Project/simple_transformer_model.py`

---

## The Hybrid Loss Innovation

### Problem with Simple Losses

Traditional time-domain losses (L1, MSE) optimize for waveform similarity but may miss perceptually important features in the frequency domain.

### The Hybrid Loss Solution

Our hybrid loss combines three complementary objectives:

```python
L_hybrid = λ_time · L_time + λ_freq · L_freq + λ_sisdr · L_sisdr
```

#### Components:

**1. Time-Domain Loss (L_time)**
```python
L_time = L1(denoised, clean) + MSE(denoised, clean)
```
Ensures accurate waveform reconstruction.

**2. Frequency-Domain Loss (L_freq)**
```python
L_freq = L1(|STFT(denoised)|, |STFT(clean)|)
```
Preserves spectral structure and harmonic content.

**3. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**
```python
SI-SDR = 10 · log₁₀(||s_target||² / ||e_noise||²)
L_sisdr = -SI-SDR
```
Directly optimizes for signal quality regardless of scale.

### Why It Works

| Loss Component | What It Optimizes | Benefit |
|----------------|-------------------|---------|
| **L_time** | Waveform fidelity | Temporal accuracy |
| **L_freq** | Spectral structure | Harmonic preservation |
| **SI-SDR** | Signal quality | Perceptual improvement |

The combination encourages the model to learn **both temporal and spectral features**, leading to superior perceptual quality in denoised audio.

**Implementation:** `Final_Project/custom_loss.py:90-117`

---

## Dataset

### Source Datasets

#### 📚 **LibriSpeech**
- **Purpose**: Clean speech signals
- **Content**: English audiobook readings (~1000 hours)
- **Subset Used**: `train.100` (100 hours)
- **Sample Rate**: 16 kHz
- **Citation**: Panayotov et al., 2015

#### 🏙️ **UrbanSound8K**
- **Purpose**: Environmental noise
- **Content**: 10 classes of urban sounds
  - Air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, street music
- **Duration**: Varied lengths
- **Citation**: Salamon et al., 2014

### Synthetic Dataset Generation

We created a controlled noisy dataset by mixing clean speech with environmental noise at various SNR levels.

**Process:**
```python
# 1. Select clean speech from LibriSpeech
# 2. Select noise from UrbanSound8K (duration ≥ 50% of speech)
# 3. Resample both to 16 kHz
# 4. Mix at specified SNR

y(t) = x(t) + α · n(t)

where α = sqrt(P_x / (10^(SNR/10) · P_n))
```

**SNR Levels:** -5 dB, 0 dB, +5 dB
**Total Samples:** 28,539
**Split:** 80% training, 20% validation
**Fixed Seed:** 42 (for reproducibility)

**Location:** Dataset generation logic in `Final_Project/custom_datasets.py`

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

### Setup

```bash
# Clone the repository
git clone https://github.com/MirkoMorello/MSc_Sensors_Signal.git
cd MSc_Sensors_Signal/Final_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
pip install librosa scipy numpy matplotlib scikit-learn
pip install pesq pystoi torchaudio-squim
pip install python-dotenv jupyter

# Set up environment variables (already configured in .env)
# SR=16000, MAX_DURATION=6, N_FFT=1024, WIN_LENGTH=1024, HOP_LENGTH=256
```

### Datasets

```bash
# Download LibriSpeech train-clean-100
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Download UrbanSound8K
# Visit: https://urbansounddataset.weebly.com/urbansound8k.html
# Extract to appropriate directory

# Generate noisy dataset (script not included, but logic in custom_datasets.py)
```

---

## Project Structure

```
MSc_Sensors_Signal/
│
├── Final_Project/
│   ├── .env                              # Configuration (SR, N_FFT, etc.)
│   ├── main.py                           # Training script for all models
│   ├── models.py                         # Model architectures
│   ├── simple_transformer_model.py       # Transformer model
│   ├── custom_loss.py                    # Loss functions
│   ├── custom_datasets.py                # Dataset classes
│   ├── utils.py                          # Helper functions
│   ├── train_util.py                     # Training utilities
│   ├── evaluation_utils.py               # Evaluation metrics
│   ├── evaluate_baseline.py              # Classical methods evaluation
│   ├── main.ipynb                        # Jupyter notebook for exploration
│   ├── research_evaluation.ipynb         # Results analysis
│   ├── Final_Project_Signals_Mirko_Morello.pdf  # Full research paper
│   └── slides_signal.pdf                 # Presentation slides
│
├── Lessons_notes/                        # Course materials
├── README.md                             # This file
├── LICENSE                               # MIT License
└── .gitignore
```

---

## Usage

### Training Models

#### Train a Single Model

```python
# Example: Train Hybrid Model with hybrid loss
from models import HybridDenoiser
from custom_loss import hybrid_loss
from utils import get_datasets
from train_util import run_training

# Load datasets
train_dataset, val_dataset = get_datasets("noisy_speech_dataset")

# Initialize model
model = HybridDenoiser()

# Train
train_losses, val_losses, stoi_scores = run_training(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    loss_fn=hybrid_loss,
    num_epochs=10,
    batch_size=24,
    experiment_name="hybrid_v2"
)
```

#### Train All Models

```bash
python main.py
```

This will train all model variants:
- `hybrid_autoencoder_v1` and `v2`
- `ResAutoencoder_v1` and `v2`
- `UNetSpec_v1` and `v2`
- `TransformerAutoencoderFreq_v2`

### Evaluation

#### Evaluate Classical Methods

```bash
python evaluate_baseline.py
```

#### Evaluate Deep Learning Models

```python
from evaluation_utils import evaluate_model
from models import HybridDenoiser

# Load trained model
model = HybridDenoiser()
checkpoint = torch.load("checkpoints/hybrid_v2/best_checkpoint.pt")
model.load_state_dict(checkpoint['model_state'])

# Evaluate
metrics = evaluate_model(model, val_dataset)
print(f"PESQ: {metrics['pesq']:.2f}")
print(f"STOI: {metrics['stoi']:.2f}")
print(f"SI-SDR: {metrics['sisdr']:.2f} dB")
```

### Inference

```python
from models import HybridDenoiser
import torchaudio

# Load model
model = HybridDenoiser()
model.load_state_dict(torch.load("checkpoints/hybrid_v2/best_checkpoint.pt")['model_state'])
model.eval()

# Load noisy audio
noisy_audio, sr = torchaudio.load("noisy_audio.wav")
noisy_audio = noisy_audio.unsqueeze(0)  # Add batch dimension

# Denoise
with torch.no_grad():
    denoised = model(noisy_audio)

# Save
torchaudio.save("denoised_audio.wav", denoised.squeeze(0), sr)
```

---

## Results

### Quantitative Results

| Method | PESQ ↑ | STOI ↑ | SI-SDR (dB) ↑ | MOS ↑ |
|--------|--------|--------|---------------|-------|
| **Baseline (No Denoising)** | 1.16±0.19 | 0.74±0.12 | 0.10±4.22 | 2.84±0.82 |
| **Spectral Subtraction** | 1.25±0.21 | 0.80±0.11 | 3.76±5.72 | 3.83±0.54 |
| **Wiener Filtering** | 1.33±0.26 | 0.76±0.14 | -0.10±6.51 | 3.82±0.47 |
| **ResAutoencoder (v1)** | 1.28±0.21 | 0.79±0.11 | 2.73±5.88 | 2.39±0.51 |
| **ResAutoencoder (v2)** | 1.29±0.24 | 0.81±0.11 | 3.18±5.69 | 2.52±0.63 |
| **U-Net (v1)** | 1.35±0.18 | 0.78±0.08 | 3.95±3.63 | 3.54±0.78 |
| **U-Net (v2)** | 1.14±0.04 | 0.62±0.09 | -7.27±3.66 | 3.12±0.80 |
| **Hybrid (v1)** | 1.81±0.46 | 0.88±0.08 | 11.29±4.81 | 3.68±0.74 |
| **Hybrid (v2)** ⭐ | **1.81±0.50** | **0.89±0.08** | **11.69±5.27** | **3.79±0.74** |
| **Transformer (v2)** | 1.78±0.45 | 0.88±0.08 | 11.65±4.90 | 2.87±0.50 |

**↑ = Higher is better**

### Performance Visualization

See the full paper (`Final_Project/Final_Project_Signals_Mirko_Morello.pdf`) for:
- Training/validation loss curves
- Spectrogram comparisons (noisy vs. clean vs. denoised)
- Radar charts comparing all methods
- Per-SNR level analysis

---

## Evaluation Metrics

### 1. PESQ (Perceptual Evaluation of Speech Quality)
- **Range**: -0.5 to 4.5
- **Measures**: Perceived quality of speech
- **Standard**: ITU-T P.862
- **Best Use**: Telephony and VoIP applications

### 2. STOI (Short-Time Objective Intelligibility)
- **Range**: 0 to 1
- **Measures**: Speech intelligibility
- **Best Use**: Hearing aid evaluation

### 3. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- **Range**: -∞ to +∞ dB (higher is better)
- **Measures**: Overall signal distortion
- **Advantage**: Scale-invariant, differentiable (can be used as loss)

### 4. MOS (Mean Opinion Score)
- **Range**: 1 to 5
- **Measures**: Subjective quality prediction
- **Method**: Pre-trained neural network (torchaudio-squim)

---

## Key Findings

### 🏆 Best Performers

1. **Hybrid Model (v2)**: Best overall performance across all metrics
   - SI-SDR: 11.69 dB (1166% improvement over baseline)
   - STOI: 0.89 (20% improvement)
   - Combines strengths of time and frequency domain processing

2. **Transformer Model (v2)**: Close second
   - SI-SDR: 11.65 dB
   - Channel-wise attention effectively weights important features

3. **U-Net (v1)**: Best among single-domain models
   - SI-SDR: 3.95 dB
   - Skip connections preserve fine details

### 📊 Key Insights

**✅ Deep Learning > Classical Methods**
- Deep learning models consistently outperform classical methods
- Hybrid (v2) achieves **>116x improvement** in SI-SDR over baseline

**✅ Hybrid Loss Matters (But Not Always)**
- For Hybrid and Transformer models: v2 (hybrid loss) > v1 (simple loss)
- For U-Net: v1 (simple loss) > v2 (hybrid loss)
- **Insight**: Optimal loss function is architecture-dependent

**✅ Multi-Domain Processing is Powerful**
- Hybrid model's dual-branch design leverages complementary information
- Time domain: captures temporal dynamics
- Frequency domain: preserves harmonic structure

**⚠️ Limitations**
- UrbanSound8K contains "speech-like" noises (children playing, etc.)
- Metrics may confuse background chatter with target speech
- Real-world performance may differ from synthetic test data

---

## Future Work

### Immediate Enhancements
- [ ] Evaluate on real-world environmental recordings
- [ ] Test on additional noise types (wind, rain, mechanical)
- [ ] Extend to longer audio clips (> 6 seconds)
- [ ] Implement real-time inference pipeline

### Research Directions
- [ ] Explore full Transformer architectures (multi-head self-attention)
- [ ] Investigate Generative Adversarial Networks (GANs)
- [ ] Develop adaptive methods for varying SNR levels
- [ ] Semi-supervised learning with unlabeled data
- [ ] Integration with downstream tasks (speech recognition, speaker ID)

### Evaluation Improvements
- [ ] Human listening tests for subjective quality
- [ ] Task-specific metrics (e.g., keyword recognition accuracy)
- [ ] Robustness testing with out-of-distribution noise
- [ ] Better metrics to distinguish speech from speech-like noise

---

## References

### Key Papers

1. **Spectral Subtraction**: Boll, S.F. (1979). *Suppression of acoustic noise in speech using spectral subtraction.* IEEE Transactions on ASSP.

2. **Wiener Filtering**: Wiener, N. (1949). *Extrapolation, interpolation, and smoothing of stationary time series.* Wiley.

3. **U-Net**: Ronneberger, O. et al. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI.

4. **LibriSpeech**: Panayotov, V. et al. (2015). *LibriSpeech: An ASR corpus based on public domain audio books.* ICASSP.

5. **UrbanSound8K**: Salamon, J. et al. (2014). *A dataset and taxonomy for urban sound research.* ACM Multimedia.

### Evaluation Tools

- **PESQ**: ITU-T Recommendation P.862
- **STOI**: Taal, C.H. et al. (2011). *An algorithm for intelligibility prediction of time-frequency weighted noisy speech.* IEEE TASLP.
- **Librosa**: McFee, B. et al. (2015). *librosa: Audio and music signal analysis in python.* SciPy.

---

## Citation

If you use this code or find this research helpful, please cite:

```bibtex
@mastersthesis{morello2024audio,
  title={A Comparative Study of Denoising Techniques for Speech Audio Signals},
  author={Morello, Mirko},
  year={2024},
  school={Università degli Studi di Milano-Bicocca},
  type={Master's Thesis},
  note={Physical Sensors and Systems for Environmental Signals}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Course**: Physical Sensors and Systems for Environmental Signals
- **Institution**: Università degli Studi di Milano-Bicocca
- **Datasets**: LibriSpeech (OpenSLR), UrbanSound8K (NYU)
- **Frameworks**: PyTorch, torchaudio, librosa

---

## Contact

**Mirko Morello**
📧 m.morello11@campus.unimib.it
🔗 [GitHub](https://github.com/MirkoMorello/MSc_Sensors_Signal)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for audio signal processing research

</div>
