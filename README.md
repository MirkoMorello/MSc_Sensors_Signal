# A Comparative Study of Deep Learning Techniques for Audio Denoising

> **Course:** Physical Sensors and Systems for Environmental Signals

## Overview

This project provides a comprehensive comparative analysis of various techniques for audio denoising, with a focus on enhancing speech signals corrupted by environmental noise. We evaluate both classical signal processing methods and a range of modern deep learning architectures to determine their relative strengths and weaknesses.

## Methods Evaluated

### Classical Signal Processing
*   **Spectral Subtraction:** A traditional method that estimates and subtracts the noise spectrum from the signal.
*   **Wiener Filtering:** A statistical method that aims to minimize the mean squared error between the clean and denoised signals.

### Deep Learning Architectures
We implemented and trained several autoencoder-style models to learn the mapping from noisy to clean audio:
*   **Residual Autoencoder:** A time-domain model that learns to predict and subtract the residual noise.
*   **U-Net (Frequency Domain):** A U-Net architecture applied to spectrograms for noise reduction.
*   **Transformer-based Autoencoder:** A U-Net style model incorporating a channel-wise attention mechanism in the bottleneck.
*   **Hybrid Model:** A novel dual-branch model that processes audio in both the time and frequency domains simultaneously.

### Key Innovation: Hybrid Loss Function
A significant contribution of this project was the design of a **hybrid loss function**. This loss combines three components:
1.  **Time-domain L1 loss** (on raw waveforms)
2.  **Frequency-domain L1 loss** (on spectrogram magnitudes)
3.  **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**
This hybrid loss encourages the model to learn both temporal and spectral features, leading to superior perceptual quality in the denoised audio. Our models trained with this loss significantly outperformed those trained with simpler losses.

## Evaluation
Models were rigorously evaluated using a suite of standard audio quality metrics:
*   **PESQ** (Perceptual Evaluation of Speech Quality)
*   **STOI** (Short-Time Objective Intelligibility)
*   **SI-SDR**

## Technologies Used

*   **Deep Learning:** Python, PyTorch, `torchaudio`
*   **Signal Processing:** Librosa, SciPy, `pystoi`, `pesq`
*   **Datasets:** LibriSpeech, UrbanSound8K
