#!/usr/bin/env python3
"""
Train speech denoising models.

Usage examples:
  - Train the hybrid model from scratch:
        python train_models.py --model hybrid
  - Train the transformer autoencoder and resume from a checkpoint:
        python train_models.py --model transformer --resume
  - Train a model and then run evaluation:
        python train_models.py --model res --evaluate

Requirements:
    pip install torch torchaudio librosa matplotlib datasets soundfile scikit-learn torchinfo tqdm pystoi pesq
"""

import os
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from tqdm import tqdm

# ---------------------------
# Utility functions & plotting
# ---------------------------
def plot_audio(audio, sr, title):
    """Plot the waveform and spectrogram of an audio signal."""
    audio = np.squeeze(audio)
    # Compute STFT and convert amplitude to dB
    D = librosa.stft(audio, n_fft=1024, win_length=512, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(12, 8))
    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # Spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

def plot_time_wave(audio, sr, title):
    """Plot only the time–domain waveform."""
    audio = np.squeeze(audio)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def griffin_lim_inversion(mag, n_iter=32, hop_length=256, win_length=512):
    """Invert a magnitude spectrogram using Griffin–Lim."""
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D array for magnitude spectrogram but got shape {mag.shape}")
    n_fft = (mag.shape[0] - 1) * 2
    return librosa.griffinlim(mag, n_iter=n_iter, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_and_prepare_datasets():
    """Load LibriSpeech (clean speech) and UrbanSound8K (noise) datasets and return sample data."""
    # Load clean speech from a local folder version of LibriSpeech
    librispeech = load_dataset(
        "audiofolder",  
        data_dir=os.path.expanduser("~/.cache/huggingface/datasets/Librispeech"),
        split="train"
    )
    sample_speech = librispeech[0]["audio"]["array"]
    sr_speech = librispeech[0]["audio"]["sampling_rate"]

    # Load noise data (UrbanSound8K)
    urbansound = load_dataset("danavery/urbansound8K", split="train")
    min_length_ratio = 0.5
    min_required_length = int(len(sample_speech) * min_length_ratio)
    valid_noise_indices = [
        i for i in range(len(urbansound))
        if len(urbansound[i]["audio"]["array"]) >= min_required_length
    ]
    if not valid_noise_indices:
        raise ValueError("No noise samples long enough found in dataset")
    noise_idx = int(np.random.choice(valid_noise_indices))
    sample_noise = urbansound[noise_idx]["audio"]["array"]
    sr_noise = urbansound[noise_idx]["audio"]["sampling_rate"]

    return librispeech, urbansound, sample_speech, sr_speech, sample_noise, sr_noise

def mix_noise(clean, noise, sr_clean, sr_noise, target_snr_db):
    """Mix clean speech and noise at a target SNR."""
    # Resample noise if needed
    if sr_noise != sr_clean:
        noise = librosa.resample(noise, orig_sr=sr_noise, target_sr=sr_clean)
    # Extend or truncate noise to match clean length
    if len(noise) < len(clean):
        noise = np.tile(noise, (len(clean) // len(noise)) + 1)
    noise = noise[:len(clean)]
    # Calculate power and scale noise to target SNR
    power_clean = np.mean(clean ** 2)
    power_noise = np.mean(noise ** 2)
    target_power_noise = power_clean / (10 ** (target_snr_db / 10))
    scaled_noise = np.sqrt(target_power_noise / power_noise) * noise
    noisy = clean + scaled_noise
    return noisy, clean

def generate_dataset(librispeech, urbansound, output_dir, num_samples=100, snr_levels=[0, 5, 10]):
    """Generate a synthetic dataset by mixing clean speech with noise at different SNRs."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_samples):
        # Get clean speech sample and its sampling rate
        speech = librispeech[i]["audio"]["array"]
        sr_speech = librispeech[i]["audio"]["sampling_rate"]
        # Randomly select a noise sample
        noise_idx = np.random.randint(0, len(urbansound))
        noise = urbansound[noise_idx]["audio"]["array"]
        sr_noise = urbansound[noise_idx]["audio"]["sampling_rate"]
        for snr in snr_levels:
            noisy, clean = mix_noise(speech, noise, sr_speech, sr_noise, snr)
            noisy_filename = os.path.join(output_dir, f"noisy_snr{snr}_sample{i}.wav")
            clean_filename = os.path.join(output_dir, f"clean_snr{snr}_sample{i}.wav")
            sf.write(noisy_filename, noisy, sr_speech)
            sf.write(clean_filename, clean, sr_speech)

# ---------------------------
# Dataset classes
# ---------------------------
SR = 16000           # Target sample rate
MAX_DURATION = 6     # seconds
MAX_LENGTH = MAX_DURATION * SR
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256

class StreamingAudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files, sr=SR):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = self._load_audio(self.noisy_files[idx])
        clean = self._load_audio(self.clean_files[idx])
        # Return as tensors with shape [1, MAX_LENGTH]
        return (
            torch.from_numpy(noisy).float().unsqueeze(0),
            torch.from_numpy(clean).float().unsqueeze(0)
        )

    def _load_audio(self, path):
        try:
            audio, _ = librosa.load(path, sr=self.sr)
            if len(audio) > MAX_LENGTH:
                audio = audio[:MAX_LENGTH]
            else:
                audio = np.pad(audio, (0, max(0, MAX_LENGTH - len(audio))), 'constant')
            return audio
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return np.zeros(MAX_LENGTH)

class StreamingSpectrogramDataset(Dataset):
    def __init__(self, noisy_files, clean_files, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = self._load_audio(self.noisy_files[idx])
        clean = self._load_audio(self.clean_files[idx])
        noisy_spec = librosa.stft(noisy, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        clean_spec = librosa.stft(clean, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        noisy_mag = np.abs(noisy_spec)
        clean_mag = np.abs(clean_spec)
        noisy_mag = torch.from_numpy(noisy_mag).float().unsqueeze(0)
        clean_mag = torch.from_numpy(clean_mag).float().unsqueeze(0)
        return noisy_mag, clean_mag

    def _load_audio(self, path):
        try:
            audio, _ = librosa.load(path, sr=self.sr)
            if len(audio) > MAX_LENGTH:
                audio = audio[:MAX_LENGTH]
            else:
                audio = np.pad(audio, (0, max(0, MAX_LENGTH - len(audio))), 'constant')
            return audio
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return np.zeros(MAX_LENGTH)

def get_datasets(data_dir, test_size=0.2):
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy_*.wav")))
    clean_files = [
        os.path.join(data_dir, f"clean_{os.path.basename(nf).split('_')[1]}_{os.path.basename(nf).split('_')[2]}")
        for nf in noisy_files
    ]
    train_noisy, val_noisy, train_clean, val_clean = train_test_split(
        noisy_files, clean_files, test_size=test_size, random_state=42
    )
    return StreamingAudioDataset(train_noisy, train_clean), StreamingAudioDataset(val_noisy, val_clean)

def get_spectrogram_datasets(data_dir, test_size=0.2):
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy_*.wav")))
    clean_files = [
        os.path.join(data_dir, f"clean_{os.path.basename(nf).split('_')[1]}_{os.path.basename(nf).split('_')[2]}")
        for nf in noisy_files
    ]
    train_noisy, val_noisy, train_clean, val_clean = train_test_split(
        noisy_files, clean_files, test_size=test_size, random_state=42
    )
    train_dataset = StreamingSpectrogramDataset(train_noisy, train_clean)
    val_dataset = StreamingSpectrogramDataset(val_noisy, val_clean)
    return train_dataset, val_dataset

# ---------------------------
# Model definitions
# ---------------------------
# 1. Simple Autoencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self, max_length):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 1, kernel_size=5, padding='same'),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 2. Autoencoder with deeper architecture
class Autoencoder(nn.Module):
    def __init__(self, max_length):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size=7, padding='same'),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Transformer-based Autoencoder
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.channel_att(x.mean(dim=-1, keepdim=True))
        return x * att

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, 5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        diff = skip.size(2) - x.size(2)
        x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_length=48000):
        super().__init__()
        self.enc1_conv = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.enc1_pool = nn.MaxPool1d(4)
        self.enc2_conv = nn.Sequential(
            nn.Conv1d(32, 64, 11, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2_pool = nn.MaxPool1d(4)
        self.enc3_conv = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3_pool = nn.MaxPool1d(4)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            AttentionBlock(256),
            nn.Conv1d(256, 128, 3, padding=1)
        )
        self.decoder = nn.ModuleDict({
            'dec1': DecoderBlock(128, 128, 64, 4),
            'dec2': DecoderBlock(64, 64, 32, 4),
            'dec3': DecoderBlock(32, 32, 1, 4)
        })
        self.final = nn.Sequential(
            nn.Conv1d(1, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1_conv(x)
        e1_pool = self.enc1_pool(e1)
        e2 = self.enc2_conv(e1_pool)
        e2_pool = self.enc2_pool(e2)
        e3 = self.enc3_conv(e2_pool)
        e3_pool = self.enc3_pool(e3)
        b = self.bottleneck(e3_pool)
        d1 = self.decoder['dec1'](b, e3)
        d2 = self.decoder['dec2'](d1, e2)
        d3 = self.decoder['dec3'](d2, e1)
        return self.final(d3)

# 4. Residual Autoencoder
class ResAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, in_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x - decoded

# 5. UNet for Spectrograms
class UNetSpec(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        in_ch = in_channels
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            in_ch = feature
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU()
        )
        rev_features = features[::-1]
        current_channels = features[-1]*2
        for feature in rev_features:
            up_conv = nn.Sequential(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2),
                nn.BatchNorm2d(feature),
                nn.ReLU()
            )
            post_conv = nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU()
            )
            self.ups.append(nn.ModuleList([up_conv, post_conv]))
            current_channels = feature
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx, (up_conv, post_conv) in enumerate(self.ups):
            x = up_conv(x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = post_conv(x)
        return self.final_conv(x)

# 6. Hybrid Denoiser (time + frequency branches)
class HybridDenoiser(nn.Module):
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.time_branch = ResAutoencoder(in_channels=1)
        self.spec_branch = UNetSpec(in_channels=1, out_channels=1, features=[32, 64, 128, 256])
        self.fusion = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: waveform [B, 1, L]
        time_out = self.time_branch(x)
        stft = torch.stft(x.squeeze(1), n_fft=self.n_fft, win_length=self.win_length,
                            hop_length=self.hop_length, return_complex=True)
        mag = stft.abs()  # [B, F, T]
        phase = stft.angle()  # [B, F, T]
        mag = mag.unsqueeze(1)  # [B, 1, F, T]
        enhanced_mag = self.spec_branch(mag)
        enhanced_spec = enhanced_mag.squeeze(1) * torch.exp(1j * phase)
        time_from_spec = torch.istft(enhanced_spec, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        time_from_spec = time_from_spec.unsqueeze(1)
        if time_out.shape[-1] != time_from_spec.shape[-1]:
            time_from_spec = F.interpolate(time_from_spec, size=time_out.shape[-1])
        fused = self.fusion(torch.cat([time_out, time_from_spec], dim=1))
        return fused

# ---------------------------
# Checkpointing and training utilities
# ---------------------------
def save_checkpoint(experiment_name, model, optimizer, scheduler, epoch, step, batch_idx,
                    train_losses, val_losses, stoi_scores, best=False):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stoi_scores': stoi_scores,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs(f'checkpoints/{experiment_name}', exist_ok=True)
    prefix = 'best_' if best else ''
    path = f'checkpoints/{experiment_name}/{prefix}checkpoint_epoch{epoch}_step{step}.pt'
    torch.save(checkpoint, path)
    with open(f'checkpoints/{experiment_name}/losses.json', 'w') as f:
        json.dump({'train': train_losses, 'val': val_losses, 'stoi': stoi_scores}, f)

def load_checkpoint(experiment_name, model, optimizer=None, scheduler=None):
    checkpoint_dir = f'checkpoints/{experiment_name}'
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') or f.startswith('best_checkpoint_')])
    if not checkpoints:
        raise ValueError("No checkpoints found")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    try:
        with open(os.path.join(checkpoint_dir, 'losses.json'), 'r') as f:
            losses = json.load(f)
            train_losses = losses.get('train', [])
            val_losses = losses.get('val', [])
            stoi_scores = losses.get('stoi', [])
    except FileNotFoundError:
        train_losses, val_losses, stoi_scores = [], [], []
    return {
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'batch_idx': checkpoint.get('batch_idx', 0),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stoi_scores': stoi_scores,
        'best_val_loss': min(val_losses) if val_losses else float('inf')
    }

# Training loop
def train(model, experiment_name, train_loader, val_loader, optimizer, scheduler, loss_fn, mse_loss,
          num_epochs, device, resume=False):
    train_losses = []
    val_losses = []
    stoi_scores = []
    start_epoch = 0
    step = 0
    best_val_loss = float('inf')
    resume_batch = 0
    if resume:
        state = load_checkpoint(experiment_name, model, optimizer, scheduler)
        start_epoch = state['epoch']
        step = state['step']
        resume_batch = state.get('batch_idx', 0)
        train_losses = state['train_losses']
        val_losses = state['val_losses']
        stoi_scores = state.get('stoi_scores', [])
        best_val_loss = state['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}, step {step}, batch {resume_batch}")
    model.to(device)
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_train_loss = 0
            train_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} [Train]', total=len(train_loader), leave=False)
            for batch_idx, (noisy_batch, clean_batch) in train_bar:
                if resume and epoch == start_epoch and batch_idx < resume_batch:
                    continue
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_batch)
                loss = loss_fn(outputs, clean_batch) + mse_loss(outputs, clean_batch)
                loss.backward()
                optimizer.step()
                step_loss = loss.item()
                epoch_train_loss += step_loss * noisy_batch.size(0)
                train_losses.append(step_loss)
                step += 1
                train_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'lr': optimizer.param_groups[0]['lr']
                })
                if step % 100 == 0:
                    save_checkpoint(experiment_name, model, optimizer, scheduler,
                                    epoch, step, batch_idx, train_losses, val_losses, stoi_scores)
            # Validation
            model.eval()
            epoch_val_loss = 0
            epoch_stoi_total = 0.0
            num_batches = 0
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', total=len(val_loader), leave=False)
            with torch.no_grad():
                for noisy_batch, clean_batch in val_bar:
                    noisy_batch = noisy_batch.to(device)
                    clean_batch = clean_batch.to(device)
                    outputs = model(noisy_batch)
                    loss = loss_fn(outputs, clean_batch) + mse_loss(outputs, clean_batch)
                    epoch_val_loss += loss.item() * noisy_batch.size(0)
                    # Using torchmetrics SToi would require extra handling;
                    # here we simply accumulate loss as a proxy.
                    batch_stoi = 0.0
                    epoch_stoi_total += batch_stoi
                    num_batches += 1
                    val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            avg_val_loss = epoch_val_loss / len(val_loader.dataset)
            avg_stoi = (epoch_stoi_total / num_batches) if num_batches > 0 else 0.0
            val_losses.append(avg_val_loss)
            stoi_scores.append(avg_stoi)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(experiment_name, model, optimizer, scheduler,
                                epoch, step, batch_idx, train_losses, val_losses, stoi_scores, best=True)
                print(f"New best model saved with val loss: {best_val_loss:.4f}")
            save_checkpoint(experiment_name, model, optimizer, scheduler,
                            epoch, step, 0, train_losses, val_losses, stoi_scores)
            print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss/len(train_loader.dataset):.4f} | Val Loss: {avg_val_loss:.4f}')
            if resume and epoch == start_epoch:
                resume = False
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(experiment_name, model, optimizer, scheduler,
                        epoch, step, batch_idx, train_losses, val_losses, stoi_scores)
    return train_losses, val_losses, stoi_scores

# ---------------------------
# Evaluation classes and functions
# ---------------------------
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

class DenoiserEvaluator:
    def __init__(self, device=torch.device("cpu"), target_sr=16000):
        self.target_sr = target_sr
        # Load pre-trained models from torchaudio-squim if available.
        # (Here we assume they are available. You may need to adjust according to your setup.)
        from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
        self.objective_model = SQUIM_OBJECTIVE.get_model().to(device)
        self.subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    
    def evaluate(self, clean, denoised, reference_nmr):
        denoised = self._preprocess(denoised)
        reference_nmr = self._preprocess(reference_nmr)
        clean = self._preprocess(clean)
        stoi, pesq, si_sdr = self.objective_model(denoised)
        mos = self.subjective_model(denoised, reference_nmr)
        return {
            'stoi': stoi.item(),
            'pesq': pesq.item(),
            'si_sdr': si_sdr.item(),
            'mos': mos.item()
        }
    def _preprocess(self, audio):
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio

# ---------------------------
# Main function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a speech denoising model.")
    parser.add_argument("--model", type=str, default="hybrid",
                        choices=["simple", "autoencoder", "transformer", "res", "unet", "hybrid"],
                        help="Which model to train.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # Data preparation
    # ---------------------------
    print("Loading LibriSpeech and UrbanSound datasets...")
    librispeech, urbansound, sample_speech, sr_speech, sample_noise, sr_noise = load_and_prepare_datasets()
    # (Optional: plot a couple of samples)
    # plot_audio(sample_speech, sr_speech, "Clean Speech (LibriSpeech)")
    # plot_audio(sample_noise, sr_noise, "Noise (UrbanSound8K)")

    # Generate synthetic noisy speech dataset if not already present
    dataset_dir = "noisy_speech_dataset"
    overwrite = False
    if not os.path.exists(dataset_dir) or overwrite:
        print("Generating synthetic noisy speech dataset...")
        generate_dataset(librispeech, urbansound, dataset_dir, num_samples=len(librispeech), snr_levels=[-5, 0, 5])
    else:
        print("Using existing synthetic dataset.")

    # ---------------------------
    # Instantiate the chosen model
    # ---------------------------
    experiment_name = f"{args.model}_denoiser"
    if args.model == "simple":
        model = SimpleAutoencoder(MAX_LENGTH)
    elif args.model == "autoencoder":
        model = Autoencoder(MAX_LENGTH)
    elif args.model == "transformer":
        model = TransformerAutoencoder(input_length=MAX_LENGTH)
    elif args.model == "res":
        model = ResAutoencoder()
    elif args.model == "unet":
        model = UNetSpec()
    elif args.model == "hybrid":
        model = HybridDenoiser()
    else:
        raise ValueError("Unknown model type.")
    print("Model summary:")
    summary(model, input_size=(1, 1, MAX_LENGTH))
    
    # ---------------------------
    # Prepare datasets and dataloaders
    # ---------------------------
    if isinstance(model, UNetSpec):
        train_dataset, val_dataset = get_spectrogram_datasets(dataset_dir)
    else:
        train_dataset, val_dataset = get_datasets(dataset_dir)
    num_workers = 4
    batch_size = 24
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---------------------------
    # Set up optimizer, scheduler, loss functions
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    loss_fn = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # ---------------------------
    # Training
    # ---------------------------
    print("Starting training...")
    train_losses, val_losses, stoi_scores = train(model, experiment_name, train_loader, val_loader,
                                                  optimizer, scheduler, loss_fn, mse_loss,
                                                  num_epochs=args.epochs, device=device, resume=args.resume)
    print("Training completed.")

    # (Optional) Plot training and validation loss curves
    epochs = range(1, len(val_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label='Val Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Evaluation (if requested)
    # ---------------------------
    if args.evaluate:
        print("Running evaluation on the validation set...")
        evaluator = DenoiserEvaluator(device=device, target_sr=SR)
        # For demonstration, we use the first sample of the validation set
        sample_noisy, sample_clean = val_dataset[0]
        # Prepare input (add batch dimension)
        model_input = sample_noisy.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(model_input)
        # Convert tensors to numpy arrays for plotting
        noisy_wave = sample_noisy.squeeze().cpu().numpy()
        clean_wave = sample_clean.squeeze().cpu().numpy()
        if output.ndim == 3:
            denoised_wave = output.squeeze().cpu().numpy()
        else:
            denoised_wave = output.cpu().numpy()
        # Plot waveforms and spectrograms
        plot_time_wave(noisy_wave, SR, "Noisy Input (Time Domain)")
        plot_audio(noisy_wave, SR, "Noisy Input (Spectrogram)")
        plot_time_wave(denoised_wave, SR, "Denoised Output (Time Domain)")
        plot_audio(denoised_wave, SR, "Denoised Output (Spectrogram)")
        plot_time_wave(clean_wave, SR, "Clean Ground Truth (Time Domain)")
        plot_audio(clean_wave, SR, "Clean Ground Truth (Spectrogram)")
        # Compute evaluation metrics (using the evaluator)
        metrics = evaluator.evaluate(
            clean=torch.from_numpy(clean_wave).unsqueeze(0),
            denoised=torch.from_numpy(denoised_wave).unsqueeze(0),
            reference_nmr=torch.from_numpy(clean_wave).unsqueeze(0)
        )
        print(f"\nDenoising Performance Report:\n"
              f"- Intelligibility (STOI): {metrics['stoi']:.2f}/1.00\n"
              f"- Quality (PESQ): {metrics['pesq']:.2f}/4.50\n"
              f"- Fidelity (SI-SDR): {metrics['si_sdr']:.2f} dB\n"
              f"- Listener Score (MOS): {metrics['mos']:.2f}/5.00\n")

if __name__ == '__main__':
    main()
