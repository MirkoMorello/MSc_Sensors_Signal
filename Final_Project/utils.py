import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from custom_datasets import StreamingAudioDataset, StreamingSpectrogramDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import librosa
import numpy as np

load_dotenv()  # Loads variables from .env

SR = int(os.getenv("SR"))
MAX_DURATION = int(os.getenv("MAX_DURATION"))
MAX_LENGTH  = MAX_DURATION * SR
N_FFT = int(os.getenv("N_FFT"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_datasets(data_dir, test_size=0.2):
    # get file paths without loading audio
    noisy_files = sorted(glob.glob(os.path.join(data_dir, "noisy_*.wav")))
    clean_files = [
        os.path.join(data_dir, f"clean_{os.path.basename(nf).split('_')[1]}_{os.path.basename(nf).split('_')[2]}")
        for nf in noisy_files
    ]
    
    # split file paths instead of loaded data
    train_noisy, val_noisy, train_clean, val_clean = train_test_split(
        noisy_files, clean_files, test_size=test_size, random_state=42
    )
    
    return (
        StreamingAudioDataset(train_noisy, train_clean),
        StreamingAudioDataset(val_noisy, val_clean)
    )

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


def save_checkpoint(experiment_name, model, optimizer, scheduler, epoch, step, batch_idx,
                    train_losses, val_losses, stoi_scores, best=False):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx,  # last processed batch index within this epoch
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
    
    # Save the losses and STOI scores to a JSON file for easy analysis.
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
    
    # Load loss and metric history
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

def compute_spectrogram(waveform):
    # waveform is expected to be a tensor of shape [B, 1, L]
    # Remove the channel dimension and compute the STFT.
    # We use return_complex=True so we can later take the magnitude.
    stft = torch.stft(waveform.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, 
                        return_complex=True)
    # Take the magnitude; shape will be [B, F, T]
    mag = torch.abs(stft)
    return mag


# Helper function to process audio tensors for the metric
def _process_audio(audio):
    audio = torch.as_tensor(audio).float()
    if audio.dim() == 1: 
        audio = audio.unsqueeze(0)  # ensure shape [B, T]
    return audio.to(device)

def plot_audio(audio, sr, title):
    # Ensure audio is a 1D array
    audio = np.squeeze(audio)
    
    # Compute STFT for the spectrogram
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Create the figure and subplots
    plt.figure(figsize=(12, 8))
    
    # Waveform (time-domain)
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Spectrogram (time-frequency)
    plt.subplot(2, 1, 2)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.show()

def plot_time_wave(audio, sr, title):
    """Plots the time-domain waveform."""
    audio = np.squeeze(audio)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def griffin_lim_inversion(mag, n_iter=32):
    """
    Given a magnitude spectrogram (numpy array of shape [F, T]),
    estimate the waveform using Griffin–Lim.
    """
    return librosa.griffinlim(mag, n_iter=n_iter, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)


def process_sample(val_dataset, sample_idx, model, device):
    """
    Process a sample from a validation dataset (time-domain or frequency-domain),
    run it through the model, and return the noisy waveform, clean waveform,
    and denoised output waveform (all as numpy arrays).

    Parameters:
        val_dataset: Dataset object supporting indexing, which returns each sample as a tuple/list.
        sample_idx (int): Index into the dataset.
        model: The trained model for inference.
        device: Torch device (e.g., 'cuda' or 'cpu').
        griffin_lim_inversion: A function that inverts a magnitude spectrogram back to a waveform.

    Returns:
        noisy_waveform (np.ndarray): Noisy input waveform.
        clean_waveform (np.ndarray): Clean (ground truth) waveform.
        denoised_wave (np.ndarray): Model output converted to a waveform.

    Raises:
        ValueError: If the sample or model output shape is not as expected.
    """
    # Retrieve the sample from the dataset.
    sample = val_dataset[sample_idx]

    # Check that the sample is a tuple or list.
    if not isinstance(sample, (list, tuple)):
        raise ValueError("Expected sample type is list or tuple.")

    # Determine if the sample is time-domain or frequency-domain by checking the tensor dimensions.
    if sample[0].ndim == 2:
        # Time-domain data.
        noisy_sample = sample[0]
        clean_sample = sample[1]
        # Prepare model input: add batch dimension and move to the specified device.
        model_input = noisy_sample.unsqueeze(0).to(device)
    elif sample[0].ndim == 3:
        # Frequency-domain data (spectrogram).
        noisy_spec = sample[0]
        clean_spec = sample[1]
        # Convert spectrograms to numpy arrays (squeezing out the channel dimension).
        noisy_spec_np = noisy_spec.squeeze(0).cpu().numpy()
        clean_spec_np = clean_spec.squeeze(0).cpu().numpy()
        # Invert the spectrograms to obtain waveform versions for evaluation.
        noisy_wave = griffin_lim_inversion(noisy_spec_np)
        clean_wave = griffin_lim_inversion(clean_spec_np)
        # Prepare model input: assume the model expects the spectrogram.
        model_input = noisy_spec.unsqueeze(0).to(device)
        # For plotting and evaluation, use the inverted waveforms (converted back to tensors).
        noisy_sample = torch.from_numpy(noisy_wave).unsqueeze(0)
        clean_sample = torch.from_numpy(clean_wave).unsqueeze(0)
    else:
        raise ValueError("Unexpected tensor dimensions in sample.")

    # Run the model on the input.
    model_output = model(model_input)
    # Remove the batch dimension.
    model_output = model_output.squeeze()

    # Process the model output:
    if model_output.ndim == 2:
        # This could be either:
        # - A time-domain waveform, or
        # - A frequency-domain spectrogram output.
        # Check the sample type to decide.
        if sample[0].ndim == 3:
            # Frequency-domain output: invert the spectrogram.
            model_output_np = model_output.detach().cpu().numpy()
            denoised_wave = griffin_lim_inversion(model_output_np)
        else:
            # Otherwise, assume it's a time-domain waveform.
            denoised_wave = model_output.detach().cpu().numpy()
    elif model_output.ndim == 1:
        # Already a 1D waveform.
        denoised_wave = model_output.detach().cpu().numpy()
    else:
        raise ValueError("Unexpected model output shape.")

    # Convert noisy and clean samples to numpy arrays for plotting.
    if noisy_sample.ndim > 1:
        noisy_waveform = noisy_sample.squeeze().cpu().numpy()
    else:
        noisy_waveform = noisy_sample

    if clean_sample.ndim > 1:
        clean_waveform = clean_sample.squeeze().cpu().numpy()
    else:
        clean_waveform = clean_sample

    return noisy_waveform, clean_waveform, denoised_wave
