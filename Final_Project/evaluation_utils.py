
# https://pytorch.org/audio/main/tutorials/squim_tutorial.html

import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import pesq
import numpy as np
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
from dotenv import load_dotenv
import os
import librosa
import matplotlib.pyplot as plt

load_dotenv()  # Loads variables from .env

SR = int(os.getenv("SR"))
MAX_DURATION = int(os.getenv("MAX_DURATION"))
MAX_LENGTH  = MAX_DURATION * SR
N_FFT = int(os.getenv("N_FFT"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))


class DenoiserEvaluator:
    def __init__(self, device=torch.device("cpu"), target_sr=16000):
        self.target_sr = target_sr
        # Load pre-trained models from torchaudio-squim and move them to the device.
        self.objective_model = SQUIM_OBJECTIVE.get_model().to(device)
        self.subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    
    def evaluate(self, clean, denoised, reference_nmr):
        """
        clean: Original clean waveform [1, T]
        denoised: Your model's output [1, T]
        reference_nmr: Non-matching reference audio [1, T]
        """
        # Ensure 16kHz, mono
        denoised = self._preprocess(denoised)
        reference_nmr = self._preprocess(reference_nmr)
        clean = self._preprocess(clean)
        
        # Compute objective metrics (STOI, PESQ, SI-SDR)
        stoi, pesq, si_sdr = self.objective_model(denoised)
        
        # Compute subjective MOS (using non-matching reference)
        mos = self.subjective_model(denoised, reference_nmr)
        
        return {
            'stoi': stoi.item(),
            'pesq': pesq.item(),
            'si_sdr': si_sdr.item(),
            'mos': mos.item()
        }

    def _preprocess(self, audio):
        """Convert to 16kHz mono if needed"""
        if audio.shape[0] > 1:  # Convert stereo to mono
            audio = audio.mean(dim=0, keepdim=True)
        # Assuming the input is already 16kHz; otherwise, add proper resampling here.
        return audio


def compute_reference_metrics(clean_wave, noisy_wave, sample_rate=16000):
    """
    Computes STOI, PESQ, SI-SDR as reference metrics using external libraries
    for a single pair of signals (clean vs. noisy).
    
    Args:
        clean_wave (numpy.ndarray): 1D array of the clean signal.
        noisy_wave (numpy.ndarray): 1D array of the noisy (or denoised) signal.
        sample_rate (int): Sampling rate, default 16kHz.
        
    Returns:
        ref_stoi (float)
        ref_pesq (float)
        ref_sisdr (float)
    """
    # Ensure both signals have the same length or handle accordingly
    min_len = min(len(clean_wave), len(noisy_wave))
    clean_wave = clean_wave[:min_len]
    noisy_wave = noisy_wave[:min_len]
    
    # STOI
    # extended=False is the standard wideband setting
    ref_stoi = stoi(clean_wave, noisy_wave, sample_rate, extended=False)
    
    # PESQ
    # mode="wb" indicates wide-band mode (for 16kHz signals)
    ref_pesq = pesq(sample_rate, clean_wave, noisy_wave, mode="wb")
    
    # SI-SDR
    ref_sisdr = compute_si_sdr(noisy_wave, clean_wave)  # see function below
    
    return ref_stoi, ref_pesq, ref_sisdr

def compute_si_sdr(estimate, reference, eps=1e-8):
    """
    Computes the scale-invariant signal-to-distortion ratio (SI-SDR) 
    between two 1D signals: estimate and reference.
    
    Adapted from various references, e.g. Asteroid, etc.
    """
    # Zero-mean
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    # Scale
    dot = np.sum(estimate * reference)
    norm_ref = np.sum(reference ** 2) + eps
    scale = dot / norm_ref
    # Projection
    projection = scale * reference
    noise = estimate - projection
    sdr = 10 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps))
    return float(sdr)



def griffin_lim_inversion(mag, n_iter=32, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Given a magnitude spectrogram (numpy array of shape [F, T]),
    estimate the waveform using Griffin–Lim.
    Raises a ValueError if mag does not have 2 dimensions.
    """
    if mag.ndim != 2:
        raise ValueError(f"Expected 2D array for magnitude spectrogram but got shape {mag.shape}")
    # Infer n_fft from the shape: librosa expects shape (n_fft//2+1, n_frames)
    n_fft = (mag.shape[0] - 1) * 2
    return librosa.griffinlim(mag, n_iter=n_iter, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def evaluate_val_set_batch(val_loader, model, evaluator, device):
    # Containers for metrics
    all_stoi = []
    all_pesq = []
    all_si_sdr = []
    all_mos = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating validation set"):
            # (Assume we already handle extracting noisy and clean as time-domain waveforms)
            if isinstance(batch, dict):
                noisy = batch['noisy_wave']
                clean = batch['clean_wave']
            elif isinstance(batch, (list, tuple)):
                noisy, clean = batch[0], batch[1]
                if noisy.ndim == 2:
                    noisy = noisy.unsqueeze(1)
                    clean = clean.unsqueeze(1)
            else:
                raise ValueError("Unknown batch type.")
            
            # Ensure shape is [B, 1, L]
            if noisy.ndim == 1:
                noisy = noisy.unsqueeze(0).unsqueeze(0)
                clean = clean.unsqueeze(0).unsqueeze(0)
            elif noisy.ndim == 2:
                noisy = noisy.unsqueeze(1)
                clean = clean.unsqueeze(1)
            
            model_input = noisy.to(device)
            
            # Run the model
            output = model(model_input)  # Expected shape: [B, 1, L]
            output = output.detach()  # [B, 1, L]
            
            # Remove channel dimension: now shape [B, L]
            if output.ndim == 3 and output.shape[1] == 1:
                output = output.squeeze(1)
            
            # Similarly, for clean, remove channel dimension (if needed)
            clean_eval = clean.to(device)
            if clean_eval.ndim == 3 and clean_eval.shape[1] == 1:
                clean_eval = clean_eval.squeeze(1)
            
            # Compute metrics using the Squim models (which expect [B, L])
            stoi_batch, pesq_batch, si_sdr_batch = evaluator.objective_model(output)
            mos_batch = evaluator.subjective_model(output, clean_eval)
            
            all_stoi.extend(stoi_batch.cpu().numpy().tolist())
            all_pesq.extend(pesq_batch.cpu().numpy().tolist())
            all_si_sdr.extend(si_sdr_batch.cpu().numpy().tolist())
            all_mos.extend(mos_batch.cpu().numpy().tolist())
    
    if len(all_stoi) == 0:
        print("No samples were processed for evaluation. Check the data format!")
        return None, [], [], [], []
    
    summary_stats = {
        'STOI': {
            'min': float(np.min(all_stoi)),
            'max': float(np.max(all_stoi)),
            'mean': float(np.mean(all_stoi)),
            'std': float(np.std(all_stoi))
        },
        'PESQ': {
            'min': float(np.min(all_pesq)),
            'max': float(np.max(all_pesq)),
            'mean': float(np.mean(all_pesq)),
            'std': float(np.std(all_pesq))
        },
        'SI-SDR': {
            'min': float(np.min(all_si_sdr)),
            'max': float(np.max(all_si_sdr)),
            'mean': float(np.mean(all_si_sdr)),
            'std': float(np.std(all_si_sdr))
        },
        'MOS': {
            'min': float(np.min(all_mos)),
            'max': float(np.max(all_mos)),
            'mean': float(np.mean(all_mos)),
            'std': float(np.std(all_mos))
        }
    }
    
    print("Validation Summary Metrics:")
    for metric, stats in summary_stats.items():
        print(f"{metric}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}")
    
    return summary_stats, all_stoi, all_pesq, all_si_sdr, all_mos

def scatter_plot_metric(ref_values, estimated_values, metric_name):
    """
    Creates a scatter plot comparing reference and estimated metric values.
    
    Args:
        ref_values (list or np.array): The reference metric values.
        estimated_values (list or np.array): The estimated metric values (e.g., from Squim).
        metric_name (str): Name of the metric (e.g., "STOI", "PESQ", "SI-SDR").
    """
    ref_values = np.array(ref_values)
    estimated_values = np.array(estimated_values)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(ref_values, estimated_values, alpha=0.6, label="Samples")
    plt.xlabel(f"Reference {metric_name}")
    plt.ylabel(f"Estimated {metric_name}")
    plt.title(f"Scatter Plot: Reference vs. Estimated {metric_name}")
    
    # Plot the diagonal line (ideal match)
    min_val = min(ref_values.min(), estimated_values.min())
    max_val = max(ref_values.max(), estimated_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")
    
    plt.legend()
    plt.grid(True)
    plt.show()