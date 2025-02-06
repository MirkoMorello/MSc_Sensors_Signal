
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
import pickle
import json
from torch.utils.data import DataLoader
from models import UNetSpec, ResAutoencoder, HybridDenoiser
from utils import load_checkpoint, get_spectrogram_datasets, get_datasets


load_dotenv()  # Loads variables from .env

SR = int(os.getenv("SR"))
MAX_DURATION = int(os.getenv("MAX_DURATION"))
MAX_LENGTH  = MAX_DURATION * SR
N_FFT = int(os.getenv("N_FFT"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = "noisy_speech_dataset"


class DenoiserEvaluator:
    def __init__(self, device=torch.device("cpu"), target_sr=16000):
        self.target_sr = target_sr
        # Load pre-trained models from torchaudio-squim and move them to the device.
        self.objective_model = SQUIM_OBJECTIVE.get_model().to(device)
        self.subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    
    def evaluate(self, clean, denoised, reference_nmr):
        """
        clean: Original clean waveform [1, T] or with extra dimensions.
        denoised: Your model's output [1, T] or with extra dimensions.
        reference_nmr: Non-matching reference audio [1, T] or with extra dimensions.
        """
        # Ensure 16kHz, mono, etc.
        denoised = self._preprocess(denoised)
        reference_nmr = self._preprocess(reference_nmr)
        clean = self._preprocess(clean)
        
        # Ensure that the inputs are 2D: [batch, time].
        if denoised.dim() == 4:
            denoised = denoised.squeeze(1).squeeze(1)
        if reference_nmr.dim() == 4:
            reference_nmr = reference_nmr.squeeze(1).squeeze(1)
        if clean.dim() == 4:
            clean = clean.squeeze(1).squeeze(1)
        
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
        """Convert to 16kHz mono if needed."""
        # If audio has more than one channel (i.e. shape[0] > 1), convert stereo to mono.
        if audio.shape[0] > 1:
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

import torch
import numpy as np
import torchaudio
from tqdm import tqdm

def naive_istft_zero_phase(mag_spec, n_fft=1024, win_length=1024, hop_length=256):
    """
    Performs a naive inverse STFT on a magnitude spectrogram using zero phase.
    Expects 'mag_spec' of shape [B, 1, F, T] and returns [B, L].
      - We create a zero-phase tensor, combine with magnitude, and call torch.istft.
    """
    if mag_spec.ndim != 4:
        raise ValueError(f"Expected 4D input [B, 1, F, T], got shape {list(mag_spec.shape)}.")

    B, C, F, T = mag_spec.shape
    if C != 1:
        raise ValueError("naive_istft_zero_phase expects channel dimension == 1.")

    # Remove channel dim => [B, F, T]
    mag_spec = mag_spec.squeeze(1)

    # Create zero-phase
    zero_phase = torch.zeros_like(mag_spec)  # same shape: [B, F, T]
    # Convert magnitude to a complex spectrogram
    complex_spec = mag_spec * torch.exp(1j * zero_phase)  # still [B, F, T], complex

    # Inverse STFT => [B, L]
    time_wave = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=False  # returns real [B, L]
    )
    return time_wave


def evaluate_val_set_batch(val_loader, model, evaluator, device,
                           n_fft=1024, win_length=1024, hop_length=256):
    """
    Evaluates a model on a validation set using SQUIM (STOI, PESQ, SI-SDR, MOS).
    - If the model output is 4D (e.g. [B, 1, F, T] spectrogram), we do a naive ISTFT.
    - If the output is time-domain [B, 1, L], we just squeeze to [B, L].
    - Then we feed [B, L] to evaluator.objective_model / evaluator.subjective_model.
    """

    model.eval()
    all_stoi, all_pesq, all_si_sdr, all_mos = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating validation set"):
            # Expecting batch = (noisy, clean), each possibly [B, 1, F, T] or [B, 1, L]
            noisy, clean = batch[0].to(device), batch[1].to(device)

            # Forward pass => model output
            output = model(noisy)

            # --- Invert model output to time-domain if needed ---
            if output.ndim == 4:  # [B, 1, F, T]
                output = naive_istft_zero_phase(output, n_fft, win_length, hop_length)
            # If [B, 1, L], remove the channel dim => [B, L]
            if output.ndim == 3 and output.shape[1] == 1:
                output = output.squeeze(1)

            # --- Also invert the 'clean' (target) if it's a spectrogram ---
            if clean.ndim == 4:  # [B, 1, F, T]
                clean = naive_istft_zero_phase(clean, n_fft, win_length, hop_length)
            if clean.ndim == 3 and clean.shape[1] == 1:
                clean = clean.squeeze(1)

            # Now output/clean should both be [B, L] => feed to SQUIM
            stoi_batch, pesq_batch, si_sdr_batch = evaluator.objective_model(output)
            mos_batch = evaluator.subjective_model(output, clean)

            # Convert to CPU .numpy()
            all_stoi.extend(stoi_batch.cpu().tolist())
            all_pesq.extend(pesq_batch.cpu().tolist())
            all_si_sdr.extend(si_sdr_batch.cpu().tolist())
            all_mos.extend(mos_batch.cpu().tolist())

    # If no samples processed, bail
    if not all_stoi:
        print("No samples were processed for evaluation. Check your data shapes!")
        return None, [], [], [], []

    # Basic summary stats
    summary_stats = {
        'STOI':  {
            'mean': float(np.mean(all_stoi)),
            'std':  float(np.std(all_stoi)),
            'min':  float(np.min(all_stoi)),
            'max':  float(np.max(all_stoi))
        },
        'PESQ':  {
            'mean': float(np.mean(all_pesq)),
            'std':  float(np.std(all_pesq)),
            'min':  float(np.min(all_pesq)),
            'max':  float(np.max(all_pesq))
        },
        'SI-SDR': {
            'mean': float(np.mean(all_si_sdr)),
            'std':  float(np.std(all_si_sdr)),
            'min':  float(np.min(all_si_sdr)),
            'max':  float(np.max(all_si_sdr))
        },
        'MOS':  {
            'mean': float(np.mean(all_mos)),
            'std':  float(np.std(all_mos)),
            'min':  float(np.min(all_mos)),
            'max':  float(np.max(all_mos))
        },
    }

    print("Validation Summary Metrics:")
    for metric, stats in summary_stats.items():
        print(f"{metric}: Mean={stats['mean']:.3f}, Std={stats['std']:.3f}, "
              f"Min={stats['min']:.3f}, Max={stats['max']:.3f}")
    
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
    
    
def evaluate_experiment(
    experiment_name,
    points_per_epoch=100,
    batch_size=80,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    sample_rate=None
):
    """
    Evaluate an experiment and return its evaluation results.
    
    This function checks if the evaluation results are already saved as a pickle
    in the folder checkpoints/<experiment_name>/evaluation_results.pkl. If so, it loads
    and returns the results. Otherwise, it computes:
    
      - Per-epoch average training loss (from the training history in losses.json)
      - A sampled training loss curve (x_coords and sampled losses)
      - Validation losses and STOI history (one per epoch)
      - SQUIM evaluation metrics on the validation set (STOI, PESQ, SI-SDR, MOS)
    
    The function then saves the results in a pickle file and returns a dictionary
    with the following keys:
      - "train_losses_per_epoch"
      - "val_losses"
      - "stoi_history"
      - "evaluation": { "summary_stats", "all_stoi", "all_pesq", "all_si_sdr", "all_mos" }
      - "sampled_train_losses"
      - "x_coords"
    
    Parameters:
      experiment_name (str): Name of the experiment (folder name under checkpoints/)
      dataset_dir (str): Directory containing your dataset.
      device (torch.device): Device on which to run evaluation.
      model (torch.nn.Module): The model to evaluate (it will have its checkpoint loaded).
      points_per_epoch (int): Number of points per epoch for training loss sampling.
      batch_size (int): Batch size for the validation DataLoader.
      n_fft, win_length, hop_length (int): STFT parameters.
      sample_rate (int or None): Sampling rate; if None, it will be loaded from the environment.
    
    Returns:
      results (dict): Dictionary with evaluation data.
    """
    # Determine sample rate.
    # Determine sample rate.
    if sample_rate is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            SR = int(os.getenv("SR"))
        except Exception:
            raise ValueError("Could not load SR from environment; please pass sample_rate explicitly.")
    else:
        SR = sample_rate

    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    eval_pickle_file = os.path.join(checkpoint_dir, "evaluation_results.pkl")
    
    # If the evaluation results have already been computed, load and return them.
    if os.path.exists(eval_pickle_file):
        print(f"[{experiment_name}] Loading evaluation results from pickle.")
        with open(eval_pickle_file, "rb") as f:
            results = pickle.load(f)
        return results

    print(f"[{experiment_name}] Computing evaluation results...")
    
    # --- Instantiate the model based on the experiment name ---
    if "UNetSpec" in experiment_name:
        model = UNetSpec()
    elif "hybrid" in experiment_name:
        model = HybridDenoiser()
    elif "ResAutoencoder" in experiment_name:
        model = ResAutoencoder()
    else:
        raise ValueError(f"Cannot determine model type from experiment name: {experiment_name}")
    model.to(device)
    
    # Load training history from losses.json.
    losses_path = os.path.join(checkpoint_dir, "losses.json")
    if not os.path.exists(losses_path):
        raise ValueError(f"Losses file not found for experiment '{experiment_name}' at {losses_path}")
    with open(losses_path, "r") as f:
        losses = json.load(f)
    
    # Load the model state from the checkpoint.
    _ = load_checkpoint(experiment_name, model)
    
    # Extract training history.
    train_losses = losses["train"]
    val_losses   = losses["val"]
    stoi_history = losses["stoi"]
    
    # --- Choose the appropriate datasets ---
    # If the model is spectrogram-based (UNetSpec), use spectrogram datasets.
    if isinstance(model, UNetSpec):
        train_dataset, val_dataset = get_spectrogram_datasets(dataset_dir)
    else:
        train_dataset, val_dataset = get_datasets(dataset_dir)
    
    # Create a DataLoader for the validation set.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # --- Compute per-epoch average training loss ---
    num_completed_epochs = len(val_losses)  # one validation measurement per epoch
    steps_per_epoch = len(train_losses) // num_completed_epochs
    train_losses_per_epoch = []
    for i in range(num_completed_epochs):
        start_idx = i * steps_per_epoch
        end_idx = (i + 1) * steps_per_epoch
        epoch_losses = train_losses[start_idx:end_idx]
        train_losses_per_epoch.append(np.mean(epoch_losses))
    
    # --- Sample training losses for a smooth training loss curve ---
    total_steps = len(train_losses)
    step_size = max(1, total_steps // (points_per_epoch * num_completed_epochs))
    sampled_train_losses = train_losses[::step_size]
    # x_coords in epoch units (each step represents step_size/steps_per_epoch epochs)
    x_coords = [i * step_size / steps_per_epoch for i in range(len(sampled_train_losses))]
    
    # --- Evaluate the model on the validation set using SQUIM ---
    evaluator = DenoiserEvaluator(device=device, target_sr=SR)
    summary_stats, all_stoi, all_pesq, all_si_sdr, all_mos = evaluate_val_set_batch(
        val_loader, model, evaluator, device,
        n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    
    # Build the results dictionary.
    results = {
        "train_losses_per_epoch": train_losses_per_epoch,
        "val_losses": val_losses,
        "stoi_history": stoi_history,
        "evaluation": {
            "summary_stats": summary_stats,
            "all_stoi": all_stoi,
            "all_pesq": all_pesq,
            "all_si_sdr": all_si_sdr,
            "all_mos": all_mos,
        },
        "sampled_train_losses": sampled_train_losses,
        "x_coords": x_coords,
    }
    
    # Save the results to a pickle file.
    with open(eval_pickle_file, "wb") as f:
        pickle.dump(results, f)
    print(f"[{experiment_name}] Evaluation results saved to {eval_pickle_file}")
    
    return results


def compute_baseline_metrics(batch_size=80, n_fft=1024, win_length=1024, hop_length=256, sample_rate=16000):
    """
    Computes baseline metrics (noisy vs. clean) for the entire validation set.
    If a pickle file 'baseline_metrics.pkl' exists, it loads and returns it.
    Otherwise, it computes the metrics and saves them.
    
    Returns a dict with keys: "stoi", "pesq", "si_sdr".
    """
    baseline_pickle = os.path.join('./checkpoints', "baseline_metrics.pkl")
    
    # If the pickle exists, load and return it.
    if os.path.exists(baseline_pickle):
        print(f"Loading baseline metrics from {baseline_pickle}")
        with open(baseline_pickle, "rb") as f:
            baseline_data = pickle.load(f)
        return baseline_data

    print("Computing baseline (Noisy vs. Clean) metrics...")
    # Here you can choose the appropriate dataset.
    # For example, if you are using time-domain audio:
    from utils import get_datasets  # or your custom dataset loader
    _, val_dataset = get_datasets(dataset_dir)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_stoi = []
    all_pesq = []
    all_sisdr = []
    
    # Loop over the validation set
    for batch in tqdm(val_loader, desc="Computing baseline metrics"):
        noisy, clean = batch[0], batch[1]
        
        # If your dataset is spectrogram-based, you need to invert it:
        # (Assuming naive_istft_zero_phase is your inversion function)
        if noisy.ndim == 4:  # spectrogram case: [B, 1, F, T]
            from utils import naive_istft_zero_phase
            noisy = naive_istft_zero_phase(noisy, n_fft, win_length, hop_length)
            clean = naive_istft_zero_phase(clean, n_fft, win_length, hop_length)
        
        # If shape is [B, 1, L], squeeze the channel dimension
        if noisy.ndim == 3 and noisy.shape[1] == 1:
            noisy = noisy.squeeze(1)
        if clean.ndim == 3 and clean.shape[1] == 1:
            clean = clean.squeeze(1)
        
        # Convert each sample to numpy and compute metrics
        for i in range(noisy.size(0)):
            noisy_np = noisy[i].numpy()
            clean_np = clean[i].numpy()
            # compute_reference_metrics is your provided function
            s_val, p_val, si_val = compute_reference_metrics(clean_np, noisy_np, sample_rate)
            all_stoi.append(s_val)
            all_pesq.append(p_val)
            all_sisdr.append(si_val)
    
    baseline_data = {
        "stoi": all_stoi,
        "pesq": all_pesq,
        "si_sdr": all_sisdr
    }
    
    # Save the baseline metrics
    with open(baseline_pickle, "wb") as f:
        pickle.dump(baseline_data, f)
    
    print(f"Baseline metrics saved to {baseline_pickle}")
    return baseline_data
