
# https://pytorch.org/audio/main/tutorials/squim_tutorial.html

import torch
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import pesq
import numpy as np
import pandas as pd
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
from dotenv import load_dotenv
import os
import librosa
import matplotlib.pyplot as plt
import pickle
import json
import multiprocessing as mp
from torch.utils.data import DataLoader
from models import UNetSpec, ResAutoencoder, HybridDenoiser
from simple_transformer_model import TransformerAutoencoderFreq
from utils import load_checkpoint, get_spectrogram_datasets, get_datasets, _process_audio


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
        self.device = device
        # Load pre-trained models from torchaudio-squim and move them to the device.
        self.objective_model = SQUIM_OBJECTIVE.get_model().to(device)
        self.subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    
    def _to_tensor(self, audio):
        """
        Converts a NumPy array to a torch.Tensor if needed.
        Ensures the tensor is moved to the evaluator's device.
        """
        # If the audio tensor has 3 dimensions, assume it's [B, C, L].
        if audio.dim() == 3:
            # Only average if the channel dimension (dim=1) has more than one channel.
            if audio.size(1) > 1:
                audio = audio.mean(dim=1, keepdim=True)
        # Otherwise (if audio is 2D or 1D) we assume it's already mono.
        return audio

    def evaluate(self, clean, denoised, reference_nmr):
        """
        Evaluates the denoised signal against the clean reference using objective and subjective metrics.
        
        Returns:
            dict: Contains 'stoi', 'pesq', 'si_sdr', and 'mos' scores.
        """
        # Convert inputs to tensors if needed.
        clean = self._to_tensor(clean)
        denoised = self._to_tensor(denoised)
        reference_nmr = self._to_tensor(reference_nmr)

        # Preprocess each signal.
        denoised = self._preprocess(denoised)
        reference_nmr = self._preprocess(reference_nmr)
        clean = self._preprocess(clean)

        # Squeeze extra dimensions:
        if denoised.dim() == 4:
            denoised = denoised.squeeze(1).squeeze(1)
        if reference_nmr.dim() == 4:
            reference_nmr = reference_nmr.squeeze(1).squeeze(1)
        if clean.dim() == 4:
            clean = clean.squeeze(1).squeeze(1)

        # If the input is 3D and has a singleton channel dimension, squeeze it.
        if denoised.dim() == 3 and denoised.size(1) == 1:
            denoised = denoised.squeeze(1)
        if reference_nmr.dim() == 3 and reference_nmr.size(1) == 1:
            reference_nmr = reference_nmr.squeeze(1)
        if clean.dim() == 3 and clean.size(1) == 1:
            clean = clean.squeeze(1)

        # Compute objective metrics (STOI, PESQ, SI-SDR)
        stoi_val, pesq_val, si_sdr_val = self.objective_model(denoised)
        # Compute subjective MOS (using non-matching reference)
        mos_val = self.subjective_model(denoised, reference_nmr)

        return {
            'stoi': stoi_val.item(),
            'pesq': pesq_val.item(),
            'si_sdr': si_sdr_val.item(),
            'mos': mos_val.item()
        }


    def _preprocess(self, audio):
        """Convert to 16kHz mono if needed."""
        # If audio has more than one channel (i.e. shape[0] > 1), convert stereo to mono.
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # Assuming the input is already 16kHz; otherwise, add proper resampling here.
        return audio


def unetspec_inference(
    noisy_wave: torch.Tensor,
    model: torch.nn.Module,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """
    Inference for magnitude-based UNetSpec:
    1) STFT the noisy time-wave
    2) Pass the magnitude into UNetSpec
    3) Combine predicted magnitude with original noisy phase
    4) iSTFT => time-domain enhanced signal

    Args:
        noisy_wave (Tensor): [B, 1, L] time-domain noisy audio
        model (nn.Module):   Your UNetSpec model
        n_fft, win_length, hop_length: STFT parameters

    Returns:
        Tensor of shape [B, 1, L] = the enhanced audio in time domain
    """
    # 1) Compute STFT on the noisy wave
    #    Remove channel dim => shape [B, L] for torch.stft
    stft_noisy = torch.stft(
        noisy_wave.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=noisy_wave.device),
        return_complex=True
    )  # => [B, F, T] (complex)

    noisy_mag = stft_noisy.abs()          # [B, F, T]
    noisy_phase = torch.angle(stft_noisy) # [B, F, T]

    # 2) Pass magnitude to UNetSpec => predicted magnitude
    #    UNetSpec expects [B, 1, F, T]
    model_input = noisy_mag.unsqueeze(1)  # => [B, 1, F, T]
    pred_mag = model(model_input)         # => [B, 1, F, T]

    # 3) Combine predicted magnitude + original phase => complex
    enhanced_complex = pred_mag.squeeze(1) * torch.exp(1j * noisy_phase) 
    # shape: [B, F, T] (complex)

    # 4) iSTFT => [B, L]
    enhanced_wave = torch.istft(
        enhanced_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=noisy_wave.device),
        return_complex=False
    )  # => [B, L]

    # Return with a channel dimension => [B, 1, L]
    return enhanced_wave.unsqueeze(1)


def compute_reference_metrics(clean_wave, noisy_wave, sample_rate=16000):
    """
    Computes STOI, PESQ, SI-SDR as reference metrics using external libraries
    for a single pair of signals (clean vs. noisy).
        
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


def evaluate_val_set_batch(val_loader, model, evaluator, device):
    """
    Evaluates a model on a validation set using SQUIM (STOI, PESQ, SI-SDR, MOS).

    This version is made to exactly mirror your training-time STOI evaluation:
      - The model output and the clean signal are both processed via _process_audio,
        which (if needed) converts a frequency-domain [B,1,F,T] tensor into a time-domain
        signal using naive_istft_zero_phase with n_fft=1024, win_length=1024, hop_length=256.
      - For UNetSpec models that receive time-domain inputs (i.e. [B,1,L]),
        we use unetspec_inference.
    """
    model.eval()
    all_stoi, all_pesq, all_si_sdr, all_mos = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating validation set"):
            noisy, clean = batch[0].to(device), batch[1].to(device)
            
            # For UNetSpec: if input is time-domain ([B,1,L]), use unetspec_inference.
            if isinstance(model, UNetSpec):
                model_output = unetspec_inference(noisy, model)
            else:
                model_output = model(noisy)
            
            # Process both model output and clean signal using the same _process_audio.
            processed_output = _process_audio(model_output)
            processed_clean = _process_audio(clean)
            
            # Compute the metrics on the processed (time-domain) signals.
            stoi_batch, pesq_batch, si_sdr_batch = evaluator.objective_model(processed_output)
            mos_batch = evaluator.subjective_model(processed_output, processed_clean)
            
            all_stoi.extend(stoi_batch.cpu().tolist())
            all_pesq.extend(pesq_batch.cpu().tolist())
            all_si_sdr.extend(si_sdr_batch.cpu().tolist())
            all_mos.extend(mos_batch.cpu().tolist())
    
    if not all_stoi:
        print("No samples processed for evaluation. Check your data shapes!")
        return None, [], [], [], []
    
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




def plot_radar_chart_normalized(experiments_dict):
    """
    Plots a radar chart of mean metrics, normalized to [0,1].
    experiments_dict: { 'ExperimentName': results_dict, ... }
        where results_dict['evaluation']['summary_stats'][metric]['mean'] is available.
    """
    # Define typical min and max for each metric.
    # Adjust these to match your data or typical known ranges.
    METRIC_RANGES = {
        "STOI":   (0.0, 1.0),
        "PESQ":   (1.0, 2),
        "SI-SDR": (-10, 15.0),
        "MOS":    (1.0, 5.0),
    }
    metrics = list(METRIC_RANGES.keys())  # ["STOI", "PESQ", "SI-SDR", "MOS"]

    # Build a dictionary of normalized metric means, keyed by experiment.
    data = {}
    for exp_name, results in experiments_dict.items():
        summary = results["evaluation"]["summary_stats"]
        # Collect normalized means for each metric
        norm_means = []
        for m in metrics:
            raw_mean = summary[m]["mean"]
            mmin, mmax = METRIC_RANGES[m]
            # Clamp if out of range:
            if raw_mean < mmin: raw_mean = mmin
            if raw_mean > mmax: raw_mean = mmax
            # Normalize:
            val = (raw_mean - mmin) / (mmax - mmin)
            norm_means.append(val)
        data[exp_name] = norm_means

    # Convert to a DataFrame
    df = pd.DataFrame(data, index=metrics)  # shape: [4 metrics, # experiments]

    # Radar chart setup
    categories = list(df.index)  # e.g. ["STOI","PESQ","SI-SDR","MOS"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # repeat the first angle to close the circle

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    # Setting the angle for each axis
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(30)
    # We'll do ticks at 0, 0.2, 0.4, ..., 1.0
    plt.yticks([0.2,0.4,0.6,0.8,1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=7)
    plt.ylim(0, 1)  # Because we normalized each metric to 0..1

    # Plot each experiment
    for exp_name in df.columns:
        values = df[exp_name].tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, label=exp_name)
        ax.fill(angles, values, alpha=0.05)

    plt.title("Radar Chart of Mean Evaluation Metrics\n\n")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def scatter_plot_metric(ref_values, estimated_values, metric_name):
    """
    Creates a scatter plot comparing reference and estimated metric values.
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
    sample_rate=None
):
    """
    Evaluate an experiment and return its evaluation results.
    
    This function:
      - Loads the model and its training history from checkpoints.
      - Chooses the appropriate dataset (spectrogram-based for UNetSpec).
      - Creates a DataLoader for the validation set.
      - Computes per-epoch training loss and a smooth training loss curve.
      - Evaluates the model on the validation set using the updated evaluate_val_set_batch,
        which applies _process_audio in the same way as during training.
      - Saves and returns a results dictionary.
    """
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
    
    if os.path.exists(eval_pickle_file):
        print(f"[{experiment_name}] Loading evaluation results from pickle.")
        with open(eval_pickle_file, "rb") as f:
            results = pickle.load(f)
        return results

    print(f"[{experiment_name}] Computing evaluation results...")
    
    if "UNetSpec" in experiment_name:
        model = UNetSpec()
    elif "hybrid" in experiment_name:
        model = HybridDenoiser()
    elif "ResAutoencoder" in experiment_name:
        model = ResAutoencoder()
    elif "TransformerAutoencoderFreq" in experiment_name:
        model = TransformerAutoencoderFreq()
    else:
        raise ValueError(f"Cannot determine model type from experiment name: {experiment_name}")
    model.to(device)
    
    losses_path = os.path.join(checkpoint_dir, "losses.json")
    if not os.path.exists(losses_path):
        raise ValueError(f"Losses file not found for experiment '{experiment_name}' at {losses_path}")
    with open(losses_path, "r") as f:
        losses = json.load(f)
    
    _ = load_checkpoint(experiment_name, model)
    
    train_losses = losses["train"]
    val_losses   = losses["val"]
    stoi_history = losses["stoi"]
    
    if isinstance(model, UNetSpec):
        train_dataset, val_dataset = get_spectrogram_datasets(dataset_dir)
    else:
        train_dataset, val_dataset = get_datasets(dataset_dir)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    num_completed_epochs = len(val_losses)
    steps_per_epoch = len(train_losses) // num_completed_epochs
    train_losses_per_epoch = []
    for i in range(num_completed_epochs):
        start_idx = i * steps_per_epoch
        end_idx = (i + 1) * steps_per_epoch
        epoch_losses = train_losses[start_idx:end_idx]
        train_losses_per_epoch.append(np.mean(epoch_losses))
    
    total_steps = len(train_losses)
    step_size = max(1, total_steps // (points_per_epoch * num_completed_epochs))
    sampled_train_losses = train_losses[::step_size]
    x_coords = [i * step_size / steps_per_epoch for i in range(len(sampled_train_losses))]
    
    evaluator = DenoiserEvaluator(device=device, target_sr=SR)
    summary_stats, all_stoi, all_pesq, all_si_sdr, all_mos = evaluate_val_set_batch(val_loader, model, evaluator, device)
    
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
    
    with open(eval_pickle_file, "wb") as f:
        pickle.dump(results, f)
    print(f"[{experiment_name}] Evaluation results saved to {eval_pickle_file}")
    
    return results



def wrap_baseline_data(baseline_data):
    summary_stats = {
        "STOI": {
            "mean": float(np.mean(baseline_data["stoi"])),
            "std":  float(np.std(baseline_data["stoi"])),
            "min":  float(np.min(baseline_data["stoi"])),
            "max":  float(np.max(baseline_data["stoi"]))
        },
        "PESQ": {
            "mean": float(np.mean(baseline_data["pesq"])),
            "std":  float(np.std(baseline_data["pesq"])),
            "min":  float(np.min(baseline_data["pesq"])),
            "max":  float(np.max(baseline_data["pesq"]))
        },
        "SI-SDR": {
            "mean": float(np.mean(baseline_data["si_sdr"])),
            "std":  float(np.std(baseline_data["si_sdr"])),
            "min":  float(np.min(baseline_data["si_sdr"])),
            "max":  float(np.max(baseline_data["si_sdr"]))
        },
        "MOS": {
            "mean": float(np.mean(baseline_data["mos"])),
            "std":  float(np.std(baseline_data["mos"])),
            "min":  float(np.min(baseline_data["mos"])),
            "max":  float(np.max(baseline_data["mos"]))
        }
    }
    
    wrapped = {
        "evaluation": {
            "summary_stats": summary_stats,
            "all_stoi": baseline_data["stoi"],
            "all_pesq": baseline_data["pesq"],
            "all_si_sdr": baseline_data["si_sdr"],
            "all_mos": baseline_data["mos"]
        }
    }
    return wrapped

def compute_baseline_metrics(batch_size=80, n_fft=1024, win_length=1024, hop_length=256, sample_rate=16000):
    baseline_pickle = os.path.join('./checkpoints', "baseline_metrics_with_mos.pkl")
    
    if os.path.exists(baseline_pickle):
        print(f"Loading baseline metrics from {baseline_pickle}")
        with open(baseline_pickle, "rb") as f:
            baseline_data = pickle.load(f)
        return wrap_baseline_data(baseline_data)

    print("Computing baseline (Noisy vs. Clean) metrics with MOS...")
    from utils import get_datasets  # your dataset loader
    _, val_dataset = get_datasets(dataset_dir)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    all_stoi, all_pesq, all_sisdr, all_mos = [], [], [], []
    
    # Instantiate the evaluator so we can compute MOS using the subjective model.
    evaluator = DenoiserEvaluator(device=device, target_sr=sample_rate)
    
    for batch in tqdm(val_loader, desc="Computing baseline metrics with MOS"):
        noisy, clean = batch[0], batch[1]
        
        # If spectrogram-based, invert to time-domain:
        if noisy.ndim == 4:  # [B, 1, F, T]
            from utils import naive_istft_zero_phase
            noisy = naive_istft_zero_phase(noisy, n_fft, win_length, hop_length)
            clean = naive_istft_zero_phase(clean, n_fft, win_length, hop_length)
        
        # Squeeze channel dimension if needed
        if noisy.ndim == 3 and noisy.shape[1] == 1:
            noisy = noisy.squeeze(1)
        if clean.ndim == 3 and clean.shape[1] == 1:
            clean = clean.squeeze(1)
        
        # For each sample in the batch, compute metrics.
        for i in range(noisy.size(0)):
            noisy_np = noisy[i].numpy()
            clean_np = clean[i].numpy()
            # Objective metrics via compute_reference_metrics
            s_val, p_val, si_val = compute_reference_metrics(clean_np, noisy_np, sample_rate)
            all_stoi.append(s_val)
            all_pesq.append(p_val)
            all_sisdr.append(si_val)
            
            # For MOS, we need to use the subjective model.
            # Here we feed the noisy signal as if it were "denoised" (i.e. unprocessed),
            # using the clean signal as both the clean and non-matching reference.
            # Convert the noisy sample back to a tensor with batch dimension.
            noisy_tensor = torch.tensor(noisy_np).unsqueeze(0).to(device)
            clean_tensor = torch.tensor(clean_np).unsqueeze(0).to(device)
            mos_val = evaluator.subjective_model(noisy_tensor, clean_tensor).item()
            all_mos.append(mos_val)
    
    baseline_data = {
        "stoi": all_stoi,
        "pesq": all_pesq,
        "si_sdr": all_sisdr,
        "mos": all_mos
    }
    
    with open(baseline_pickle, "wb") as f:
        pickle.dump(baseline_data, f)
    
    print(f"Baseline metrics (with MOS) saved to {baseline_pickle}")
    return wrap_baseline_data(baseline_data)
