import os
import pickle
import torch
import numpy as np
import librosa
from scipy.signal import wiener
from tqdm import tqdm
from evaluation_utils import DenoiserEvaluator, wrap_baseline_data
from utils import get_datasets, process_sample


dataset_dir = "noisy_speech_dataset"
SR = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Limit evaluation to a subset for faster turnaround.


def wiener_filter(noisy_audio, mysize=512):
    """Apply Wiener filter in the time domain."""
    denoised_audio = wiener(noisy_audio, mysize=mysize)
    return denoised_audio

def spectral_subtraction(noisy_audio, sr, n_fft=1024, win_length=1024, hop_length=256,
                         noise_frame_start=0, noise_frame_end=10):
    """Perform spectral subtraction denoising."""
    # Compute STFT of the noisy audio.
    D = librosa.stft(noisy_audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise profile from the first few frames.
    noise_profile = np.mean(magnitude[:, noise_frame_start:noise_frame_end], axis=1, keepdims=True)
    
    # Subtract noise from the magnitude spectrum.
    denoised_magnitude = np.maximum(magnitude - noise_profile, 0)
    
    # Reconstruct the denoised audio.
    denoised_stft = denoised_magnitude * np.exp(1j * phase)
    denoised_audio = librosa.istft(denoised_stft, win_length=win_length, hop_length=hop_length)
    return denoised_audio

def to_tensor(x):
    """Convert a numpy waveform to a torch tensor with a batch dimension."""
    return torch.from_numpy(x).float().unsqueeze(0)

def baseline_denoise(noisy_waveform, sr, method="wiener", **kwargs):
    """
    Denoise a given waveform using the specified baseline method.
    """
    if method == "wiener":
        denoised = wiener_filter(noisy_waveform, **kwargs)
    elif method == "spectral_subtraction":
        denoised = spectral_subtraction(noisy_waveform, sr, **kwargs)
    else:
        raise ValueError("Unknown method: choose 'wiener' or 'spectral_subtraction'")
    return denoised

def evaluate_baseline_experiment(experiment_name, force_recompute=False, max_samples=None):
    """
    Evaluate baseline denoising methods (Wiener filtering and spectral subtraction)
    on the validation set. Evaluation metrics are saved to (or loaded from) disk using pickle.
    """
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "baseline_metrics.pkl")
    
    # If saved metrics exist and we're not forcing a recompute, load and return them.
    if not force_recompute and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            baseline_metrics = pickle.load(f)
            converted_baseline_metrics = {}
            for method, metrics in baseline_metrics.items():
                converted_baseline_metrics[method] = wrap_baseline_data(baseline_metrics[method])
        print(f"Loaded baseline metrics from {save_path}")
        return converted_baseline_metrics

    _, val_dataset = get_datasets(dataset_dir)
    
    evaluator = DenoiserEvaluator()
    
    # Define the baseline methods to evaluate.
    methods = ["wiener", "spectral_subtraction"]
    baseline_metrics = {method: {"pesq": [], "stoi": [], "si_sdr": [], "mos": []} for method in methods}
    
    # Define a dummy model since our baseline methods operate directly on the waveform.
    class DummyModel:
        def __call__(self, x):
            return x
    dummy_model = DummyModel()
    
    num_samples = len(val_dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    
    # Loop over the validation set.
    for i in tqdm(range(num_samples), desc="Evaluating Baselines"):
        noisy_waveform, clean_waveform, _ = process_sample(val_dataset, i, dummy_model, device)
        
        for method in methods:
            if method == "wiener":
                denoised = baseline_denoise(noisy_waveform, SR, method="wiener", mysize=1024)
            elif method == "spectral_subtraction":
                denoised = baseline_denoise(noisy_waveform, SR, method="spectral_subtraction",
                                              n_fft=1024, win_length=1024, hop_length=256)
            
            # Convert signals to tensors (with a batch dimension).
            tensor_noisy = to_tensor(noisy_waveform)
            tensor_clean = to_tensor(clean_waveform)
            tensor_denoised = to_tensor(denoised)
            
            # Compute evaluation metrics.
            metrics = evaluator.evaluate(clean=tensor_clean,
                                         denoised=tensor_denoised,
                                         reference_nmr=tensor_noisy)
            
            # Append metrics to our dictionary.
            for key in baseline_metrics[method]:
                baseline_metrics[method][key].append(metrics[key])
    
    # Save the computed metrics to disk using pickle.
    with open(save_path, "wb") as f:
        pickle.dump(baseline_metrics, f)
    print(f"Saved baseline metrics to {save_path}")
    
    converted_baseline_metrics = {}
    for method, metrics in baseline_metrics.items():
        converted_baseline_metrics[method] = wrap_baseline_data(baseline_metrics[method])
    
    return converted_baseline_metrics

######################################
# Main Block
######################################

if __name__ == "__main__":
    
    _ = evaluate_baseline_experiment("simple_methods_full", force_recompute=True, max_samples=None)
