import os
import pickle
import torch
import numpy as np
import librosa
import multiprocessing as mp
from scipy.signal import wiener
from tqdm import tqdm
from evaluation_utils import DenoiserEvaluator
from utils import get_datasets, process_sample

#############################
# Baseline Denoising Methods
#############################

def wiener_filter(noisy_audio, mysize=512):
    """Wiener filter in time domain."""
    denoised_audio = wiener(noisy_audio, mysize=mysize)
    return denoised_audio

def spectral_subtraction(noisy_audio, sr, n_fft=1024, win_length=1024, hop_length=256,
                         noise_frame_start=0, noise_frame_end=10):
    """Spectral subtraction denoising."""
    # Compute STFT of noisy audio
    D = librosa.stft(noisy_audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise profile from first few frames
    noise_profile = np.mean(magnitude[:, noise_frame_start:noise_frame_end], axis=1, keepdims=True)
    
    # Subtract noise from magnitude spectrum
    denoised_magnitude = np.maximum(magnitude - noise_profile, 0)
    
    # Reconstruct audio from modified magnitude and original phase
    denoised_stft = denoised_magnitude * np.exp(1j * phase)
    denoised_audio = librosa.istft(denoised_stft, win_length=win_length, hop_length=hop_length)
    
    return denoised_audio

##########################
# Helper Conversions
##########################

def to_tensor(x):
    """Convert a numpy waveform to a torch tensor with a batch dimension."""
    return torch.from_numpy(x).float().unsqueeze(0)

def baseline_denoise(noisy_waveform, sr, method="wiener", **kwargs):
    """
    Denoise a given noisy waveform using a specified baseline method.
    """
    if method == "wiener":
        denoised = wiener_filter(noisy_waveform, **kwargs)
    elif method == "spectral_subtraction":
        denoised = spectral_subtraction(noisy_waveform, sr, **kwargs)
    else:
        raise ValueError("Unknown method: choose 'wiener' or 'spectral_subtraction'")
    return denoised

######################################
# Per-Sample Evaluation (Multiprocessing)
######################################

def evaluate_sample(sample_idx, method, SR, device, val_dataset):
    """
    Evaluate a single sample using the specified baseline method.
    """
    # Dummy model since our baseline methods operate on waveforms directly.
    class DummyModel:
        def __call__(self, x):
            return x
    dummy_model = DummyModel()
    
    # Obtain the noisy and clean waveforms using your process_sample helper.
    noisy_waveform, clean_waveform, _ = process_sample(val_dataset, sample_idx, dummy_model, device)
    
    # Apply the appropriate baseline denoising method.
    if method == "wiener":
        denoised = baseline_denoise(noisy_waveform, SR, method="wiener", mysize=512)
    elif method == "spectral_subtraction":
        denoised = baseline_denoise(noisy_waveform, SR, method="spectral_subtraction",
                                      n_fft=1024, win_length=512, hop_length=256)
    else:
        raise ValueError("Unknown method")
    
    # Convert signals to tensors (with a batch dimension)
    tensor_noisy = to_tensor(noisy_waveform)
    tensor_clean = to_tensor(clean_waveform)
    tensor_denoised = to_tensor(denoised)
    
    # Evaluate using DenoiserEvaluator.
    evaluator = DenoiserEvaluator()
    metrics = evaluator.evaluate(clean=tensor_clean,
                                 denoised=tensor_denoised,
                                 reference_nmr=tensor_noisy)
    return metrics

# Global wrapper function (avoiding lambda pickling issues).
def evaluate_sample_wrapper(args):
    return evaluate_sample(*args)

######################################
# Main Evaluation Function
######################################

def evaluate_baseline_experiment(experiment_name, force_recompute=False, max_samples=None):
    """
    Evaluate baseline denoising methods (Wiener filtering and spectral subtraction)
    on the validation set. Uses multiprocessing for faster evaluation, displays progress
    with tqdm, saves results to disk using pickle, and loads previously computed metrics if available.
    """

    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "baseline_metrics.pkl")
    
    # Load from disk if available and not forcing recompute.
    if not force_recompute and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            baseline_metrics = pickle.load(f)
        print(f"Loaded baseline metrics from {save_path}")
        return baseline_metrics

    # Load your validation dataset using your helper.
    _, val_dataset = get_datasets(dataset_dir)
    
    # Check if the dataset is empty.
    if len(val_dataset) == 0:
        raise ValueError("The validation dataset is empty. Please check the dataset directory and files.")
    
    # If max_samples is specified, limit the number of samples.
    total_samples = len(val_dataset)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    # Define the baseline methods to evaluate.
    methods = ["wiener", "spectral_subtraction"]
    baseline_metrics = {method: {"pesq": [], "stoi": [], "si_sdr": [], "mos": []} for method in methods}
    
    # Set up a multiprocessing pool.
    num_workers = mp.cpu_count() 
    pool = mp.Pool(num_workers)
    
    # Evaluate each method.
    for method in methods:
        # Prepare tasks: each task is a tuple (sample_idx, method, SR, device, val_dataset)
        tasks = [(i, method, SR, device, val_dataset) for i in range(total_samples)]
        # Use imap_unordered wrapped in tqdm for progress.
        results = list(tqdm(pool.imap_unordered(evaluate_sample_wrapper, tasks),
                            total=total_samples,
                            desc=f"Evaluating {method}"))
        # Aggregate metrics from all samples.
        for metrics in results:
            for key in baseline_metrics[method]:
                baseline_metrics[method][key].append(metrics[key])
    
    pool.close()
    pool.join()
    
    # Save the computed metrics to disk using pickle.
    with open(save_path, "wb") as f:
        pickle.dump(baseline_metrics, f)
    print(f"Saved baseline metrics to {save_path}")
    
    return baseline_metrics

######################################
# Main Block
######################################

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    dataset_dir = "noisy_speech_dataset"
    SR = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # limit evaluation to a subset for faster turnaround.
    max_samples = 1000  
    
    # run the evaluation.
    _ = evaluate_baseline_experiment("simple_methods", force_recompute=True, max_samples=max_samples)
    