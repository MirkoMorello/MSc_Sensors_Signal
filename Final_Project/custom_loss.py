import torch
import torch.nn.functional as F
import torch.nn as nn

def simple_loss(denoised, clean):
    """
    Computes a simple loss as the sum of L1 loss and MSE loss.
    
    Args:
        denoised (torch.Tensor): e.g., shape [B, 1, L] (time) or [B, 1, F, T] (freq)
        clean (torch.Tensor): Same shape as denoised
    Returns:
        torch.Tensor: A scalar loss value.
    """
    l1 = nn.L1Loss()(denoised, clean)
    mse = nn.MSELoss()(denoised, clean)
    return l1 + mse


def compute_spectrogram(waveform, n_fft=1024, win_length=512, hop_length=256):
    """
    Computes the magnitude spectrogram for a batch of waveforms.
    Assumes waveform shape is [B, 1, L]. The output shape will be [B, F, T].
    """
    # Remove the channel dimension: [B, L]
    waveform = waveform.squeeze(1)
    # Compute the STFT with return_complex=True (PyTorch 1.7+)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        return_complex=True
    )
    # Return the magnitude
    return stft.abs()


def compute_si_sdr_loss(denoised, clean, eps=1e-8):
    """
    Computes the scale-invariant SDR (SI-SDR) loss for time-domain signals.
    Both inputs are expected to have shape [B, 1, L].
    Returns the negative mean SI-SDR (a scalar).
    """
    # Remove channel dimension: [B, L]
    denoised = denoised.squeeze(1)
    clean = clean.squeeze(1)
    
    # Zero-mean each signal
    denoised = denoised - torch.mean(denoised, dim=1, keepdim=True)
    clean = clean - torch.mean(clean, dim=1, keepdim=True)
    
    # Compute projection of denoised onto clean
    dot = torch.sum(denoised * clean, dim=1, keepdim=True)
    norm_clean = torch.sum(clean ** 2, dim=1, keepdim=True) + eps
    scale = dot / norm_clean
    projection = scale * clean
    
    noise = denoised - projection
    ratio = torch.sum(projection ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)
    
    return -si_sdr.mean()

def naive_istft_zero_phase(mag_spec, n_fft=1024, win_length=1024, hop_length=256):
    """
    Performs a naive inverse STFT on a magnitude spectrogram using zero phase.
    Expects mag_spec of shape [B, 1, F, T] and returns a time-domain tensor [B, 1, L].
    """
    # Remove the channel dimension: [B, F, T]
    mag = mag_spec.squeeze(1)
    # Create a zero-phase tensor (same shape as mag)
    zeros_phase = torch.zeros_like(mag)
    # Combine magnitude with zero phase to form a complex tensor
    complex_spec = mag * torch.exp(1j * zeros_phase)
    # Compute the inverse STFT; the output shape L is determined by the parameters
    time_wave = torch.istft(
        complex_spec,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        return_complex=False  # returns a real-valued tensor
    )
    # Add the channel dimension back: [B, 1, L]
    return time_wave.unsqueeze(1)




def hybrid_loss(denoised, clean,
                simple_loss_fn=simple_loss,
                lambda_time=1.0, lambda_freq=0.5, lambda_sisdr=0.5,
                n_fft=1024, win_length=512, hop_length=256):
    """
    A "hybrid" loss combining:
      - time-domain loss (simple_loss on raw waveforms),
      - frequency-domain loss (simple_loss on magnitude spectrograms),
      - SI-SDR loss.
    
    Expects both denoised and clean to be time-domain waveforms of shape [B, 1, L].
    You must have a dataset that loads [B, 1, L] (e.g., StreamingAudioDataset).
    """
    # ----- Time-Domain -----
    time_loss = simple_loss_fn(denoised, clean)
    
    # ----- Frequency-Domain -----
    denoised_spec = compute_spectrogram(denoised, n_fft, win_length, hop_length)
    clean_spec = compute_spectrogram(clean, n_fft, win_length, hop_length)
    freq_loss = simple_loss_fn(denoised_spec, clean_spec)
    
    # ----- SI-SDR -----
    sisdr_loss = compute_si_sdr_loss(denoised, clean)
    
    total_loss = (lambda_time * time_loss
                  + lambda_freq * freq_loss
                  + lambda_sisdr * sisdr_loss)
    return total_loss


def hybrid_loss_freq(denoised_spec, clean_spec,
                     simple_loss_fn=simple_loss,
                     lambda_time=1.0,
                     lambda_freq=0.5,
                     lambda_sisdr=0.5,
                     n_fft=1024,
                     win_length=512,
                     hop_length=256):
    """
    "Hybrid" loss for frequency-based datasets/models, shape [B, 1, F, T].
    We do:
      - freq_loss: compare denoised_spec & clean_spec in freq domain
      - time_loss: naive iSTFT (zero phase) both, then do L1+MSE
      - si_sdr_loss: naive iSTFT both, then do SI-SDR in time domain

    Because the dataset only has magnitude, we have no phase and 
    do iSTFT with zero phase. This is naive but allows time metrics.
    """
    # 1) Frequency-domain loss
    freq_loss = simple_loss_fn(denoised_spec, clean_spec)
    
    # 2) Convert both to time domain (zero phase)
    denoised_time = naive_istft_zero_phase(denoised_spec, n_fft, win_length, hop_length)
    clean_time = naive_istft_zero_phase(clean_spec, n_fft, win_length, hop_length)

    # 3) Time-domain L1+MSE
    time_loss = simple_loss_fn(denoised_time, clean_time)

    # 4) SI-SDR
    sisdr_loss = compute_si_sdr_loss(denoised_time, clean_time)

    total_loss = (lambda_time * time_loss
                  + lambda_freq * freq_loss
                  + lambda_sisdr * sisdr_loss)
    return total_loss
