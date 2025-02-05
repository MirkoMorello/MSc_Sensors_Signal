import torch
import torch.nn.functional as F
import torch.nn as nn

def simple_loss(denoised, clean):
    """
    Computes a simple loss as the sum of L1 loss and MSE loss on the time-domain waveforms.
    
    Args:
        denoised (torch.Tensor): Denoised output waveform (e.g., shape [B, 1, L])
        clean (torch.Tensor): Ground truth waveform (same shape as denoised)
        
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
    stft = torch.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length, return_complex=True)
    # Return the magnitude
    return stft.abs()


def compute_si_sdr_loss(denoised, clean, eps=1e-8):
    """
    Computes the scale-invariant SDR (SI-SDR) loss.
    Both inputs are expected to have shape [B, 1, L]. The function returns the mean SI-SDR over the batch.
    Since we want to maximize SI-SDR, we use its negative as a loss.
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
    
    # We want to maximize SI-SDR, so as a loss we use its negative.
    return -si_sdr.mean()


def hybrid_loss(denoised, clean, simple_loss_fn = simple_loss, 
                lambda_time=1.0, lambda_freq=0.5, lambda_sisdr=0.5,
                n_fft=1024, win_length=512, hop_length=256):
    """
    Computes a hybrid loss combining time-domain loss, frequency-domain loss, and SI-SDR loss.
    
    Args:
      denoised: Denoised output waveform, tensor of shape [B, 1, L]
      clean: Ground truth waveform, tensor of shape [B, 1, L]
      loss_fn: e.g. torch.nn.L1Loss()
      mse_loss: e.g. torch.nn.MSELoss()
      lambda_time, lambda_freq, lambda_sisdr: weights for each loss component.
      n_fft, win_length, hop_length: parameters for the STFT.
      
    Returns:
      A scalar loss.
    """
    # Time-domain loss (simple loss on the waveform)
    time_loss = simple_loss_fn(denoised, clean)
    
    # Frequency-domain loss: compute spectrograms and then compute simple loss between them.
    denoised_spec = compute_spectrogram(denoised, n_fft, win_length, hop_length)
    clean_spec = compute_spectrogram(clean, n_fft, win_length, hop_length)
    freq_loss = simple_loss_fn(denoised_spec, clean_spec)
    
    # SI-SDR loss (differentiable)
    sisdr_loss = compute_si_sdr_loss(denoised, clean)
    
    total_loss = lambda_time * time_loss + lambda_freq * freq_loss + lambda_sisdr * sisdr_loss
    return total_loss
