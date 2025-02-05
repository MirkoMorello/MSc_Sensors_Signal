from torch.utils.data import Dataset
import torch
import librosa
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env

SR = int(os.getenv("SR"))
MAX_DURATION = int(os.getenv("MAX_DURATION"))
MAX_LENGTH  = MAX_DURATION * SR
N_FFT = int(os.getenv("N_FFT"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))



class StreamingAudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files, sr=SR):
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr
        
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        # Load and process one audio pair at a time
        noisy = self._load_audio(self.noisy_files[idx])
        clean = self._load_audio(self.clean_files[idx])
        
        # Convert to tensors and add a channel dimension: shape becomes [1, MAX_LENGTH]
        return (
            torch.from_numpy(noisy).float().unsqueeze(0),
            torch.from_numpy(clean).float().unsqueeze(0)
        )
    
    def _load_audio(self, path):
        try:
            audio, _ = librosa.load(path, sr=self.sr)
            # Truncate or pad the audio to MAX_LENGTH
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
        
        # compute the STFT and take the magnitude (discard phase)
        noisy_spec = librosa.stft(noisy, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        clean_spec = librosa.stft(clean, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        
        # magnitude spectrograms
        noisy_mag = np.abs(noisy_spec)
        clean_mag = np.abs(clean_spec)
        
        # convert to tensor and add a channel dimension, resulting in shape [1, F, T]
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
