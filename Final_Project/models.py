import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    def __init__(self, max_length):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding='same'),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class Autoencoder(nn.Module):
    def __init__(self, max_length):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=7, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.decoder = nn.Sequential(
            # First upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Second upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=5, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Final upsampling block
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size=7, padding='same'),
            nn.Tanh()  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_length=48000):
        super().__init__()
        
        # Encoder
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            AttentionBlock(256),
            nn.Conv1d(256, 128, 3, padding=1)
        )

        # Decoder with corrected channels
        self.decoder = nn.ModuleDict({
            'dec1': DecoderBlock(128, 128, 64, 4),  # Input: 128 (bottleneck) + 128 (skip)
            'dec2': DecoderBlock(64, 64, 32, 4),    # Input: 64 + 64
            'dec3': DecoderBlock(32, 32, 1, 4)      # Input: 32 + 32
        })

        self.final = nn.Sequential(
            nn.Conv1d(1, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1_conv(x)        # [B, 32, L]
        e1_pool = self.enc1_pool(e1)  # [B, 32, L//4]
        
        e2 = self.enc2_conv(e1_pool)  # [B, 64, L//4]
        e2_pool = self.enc2_pool(e2)  # [B, 64, L//16]
        
        e3 = self.enc3_conv(e2_pool)  # [B, 128, L//16]
        e3_pool = self.enc3_pool(e3)  # [B, 128, L//64]

        # Bottleneck
        b = self.bottleneck(e3_pool)  # [B, 128, L//64]

        # Decoder with proper channel handling
        d1 = self.decoder['dec1'](b, e3)  # [B, 64, L//16]
        d2 = self.decoder['dec2'](d1, e2) # [B, 32, L//4]
        d3 = self.decoder['dec3'](d2, e1) # [B, 1, L]
        
        return self.final(d3)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, 
                                  mode='linear',
                                  align_corners=True)
        self.conv = nn.Sequential(
            # Input channels: in_channels + skip_channels
            nn.Conv1d(in_channels + skip_channels, out_channels, 5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatches
        diff = skip.size(2) - x.size(2)
        x = F.pad(x, (diff//2, diff - diff//2))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.Conv1d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv1d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        att = self.channel_att(x.mean(dim=-1, keepdim=True))
        return x * att

class ResAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Encoder: using a few convolutional layers
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
        # Decoder: using ConvTranspose1d for learnable upsampling
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
        # Residual connection: model predicts a residual that is subtracted from x.
        return x - decoded
    
class UNetSpec(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Downsampling path
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU()
        )
        
        # Upsampling path
        rev_features = features[::-1]
        current_channels = features[-1] * 2  # Output channels from bottleneck
        for feature in rev_features:
            # Upsample block
            up_conv = nn.Sequential(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2),
                nn.BatchNorm2d(feature),
                nn.ReLU()
            )
            # Post-concatenation block
            post_conv = nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU()
            )
            self.ups.append(nn.ModuleList([up_conv, post_conv]))
            current_channels = feature  # Update current_channels for next layer
        
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
            # Handle potential shape mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = post_conv(x)
            
        return self.final_conv(x)
    

class HybridDenoiser(nn.Module):
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        # Time-domain branch
        self.time_branch = ResAutoencoder(in_channels=1)
        
        # Frequency-domain branch
        self.spec_branch = UNetSpec(in_channels=1, out_channels=1, features=[32, 64, 128, 256])
        
        # Fusion: fuse along the channel dimension and reduce back to one channel.
        self.fusion = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x: waveform [B, 1, L]
        # Time-domain denoising
        time_out = self.time_branch(x)
        
        # Frequency-domain processing:
        # Compute STFT; here we assume x is real-valued.
        stft = torch.stft(x.squeeze(1), n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, return_complex=True)
        mag = stft.abs()  # [B, F, T]
        phase = stft.angle()  # [B, F, T]
        
        # Add a channel dimension: [B, 1, F, T]
        mag = mag.unsqueeze(1)
        enhanced_mag = self.spec_branch(mag)
        
        # Reconstruct the complex spectrogram from the enhanced magnitude and original phase
        enhanced_spec = enhanced_mag.squeeze(1) * torch.exp(1j * phase)
        # Inverse STFT
        time_from_spec = torch.istft(enhanced_spec,
                                     n_fft=self.n_fft,
                                     win_length=self.win_length,
                                     hop_length=self.hop_length, 
                                     window = torch.hann_window(window_length=self.win_length, device=x.device))
        # output has shape [B, 1, L]
        time_from_spec = time_from_spec.unsqueeze(1)
        
        # Fuse the time-branch and frequency-branch outputs.
        # Ensure both outputs have the same length. If not, use interpolation/padding.
        if time_out.shape[-1] != time_from_spec.shape[-1]:
            time_from_spec = F.interpolate(time_from_spec, size=time_out.shape[-1])
            
        fused = self.fusion(torch.cat([time_out, time_from_spec], dim=1))
        return fused