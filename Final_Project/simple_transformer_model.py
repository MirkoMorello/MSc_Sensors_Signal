import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock2D(nn.Module):
    """
    A 2D channel-attention block that computes a channel attention map by
    aggregating information across spatial (frequency x time) dimensions.
    """
    def __init__(self, channels):
        super().__init__()
        # Using 1x1 "conv" to create the attention, similar to your 1D version.
        # This is channel-attention only, so we do global spatial pooling.
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C, F, T]
        # Global average pool across frequency and time
        att = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        att = self.channel_att(att)            # [B, C, 1, 1]
        return x * att

class EncoderBlock2D(nn.Module):
    """
    A helper block for the 2D encoder stage (Conv2D -> BN -> activation -> Pool).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, pool_size=2, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        # We’ll do two convolutions per stage if you want deeper features.
        # You can adjust this based on your preference for capacity.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            activation
        )
        self.pool = nn.MaxPool2d(pool_size)
        
    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled  # Return both for skip connection

class DecoderBlock2D(nn.Module):
    """
    A helper block for the 2D decoder stage (Upsample -> skip connection -> conv).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, scale_factor=2, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            activation
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # Make sure shapes match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)  # along channel dimension
        return self.conv(x)

class TransformerAutoencoderFreq(nn.Module):
    """
    A transformer-style autoencoder that processes signals in the frequency domain.
    1) Perform STFT to get magnitude & phase
    2) Encode/Decode the magnitude spectrogram with a 2D CNN + channel-attention
    3) Use the original phase for reconstruction in iSTFT
    """
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # --------------------------
        # Encoder
        # --------------------------
        # Example feature sizes: [1 -> 32 -> 64 -> 128 -> 256]
        # Adjust as needed depending on how large your STFT is.
        self.enc1 = EncoderBlock2D(in_ch=1,   out_ch=32,  kernel_size=3, pool_size=2)
        self.enc2 = EncoderBlock2D(in_ch=32,  out_ch=64,  kernel_size=3, pool_size=2)
        self.enc3 = EncoderBlock2D(in_ch=64,  out_ch=128, kernel_size=3, pool_size=2)

        # --------------------------
        # Bottleneck with attention
        # --------------------------
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.attention_block = AttentionBlock2D(256)
        self.bottleneck_out = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # back to 128 channels

        # --------------------------
        # Decoder
        # --------------------------
        self.dec1 = DecoderBlock2D(in_ch=128 + 128, out_ch=64,  kernel_size=3, scale_factor=2)
        self.dec2 = DecoderBlock2D(in_ch=64 + 64,   out_ch=32,  kernel_size=3, scale_factor=2)
        self.dec3 = DecoderBlock2D(in_ch=32 + 32,   out_ch=16,  kernel_size=3, scale_factor=2)

        # Final conv to get a single channel for magnitude
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # or Tanh, or ReLU — depends on your magnitude scaling
        )

    def forward(self, x):
        """
        x is the time-domain waveform of shape [B, 1, L].
        We'll:
          1) Compute STFT -> [B, freq, time], separate magnitude and phase
          2) Pass magnitude through the 2D autoencoder
          3) Recombine with original phase
          4) iSTFT to get the time-domain signal
        """
        # ---------------------------
        # 1) STFT
        # ---------------------------
        stft_complex = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )  # shape: [B, freq, time]

        mag = stft_complex.abs()    # [B, freq, time]
        phase = stft_complex.angle()  # [B, freq, time]

        # Add channel dimension for CNN: [B, 1, freq, time]
        mag = mag.unsqueeze(1)

        # ---------------------------
        # 2) Encoder
        # ---------------------------
        e1, p1 = self.enc1(mag)  # p1 is pooled
        e2, p2 = self.enc2(p1)
        e3, p3 = self.enc3(p2)

        # ---------------------------
        # 3) Bottleneck
        # ---------------------------
        b = self.bottleneck_conv(p3)
        b = self.attention_block(b)
        b = self.bottleneck_out(b)

        # ---------------------------
        # 4) Decoder with skip connections
        # ---------------------------
        d1 = self.dec1(b, e3)  # combine with skip e3
        d2 = self.dec2(d1, e2) # combine with skip e2
        d3 = self.dec3(d2, e1) # combine with skip e1

        # Final 1-channel output (enhanced magnitude)
        enhanced_mag = self.final_conv(d3)  # [B, 1, freq, time]

        # ---------------------------
        # 5) Reconstruct time-domain
        # ---------------------------
        # If you used Sigmoid, your magnitude is in [0,1].
        # Typically you'd want to scale it back to match original scale,
        # or you might have used a direct model that regresses amplitude.
        #
        # For demonstration, we'll just multiply the original phase:
        enhanced_spec = enhanced_mag.squeeze(1) * torch.exp(1j * phase)  # [B, freq, time]
        out = torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
             window = torch.hann_window(window_length=self.win_length, device=x.device)
        )  # [B, L]
        return out.unsqueeze(1)  # shape [B, 1, L]
