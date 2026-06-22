import torch
from torch import nn
from torch.nn import functional as F
import math

class WN(nn.Module):
    """
    WaveNet-like residual blocks inside coupling layers.
    """
    def __init__(self, hidden_channels, kernel_size, dilation_rate, num_layers, p_dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(num_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            )
            self.res_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, 1)
            )

    def forward(self, x, x_mask=None):
        output = torch.zeros_like(x)
        for i in range(self.num_layers):
            x_in = self.in_layers[i](x)
            y_1, y_2 = torch.split(x_in, self.hidden_channels, dim=1)
            y = torch.tanh(y_1) * torch.sigmoid(y_2)
            y = self.drop(y)
            
            res = self.res_layers[i](y)
            if x_mask is not None:
                res = res * x_mask
            
            x = (x + res) * 0.7071
            output = output + res
            
        return output

class ResidualCouplingLayer(nn.Module):
    """
    Normalizing Flow Coupling Layer.
    """
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, num_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, num_layers, p_dropout)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)
        
    def forward(self, x, x_mask=None, reverse=False):
        x0, x1 = torch.split(x, self.half_channels, dim=1)
        if not reverse:
            h = self.pre(x0) * x_mask if x_mask is not None else self.pre(x0)
            h = self.enc(h, x_mask)
            stats = self.post(h)
            x1 = (x1 + stats) * x_mask if x_mask is not None else x1 + stats
            return torch.cat([x0, x1], dim=1)
        else:
            h = self.pre(x0) * x_mask if x_mask is not None else self.pre(x0)
            h = self.enc(h, x_mask)
            stats = self.post(h)
            x1 = (x1 - stats) * x_mask if x_mask is not None else x1 - stats
            return torch.cat([x0, x1], dim=1)

class TextEncoder(nn.Module):
    """
    Simple Transformer/Convolution-based Text Encoder mapping phoneme IDs to hidden space.
    """
    def __init__(self, num_vocab, out_channels, hidden_channels, kernel_size, num_layers):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, hidden_channels)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(num_layers)
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x).transpose(1, 2) # [B, H, T]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        for conv in self.convs:
            x = F.relu(conv(x)) * x_mask
            
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, stats.size(1) // 2, dim=1)
        return m, logs, x_mask

class PosteriorEncoder(nn.Module):
    """
    Encodes linear spectrogram frames to the latent space distribution of VAE.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, num_layers):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, num_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask=None):
        x = self.pre(x) * x_mask if x_mask is not None else self.pre(x)
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask if x_mask is not None else self.proj(x)
        m, logs = torch.split(stats, stats.size(1) // 2, dim=1)
        return m, logs

class Generator(nn.Module):
    """
    HiFi-GAN vocoder generator which reconstructs raw waveforms from latents.
    """
    def __init__(self, initial_channels, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilations):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channels, 512, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(512 // (2**i), 512 // (2**(i+1)), k, u, padding=(k-u)//2)
            )

        self.conv_post = nn.Conv1d(512 // (2**self.num_upsamples), 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for up in self.ups:
            x = F.leaky_relu(up(x), 0.1)
        x = torch.tanh(self.conv_post(x))
        return x

class SynthesizerTrn(nn.Module):
    """
    VITS End-to-End Synthesizer model integrating TextEncoder, Flow, PosteriorEncoder, and Generator.
    """
    def __init__(self, num_vocab, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock_kernel_sizes,
                 resblock_dilations, upsample_rates, upsample_kernel_sizes, c_g=192):
        super().__init__()
        self.segment_size = segment_size
        
        self.enc_p = TextEncoder(num_vocab, inter_channels, hidden_channels, kernel_size, n_layers)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16)
        self.dec = Generator(inter_channels, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes, resblock_dilations)
        
        # Flows
        self.flow = ResidualCouplingLayer(inter_channels, hidden_channels, 5, 1, 4)
        
        # Calculate hop length dynamically from upsample rates
        self.hop_length = 1
        for r in upsample_rates:
            self.hop_length *= r

    def forward(self, x, x_lengths, y, y_lengths):
        # 1. Encode text
        m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        # 2. Encode linear spectrograms
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        m_q, logs_q = self.enc_q(y, y_mask)
        
        # 3. Sample from posterior distribution
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        z = z * y_mask
        
        # 4. Transform via Normalizing Flow
        z_p = self.flow(z, y_mask, reverse=False)
        
        # Align lengths or slice for generator (during training)
        z_sliced, slice_ids = rand_slice_segments(z, y_lengths, self.segment_size // self.hop_length)
        o = self.dec(z_sliced)
        
        return o, slice_ids, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, noise_scale=0.667):
        m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        # In inference, we map directly from TextEncoder mean to generator using Flows
        # Generate raw duration predictions or mapping.
        # For simplicity, we define dynamic alignment or use text length mapping:
        y_lengths = (x_lengths * 2.5).int() # Simple duration stretch factor for demo inference
        max_y_len = y_lengths.max().item()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, max_y_len), 1).to(x_mask.dtype)
        
        # Upsample mean to target frame length
        m_p_up = F.interpolate(m_p, size=(max_y_len,), mode='nearest')
        logs_p_up = F.interpolate(logs_p, size=(max_y_len,), mode='nearest')
        
        # Sample in latent space
        z_p = m_p_up + torch.randn_like(m_p_up) * torch.exp(logs_p_up) * noise_scale
        z_p = z_p * y_mask
        
        # Reverse flow to map from simple distribution to complex spectrogram space
        z = self.flow(z_p, y_mask, reverse=True)
        
        # Synthesize audio waveform
        o = self.dec(z)
        return o, y_mask

# Helper math functions
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def rand_slice_segments(x, ids_str, segment_size=4):
    b, c, t = x.size()
    slice_ids = []
    output = torch.zeros(b, c, segment_size, dtype=x.dtype, device=x.device)
    for i, length in enumerate(ids_str):
        max_start = max(0, length - segment_size)
        start = random_start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        slice_ids.append(start)
        output[i] = x[i, :, start:start+segment_size]
    return output, slice_ids
