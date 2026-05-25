#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference_fs2.py
----------------
Inference pipeline for the FastSpeech 2 + HiFi-GAN vocoder baseline.
Takes normalized text, predicts the mel spectrogram, and synthesizes speech.
"""

import os
import json
import argparse
import torch
from torch import nn
import numpy as np
import soundfile as sf

from models.fastspeech2.train_fs2 import FastSpeech2Baseline
from models.vits.dataset import text_to_sequence
from data_pipeline.normalizer import normalize_text

class HiFiGANVocoder(nn.Module):
    """
    Mock HiFi-GAN vocoder mapping mel spectrograms back to raw waveforms.
    """
    def __init__(self, in_channels=80):
        super().__init__()
        # Simple transposed convolution layer mapping 80-channel mel to raw wave (upsampling ratio ~256)
        self.generator = nn.ConvTranspose1d(in_channels, 1, kernel_size=512, stride=256, padding=128)

    def forward(self, mel):
        # mel: [B, mel_channels, T_mel]
        # output: [B, 1, T_wave]
        return torch.tanh(self.generator(mel))

def synthesize_fastspeech2(text, fs2_checkpoint, output_wav, config_path):
    """
    Synthesizes speech using FastSpeech 2 (Acoustic model) + HiFi-GAN (Vocoder).
    """
    # 1. Normalize text
    norm_text = normalize_text(text)
    print(f"[*] Original text:  \"{text}\"")
    print(f"[*] Normalized text: \"{norm_text}\"")
    
    seq = text_to_sequence(norm_text)
    if not seq:
        print("[-] Error: Input text contains no valid characters.")
        return
        
    x = torch.LongTensor(seq).unsqueeze(0) # Batch size 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference running on: {device}")
    
    # 2. Instantiate and load FastSpeech 2 acoustic model
    acoustic_model = FastSpeech2Baseline(num_vocab=150).to(device)
    if os.path.exists(fs2_checkpoint):
        try:
            acoustic_model.load_state_dict(torch.load(fs2_checkpoint, map_location=device))
            print(f"[+] Loaded FastSpeech 2 weights from: {fs2_checkpoint}")
        except Exception as e:
            print(f"[!] Warning: Error loading weights: {e}. Using randomly initialized model.")
    else:
        print("[!] Warning: FastSpeech 2 checkpoint not found. Running with random initialization.")
        
    # 3. Instantiate vocoder
    vocoder = HiFiGANVocoder().to(device)
    
    acoustic_model.eval()
    vocoder.eval()
    
    x = x.to(device)
    
    with torch.no_grad():
        # Predict mel spectrogram
        # returns mel_out, pred_dur, pred_pitch, pred_energy
        mel_out, _, _, _ = acoustic_model(x)
        
        # Reshape to [B, mel_channels, T_mel] for HiFi-GAN vocoder input
        mel_transposed = mel_out.transpose(1, 2)
        
        # Convert mel to raw waveform
        wave_out = vocoder(mel_transposed)
        
        audio = wave_out.squeeze().cpu().numpy()
        # Clamp amplitude
        audio = np.clip(audio, -1.0, 1.0)
        
    # 4. Save to WAV
    sample_rate = 22050
    os.makedirs(os.path.dirname(output_wav) or '.', exist_ok=True)
    sf.write(output_wav, audio, sample_rate)
    print(f"[+] Audio successfully synthesized and saved: {output_wav}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech using FastSpeech 2 + HiFi-GAN.")
    parser.add_argument("--text", type=str, required=True, help="Input text sentence to speak.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fastspeech2_latest.pth", help="Path to FastSpeech 2 acoustic checkpoint.")
    parser.add_argument("--output", type=str, default="outputs/synthesized_fs2.wav", help="Path to output WAV file.")
    parser.add_argument("--config", type=str, default="configs/fs2_config.json", help="Path to config file.")
    
    args = parser.parse_args()
    
    synthesize_fastspeech2(
        text=args.text,
        fs2_checkpoint=args.checkpoint,
        output_wav=args.output,
        config_path=args.config
    )

if __name__ == "__main__":
    main()
