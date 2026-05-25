#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spectrogram_plotter.py
----------------------
Extracts and draws side-by-side Mel-spectrogram comparisons (Before vs After)
using librosa and matplotlib. Outputs high-resolution plots for report insertion.
"""

import os
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_before_after_spectrograms(before_path, after_path, output_img="outputs/spectrogram_comparison.png"):
    """
    Loads raw and cleaned audio, computes Mel-Spectrograms, and plots them side-by-side.
    """
    if not os.path.exists(before_path):
        print(f"[-] Error: Before file not found: {before_path}")
        return
    if not os.path.exists(after_path):
        print(f"[-] Error: After file not found: {after_path}")
        return
        
    print(f"[*] Loading audio files:")
    print(f"    - Before: {before_path}")
    print(f"    - After:  {after_path}")
    
    # Load audio clips
    y_before, sr_before = librosa.load(before_path, sr=None)
    y_after, sr_after = librosa.load(after_path, sr=None)
    
    # Compute short-time Fourier transforms (STFT)
    stft_before = librosa.stft(y_before)
    stft_after = librosa.stft(y_after)
    
    # Convert amplitude to Decibel scale
    db_before = librosa.amplitude_to_db(abs(stft_before), ref=librosa.max)
    db_after = librosa.amplitude_to_db(abs(stft_after), ref=librosa.max)
    
    # Create Side-by-Side figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    
    # Plot Before Spectrogram
    img1 = librosa.display.specshow(
        db_before, sr=sr_before, x_axis='time', y_axis='hz', ax=ax1, cmap='inferno'
    )
    ax1.set_title("Phổ âm thanh gốc (Có tạp âm)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Thời gian (giây)", fontsize=10)
    ax1.set_ylabel("Tần số (Hz)", fontsize=10)
    fig.colorbar(img1, ax=ax1, format="%+2.0f dB")
    
    # Plot After Spectrogram
    img2 = librosa.display.specshow(
        db_after, sr=sr_after, x_axis='time', y_axis='hz', ax=ax2, cmap='inferno'
    )
    ax2.set_title("Phổ âm thanh sau khi làm sạch (DeepFilterNet)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Thời gian (giây)", fontsize=10)
    fig.colorbar(img2, ax=ax2, format="%+2.0f dB")
    
    # Tight layout adjustments
    plt.tight_layout()
    
    # Save chart image
    os.makedirs(os.path.dirname(output_img) or '.', exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Side-by-side spectrogram comparison saved to: {output_img}")

def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side spectrograms comparing noisy and cleaned audio files.")
    parser.add_argument("--before", type=str, required=True, help="Path to raw/noisy WAV audio file.")
    parser.add_argument("--after", type=str, required=True, help="Path to enhanced/cleaned WAV audio file.")
    parser.add_argument("--output", type=str, default="outputs/spectrogram_comparison.png", help="Path to save output comparison image.")
    
    args = parser.parse_args()
    
    plot_before_after_spectrograms(
        before_path=args.before,
        after_path=args.after,
        output_img=args.output
    )

if __name__ == "__main__":
    main()
