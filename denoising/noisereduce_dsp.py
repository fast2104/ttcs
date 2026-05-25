#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
noisereduce_dsp.py
------------------
Uses DSP Spectral Gating (noisereduce library) to clean noise from audio clips.
Acts as a baseline algorithm to contrast traditional DSP vs Deep Learning approaches.
"""

import os
import argparse
import glob
import numpy as np
import soundfile as sf
import noisereduce as nr
from tqdm import tqdm

def denoise_noisereduce(input_dir, output_dir, stationary=True):
    """
    Cleans audio noise in batch using the noisereduce library.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"[-] No WAV files found in the directory: {input_dir}")
        return
        
    print(f"[*] Starting DSP noisereduce (Spectral Gating) for {len(wav_files)} files...")
    print(f"[*] Mode: {'Stationary Noise (Global Profile)' if stationary else 'Non-stationary Noise (Dynamic)'}")
    print(f"[*] Output directory: {os.path.abspath(output_dir)}")
    
    for wav_path in tqdm(wav_files, desc="DSP Denoising"):
        try:
            # Read audio data
            data, sr = sf.read(wav_path)
            
            # Apply noisereduce
            # stationary=True assumes the noise is uniform across the file,
            # which is suitable for fan noises or constant aircon hums.
            reduced_noise = nr.reduce_noise(y=data, sr=sr, stationary=stationary)
            
            # Save audio
            out_path = os.path.join(output_dir, os.path.basename(wav_path))
            sf.write(out_path, reduced_noise, sr)
        except Exception as e:
            print(f"[!] Error processing {os.path.basename(wav_path)}: {e}")
            
    print("[+] DSP noisereduce complete.")

def main():
    parser = argparse.ArgumentParser(description="Batch denoise audio using DSP Spectral Gating.")
    parser.add_argument("--input_dir", type=str, default="processed_data/wavs_need_denoise", help="Input directory of noisy audio files.")
    parser.add_argument("--output_dir", type=str, default="processed_data/wavs_noisereduce", help="Output directory to save cleaned audio files.")
    parser.add_argument("--non_stationary", action="store_true", help="Enable non-stationary mode for dynamic noise profile reduction.")
    
    args = parser.parse_args()
    
    denoise_noisereduce(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stationary=not args.non_stationary
    )

if __name__ == "__main__":
    main()
