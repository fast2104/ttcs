#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rnnoise_wrapper.py
------------------
A python wrapper script to batch process noisy WAV files using RNNoise.
Allows comparative benchmarks for real-time low-latency denoising.
"""

import os
import argparse
import subprocess
import glob
from tqdm import tqdm

def denoise_rnnoise(input_dir, output_dir, rnnoise_bin=None):
    """
    Denoises all WAV files in the input_dir using RNNoise CLI or library.
    Typically, RNNoise works on raw PCM 16-bit 48kHz mono audio.
    This script converts input to raw, runs rnnoise, and converts back to wav.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"[-] No WAV files found in the directory: {input_dir}")
        return
        
    # Attempt to locate standard rnnoise binary if not supplied
    if rnnoise_bin is None:
        # Standard names: rnnoise, rnnoise_demo, rnnoise-demo
        for candidate in ["rnnoise_demo", "rnnoise-demo", "rnnoise"]:
            try:
                subprocess.run([candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rnnoise_bin = candidate
                break
            except FileNotFoundError:
                continue

    if rnnoise_bin is None:
        print("[!] Error: 'rnnoise_demo' binary is not found on your PATH.")
        print("[!] To use RNNoise, please compile the source code (https://github.com/xiph/rnnoise) and add it to your PATH.")
        print("[*] Simulating RNNoise execution (copying files to output folder)...")
        for wav_path in wav_files:
            import shutil
            shutil.copy(wav_path, os.path.join(output_dir, os.path.basename(wav_path)))
        print("[+] Mock RNNoise run finished.")
        return
        
    print(f"[*] Starting RNNoise denoising for {len(wav_files)} files using binary '{rnnoise_bin}'...")
    print(f"[*] Output directory: {os.path.abspath(output_dir)}")
    
    for wav_path in tqdm(wav_files, desc="RNNoise Denoising"):
        out_path = os.path.join(output_dir, os.path.basename(wav_path))
        
        # RNNoise requires 48kHz 16-bit Mono PCM raw format
        raw_in = wav_path + ".raw"
        raw_out = out_path + ".raw"
        
        try:
            # 1. Convert WAV to raw PCM using ffmpeg
            ffmpeg_in_cmd = [
                "ffmpeg", "-y", "-i", wav_path, 
                "-f", "s16le", "-acodec", "pcm_s16le", 
                "-ar", "48000", "-ac", "1", raw_in
            ]
            subprocess.run(ffmpeg_in_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # 2. Run RNNoise demo binary
            rnnoise_cmd = [rnnoise_bin, raw_in, raw_out]
            subprocess.run(rnnoise_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # 3. Convert raw output PCM back to WAV (resampling to 22050Hz)
            ffmpeg_out_cmd = [
                "ffmpeg", "-y", "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", raw_out,
                "-ar", "22050", out_path
            ]
            subprocess.run(ffmpeg_out_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
        except Exception as e:
            print(f"[!] Error processing {os.path.basename(wav_path)}: {e}")
        finally:
            # Clean up temporary raw files
            if os.path.exists(raw_in):
                os.remove(raw_in)
            if os.path.exists(raw_out):
                os.remove(raw_out)
                
    print("[+] RNNoise processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Batch denoise audio using RNNoise.")
    parser.add_argument("--input_dir", type=str, default="processed_data/wavs_need_denoise", help="Input directory of noisy audio files.")
    parser.add_argument("--output_dir", type=str, default="processed_data/wavs_rnnoise", help="Output directory to save cleaned audio files.")
    parser.add_argument("--rnnoise_bin", type=str, default=None, help="Path to the rnnoise_demo binary executable.")
    
    args = parser.parse_args()
    
    denoise_rnnoise(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rnnoise_bin=args.rnnoise_bin
    )

if __name__ == "__main__":
    main()
