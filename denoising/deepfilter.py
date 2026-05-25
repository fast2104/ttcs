#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
deepfilter.py
-------------
Uses DeepFilterNet to denoise and clean wav audio segments in batch mode.
Optimized to run on Google Colab or locally.
"""

import os
import argparse
import subprocess
import glob
from tqdm import tqdm

def denoise_deepfilter(input_dir, output_dir):
    """
    Denoises all WAV files in the input_dir using DeepFilterNet.
    Attempts importing the library first, falling back to the df-enhance CLI command.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"[-] No WAV files found in the directory: {input_dir}")
        return
        
    print(f"[*] Starting DeepFilterNet denoising for {len(wav_files)} files...")
    print(f"[*] Output directory: {os.path.abspath(output_dir)}")
    
    # Try importing DeepFilterNet programmatic API
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        # Initialize model
        print("[*] Initializing DeepFilterNet deep learning model in-memory...")
        model, df_state, _ = init_df()
        
        for wav_path in tqdm(wav_files, desc="Denoising (Python API)"):
            try:
                # Load audio
                audio, meta = load_audio(wav_path, sr=df_state.sr())
                # Enhance
                enhanced_audio = enhance(model, df_state, audio)
                # Output filename
                out_name = os.path.basename(wav_path)
                out_path = os.path.join(output_dir, out_name)
                # Save audio
                save_audio(out_path, enhanced_audio, df_state.sr())
            except Exception as e:
                print(f"[!] Error processing {os.path.basename(wav_path)}: {e}")
        print("[+] DeepFilterNet denoising complete using Python API.")
        
    except ImportError:
        # Fallback to CLI command "df-enhance" via subprocess
        print("[!] DeepFilterNet python API not found in this environment. Falling back to CLI wrapper...")
        
        # Verify df-enhance is installed on path
        try:
            subprocess.run(["df-enhance", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            has_cli = True
        except Exception:
            has_cli = False
            
        if not has_cli:
            print("[-] Error: 'df-enhance' CLI command not found. Please install with 'pip install deepfilternet'.", file=subprocess.stderr)
            return
            
        for wav_path in tqdm(wav_files, desc="Denoising (CLI)"):
            out_path = os.path.join(output_dir, os.path.basename(wav_path))
            cmd = ["df-enhance", "--output-dir", output_dir, wav_path]
            try:
                # Run df-enhance on the specific file
                # It exports as filename_clean.wav, so we rename it to preserve names
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Rename output file back to original file name
                base_clean = os.path.splitext(os.path.basename(wav_path))[0] + "_clean.wav"
                temp_clean_path = os.path.join(output_dir, base_clean)
                if os.path.exists(temp_clean_path):
                    shutil_dest = os.path.join(output_dir, os.path.basename(wav_path))
                    # Overwrite if exists
                    if os.path.exists(shutil_dest):
                        os.remove(shutil_dest)
                    os.rename(temp_clean_path, shutil_dest)
            except Exception as e:
                print(f"[!] Error processing {os.path.basename(wav_path)} with CLI: {e}")
        print("[+] DeepFilterNet denoising complete using CLI wrapper.")

def main():
    parser = argparse.ArgumentParser(description="Batch denoise audio clips using DeepFilterNet.")
    parser.add_argument("--input_dir", type=str, default="processed_data/wavs_need_denoise", help="Input directory of noisy audio files.")
    parser.add_argument("--output_dir", type=str, default="processed_data/wavs_denoised", help="Output directory to save cleaned audio files.")
    
    args = parser.parse_args()
    
    denoise_deepfilter(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
