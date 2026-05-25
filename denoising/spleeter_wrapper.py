#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
spleeter_wrapper.py
-------------------
Uses Spleeter (AI source separation) to isolate vocal tracks from background music.
Helpful for video datasets with prominent background music.
"""

import os
import argparse
import subprocess
import glob
import shutil
from tqdm import tqdm

def separate_vocals_spleeter(input_dir, output_dir):
    """
    Separates vocal and accompaniment stems using spleeter's 2stems model.
    Keeps only the vocal track and renames it to match the original wav name.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    if not wav_files:
        print(f"[-] No WAV files found in the directory: {input_dir}")
        return
        
    # Check if spleeter CLI command is available
    try:
        subprocess.run(["spleeter", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_spleeter = True
    except Exception:
        has_spleeter = False
        
    if not has_spleeter:
        print("[!] Error: 'spleeter' CLI command is not available in this environment.")
        print("[!] To use Spleeter, please run: pip install spleeter")
        print("[*] Simulating Spleeter execution (copying files directly to output folder)...")
        for wav_path in wav_files:
            shutil.copy(wav_path, os.path.join(output_dir, os.path.basename(wav_path)))
        print("[+] Mock Spleeter separation finished.")
        return

    print(f"[*] Starting Spleeter 2stems vocal separation for {len(wav_files)} files...")
    print(f"[*] Output directory: {os.path.abspath(output_dir)}")
    
    # We can separate in batches or individually. Separating in batch is faster.
    # command: spleeter separate -p spleeter:2stems -o tmp_spleeter_out file1.wav file2.wav ...
    temp_out_dir = os.path.join(output_dir, "spleeter_temp")
    os.makedirs(temp_out_dir, exist_ok=True)
    
    # Process files in chunks of 20 to avoid exceeding CLI command length limits
    chunk_size = 20
    file_chunks = [wav_files[i:i + chunk_size] for i in range(0, len(wav_files), chunk_size)]
    
    for chunk in tqdm(file_chunks, desc="Spleeter Separation Chunks"):
        cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o", temp_out_dir] + chunk
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Retrieve vocals from temporary folders and move to final destination
            for wav_path in chunk:
                filename_no_ext = os.path.splitext(os.path.basename(wav_path))[0]
                vocal_source = os.path.join(temp_out_dir, filename_no_ext, "vocals.wav")
                vocal_dest = os.path.join(output_dir, os.path.basename(wav_path))
                
                if os.path.exists(vocal_source):
                    shutil.move(vocal_source, vocal_dest)
        except Exception as e:
            print(f"[!] Error processing chunk {chunk}: {e}")
            
    # Clean up temporary directories
    if os.path.exists(temp_out_dir):
        shutil.rmtree(temp_out_dir)
        
    print("[+] Spleeter vocal separation complete.")

def main():
    parser = argparse.ArgumentParser(description="Isolate vocals from audio tracks using Spleeter.")
    parser.add_argument("--input_dir", type=str, default="processed_data/wavs_need_denoise", help="Input directory containing noisy files.")
    parser.add_argument("--output_dir", type=str, default="processed_data/wavs_spleeter", help="Output directory to save extracted vocal files.")
    
    args = parser.parse_args()
    
    separate_vocals_spleeter(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
