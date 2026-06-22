#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_4_filters.py
--------------------
Runs all 4 denoising models on a sample noisy file, outputs a combined comparison
spectrogram plot and saves the cleaned wav files for listening.
"""

import os
import glob
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Import wrappers
from denoising.deepfilter import denoise_deepfilter
from denoising.noisereduce_dsp import denoise_noisereduce
from denoising.rnnoise_wrapper import denoise_rnnoise
from denoising.spleeter_wrapper import separate_vocals_spleeter

def plot_all_spectrograms(audio_paths, output_img):
    """
    Plots a grid of spectrograms comparing the original and all 4 denoised files.
    """
    fig, axes = plt.subplots(len(audio_paths), 1, figsize=(12, 3.5 * len(audio_paths)), sharex=True)
    
    for idx, (name, path) in enumerate(audio_paths.items()):
        ax = axes[idx]
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"File not found: {os.path.basename(path)}", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{name} (Failed/Not Run)", fontsize=12, fontweight='bold')
            continue
            
        # Load audio
        y, sr = librosa.load(path, sr=None)
        # Compute Spectrogram
        stft = librosa.stft(y)
        db = librosa.amplitude_to_db(abs(stft), ref=librosa.max)
        
        # Plot Spec
        img = librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='inferno')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Freq (Hz)")
        if idx == len(audio_paths) - 1:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xlabel("")
            
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Spectrogram comparison plot saved to: {output_img}")

def main():
    # 1. Locate a sample noisy file
    sample_file = None
    search_dirs = [
        "processed_data/wavs_need_denoise",
        "merged_dataset/wavs",
        "wavs",
        "."
    ]
    
    for folder in search_dirs:
        wavs = glob.glob(os.path.join(folder, "*.wav"))
        if wavs:
            # Avoid picking already processed files if in root
            valid_wavs = [w for w in wavs if "comparison" not in w and "synthesized" not in w]
            if valid_wavs:
                sample_file = valid_wavs[0]
                break
                
    if not sample_file:
        print("[-] Error: No sample WAV file found to run comparison.")
        # Create a dummy noisy wave for testing
        print("[*] Creating a dummy noisy wave for testing...")
        sample_file = "noisy_sample_test.wav"
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440Hz sine wave + white noise
        speech_signal = np.sin(2 * np.pi * 440 * t) * 0.5
        noise = np.random.normal(0, 0.1, len(t))
        sf.write(sample_file, speech_signal + noise, sr)
        
    print(f"[+] Using sample file: {sample_file}")
    
    # 2. Setup folders
    comp_dir = "outputs/comparisons"
    os.makedirs(comp_dir, exist_ok=True)
    
    # Copy original file to output directory
    orig_copy = os.path.join(comp_dir, "0_original.wav")
    shutil.copy(sample_file, orig_copy)
    
    # We will use temporary directories because our wrappers process directories
    temp_in = "temp_compare_in"
    os.makedirs(temp_in, exist_ok=True)
    shutil.copy(sample_file, os.path.join(temp_in, os.path.basename(sample_file)))
    
    # Paths for cleaned output files
    output_files = {
        "1. noisereduce (DSP)": os.path.join(comp_dir, "1_noisereduce.wav"),
        "2. DeepFilterNet (AI)": os.path.join(comp_dir, "2_deepfilter.wav"),
        "3. RNNoise (RNN)": os.path.join(comp_dir, "3_rnnoise.wav"),
        "4. Spleeter (Separation)": os.path.join(comp_dir, "4_spleeter.wav")
    }
    
    # 3. Run each model
    # noisereduce
    print("\n--- Running noisereduce ---")
    temp_nr = "temp_nr_out"
    denoise_noisereduce(temp_in, temp_nr)
    nr_res = glob.glob(os.path.join(temp_nr, "*.wav"))
    if nr_res:
        shutil.copy(nr_res[0], output_files["1. noisereduce (DSP)"])
    shutil.rmtree(temp_nr, ignore_errors=True)
    
    # DeepFilterNet
    print("\n--- Running DeepFilterNet ---")
    temp_df = "temp_df_out"
    try:
        denoise_deepfilter(temp_in, temp_df)
        df_res = glob.glob(os.path.join(temp_df, "*.wav"))
        if df_res:
            shutil.copy(df_res[0], output_files["2. DeepFilterNet (AI)"])
    except Exception as e:
        print(f"[-] DeepFilterNet failed: {e}")
    shutil.rmtree(temp_df, ignore_errors=True)
    
    # RNNoise
    print("\n--- Running RNNoise ---")
    temp_rn = "temp_rn_out"
    denoise_rnnoise(temp_in, temp_rn)
    rn_res = glob.glob(os.path.join(temp_rn, "*.wav"))
    if rn_res:
        shutil.copy(rn_res[0], output_files["3. RNNoise (RNN)"])
    shutil.rmtree(temp_rn, ignore_errors=True)
    
    # Spleeter
    print("\n--- Running Spleeter ---")
    temp_sp = "temp_sp_out"
    separate_vocals_spleeter(temp_in, temp_sp)
    sp_res = glob.glob(os.path.join(temp_sp, "*.wav"))
    if sp_res:
        shutil.copy(sp_res[0], output_files["4. Spleeter (Separation)"])
    shutil.rmtree(temp_sp, ignore_errors=True)
    
    # Clean up input temp dir
    shutil.rmtree(temp_in, ignore_errors=True)
    
    # 4. Generate Spectrogram comparison plot
    all_paths = {"0. Original Noisy": orig_copy}
    all_paths.update(output_files)
    
    print("\n--- Plotting Spectrograms ---")
    plot_all_spectrograms(all_paths, os.path.join(comp_dir, "spectrogram_comparison_all.png"))
    
    print("\n" + "="*50)
    print("           COMPARISON SUITE COMPLETE")
    print("="*50)
    print(f"Outputs are stored in: {comp_dir}")
    for name, path in all_paths.items():
        if os.path.exists(path):
            print(f" - {name}: {path} ({os.path.getsize(path)/1024:.1f} KB)")
            
    # Python code output hint for Colab Audio playback
    print("\nCopy the following Python code to a Colab cell to play them side-by-side:")
    print("""
import os
from IPython.display import Audio, display, Image

# Show spectrogram
display(Image("outputs/comparisons/spectrogram_comparison_all.png"))

files = {
    "0. Original": "outputs/comparisons/0_original.wav",
    "1. noisereduce": "outputs/comparisons/1_noisereduce.wav",
    "2. DeepFilterNet": "outputs/comparisons/2_deepfilter.wav",
    "3. RNNoise": "outputs/comparisons/3_rnnoise.wav",
    "4. Spleeter": "outputs/comparisons/4_spleeter.wav"
}

for name, path in files.items():
    if os.path.exists(path):
        print(f"\\n🔊 Play: {name}")
        display(Audio(path))
    """)

if __name__ == "__main__":
    main()
