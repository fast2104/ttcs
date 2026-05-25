#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_denoise.py
--------------------
Measures computational performance (Real-Time Factor, peak memory usage) of the 4
denoising algorithms: DeepFilterNet, noisereduce (DSP), RNNoise, and Spleeter.
Outputs a comparative Markdown table.
"""

import os
import time
import argparse
import glob
import tracemalloc
import soundfile as sf
import pandas as pd

# Import our denoising runners
# We load them wrapper-style so this benchmarking script remains executable
from denoising.deepfilter import denoise_deepfilter
from denoising.noisereduce_dsp import denoise_noisereduce
from denoising.rnnoise_wrapper import denoise_rnnoise
from denoising.spleeter_wrapper import separate_vocals_spleeter

def get_audio_duration(file_list):
    """
    Computes total duration in seconds of all WAV files in the list.
    """
    total_duration = 0.0
    for path in file_list:
        try:
            info = sf.info(path)
            total_duration += info.duration
        except Exception:
            continue
    return total_duration

def run_benchmark(input_dir, output_root):
    """
    Runs each denoiser on the input_dir, measuring execution time and peak memory.
    """
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"[-] Error: No WAV files found in input directory '{input_dir}' for benchmarking.")
        return
        
    total_dur = get_audio_duration(wav_files)
    print(f"[*] Starting benchmark suite on {len(wav_files)} files.")
    print(f"[*] Total audio duration: {total_dur:.2f} seconds\n")
    
    results = []
    
    # Define models to run
    models = [
        {
            "name": "noisereduce (DSP)",
            "func": lambda inp, out: denoise_noisereduce(inp, out, stationary=True),
            "out_subdir": "noisereduce"
        },
        {
            "name": "DeepFilterNet (AI)",
            "func": denoise_deepfilter,
            "out_subdir": "deepfilter"
        },
        {
            "name": "RNNoise (RNN)",
            "func": denoise_rnnoise,
            "out_subdir": "rnnoise"
        },
        {
            "name": "Spleeter (AI Separation)",
            "func": separate_vocals_spleeter,
            "out_subdir": "spleeter"
        }
    ]
    
    for model in models:
        name = model["name"]
        func = model["func"]
        out_dir = os.path.join(output_root, model["out_subdir"])
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"[*] Benchmarking model: {name}...")
        
        # Start tracking memory and time
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            # Execute denoiser
            func(input_dir, out_dir)
            
            end_time = time.perf_counter()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            elapsed_time = end_time - start_time
            # RTF = processing_time / total_audio_duration
            rtf = elapsed_time / total_dur if total_dur > 0 else 0.0
            peak_mem_mb = peak_mem / (1024 * 1024) # Convert to MB
            
            results.append({
                "Algorithm": name,
                "Time (s)": f"{elapsed_time:.2f}",
                "RTF": f"{rtf:.4f}",
                "Peak Memory (MB)": f"{peak_mem_mb:.2f}",
                "Status": "Success"
            })
            
        except Exception as e:
            tracemalloc.stop()
            print(f"[-] Error running {name}: {e}")
            results.append({
                "Algorithm": name,
                "Time (s)": "N/A",
                "RTF": "N/A",
                "Peak Memory (MB)": "N/A",
                "Status": f"Failed ({type(e).__name__})"
            })
            
    # Output markdown report table
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("           DENOISING BENCHMARK RESULTS")
    print("="*50)
    print(df.to_markdown(index=False))
    print("="*50)
    
    # Save results to csv
    df.to_csv(os.path.join(output_root, "benchmark_report.csv"), index=False)
    print(f"[+] Benchmark report saved to: {os.path.join(output_root, 'benchmark_report.csv')}")

def main():
    parser = argparse.ArgumentParser(description="Run speed and memory benchmarks on speech enhancement models.")
    parser.add_argument("--input_dir", type=str, default="processed_data/wavs_need_denoise", help="Folder containing noisy test WAVs.")
    parser.add_argument("--output_root", type=str, default="processed_data/benchmarks", help="Folder to save clean outputs and reports.")
    
    args = parser.parse_args()
    
    run_benchmark(args.input_dir, args.output_root)

if __name__ == "__main__":
    main()
