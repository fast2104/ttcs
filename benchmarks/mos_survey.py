#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mos_survey.py
-------------
A script to host a local terminal-based Mean Opinion Score (MOS) survey.
Plays audio clips (synthesized samples) to listeners, collects ratings (1 to 5)
for naturalness, intelligibility, and audio quality, and saves results to a CSV.
"""

import os
import argparse
import glob
import random
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play

def play_audio(filepath):
    """
    Plays WAV audio file.
    """
    try:
        sound = AudioSegment.from_wav(filepath)
        print(f"[*] Playing audio clip: {os.path.basename(filepath)}...")
        play(sound)
    except Exception as e:
        print(f"[!] Error playing audio: {e}. Make sure simpleaudio/pyaudio is installed, or play manually.")

def run_mos_survey(audio_folder, output_csv="outputs/mos_results.csv"):
    """
    Hosts the MOS scoring session.
    """
    wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    if not wav_files:
        print(f"[-] No wav files found in target folder: {audio_folder}")
        return
        
    print("="*60)
    print("      TTS MEAN OPINION SCORE (MOS) EVALUATION SURVEY")
    print("="*60)
    print("Instruction for Evaluator:")
    print("For each sample, listen carefully, then rate from 1 to 5:")
    print("  1: Very Poor (Robot, heavily distorted, unintelligible)")
    print("  2: Poor (Reedy, metallic, hard to understand)")
    print("  3: Fair (Understandable, minor distortion/robotic tone)")
    print("  4: Good (Natural pronunciation, clean background)")
    print("  5: Excellent (Human-like naturalness, studio quality)")
    print("="*60)
    
    listener_name = input("Enter Listener / Evaluator Name: ").strip()
    if not listener_name:
        listener_name = "Anonymous"
        
    # Shuffle files to implement blind test (listener does not know filenames/models)
    random.shuffle(wav_files)
    
    results = []
    
    for idx, filepath in enumerate(wav_files, 1):
        print(f"\n[{idx}/{len(wav_files)}] Sample ID: {idx:03d} (Blind Evaluation)")
        
        # Play the audio
        play_audio(filepath)
        
        # Get rating inputs
        while True:
            try:
                nat = int(input("Rate Naturalness (1-5): "))
                if 1 <= nat <= 5: break
                print("[!] Score must be an integer between 1 and 5.")
            except ValueError:
                print("[!] Invalid input. Enter integer between 1 and 5.")
                
        while True:
            try:
                intel = int(input("Rate Intelligibility/Clarity (1-5): "))
                if 1 <= intel <= 5: break
                print("[!] Score must be an integer between 1 and 5.")
            except ValueError:
                print("[!] Invalid input. Enter integer between 1 and 5.")
                
        while True:
            try:
                qual = int(input("Rate Audio Quality/Denoising (1-5): "))
                if 1 <= qual <= 5: break
                print("[!] Score must be an integer between 1 and 5.")
            except ValueError:
                print("[!] Invalid input. Enter integer between 1 and 5.")
                
        results.append({
            "evaluator": listener_name,
            "filename": os.path.basename(filepath),
            "naturalness": nat,
            "intelligibility": intel,
            "quality": qual
        })
        
    # Save results
    df_new = pd.DataFrame(results)
    header = not os.path.exists(output_csv)
    
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_new.to_csv(output_csv, mode='a', index=False, header=header, encoding='utf-8')
    print(f"\n[+] Score logging complete! Results appended to: {output_csv}")
    
    # Show running average
    df_all = pd.read_csv(output_csv)
    print("\n" + "-"*40)
    print("      CURRENT MOS STATISTICS SUMMARY")
    print("-"*40)
    print(f"Total Evaluations logged: {len(df_all)}")
    print(f"Mean Naturalness Score:   {df_all['naturalness'].mean():.2f} / 5.0")
    print(f"Mean Intelligibility:     {df_all['intelligibility'].mean():.2f} / 5.0")
    print(f"Mean Audio Quality:       {df_all['quality'].mean():.2f} / 5.0")
    print("-"*40)

def main():
    parser = argparse.ArgumentParser(description="Run terminal-based interactive MOS evaluation survey.")
    parser.add_argument("--audio_dir", type=str, default="outputs/eval_samples", help="Directory containing synthesized test WAV files.")
    parser.add_argument("--output", type=str, default="outputs/mos_results.csv", help="CSV file path to record rating records.")
    
    args = parser.parse_args()
    
    run_mos_survey(args.audio_dir, args.output)

if __name__ == "__main__":
    main()
