#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_drive_data.py
-------------------
A specialized tool to merge all Google Drive / OneDrive metadata CSV sheets
(verified, noise, recovered, final, etc.) and copy all corresponding WAV audio
files from various folders (wavs, wavs_need_denoise) while explicitly excluding
trash folders.

It also verifies file existence, updates path references, and generates
train/val splits for TTS training.
"""

import os
import glob
import shutil
import argparse
import random
import pandas as pd
from tqdm import tqdm

def merge_drive_dataset(input_dir, output_dir, split_ratio=0.95, seed=42):
    """
    1. Merges all metadata CSVs (verified, noise, recovered, metadata.csv).
    2. Gathers and copies WAV files from all folders under input_dir, excluding folders with 'trash'.
    3. Aligns metadata with the merged wavs, removing references to missing or trash files.
    4. Splits into train.txt and val.txt for TTS training.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(out_wav_dir, exist_ok=True)
    
    print("="*60)
    print("      TTS DATASET MERGER & ALIGNMENT TOOL")
    print("="*60)
    print(f"[*] Input Directory:  {os.path.abspath(input_dir)}")
    print(f"[*] Output Directory: {os.path.abspath(output_dir)}")
    
    # ----------------------------------------------------
    # Step 1: Find and Merge all CSV files
    # ----------------------------------------------------
    csv_pattern = os.path.join(input_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    if not csv_files:
        # Check non-recursively
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
    if not csv_files:
        print("[-] Error: No CSV files found in the input directory.")
        return False
        
    print(f"\n[*] Found {len(csv_files)} metadata CSV files to merge:")
    for f in csv_files:
        print(f"    - {os.path.relpath(f, input_dir)}")
        
    dfs = []
    for f_path in csv_files:
        try:
            df_temp = pd.read_csv(f_path, encoding='utf-8-sig')
            
            # Find audio path and text columns (case-insensitive search)
            audio_col = None
            text_col = None
            for col in df_temp.columns:
                col_lower = col.lower()
                if 'audio' in col_lower or 'path' in col_lower or 'file' in col_lower:
                    audio_col = col
                elif 'text' in col_lower or 'transcript' in col_lower or 'content' in col_lower:
                    text_col = col
                    
            # Fallback to defaults
            if audio_col is None:
                audio_col = df_temp.columns[0]
            if text_col is None:
                text_col = df_temp.columns[1] if len(df_temp.columns) > 1 else df_temp.columns[0]
                
            df_filtered = df_temp[[audio_col, text_col]].copy()
            df_filtered.columns = ['audio_path', 'text']
            
            # Record source CSV for debugging/transparency
            df_filtered['source_file'] = os.path.basename(f_path)
            dfs.append(df_filtered)
        except Exception as e:
            print(f"[!] Warning: Could not parse CSV {f_path}: {e}")
            
    if not dfs:
        print("[-] Error: No valid data frames loaded.")
        return False
        
    merged_df = pd.concat(dfs, ignore_index=True)
    initial_len = len(merged_df)
    print(f"[+] Loaded {initial_len} total entries from CSVs.")
    
    # ----------------------------------------------------
    # Step 2: Merge audio folders (Exclude trash)
    # ----------------------------------------------------
    print("\n[*] Scanning and merging audio folders...")
    # Find all wav files in all subfolders
    all_wavs_pattern = os.path.join(input_dir, "**", "*.wav")
    all_wav_files = glob.glob(all_wavs_pattern, recursive=True)
    
    copied_count = 0
    copied_basenames = {} # basename -> path
    
    # Track files in trash folders to explicitly exclude
    trash_files = set()
    
    for wav_path in tqdm(all_wav_files, desc="Processing Audio Files"):
        # Check if file resides in a trash folder
        parts = os.path.normpath(wav_path).split(os.sep)
        is_trash = any('trash' in part.lower() for part in parts)
        
        basename = os.path.basename(wav_path)
        
        if is_trash:
            trash_files.add(basename)
            continue
            
        # Copy to output wav folder
        dest_path = os.path.join(out_wav_dir, basename)
        
        # In case of duplicate filenames across different directories (e.g. wavs vs wavs_need_denoise)
        if basename in copied_basenames:
            # If they are identical or we prefer the one in verified folder, copy it
            # For safety, we keep the one already copied, or overwrite if new one is from verified
            is_verified = any('verified' in part.lower() or 'wavs' == part.lower() for part in parts)
            if is_verified:
                try:
                    shutil.copy2(wav_path, dest_path)
                    copied_basenames[basename] = wav_path
                except Exception as e:
                    print(f"[!] Error copying {basename}: {e}")
        else:
            try:
                shutil.copy2(wav_path, dest_path)
                copied_basenames[basename] = wav_path
                copied_count += 1
            except Exception as e:
                print(f"[!] Error copying {basename}: {e}")
                
    print(f"[+] Successfully copied {copied_count} unique WAV files to: {out_wav_dir}")
    print(f"[+] Excluded {len(trash_files)} WAV files from trash folders.")
    
    # ----------------------------------------------------
    # Step 3: Align and clean metadata
    # ----------------------------------------------------
    print("\n[*] Aligning and cleaning metadata...")
    
    # Clean transcripts
    merged_df['text'] = merged_df['text'].astype(str).str.strip()
    merged_df.dropna(subset=['audio_path', 'text'], inplace=True)
    
    cleaned_rows = []
    seen_basenames = set()
    
    # We iterate over the dataframe rows
    # We map the audio path to just its basename, and verify if it was copied
    for idx, row in merged_df.iterrows():
        orig_path = row['audio_path']
        text = row['text']
        source = row['source_file']
        
        basename = os.path.basename(str(orig_path))
        
        # Skip if the file is in trash
        if basename in trash_files:
            continue
            
        # Skip if the file was not copied (doesn't exist in our merged directory)
        if basename not in copied_basenames:
            continue
            
        # Deduplicate: if we've seen this filename already, resolve duplicates
        # verified metadata takes priority over noise
        if basename in seen_basenames:
            # We already have an entry for this file. Let's find it in our cleaned list
            # and verify if this new entry is better (e.g. from verified file)
            # If so, update it, otherwise skip.
            is_verified_new = 'verified' in source.lower()
            for r_idx, r in enumerate(cleaned_rows):
                if r['audio_name'] == basename:
                    is_verified_old = 'verified' in r['source_file'].lower()
                    # If the new one is verified and old one is not, overwrite text
                    if is_verified_new and not is_verified_old:
                        cleaned_rows[r_idx]['text'] = text
                        cleaned_rows[r_idx]['source_file'] = source
                    break
            continue
            
        seen_basenames.add(basename)
        cleaned_rows.append({
            'audio_name': basename,
            'audio_path': f"wavs/{basename}", # standardized path relative to output directory
            'text': text,
            'source_file': source
        })
        
    cleaned_df = pd.DataFrame(cleaned_rows)
    print(f"[+] Deduplicated & cleaned metadata size: {len(cleaned_df)} rows (from original {initial_len})")
    
    if cleaned_df.empty:
        print("[-] Error: No valid metadata matches remaining after alignment.")
        return False
        
    # Save master CSV
    master_csv = os.path.join(output_dir, "final_metadata.csv")
    cleaned_df[['audio_path', 'text']].to_csv(master_csv, index=False, encoding='utf-8')
    print(f"[+] Final Master CSV saved to: {master_csv}")
    
    # ----------------------------------------------------
    # Step 4: Random split (Train/Val)
    # ----------------------------------------------------
    random.seed(seed)
    records = list(cleaned_df[['audio_path', 'text']].itertuples(index=False, name=None))
    random.shuffle(records)
    
    split_idx = int(len(records) * split_ratio)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    train_txt = os.path.join(output_dir, "train.txt")
    val_txt = os.path.join(output_dir, "val.txt")
    
    with open(train_txt, "w", encoding="utf-8") as f:
        for audio_path, text in train_records:
            f.write(f"{audio_path}|{text}\n")
            
    with open(val_txt, "w", encoding="utf-8") as f:
        for audio_path, text in val_records:
            f.write(f"{audio_path}|{text}\n")
            
    print(f"\n[+] Created train/val splits (seed={seed}, ratio={split_ratio}):")
    print(f"    - Train split size: {len(train_records)} rows -> {train_txt}")
    print(f"    - Val split size:   {len(val_records)} rows -> {val_txt}")
    print("\n[+] Dataset merging and alignment complete!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge Google Drive metadata sheets and wav directories, excluding trash.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to downloaded/synced Google Drive directory.")
    parser.add_argument("--output_dir", type=str, default="merged_dataset", help="Output directory for merged wavs and split files.")
    parser.add_argument("--split_ratio", type=float, default=0.95, help="Train/val split ratio (default: 0.95).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    
    args = parser.parse_args()
    
    merge_drive_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
