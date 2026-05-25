#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_and_split.py
------------------
A script to search and merge all CSV metadata files inside a specified directory,
remove duplicates, output a single master metadata CSV, and randomly split it into
training and validation text lists suitable for VITS and FastSpeech 2 models.
"""

import os
import glob
import argparse
import random
import pandas as pd

def merge_and_split_dataset(input_dir, output_dir, split_ratio=0.95, seed=42, 
                            audio_col='audio_path', text_col='text'):
    """
    Finds all CSV files in the input_dir, merges them, de-duplicates,
    and splits them into train.txt and val.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate all CSV files recursively
    search_path = os.path.join(input_dir, "**", "*.csv")
    csv_files = glob.glob(search_path, recursive=True)
    
    if not csv_files:
        # Check non-recursively as fallback
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
    if not csv_files:
        print(f"[-] Error: No CSV files found in the directory: {input_dir}")
        return False
        
    print(f"[*] Found {len(csv_files)} CSV file(s) to merge:")
    for f_path in csv_files:
        print(f"    - {f_path}")
        
    dfs = []
    for f_path in csv_files:
        try:
            # Let's read the CSV file. Handle common encodings like utf-8 or utf-8-sig
            df_temp = pd.read_csv(f_path, encoding='utf-8-sig')
            
            # Check if columns match
            if audio_col not in df_temp.columns or text_col not in df_temp.columns:
                print(f"[!] Warning: Columns '{audio_col}' and/or '{text_col}' not found in {f_path}. Skipping.")
                print(f"    Available columns: {list(df_temp.columns)}")
                continue
                
            # Keep only the columns we need to prevent shape conflicts
            df_filtered = df_temp[[audio_col, text_col]].copy()
            dfs.append(df_filtered)
        except Exception as e:
            print(f"[!] Warning: Error reading {f_path}: {e}")
            
    if not dfs:
        print("[-] Error: No valid data frames could be merged.")
        return False
        
    # Merge all datasets
    merged_df = pd.concat(dfs, ignore_index=True)
    initial_len = len(merged_df)
    
    # Remove exact duplicate rows
    merged_df.drop_duplicates(subset=[audio_col, text_col], inplace=True)
    # Also clean up NaN or empty text/path
    merged_df.dropna(subset=[audio_col, text_col], inplace=True)
    merged_df[text_col] = merged_df[text_col].astype(str).str.strip()
    merged_df = merged_df[merged_df[text_col] != ""]
    final_len = len(merged_df)
    
    print(f"\n[+] Merging complete:")
    print(f"    - Total rows loaded: {initial_len}")
    print(f"    - Cleaned duplicates/nulls: {initial_len - final_len} rows removed")
    print(f"    - Total unique samples remaining: {final_len}")
    
    # Output merged CSV
    master_csv_path = os.path.join(output_dir, "final_metadata.csv")
    merged_df.to_csv(master_csv_path, index=False, encoding="utf-8")
    print(f"[+] Merged Master CSV saved to: {master_csv_path}")
    
    # Shuffle and Split randomly
    # Set seed for reproducibility
    random.seed(seed)
    
    # Convert dataframe to list of tuples
    records = list(merged_df.itertuples(index=False, name=None))
    random.shuffle(records)
    
    split_index = int(len(records) * split_ratio)
    train_records = records[:split_index]
    val_records = records[split_index:]
    
    print(f"\n[*] Splitting dataset (Ratio: {split_ratio * 100:.1f}% Train / {(1 - split_ratio) * 100:.1f}% Val):")
    print(f"    - Training samples: {len(train_records)}")
    print(f"    - Validation samples: {len(val_records)}")
    
    # Save as VITS format files (Format: filepath|text)
    train_txt_path = os.path.join(output_dir, "train.txt")
    val_txt_path = os.path.join(output_dir, "val.txt")
    
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for audio_path, text in train_records:
            # Replace backslashes with forward slashes for cross-platform compatibility
            cleaned_path = str(audio_path).replace("\\", "/")
            f.write(f"{cleaned_path}|{text}\n")
            
    with open(val_txt_path, "w", encoding="utf-8") as f:
        for audio_path, text in val_records:
            cleaned_path = str(audio_path).replace("\\", "/")
            f.write(f"{cleaned_path}|{text}\n")
            
    print(f"[+] VITS training list exported: {train_txt_path}")
    print(f"[+] VITS validation list exported: {val_txt_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Merge CSV metadata files from a directory and split into train/val datasets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CSV files (e.g. OneDrive/Google Drive synced folder).")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Directory where final metadata, train.txt, and val.txt will be saved.")
    parser.add_argument("--split_ratio", type=float, default=0.95, help="Proportion of the dataset to allocate to training. Default: 0.95 (95%).")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for the random splits. Default: 42.")
    parser.add_argument("--audio_col", type=str, default="audio_path", help="Name of the audio file column in CSVs. Default: 'audio_path'.")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text transcript column in CSVs. Default: 'text'.")
    
    args = parser.parse_args()
    
    merge_and_split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed,
        audio_col=args.audio_col,
        text_col=args.text_col
    )

if __name__ == "__main__":
    main()
