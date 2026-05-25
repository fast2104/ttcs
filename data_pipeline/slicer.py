#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
slicer.py
---------
Parses .vtt files (YouTube subtitles) and slices the corresponding master .wav file
into short speech segments using pydub.
"""

import os
import re
import glob
import argparse
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

def parse_time_to_ms(time_str):
    """
    Converts VTT timestamp format (HH:MM:SS.mmm or MM:SS.mmm) to milliseconds.
    """
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0.0
        minutes = float(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Invalid timestamp format: {time_str}")
    
    return int((hours * 3600 + minutes * 60 + seconds) * 1000)

def clean_vtt_text(text):
    """
    Cleans up formatting, tags, and inline timestamps from VTT text.
    """
    # Remove HTML-like tags (e.g., <c>, <b>, <i>, </c>)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove inline timestamps (e.g., <00:00:01.000>)
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
    # Remove brackets/parentheses and contents inside (e.g., [tiếng cười], (music))
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^\)]*\)', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def slice_audio(raw_dir, output_dir, padding_ms=100, min_duration_sec=1.0, max_duration_sec=15.0):
    """
    Scans raw_dir for pairs of .wav and .vtt files, slices the wavs according to the timestamps,
    and saves them in the output_dir. Generates an initial metadata CSV file.
    """
    wav_output_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_output_dir, exist_ok=True)
    
    vtt_files = glob.glob(os.path.join(raw_dir, "*.vtt"))
    if not vtt_files:
        print(f"[-] No VTT subtitle files found in {raw_dir}")
        return

    metadata_records = []
    
    print(f"[*] Found {len(vtt_files)} VTT subtitle files. Starting slicing...")
    
    for vtt_path in vtt_files:
        # Find matching wav file
        base_name = os.path.basename(vtt_path).split('.')[0]
        wav_path = os.path.join(raw_dir, f"{base_name}.wav")
        
        if not os.path.exists(wav_path):
            print(f"[!] Warning: Audio file not found for subtitles: {wav_path}. Skipping.")
            continue
            
        print(f"[*] Processing audio file: {os.path.basename(wav_path)}")
        try:
            audio = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(f"[-] Error loading wav {wav_path}: {e}. Skipping.")
            continue
            
        # Parse VTT file
        with open(vtt_path, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
            
        # VTT cue regex pattern
        # Matches: timestamp_start --> timestamp_end followed by the cue text
        cue_pattern = re.compile(
            r'(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\s*\n|\n\d|\Z)', 
            re.DOTALL
        )
        
        cues = cue_pattern.findall(vtt_content)
        print(f"    - Extracted {len(cues)} potential speech cues.")
        
        # Audio slicing loop
        for idx, (start_str, end_str, text_raw) in enumerate(tqdm(cues, desc=f"Slicing {base_name}")):
            text = clean_vtt_text(text_raw)
            if not text:
                continue
                
            try:
                start_ms = parse_time_to_ms(start_str)
                end_ms = parse_time_to_ms(end_str)
            except Exception as e:
                print(f"[!] Error parsing time for cue {idx} in {base_name}: {e}")
                continue
                
            # Apply padding but ensure boundaries are within audio duration
            start_padded = max(0, start_ms - padding_ms)
            end_padded = min(len(audio), end_ms + padding_ms)
            duration_sec = (end_padded - start_padded) / 1000.0
            
            # Check duration limits
            if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
                continue
                
            # Perform audio slice
            segment = audio[start_padded:end_padded]
            segment_filename = f"{base_name}_segment_{idx:05d}.wav"
            segment_filepath = os.path.join(wav_output_dir, segment_filename)
            
            try:
                segment.export(segment_filepath, format="wav")
                metadata_records.append({
                    "audio_path": f"processed_data/wavs/{segment_filename}",
                    "text": text,
                    "duration_sec": duration_sec
                })
            except Exception as e:
                print(f"[!] Error exporting slice {segment_filename}: {e}")
                
    # Save base metadata
    if metadata_records:
        df = pd.DataFrame(metadata_records)
        metadata_out = os.path.join(output_dir, "raw_sliced_metadata.csv")
        df.to_csv(metadata_out, index=False, encoding="utf-8")
        print(f"\n[+] Slicing complete! Exported {len(metadata_records)} segments.")
        print(f"[+] Slices stored in: {wav_output_dir}")
        print(f"[+] Sliced metadata CSV written to: {metadata_out}")
    else:
        print("\n[-] Slicing completed, but no segments were exported.")

def main():
    parser = argparse.ArgumentParser(description="Parse VTT subtitles and slice master audio files into dataset chunks.")
    parser.add_argument("--raw_dir", type=str, default="raw_data", help="Directory containing raw wav/vtt files.")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory to store sliced audio and metadata.")
    parser.add_argument("--padding", type=int, default=100, help="Padding in milliseconds added at start and end of slices. Default: 100ms.")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum duration of slice in seconds. Default: 1.0s.")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum duration of slice in seconds. Default: 15.0s.")
    
    args = parser.parse_args()
    
    slice_audio(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        padding_ms=args.padding,
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration
    )

if __name__ == "__main__":
    main()
