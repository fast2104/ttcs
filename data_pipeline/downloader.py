#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
downloader.py
-------------
Automates the retrieval of raw audio and subtitle files (.vtt) from YouTube
using yt-dlp.
"""

import os
import argparse
import sys
import yt_dlp

def download_youtube_data(url_list, output_dir="raw_data", sample_rate=22050):
    """
    Downloads audio and manual subtitles (.vtt) from a list of YouTube URLs.
    
    Parameters:
    -----------
    url_list : list of str
        List of YouTube URLs or video IDs.
    output_dir : str
        Directory to save the raw audio and subtitle files.
    sample_rate : int
        Target sample rate for extracted audio.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        # Post-processors to convert to WAV and set sample rate/mono
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }, {
            'key': 'FFmpegModifyMetadata',
        }],
        # Configure FFmpeg to output 22050Hz (standard for VITS) and Mono
        'postprocessor_args': [
            '-ar', str(sample_rate),
            '-ac', '1'
        ],
        # Subtitle settings: download manual subtitles, prioritize Vietnamese ('vi')
        'writesubtitles': True,
        'writeautomaticsub': False,  # Only manual subtitles are clean enough
        'subtitleslangs': ['vi'],
        'subtitlesformat': 'vtt',
        'quiet': False,
        'no_warnings': False
    }

    print(f"[*] Starting download of {len(url_list)} video(s)...")
    print(f"[*] Saving files to: {os.path.abspath(output_dir)}")
    print(f"[*] Resampling audio to {sample_rate}Hz, Mono WAV")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for index, url in enumerate(url_list, 1):
            url = url.strip()
            if not url:
                continue
            print(f"\n[{index}/{len(url_list)}] Processing: {url}")
            try:
                ydl.download([url])
                print("[+] Successfully processed.")
            except Exception as e:
                print(f"[-] Error downloading {url}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Download audio and subtitles (.vtt) from YouTube for TTS datasets.")
    parser.add_argument("--urls", type=str, nargs="+", help="Space-separated YouTube URLs or video IDs.")
    parser.add_argument("--file", type=str, help="Path to text file containing one URL per line.")
    parser.add_argument("--output_dir", type=str, default="raw_data", help="Directory to save downloaded files.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate (e.g. 22050 or 44100). Default: 22050.")
    
    args = parser.parse_args()
    
    urls = []
    if args.urls:
        urls.extend(args.urls)
    
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                urls.extend([line.strip() for line in f if line.strip()])
        else:
            print(f"[-] Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
            
    if not urls:
        print("[-] Error: No URLs provided. Please use --urls or --file.")
        parser.print_help()
        sys.exit(1)
        
    download_youtube_data(urls, output_dir=args.output_dir, sample_rate=args.sample_rate)

if __name__ == "__main__":
    main()
