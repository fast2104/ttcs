#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_checkpoints.py
-----------------------
Utility script to programmatically download pre-trained weights for VITS
and FastSpeech 2 models from public HuggingFace repositories.
"""

import os
import sys
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """
    Downloads a file from a URL with a visual progress bar.
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define models to download
    # Standard public pre-trained Vietnamese VITS models hosted on Hugging Face
    models = {
        "vits_latest.pth": "https://huggingface.co/cdtdung/vits_vi/resolve/main/G_280000.pth",
        "fastspeech2_latest.pth": "https://huggingface.co/cdtdung/vietnamese-tts-fastspeech2/resolve/main/fastspeech2_vivos.pth"
    }
    
    print("="*60)
    print("      PRE-TRAINED TTS CHECKPOINT DOWNLOADER")
    print("="*60)
    print(f"[*] Checkpoints will be saved to: {os.path.abspath(checkpoint_dir)}")
    print("[*] Note: These files are 150MB-300MB each. Downloading may take a few minutes.\n")
    
    for filename, url in models.items():
        dest_path = os.path.join(checkpoint_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"[+] File '{filename}' already exists. Skipping download.")
            continue
            
        print(f"[*] Downloading {filename}...")
        try:
            download_url(url, dest_path)
            print(f"[+] Successfully downloaded {filename}\n")
        except Exception as e:
            print(f"[-] Error downloading {filename}: {e}", file=sys.stderr)
            print("[!] Please check your internet connection or download manually from:")
            print(f"    {url}\n", file=sys.stderr)

    print("[+] Downloader finished.")

if __name__ == "__main__":
    main()
