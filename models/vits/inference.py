#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference.py
------------
Synthesis test execution script using VITS. Takes raw text input, passes it
through the normalizer, and synthesizes speech waveforms.
"""

import os
import json
import argparse
import torch
import soundfile as sf
import numpy as np

from models.vits.model import SynthesizerTrn
from models.vits.dataset import text_to_sequence, VIETNAMESE_SYMBOLS
from models.vits.utils import load_checkpoint
from data_pipeline.normalizer import normalize_text

def synthesize_text(text, checkpoint_path, config_path, output_wav):
    """
    Synthesizes input text to speech using a trained VITS model.
    """
    # 1. Normalize the text (expands numbers, abbreviations, handles casing)
    normalized_str = normalize_text(text)
    print(f"[*] Original text:  \"{text}\"")
    print(f"[*] Normalized text: \"{normalized_str}\"")
    
    # Translate to integer sequence
    seq = text_to_sequence(normalized_str)
    if not seq:
        print("[-] Error: Input text contains no valid phonemes or characters after normalization.")
        return
        
    x = torch.LongTensor(seq).unsqueeze(0) # Batch size 1
    x_lengths = torch.LongTensor([len(seq)])
    
    # 2. Load model configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    spec_params = config.get("spectrogram", {})
    model_params = config.get("model", {})
    
    sample_rate = spec_params.get("sample_rate", 22050)
    n_fft = spec_params.get("n_fft", 1024)
    spec_channels = n_fft // 2 + 1
    num_vocab = len(VIETNAMESE_SYMBOLS)
    
    # 3. Instantiate and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on device: {device}")
    
    model = SynthesizerTrn(
        num_vocab=num_vocab,
        spec_channels=spec_channels,
        segment_size=model_params.get("segment_size", 8192),
        inter_channels=model_params.get("inter_channels", 192),
        hidden_channels=model_params.get("hidden_channels", 192),
        filter_channels=model_params.get("filter_channels", 768),
        n_heads=model_params.get("n_heads", 2),
        n_layers=model_params.get("n_layers", 6),
        kernel_size=model_params.get("kernel_size", 3),
        p_dropout=model_params.get("p_dropout", 0.1),
        resblock_kernel_sizes=model_params.get("resblock_kernel_sizes", [3, 7, 11]),
        resblock_dilations=model_params.get("resblock_dilations", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        upsample_rates=model_params.get("upsample_rates", [8, 8, 2, 2]),
        upsample_kernel_sizes=model_params.get("upsample_kernel_sizes", [16, 16, 4, 4])
    ).to(device)
    
    # Load weights
    res = load_checkpoint(checkpoint_path, model, optimizer=None)
    if res is None:
        print("[-] Error: Model checkpoint weights could not be loaded. Cannot synthesize.")
        return
        
    model.eval()
    
    # Move parameters to device
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    
    # 4. Forward inference pass
    with torch.no_grad():
        # inference returns (reconstructed wave, y_mask)
        wav_out, _ = model.infer(x, x_lengths)
        
        # Postprocess wave array
        # Clamp between -1.0 and 1.0 to avoid clipping distortion
        audio = wav_out.squeeze().cpu().numpy()
        audio = np.clip(audio, -1.0, 1.0)
        
    # 5. Write to WAV file
    os.makedirs(os.path.dirname(output_wav) or '.', exist_ok=True)
    sf.write(output_wav, audio, sample_rate)
    print(f"[+] Audio successfully synthesized and saved to: {output_wav}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize text to speech using a trained VITS checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Sentence/Text string to synthesize.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vits_latest.pth", help="Path to model checkpoint (.pth).")
    parser.add_argument("--config", type=str, default="configs/vits_config.json", help="Path to config JSON file.")
    parser.add_argument("--output", type=str, default="outputs/synthesized_sample.wav", help="Path to save synthesized audio (.wav).")
    
    args = parser.parse_args()
    
    synthesize_text(
        text=args.text,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_wav=args.output
    )

if __name__ == "__main__":
    main()
