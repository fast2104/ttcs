#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
Training and fine-tuning execution script for the VITS model.
Configured for local execution and Google Colab environments, supports TensorBoard tracking.
"""

import os
import json
import argparse
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vits.model import SynthesizerTrn
from models.vits.dataset import TextAudioDataset, TextAudioCollate, VIETNAMESE_SYMBOLS
from models.vits.utils import save_checkpoint, load_checkpoint

def train_and_fine_tune(config_path, train_list, val_list, output_dir, checkpoint_path=None,
                        epochs=100, batch_size=8, lr=1e-4, log_interval=10, save_interval=1000):
    """
    Main training and fine-tuning loop for VITS.
    """
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # Load config file parameters
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # Get hyperparameters
    spec_params = config.get("spectrogram", {})
    model_params = config.get("model", {})
    
    sample_rate = spec_params.get("sample_rate", 22050)
    n_fft = spec_params.get("n_fft", 1024)
    hop_length = spec_params.get("hop_length", 256)
    win_length = spec_params.get("win_length", 1024)
    spec_channels = n_fft // 2 + 1
    
    print("[*] Preparing PyTorch datasets...")
    # Initialize datasets
    train_dataset = TextAudioDataset(
        metadata_file=train_list,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    val_dataset = TextAudioDataset(
        metadata_file=val_list,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
    
    collate_fn = TextAudioCollate()
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        num_workers=2 if os.name == 'posix' else 0, # Colab supports workers
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    # Instantiate the Synthesizer model
    # Vocabulary size based on VIETNAMESE_SYMBOLS
    num_vocab = len(VIETNAMESE_SYMBOLS)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on target device: {device}")
    
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
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Check fine-tuning / resume options
    start_epoch = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        res = load_checkpoint(checkpoint_path, model, optimizer)
        if res:
            start_epoch, lr = res
            start_epoch += 1 # Continue from next epoch
            # Adjust learning rate if fine-tuning config overrides it
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        print("[*] Training from scratch (randomly initialized weights).")
        
    global_step = 0
    # Simple training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        print(f"\n--- Epoch {epoch} / {epochs} ---")
        epoch_loss = 0.0
        
        # Wrapping iterator in tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} iteration")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move items to GPU
            x, x_lengths, y, y_lengths, wave, wave_lengths = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            # o: output reconstructed wave segment
            # slice_ids: indices of cropped spectrogram segments for wave synthesizer
            # stats: intermediate hidden representations (z, z_p, m_p, logs_p, m_q, logs_q)
            o, slice_ids, x_mask, y_mask, stats = model(x, x_lengths, y, y_lengths)
            
            # Unpack stats for loss calculations
            z, z_p, m_p, logs_p, m_q, logs_q = stats
            
            # 1. Reconstruction Wave Loss (L1/L2 on synthesized raw audio slice vs raw ground truth slice)
            # Reconstruct targets by slicing original waveforms accordingly
            target_segment_size = model.segment_size
            wave_slices = []
            for idx, start_id in enumerate(slice_ids):
                # hop size ratio is 256 (upsample_rates: 8 * 8 * 2 * 2 = 256)
                wave_start = start_id * hop_length
                wave_slices.append(wave[idx, wave_start:wave_start + target_segment_size])
            wave_sliced = torch.stack(wave_slices, dim=0).unsqueeze(1)
            
            # Compare synthesized segment with original segment
            loss_recon = F.l1_loss(o, wave_sliced)
            
            # 2. KL Divergence loss on prior/posterior spaces (Flow alignment)
            # D_kl(q(z|y) || p(z|x))
            # Interpolate prior parameters (m_p, logs_p) to match posterior size (z_p, logs_q)
            t_audio = z_p.size(2)
            m_p_up = F.interpolate(m_p, size=t_audio, mode='nearest')
            logs_p_up = F.interpolate(logs_p, size=t_audio, mode='nearest')
            
            kl_loss = torch.sum(logs_p_up - logs_q - 0.5 + ((z_p - m_p_up) ** 2) * torch.exp(-2.0 * logs_p_up) * 0.5) / torch.sum(y_mask)
            
            # Total Loss
            loss = loss_recon + 1.0 * kl_loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Print loss in progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "recon": f"{loss_recon.item():.4f}", 
                "kl": f"{kl_loss.item():.4f}"
            })
            
            # TensorBoard logging
            if global_step % log_interval == 0:
                writer.add_scalar("Loss/Train_Total", loss.item(), global_step)
                writer.add_scalar("Loss/Train_Recon", loss_recon.item(), global_step)
                writer.add_scalar("Loss/Train_KL", kl_loss.item(), global_step)
                
            # Periodic model checkpoints saving
            if global_step % save_interval == 0:
                ckpt_file = os.path.join(output_dir, f"vits_step_{global_step}.pth")
                save_checkpoint(model, optimizer, lr, epoch, ckpt_file)
                
        # Epoch-level validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, x_lengths, y, y_lengths, wave, wave_lengths = [b.to(device) for b in batch]
                # During validation, evaluate directly without cropping
                o, _, x_mask, y_mask, stats = model(x, x_lengths, y, y_lengths)
                z, z_p, m_p, logs_p, m_q, logs_q = stats
                
                # Check validation sample rate
                # Compare full reconstructed output with padded input waves
                loss_recon = F.l1_loss(o, wave.unsqueeze(1)[:, :, :o.size(2)])
                
                # Interpolate prior parameters for validation
                t_audio_val = z_p.size(2)
                m_p_up_val = F.interpolate(m_p, size=t_audio_val, mode='nearest')
                logs_p_up_val = F.interpolate(logs_p, size=t_audio_val, mode='nearest')
                
                kl_loss = torch.sum(logs_p_up_val - logs_q - 0.5 + ((z_p - m_p_up_val) ** 2) * torch.exp(-2.0 * logs_p_up_val) * 0.5) / torch.sum(y_mask)
                val_loss += (loss_recon + kl_loss).item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"[+] Validation complete. Avg Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val_Total", avg_val_loss, epoch)
        
        # Save epoch checkpoint
        ckpt_epoch_path = os.path.join(output_dir, "vits_latest.pth")
        save_checkpoint(model, optimizer, lr, epoch, ckpt_epoch_path)
        
    writer.close()
    print("[+] Training pipeline finished.")

def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune VITS models.")
    parser.add_argument("--config", type=str, default="configs/vits_config.json", help="Path to config JSON file.")
    parser.add_argument("--train_list", type=str, default="processed_data/train.txt", help="Path to train.txt metadata file.")
    parser.add_argument("--val_list", type=str, default="processed_data/val.txt", help="Path to val.txt metadata file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints and logs.")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing model checkpoint (.pth) to resume training or fine-tune.")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training. Decrease if running out of VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between training logs.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between periodic checkpoints.")
    
    args = parser.parse_args()
    
    train_and_fine_tune(
        config_path=args.config,
        train_list=args.train_list,
        val_list=args.val_list,
        output_dir=args.output_dir,
        checkpoint_path=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )

if __name__ == "__main__":
    main()
