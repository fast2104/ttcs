import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np

# Vocabulary mapping for Vietnamese (fallback character set and tones)
VIETNAMESE_SYMBOLS = [
    '_', 'sp', 'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y',
    'à', 'ằ', 'ầ', 'è', 'ề', 'ì', 'ò', 'ồ', 'ờ', 'ù', 'ừ', 'ỳ',
    'á', 'ắ', 'ấ', 'é', 'ế', 'í', 'ó', 'ố', 'ớ', 'ú', 'ứ', 'ý',
    'ả', 'ẳ', 'ẩ', 'ẻ', 'ể', 'ỉ', 'ỏ', 'ổ', 'ở', 'ủ', 'ử', 'ỷ',
    'ã', 'ẵ', 'ẫ', 'ẽ', 'ễ', 'ĩ', 'õ', 'ỗ', 'ỡ', 'ũ', 'ữ', 'ỹ',
    'ạ', 'ặ', 'ậ', 'ẹ', 'ệ', 'ị', 'ọ', 'ộ', 'ợ', 'ụ', 'ự', 'ỵ',
    ' ', '.', ',', '?', '!', '-', '\''
]

_symbol_to_id = {s: i for i, s in enumerate(VIETNAMESE_SYMBOLS)}
_id_to_symbol = {i: s for i, s in enumerate(VIETNAMESE_SYMBOLS)}

def text_to_sequence(text):
    """
    Converts a string of Vietnamese characters to a sequence of integer IDs.
    Unknown characters are ignored.
    """
    clean_text = text.lower().strip()
    sequence = []
    for char in clean_text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
    return sequence

class TextAudioDataset(Dataset):
    """
    Custom PyTorch Dataset that loads audio/text pairs and extracts linear spectrograms.
    """
    def __init__(self, metadata_file, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Load file list
        self.records = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    audio_path, text = parts
                    self.records.append((audio_path, text))
                    
        # Initialize STFT processor
        self.stft = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1.0,
            normalized=False
        )

    def load_audio_to_tensor(self, path):
        """
        Loads audio, checks channel count, resamples if necessary.
        """
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        # Average channels to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0)

    def get_spectrogram(self, waveform):
        """
        Extracts linear amplitude spectrogram from raw waveform.
        """
        # Add batch dim for transform
        spec = self.stft(waveform.unsqueeze(0)).squeeze(0)
        # Apply log compression to avoid numerical instability
        log_spec = torch.log(torch.clamp(spec, min=1e-5))
        return log_spec

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        audio_path, text = self.records[index]
        
        # 1. Load waveform
        try:
            waveform = self.load_audio_to_tensor(audio_path)
        except Exception as e:
            # Fallback if audio file is corrupted during run
            print(f"[!] Error loading file {audio_path}: {e}")
            # Generate a 1-second silent tensor
            waveform = torch.zeros(self.sample_rate)
            
        # 2. Extract linear spectrogram
        spec = self.get_spectrogram(waveform)
        
        # 3. Convert text to integer sequences
        text_seq = torch.LongTensor(text_to_sequence(text))
        
        return (text_seq, spec, waveform)

class TextAudioCollate:
    """
    Zero-pads text sequences and spectrograms to match batch lengths dynamically.
    """
    def __call__(self, batch):
        # Sort batch by text length descending (beneficial for RNN processing)
        batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        
        max_text_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wave_len = max([x[2].size(0) for x in batch])
        
        num_features = batch[0][1].size(0)
        
        # Allocate padded tensors
        texts = torch.LongTensor(len(batch), max_text_len).zero_()
        text_lengths = torch.LongTensor(len(batch))
        
        specs = torch.FloatTensor(len(batch), num_features, max_spec_len).zero_()
        spec_lengths = torch.LongTensor(len(batch))
        
        waves = torch.FloatTensor(len(batch), max_wave_len).zero_()
        wave_lengths = torch.LongTensor(len(batch))
        
        for idx, (text, spec, wave) in enumerate(batch):
            # Pad text
            text_len = text.size(0)
            texts[idx, :text_len] = text
            text_lengths[idx] = text_len
            
            # Pad spec
            spec_len = spec.size(1)
            specs[idx, :, :spec_len] = spec
            spec_lengths[idx] = spec_len
            
            # Pad wave
            wave_len = wave.size(0)
            waves[idx, :wave_len] = wave
            wave_lengths[idx] = wave_len
            
        return texts, text_lengths, specs, spec_lengths, waves, wave_lengths
