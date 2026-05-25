import os
import torch
import numpy as np

def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path):
    """
    Saves a training checkpoint.
    """
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'epoch': epoch
    }
    torch.save(state_dict, checkpoint_path)
    print(f"[+] Saved checkpoint to: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads model state from a saved checkpoint, supporting transfer learning/fine-tuning.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found at: {checkpoint_path}")
        return None
        
    print(f"[*] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model weights
    state_dict = checkpoint['model']
    model_state = model.state_dict()
    
    # Filter state_dict keys to allow loading from models with slight configuration changes
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"[!] Warning: Skipping parameter {k} due to shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape})")
        else:
            print(f"[!] Warning: Parameter {k} in checkpoint is not present in model structure. Skipping.")
            
    model_state.update(filtered_state)
    model.load_state_dict(model_state)
    
    # Load optimizer state if supplied
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[*] Optimizer weights restored.")
        except Exception as e:
            print(f"[!] Warning: Could not restore optimizer weights: {e}")
            
    epoch = checkpoint.get('epoch', 0)
    learning_rate = checkpoint.get('learning_rate', 1e-4)
    print(f"[+] Successfully loaded checkpoint (Epoch: {epoch}, LR: {learning_rate})")
    
    return epoch, learning_rate

def plot_spectrogram(spectrogram):
    """
    Utility to format spectrogram for tensorboard mapping.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   cmap="inferno", interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
