# -*- coding: utf-8 -*-
"""
train_on_colab.py
-----------------
Script chạy trực tiếp trên Google Colab để tự động hóa toàn bộ quy trình:
1. Mount Google Drive.
2. Cài đặt các thư viện phụ thuộc (requirements).
3. Tải Pre-trained Checkpoint VITS tiếng Việt hoạt động tốt (thay cho link die).
4. Đồng bộ/gộp dữ liệu từ Google Drive vào môi trường Colab.
5. Kích hoạt lệnh Train/Fine-tune VITS.
"""

import os
import sys
import subprocess

def run_command(cmd):
    print(f"[*] Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"[-] Command failed with return code {process.returncode}")
        return False
    return True

def main():
    # 1. Kiểm tra môi trường Colab và Mount Google Drive
    in_colab = False
    try:
        import google.colab
        in_colab = True
    except ImportError:
        pass

    if in_colab:
        print("[*] Detecting Google Colab environment. Mounting Google Drive...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print(f"[!] Warning: Drive mount failed: {e}. Checking if already mounted...")

    # Xác định đường dẫn thư mục hiện tại của script để làm project root
    # Hỗ trợ cả môi trường chạy file script (.py) và chạy cell Jupyter Notebook (.ipynb)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Trường hợp chạy trực tiếp code trong cell Jupyter/Colab, __file__ sẽ không tồn tại
        script_dir = os.getcwd()
        if script_dir == "/content" and os.path.exists("/content/ttcs"):
            script_dir = "/content/ttcs"
            
    os.chdir(script_dir)
    print(f"[+] Working directory set to: {os.getcwd()}")

    drive_dataset_dir = "/content/drive/MyDrive/TTS_Dataset_Project"

    # 2. Cài đặt thư viện yêu cầu (Xóa bỏ df-apicep lỗi, thay bằng deepfilternet và noisereduce)
    print("\n=== STEP 1: Installing Dependencies ===")
    if os.path.exists("requirements.txt"):
        # Đọc requirements.txt nhưng bỏ qua deepfilternet để tránh lỗi biên dịch Rust trên Colab
        with open("requirements.txt", "r") as f:
            lines = f.readlines()
        filtered_reqs = []
        for line in lines:
            if "deepfilternet" not in line and "df-apicep" not in line:
                filtered_reqs.append(line.strip())
        with open("requirements_colab.txt", "w") as f:
            f.write("\n".join(filtered_reqs))
        run_command("pip install -r requirements_colab.txt")
    else:
        run_command("pip install torch torchvision torchaudio librosa soundfile pydub tqdm tensorboard noisereduce")

    # 3. Tải Pre-trained Checkpoint VITS
    print("\n=== STEP 2: Downloading Pre-trained Checkpoint VITS ===")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_checkpoint_path = os.path.join(checkpoint_dir, "vits_base.pth")
    
    use_resume = False
    
    # Cách 1: Ưu tiên tìm checkpoint trên Google Drive do người dùng tự tải lên (Bypass lỗi 401 HF)
    drive_checkpoint_path = os.path.join(drive_dataset_dir, "vits_base.pth")
    if in_colab and os.path.exists(drive_checkpoint_path):
        print(f"[+] Found pre-trained checkpoint on Google Drive: {drive_checkpoint_path}")
        print("[*] Copying checkpoint to Colab environment...")
        import shutil
        try:
            shutil.copy(drive_checkpoint_path, base_checkpoint_path)
            if os.path.exists(base_checkpoint_path) and os.path.getsize(base_checkpoint_path) > 1000:
                print("[+] Successfully loaded pre-trained checkpoint from Google Drive!")
                use_resume = True
        except Exception as e:
            print(f"[-] Failed to copy checkpoint from Drive: {e}")

    # Cách 2: Nếu không có trên Drive, tự động thử tải từ Link Public của cộng đồng
    if not use_resume:
        vits_public_url = "https://huggingface.co/dang1412/vits-vietnamese/resolve/main/G_920000.pth"
        
        if not os.path.exists(base_checkpoint_path):
            print(f"[*] Attempting to download pre-trained VITS checkpoint from Hugging Face...")
            success = run_command(f"wget -q --show-progress -O {base_checkpoint_path} {vits_public_url}")
            if not success:
                success = run_command(f"curl -L -o {base_checkpoint_path} {vits_public_url}")
            
            if success and os.path.exists(base_checkpoint_path) and os.path.getsize(base_checkpoint_path) > 1000:
                print("[+] Successfully downloaded base checkpoint from Hugging Face.")
                use_resume = True
            else:
                print("[!] Warning: Could not download from Hugging Face (auth 401/private).")
                print("[!] To use pre-trained model: Please download it manually and upload to Google Drive path:")
                print(f"    Drive của tôi/TTS_Dataset_Project/vits_base.pth")
                print("[!] Script will proceed to train FROM SCRATCH (without pre-trained weights) for now.")
                if os.path.exists(base_checkpoint_path):
                    os.remove(base_checkpoint_path)
        else:
            print("[+] Pre-trained checkpoint already exists.")
            use_resume = True

    # 4. Gộp và Chuẩn bị dữ liệu từ Google Drive (Đã gộp từ trước, chỉ cần copy vào Colab để tăng tốc độ I/O)
    print("\n=== STEP 3: Copying Merged Dataset from Google Drive ===")
    local_dataset_dir = "merged_dataset"
    drive_merged_dir = os.path.join(drive_dataset_dir, "merged_dataset")
    drive_merged_zip = os.path.join(drive_dataset_dir, "merged_dataset.zip")
    
    if in_colab:
        # Cách 1: Ưu tiên copy file nén .zip từ Drive và giải nén (Tốc độ cực nhanh, chỉ mất vài giây)
        if os.path.exists(drive_merged_zip):
            print(f"[+] Found merged dataset zip file on Google Drive: {drive_merged_zip}")
            if not os.path.exists(local_dataset_dir):
                print("[*] Copying and extracting ZIP file (this is 20x faster than copying individual files)...")
                import zipfile
                local_zip_path = "merged_dataset.zip"
                try:
                    # Copy file zip đơn lẻ (nhanh)
                    import shutil
                    shutil.copy(drive_merged_zip, local_zip_path)
                    # Giải nén
                    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(".")
                    os.remove(local_zip_path)
                    print("[+] Dataset copied and extracted successfully!")
                except Exception as e:
                    print(f"[-] ZIP copy/extract failed: {e}. Falling back to copytree...")
            else:
                print("[+] Merged dataset already exists in Colab storage.")
                
        # Cách 2: Dự phòng nếu không có file .zip thì copy từng file (Rất chậm do Drive API overhead)
        elif os.path.exists(drive_merged_dir):
            print(f"[+] Found merged dataset directory on Google Drive: {drive_merged_dir}")
            if not os.path.exists(local_dataset_dir):
                print("[!] Warning: Copying folder containing thousands of small wav files one-by-one is very slow due to Google Drive network overhead.")
                print("[!] Tip: For next time, please zip the folder to 'merged_dataset.zip' and upload it to Google Drive.")
                print("[*] Copying merged dataset to Colab local storage...")
                import shutil
                try:
                    shutil.copytree(drive_merged_dir, local_dataset_dir)
                    print("[+] Dataset copied successfully to Colab local storage.")
                except Exception as e:
                    print(f"[!] Warning: Copying dataset failed: {e}. Will use dataset directly from Google Drive.")
                    local_dataset_dir = drive_merged_dir
            else:
                print("[+] Merged dataset already copied to Colab local storage.")
        else:
            # Nếu không tìm thấy thư mục 'merged_dataset' con, kiểm tra xem chính 'TTS_Dataset_Project' có phải là thư mục chứa wavs/train.txt không
            if os.path.exists(os.path.join(drive_dataset_dir, "train.txt")):
                print(f"[+] Found dataset files directly in: {drive_dataset_dir}")
                local_dataset_dir = drive_dataset_dir
            else:
                print(f"[-] Error: Could not find merged dataset at {drive_merged_dir}, {drive_merged_zip} or {drive_dataset_dir}")
                print("[!] Please check your Google Drive path and make sure it has 'train.txt' and 'val.txt'.")
                sys.exit(1)
    else:
        print("[*] Running locally, using existing merged_dataset or processed_data.")

    # 5. Kích hoạt huấn luyện VITS
    print("\n=== STEP 4: Launching Training ===")
    config_path = "configs/vits_config.json"
    train_list = os.path.join(local_dataset_dir, "train.txt") if in_colab else "processed_data/train.txt"
    val_list = os.path.join(local_dataset_dir, "val.txt") if in_colab else "processed_data/val.txt"
    
    # Kiểm tra xem mã nguồn train.py có tồn tại không trước khi chạy
    train_script_path = "models/vits/train.py"
    if not os.path.exists(train_script_path):
        print(f"[-] Error: Could not find training script at {train_script_path}")
        print("[!] Please upload the 'models', 'configs', and 'data_pipeline' folders from your local machine to Colab.")
        print("[!] Tip: You can zip your local folder, upload it, and unzip it in Colab: !unzip ttcs.zip")
        sys.exit(1)
        
    # Tạo lệnh train (Thêm PYTHONPATH=. vào trước để Python nhận diện đúng thư mục root chứa package 'models')
    train_cmd = (
        f"PYTHONPATH=. python {train_script_path} "
        f"--config {config_path} "
        f"--train_list {train_list} "
        f"--val_list {val_list} "
        f"--output_dir {checkpoint_dir} "
        f"--epochs 150 "
        f"--batch_size 8"
    )
    
    if use_resume:
        train_cmd += f" --resume {base_checkpoint_path}"
        
    run_command(train_cmd)

if __name__ == "__main__":
    main()

