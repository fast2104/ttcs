# BẢNG TIẾN ĐỘ VÀ DANH SÁCH CÔNG VIỆC HOÀN THIỆN ĐỒ ÁN TTS & SPEECH ENHANCEMENT

Chào Phát, dưới đây là toàn bộ lộ trình và danh sách các việc cần làm (To-Do List) được cập nhật chi tiết theo cấu trúc mã nguồn tự động của nhóm và phân chia môi trường thực hiện (Google Colab cho khâu xử lý data và Kaggle cho khâu huấn luyện).

Bạn có thể lưu file `.md` này hoặc theo dõi tiến độ bằng cách tích chọn `[x]`.

---

## 📌 GIAI ĐOẠN 1: GỘP KHO DỮ LIỆU & LÀM SẠCH (THỰC HIỆN TRÊN GOOGLE COLAB)
*Mục tiêu: Đóng gói toàn bộ dữ liệu duyệt từ Drive, chạy khử nhiễu mô hình học sâu, gộp CSV và chia tập train/val.*

- [ ] **1.1. Khử nhiễu hàng loạt tập audio bị ồn (Speech Enhancement)**
  - [ ] Di chuyển các file audio ồn trong thư mục `wavs_need_denoise` qua mô hình DeepFilterNet sử dụng file [deepfilter.py](file:///c:/code/ttcs/denoising/deepfilter.py) để rửa nhiễu:
    ```bash
    python denoising/deepfilter.py --input_dir processed_data/wavs_need_denoise --output_dir processed_data/wavs_denoised
    ```
  - [ ] Copy đè các file âm thanh đã lọc sạch trong `wavs_denoised/` về chung thư mục gốc `processed_data/wavs/` để đồng bộ.
- [ ] **1.2. Hợp nhất dữ liệu toàn hệ thống & Phân chia Train/Val**
  - [ ] Chạy file gộp [merge_drive_data.py](file:///c:/code/ttcs/data_pipeline/merge_drive_data.py) để tự động gộp các file `metadata_verified_X.csv`, `metadata_noise_X.csv`, `metadata_noise_recovered.csv`, v.v.
  - [ ] Script sẽ tự động loại bỏ bản ghi trùng lặp, lọc bỏ các file âm thanh bị dán nhãn rác trong thư mục `wavs_trash/`, và chia ngẫu nhiên tỉ lệ 95% Train / 5% Val:
    ```bash
    python data_pipeline/merge_drive_data.py --input_dir processed_data --output_dir merged_dataset --split_ratio 0.95
    ```
- [ ] **1.3. Thống kê thông số Dataset đầu vào (Phục vụ viết báo cáo)**
  - [ ] Thống kê số lượng câu thô, câu sạch và thời lượng tổng cộng của dataset sau khi gộp (in ra từ log chạy của file gộp).
- [ ] **1.4. Đóng gói chuẩn bị cho Kaggle**
  - [ ] Nén thư mục `merged_dataset` thành tệp `merged_dataset.zip` và upload làm Dataset trên Kaggle.

---

## 🤖 GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH TTS & THỬ NGHIỆM (THỰC HIỆN TRÊN KAGGLE GPU)
*Mục tiêu: Tận dụng GPU T4 miễn phí trên Kaggle để chạy thử nghiệm và fine-tune mô hình VITS & FastSpeech 2.*

- [ ] **2.1. Cài đặt môi trường và tải Pre-trained models**
  - [ ] Tạo Kaggle Notebook mới, chọn Accelerator là GPU T4, và import dataset `merged_dataset.zip`.
  - [ ] Clone repo dự án và cài đặt dependencies `pip install -r requirements.txt`.
  - [ ] Chạy tool tải tự động trọng số pre-trained tiếng Việt từ HuggingFace [download_checkpoints.py](file:///c:/code/ttcs/data_pipeline/download_checkpoints.py):
    ```bash
    python data_pipeline/download_checkpoints.py
    ```
  - [ ] Tạo liên kết ảo (Symlink) để đồng bộ đường dẫn tập dữ liệu:
    ```bash
    !ln -s /kaggle/input/ten-dataset-cua-ban/merged_dataset/wavs /kaggle/working/wavs
    ```
- [ ] **2.2. Khởi chạy chiến thuật Fine-tuning VITS**
  - [ ] Chạy file huấn luyện [train.py](file:///c:/code/ttcs/models/vits/train.py) trỏ đến chỉ mục dữ liệu trên Kaggle và nạp trọng số đã học tiếng Việt:
    ```bash
    !python ttcs/models/vits/train.py --config ttcs/configs/vits_config.json --train_list /kaggle/working/train.txt --val_list /kaggle/working/val.txt --resume checkpoints/vits_latest.pth --epochs 10 --batch_size 16
    ```
  - [ ] Theo dõi các hàm mất mát (Loss curves) trực tiếp trên giao diện console hoặc Tensorboard.
- [ ] **2.3. Kiểm thử sinh giọng đọc (Inference)**
  - [ ] Chạy thử nghiệm tổng hợp tiếng Việt bằng [inference.py](file:///c:/code/ttcs/models/vits/inference.py) để xem mức độ bắt chước giọng nói mới có tự nhiên hay không:
    ```bash
    !python ttcs/models/vits/inference.py --text "Hôm nay tôi muốn thử nghiệm mô hình VITS tiếng Việt trên Kaggle." --checkpoint checkpoints/vits_latest.pth --output outputs/test_vits.wav
    ```

---

## 📊 GIAI ĐOẠN 3: ĐO ĐẠC THỰC NGHIỆM & TRÍCH XUẤT BIỂU ĐỒ (Dự kiến: 2 - 3 ngày)
*Mục tiêu: Thu thập số liệu kỹ thuật khách quan và hình ảnh trực quan đưa vào báo cáo.*

- [ ] **3.1. Vẽ biểu đồ phổ âm Spectrogram so sánh**
  - [ ] Sử dụng công cụ [spectrogram_plotter.py](file:///c:/code/ttcs/benchmarks/spectrogram_plotter.py) vẽ đồ thị so sánh side-by-side phổ âm của một câu thoại trước và sau khi lọc nhiễu để đưa vào quyển:
    ```bash
    python benchmarks/spectrogram_plotter.py --before path/to/noisy.wav --after path/to/clean.wav --output outputs/spectrogram_comparison.png
    ```
- [ ] **3.2. Chạy Benchmark so sánh tốc độ và tài nguyên**
  - [ ] Chạy đo đạc hệ số thời gian thực (RTF) và RAM/VRAM tiêu thụ giữa 4 mô hình bằng [benchmark_denoise.py](file:///c:/code/ttcs/benchmarks/benchmark_denoise.py):
    ```bash
    python benchmarks/benchmark_denoise.py --input_dir path/to/noisy_wavs --output_root outputs/benchmarks
    ```
- [ ] **3.3. Tổ chức chấm điểm cảm nhận thực tế (MOS)**
  - [ ] Tải các file audio mẫu về máy tính cá nhân, chạy khảo sát mù với các bạn trong lớp bằng [mos_survey.py](file:///c:/code/ttcs/benchmarks/mos_survey.py) để ghi nhận điểm số trung bình về độ tự nhiên, độ rõ tiếng của mô hình.

---

## 📝 GIAI ĐOẠN 4: SOẠN THẢO QUYỂN BÁO CÁO ĐỒ ÁN & THIẾT KẾ SLIDE
*Mục tiêu: Đóng gói toàn bộ đề tài thành sản phẩm học thuật chuẩn PTIT.*

- [ ] **4.1. Hoàn thiện Chương 3 (Thiết kế hệ thống / Data Pipeline)**
  - [ ] Đưa sơ đồ khối Mermaid (Pipeline) vào quyển.
  - [ ] Giải thích nguyên lý hoạt động của normalizer số sang chữ và tool gộp Drive.
- [ ] **4.2. Hoàn thiện Chương 4 (Thực nghiệm & Kết quả)**
  - [ ] Chèn biểu đồ phổ âm trước/sau lọc nhiễu.
  - [ ] Chèn bảng so sánh RTF và RAM của các mô hình khử nhiễu.
  - [ ] Chèn bảng so sánh điểm MOS tự nhiên thu được của VITS và FastSpeech 2.