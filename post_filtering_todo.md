# BẢNG TIẾN ĐỘ & CÁC VIỆC CẦN LÀM SAU KHI LỌC DỮ LIỆU (POST-FILTERING TO-DO LIST)

Chào Phát và Đăng, đây là checklist chi tiết các công việc cần làm kể từ khi các bạn hoàn tất khâu lọc nhiễu dữ liệu. Bảng này hướng dẫn từ bước gộp kho dữ liệu trên Drive, kiểm thử giọng đọc, đo đạc thông số thực nghiệm, cho đến khi hoàn thành báo cáo khoa học.

Bạn có thể tích chọn `[x]` để theo dõi tiến độ hàng ngày trực tiếp trong VS Code hoặc GitHub.

---

## 🔗 GIAI ĐOẠN 1: GỘP KHO DỮ LIỆU SẠCH (THỰC HIỆN TRÊN GOOGLE COLAB)
*Mục tiêu: Đóng gói và gộp các file audio sạch, gộp CSV đã duyệt trên Drive và nén lại thành tệp dataset duy nhất.*

- [ ] **1.1. Chuẩn bị môi trường trên Google Colab**
  - [ ] Tạo mới một file Notebook trên Google Colab.
  - [ ] Kết nối (Mount) Google Drive chứa thư mục dữ liệu đã duyệt:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
  - [ ] Clone code dự án mới nhất từ GitHub của bạn:
    ```bash
    !git clone https://github.com/fast2104/ttcs.git
    ```
- [ ] **1.2. Chạy tool gộp dữ liệu tự động (`merge_drive_data.py`)**
  - [ ] Chạy lệnh gộp dữ liệu directly trên Colab (thay đổi đường dẫn thư mục Drive thực tế của bạn):
    ```bash
    !python ttcs/data_pipeline/merge_drive_data.py \
      --input_dir "/content/drive/MyDrive/TTS_Dataset_Project/processed_data" \
      --output_dir "/content/drive/MyDrive/TTS_Dataset_Project/merged_dataset" \
      --split_ratio 0.95
    ```
  - [ ] *Xác nhận kết quả trên Google Drive*: Thư mục `/merged_dataset` đã được tạo ra đầy đủ gồm:
    - [ ] Sổ tay tổng hợp sạch: `final_metadata.csv`
    - [ ] File chỉ mục train/val: `train.txt` và `val.txt`
    - [ ] Folder chứa toàn bộ âm thanh sạch: `wavs/`
- [ ] **1.3. Nén dữ liệu thành file `.zip` để chuẩn bị upload lên Kaggle**
  - [ ] Chạy lệnh nén folder đầu ra trên Colab:
    ```bash
    !zip -q -r /content/drive/MyDrive/TTS_Dataset_Project/merged_dataset.zip /content/drive/MyDrive/TTS_Dataset_Project/merged_dataset
    ```

---

## 📥 GIAI ĐOẠN 2: HUẤN LUYỆN & THỬ NGHIỆM TTS (THỰC HIỆN TRÊN KAGGLE)
*Mục tiêu: Đẩy dữ liệu sạch lên Kaggle, tận dụng GPU T4 miễn phí để chạy thử nghiệm và huấn luyện mô hình.*

- [ ] **2.1. Đưa tập dữ liệu (Dataset) lên Kaggle**
  - [ ] Truy cập Kaggle, chọn mục **Create -> New Dataset**.
  - [ ] Upload tệp `merged_dataset.zip` vừa nén trên Drive về máy hoặc liên kết trực tiếp để tạo dataset trên Kaggle với tên (ví dụ: `vits-vietnamese-dataset`).
- [ ] **2.2. Khởi tạo Kaggle Notebook**
  - [ ] Tạo một **New Notebook** trên Kaggle.
  - [ ] Cấu hình phần cứng (Accelerator): Chọn **GPU T4 x2** hoặc **GPU T4** trong cài đặt Notebook.
  - [ ] Thêm Dataset đã upload ở bước 2.1 vào Notebook (nó sẽ được giải nén tự động tại thư mục `/kaggle/input/vits-vietnamese-dataset`).
- [ ] **2.3. Thiết lập mã nguồn và tạo liên kết thư mục (Symlink)**
  - [ ] Tải mã nguồn dự án trên Kaggle Notebook:
    ```bash
    !git clone https://github.com/fast2104/ttcs.git
    ```
  - [ ] Cài đặt các thư viện phụ thuộc:
    ```bash
    !pip install -r ttcs/requirements.txt
    ```
  - [ ] **Mẹo đồng bộ đường dẫn**: Do file `train.txt` ghi đường dẫn dạng `wavs/file.wav` nhưng trên Kaggle dữ liệu nằm ở `/kaggle/input/...`, hãy tạo một liên kết thư mục (Symbolic Link) để code chạy không bị lỗi:
    ```bash
    !ln -s /kaggle/input/vits-vietnamese-dataset/merged_dataset/wavs /kaggle/working/wavs
    ```
- [ ] **2.4. Tải trọng số pre-trained để kiểm thử giọng đọc**
  - [ ] Chạy script tải trọng số trên Kaggle:
    ```bash
    !python ttcs/data_pipeline/download_checkpoints.py
    ```
  - [ ] *Kiểm thử sinh giọng nói (Inference VITS)*:
    ```bash
    !python ttcs/models/vits/inference.py \
      --text "Hôm nay chúng tôi thử nghiệm hệ thống trên môi trường Kaggle." \
      --checkpoint checkpoints/vits_latest.pth \
      --output outputs/test_vits.wav
    ```
- [ ] **2.5. Huấn luyện / Fine-tune mô hình trên Kaggle**
  - [ ] Chạy lệnh huấn luyện sử dụng GPU:
    ```bash
    !python ttcs/models/vits/train.py \
      --config ttcs/configs/vits_config.json \
      --train_list /kaggle/input/vits-vietnamese-dataset/merged_dataset/train.txt \
      --val_list /kaggle/input/vits-vietnamese-dataset/merged_dataset/val.txt \
      --output_dir /kaggle/working/checkpoints \
      --resume checkpoints/vits_latest.pth \
      --epochs 100 \
      --batch_size 16
    ```

---

## 📊 GIAI ĐOẠN 3: ĐO ĐẠC THỰC NGHIỆM TRÊN KAGGLE
*Mục tiêu: Chạy các script so sánh hiệu năng, vẽ biểu đồ phổ âm và lưu kết quả về máy.*

- [ ] **3.1. Trích xuất phổ âm Spectrogram so sánh**
  - [ ] Chạy script plotter để xuất ảnh phổ đồ (Before/After):
    ```bash
    !python ttcs/benchmarks/spectrogram_plotter.py \
      --before /kaggle/input/vits-vietnamese-dataset/merged_dataset/wavs/file_on.wav \
      --after /kaggle/working/wavs/file_sach.wav \
      --output /kaggle/working/outputs/spectrogram_comparison.png
    ```
- [ ] **3.2. Chạy so sánh RTF và Peak Memory các thuật toán lọc nhiễu**
  - [ ] ```bash
    !python ttcs/benchmarks/benchmark_denoise.py \
      --input_dir /kaggle/input/vits-vietnamese-dataset/merged_dataset/wavs \
      --output_root /kaggle/working/outputs/benchmarks
    ```
- [ ] **3.3. Tải các file kết quả thực nghiệm về máy cá nhân**
  - [ ] Tải file ảnh phổ âm `spectrogram_comparison.png` và các checkpoint `.pth` cùng báo cáo CSV trong thư mục `/kaggle/working/outputs` về máy để làm slide bảo vệ và chèn vào quyển Word.
- [ ] **3.4. Chạy MOS Survey khảo sát điểm chất lượng**
  - [ ] Dùng các file audio `.wav` đã tải về máy chạy MOS local để khảo sát lấy điểm trực tiếp trên máy tính:
    ```bash
    python benchmarks/mos_survey.py --audio_dir outputs/eval_samples --output outputs/mos_results.csv
    ```

---

## 📝 GIAI ĐOẠN 4: HOÀN THIỆN QUYỂN BÁO CÁO & SLIDE BẢO VỆ (Dự kiến: Song song)
*Mục tiêu: Đóng gói toàn bộ lý thuyết và số liệu thực nghiệm thành sản phẩm báo cáo PTIT hoàn chỉnh.*

- [ ] **4.1. Hoàn thiện Chương 3 (Quy trình xây dựng Dataset)**
  - [ ] Đưa sơ đồ khối Mermaid (Pipeline xử lý dữ liệu) vào quyển Word.
  - [ ] Giải thích nguyên lý hoạt động của tool chuẩn hóa số `normalizer.py` và công cụ gộp tự động `merge_drive_data.py`.
  - [ ] Thống kê số lượng mẫu, tổng thời lượng dataset sạch sau khi gộp.
- [ ] **4.2. Hoàn thiện Chương 4 (Đánh giá Thực nghiệm & Thử nghiệm)**
  - [ ] Chèn hình ảnh so sánh Spectrogram (Before/After) của DeepFilterNet.
  - [ ] Lập bảng so sánh RTF và Peak Memory giữa các mô hình khử nhiễu (Dữ liệu lấy từ Bước 3.2).
  - [ ] Chèn bảng so sánh tính năng VITS vs FastSpeech 2.
  - [ ] Trình bày kết quả điểm khảo sát MOS (Độ tự nhiên, độ rõ chữ) thu được từ Bước 3.3.
- [ ] **4.3. Rà soát căn lề định dạng văn bản (Format PTIT)**
  - [ ] Căn lề chuẩn: Trên 2cm, Dưới 2cm, Trái 3cm, Phải 2cm. Font: Times New Roman cỡ 13, giãn dòng 1.5 line.
  - [ ] Kiểm tra mục lục tự động, danh mục hình vẽ và bảng biểu.
- [ ] **4.4. Chuẩn bị Slide thuyết trình bảo vệ**
  - [ ] Thiết kế slide ngắn gọn (12 - 15 slides).
  - [ ] Đính kèm trực tiếp các tệp âm thanh demo trước/sau khi khử nhiễu và audio TTS để bật trực tiếp cho Hội đồng nghe lúc bảo vệ đồ án.
