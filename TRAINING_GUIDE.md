# HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH (TRAINING & FINE-TUNING GUIDE)
*Tài liệu hướng dẫn từng bước sau khi hoàn tất Crawl dữ liệu thô*

Tài liệu này cung cấp hướng dẫn chi tiết giúp bạn và nhóm xử lý tập dữ liệu thô, tiến hành gán nhãn, chạy lọc tạp âm bằng mô hình học sâu, và cấu hình huấn luyện/fine-tune mô hình TTS VITS & FastSpeech 2.

---

## 📅 BƯỚC 1: CẮT NHỎ AUDIO THEO PHỤ ĐỀ (SLICING)

Sau khi tải xong video và phụ đề bằng `downloader.py` vào thư mục `raw_data`, chạy script để cắt tự động thành các câu thoại ngắn (từ 1 đến 15 giây):

```bash
python data_pipeline/slicer.py --raw_dir raw_data --output_dir processed_data --padding 100 --min_duration 1.0 --max_duration 12.0
```
* **Ý nghĩa thông số**:
  * `--padding 100`: Thêm 100ms đệm vào đầu và cuối mỗi phân đoạn để tránh bị mất chữ/cụt âm do phụ đề trễ.
  * `--min_duration 1.0` và `--max_duration 12.0`: Giới hạn độ dài để loại các câu quá ngắn (tiếng thở, ừm) hoặc quá dài (khó học đối với VITS).
* **Kết quả**: Thư mục `processed_data/wavs` chứa các file wav ngắn và file mục lục ban đầu `processed_data/raw_sliced_metadata.csv`.

---

## 🎨 BƯỚC 2: KIỂM ĐỊNH & PHÂN LOẠI DỮ LIỆU QUA GIAO DIỆN UI (COLAB)

Để chia sẻ khối lượng duyệt dữ liệu cho các thành viên trong nhóm (Ví dụ: Phát duyệt chỉ số chẵn, Đăng duyệt chỉ số lẻ), hãy upload thư mục `processed_data/` lên Google Drive hoặc OneDrive và mở notebook `data_pipeline/annotator_ui.ipynb`.

1. **Chạy các Cell thiết lập**: Trình duyệt sẽ hiển thị bảng điều khiển trực quan.
2. **Thực hiện duyệt**:
   * Hệ thống tự động phát đoạn âm thanh.
   * Chỉnh sửa chính tả trực tiếp trên hộp thoại `Văn bản` nếu phát hiện phụ đề YouTube bị sai hoặc viết tắt.
   * Chọn nút **Chuẩn / Lưu**: Dữ liệu sạch sẽ được lưu vào sổ tay `metadata_verified.csv`.
   * Chọn nút **Ồn / Nhạc**: Các file ồn sẽ được di chuyển sang thư mục `wavs_need_denoise/` và ghi log vào `metadata_noise.csv`.
   * Chọn nút **Rác (Lỗi)**: Các file bị nói lắp, tạp âm quá nặng không cứu được sẽ chuyển sang `wavs_trash/` để loại bỏ khỏi tập train.
   * Nút **Hoàn tác (Undo)**: Bấm khi lỡ nhấn nhầm nút, hệ thống sẽ khôi phục file audio và xóa dòng log bị ghi sai trong CSV.

---

## 🧹 BƯỚC 3: KHỬ NHIỄU PHẦN DATA BỊ ỒN BẰNG DEEPFILTERNET

Các file audio trong danh sách `Ồn / Nhạc` cần được đi qua bộ lọc AI để trả về tiêu chuẩn phòng thu. Chạy lệnh:

```bash
python denoising/deepfilter.py --input_dir processed_data/wavs_need_denoise --output_dir processed_data/wavs_denoised
```
* **Hành động**: DeepFilterNet sẽ triệt tiêu tiếng gió, tiếng máy lạnh, quạt và nhạc nền nhẹ, xuất các file đã sạch vào thư mục `wavs_denoised/`.
* *Lưu ý*: Sao chép/Di chuyển toàn bộ các file sạch này đè ngược lại vào thư mục gốc `processed_data/wavs/` để đồng bộ dữ liệu.

---

## 🔗 BƯỚC 4: GỘP SỔ TAY VÀ PHÂN CHIA TẬP TRAIN / VALIDATE

Khi cả nhóm đã hoàn tất duyệt và khử nhiễu, hãy tải toàn bộ thư mục `processed_data` từ Drive/OneDrive về máy tính và tiến hành gộp dữ liệu. Nhóm có hai lựa chọn công cụ gộp:

### Lựa chọn A: Gộp nhanh dữ liệu cục bộ (Chỉ gộp CSV)
Nếu các file âm thanh đã được bạn gộp thủ công vào thư mục `processed_data/wavs`, chỉ cần gộp các file CSV:
```bash
python data_pipeline/merge_and_split.py --input_dir processed_data --output_dir processed_data --split_ratio 0.95 --seed 42
```

### Lựa chọn B: Gộp tự động toàn bộ dữ liệu Drive (Khuyên dùng)
Công cụ này sẽ quét tất cả các file metadata (`verified`, `noise`, `recovered`, `final`), tự động gộp và loại trùng lặp. Đồng thời quét qua các thư mục audio, sao chép toàn bộ file `.wav` trừ thư mục chứa chữ `trash` (ví dụ `wavs_trash`) sang thư mục đầu ra thống nhất, tự động đồng bộ lại đường dẫn trong file text:
```bash
python data_pipeline/merge_drive_data.py --input_dir path/to/sync/processed_data --output_dir merged_dataset --split_ratio 0.95 --seed 42
```
* **Kết quả đầu ra**:
  * Thư mục `merged_dataset/wavs` chứa toàn bộ file âm thanh sạch từ `wavs` và `wavs_need_denoise` (Đã loại bỏ các file rác trong `wavs_trash`).
  * `merged_dataset/final_metadata.csv` (Sổ tay dữ liệu tổng hợp sạch).
  * `merged_dataset/train.txt` & `merged_dataset/val.txt` (Danh sách tập train/val định dạng chuẩn `wavs/file.wav|text` sẵn sàng để đẩy vào VITS).

---

## 🤖 BƯỚC 5: TIẾN HÀNH FINE-TUNE / HUẤN LUYỆN MÔ HÌNH VITS

### 5.1 Cấu hình file `configs/vits_config.json`
Đảm bảo các đường dẫn và tần số lấy mẫu (Sample Rate) trùng khớp với dataset của bạn:
* `"sample_rate": 22050` (Tần số chuẩn cho VITS).
* `"segment_size": 8192` (Độ dài slice nạp vào GPU). Nếu card đồ họa bị báo lỗi hết bộ nhớ (Out-Of-Memory - OOM), hãy hạ `batch_size` xuống còn `4` hoặc `2`.

### 5.2 Tải Base Model (Pre-trained model)
Để mô hình học nhanh và phát âm tiếng Việt chuẩn xác có cảm xúc, bạn nên nạp trọng số (weights) của một mô hình VITS tiếng Việt nền tảng đã huấn luyện sẵn.
* Đặt file Base Model vào thư mục `checkpoints/vits_base.pth`.

### 5.3 Chạy lệnh Train / Fine-tune
```bash
python models/vits/train.py --config configs/vits_config.json --train_list processed_data/train.txt --val_list processed_data/val.txt --output_dir checkpoints --resume checkpoints/vits_base.pth --epochs 100 --batch_size 8
```
* Script tự động nạp các lớp tương thích từ `vits_base.pth` và tiếp tục tối ưu hóa trọng số dựa trên giọng đọc mới trong dataset của bạn.

---

## 📈 BƯỚC 6: THEO DÕI BIỂU ĐỒ LOSS TRÊN TENSORBOARD

Trong quá trình huấn luyện, mở terminal mới và khởi chạy Tensorboard để kiểm tra mức độ hội tụ của mô hình:

```bash
tensorboard --logdir checkpoints/logs
```
* **Các chỉ số cần quan tâm**:
  * `Loss/Train_Recon` (L1 loss của sóng âm): Phải giảm dần theo thời gian, chứng tỏ AI đang học tái tạo âm sắc giọng người tốt.
  * `Loss/Train_KL` (Sai biệt phân phối prior/posterior): Giúp ổn định ngữ điệu đọc.
  * `Loss/Val_Total`: Đánh giá trên tập validate độc lập, nếu biểu đồ này đi ngang hoặc tăng trong khi loss train vẫn giảm thì mô hình bắt đầu bị hiện tượng học vẹt (Overfitting) -> nên dừng train để lấy checkpoint trước đó.

---

## 🔊 BƯỚC 7: KIỂM THỬ GIỌNG ĐỌC & ĐÁNH GIÁ CHẤT LƯỢNG (MOS)

### 7.1 Chạy sinh giọng đọc tự động (Inference)
Nhập một câu văn bất kỳ chưa từng có trong tập dữ liệu huấn luyện để thử nghiệm độ mượt mà:

```bash
python models/vits/inference.py --text "Trí tuệ nhân tạo đang làm thay đổi thế giới xử lý âm thanh kỹ thuật số." --checkpoint checkpoints/vits_latest.pth --output outputs/test_synthesis.wav
```
* **Hành động**: Lệnh sẽ gọi bộ dịch số sang chữ (`normalizer.py`), chuyển hóa ngữ âm và tổng hợp ra file `test_synthesis.wav`.

### 7.2 Tổ chức khảo sát chấm điểm MOS (Mean Opinion Score)
Lấy các mẫu âm thanh được tạo ra từ mô hình của nhóm, đặt vào thư mục `outputs/eval_samples/` và chạy công cụ khảo sát:

```bash
python benchmarks/mos_survey.py --audio_dir outputs/eval_samples --output outputs/mos_results.csv
```
* **Quy trình**: Công cụ sẽ phát mù ngẫu nhiên (Blind Test) từng file. Người nghe (các bạn cùng lớp) chấm từ 1 đến 5 điểm. Kết quả thống kê điểm số trung bình (Mean) sẽ được in trực tiếp làm tài liệu minh chứng đưa vào chương 4 của Báo cáo đồ án.
