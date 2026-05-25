# BẢNG TIẾN ĐỘ VÀ DANH SÁCH CÔNG VIỆC HOÀN THIỆN ĐỒ ÁN TTS & SPEECH ENHANCEMENT

Chào Phát, dưới đây là toàn bộ lộ trình và danh sách các việc cần làm (To-Do List) được thiết kế chi tiết cho bạn và Đăng để hoàn thiện từ bước xử lý dữ liệu cuối cùng, huấn luyện mô hình VITS cho đến lúc viết xong quyển báo cáo nộp Hội đồng PTIT. 

Bạn có thể lưu file `.md` này vào các công cụ như Notion, Obsidian, VS Code hoặc GitHub để tích chọn `[x]` theo dõi tiến độ hàng ngày.

---

## 📌 GIAI ĐOẠN 1: CHỐT SỔ & HỢP NHẤT DATASET SẠCH (Dự kiến: 3 - 5 ngày)
*Mục tiêu: Đóng gói toàn bộ công sức duyệt data của Phát và Đăng thành một bộ nguyên liệu duy nhất, không lỗi số, không trùng lặp.*

- [ ] **1.1. Chạy rà soát số lần cuối (Cả ca A và ca B)**
  - [ ] Mở tool quét số 2-trong-1 đã viết, chạy cho **Sổ ỔN** (`metadata_verified_A.csv` và `_B.csv`) để dọn sạch các chữ số lọt lưới.
  - [ ] Chạy tiếp cho **Sổ ỒN** (`metadata_noise_A.csv` và `_B.csv`) để đảm bảo phần text của các file cần lọc nhiễu cũng đã được dịch hoàn toàn sang chữ.
- [ ] **1.2. Hợp nhất dữ liệu toàn hệ thống (Merge Master)**
  - [ ] Viết script gộp `metadata_verified_A` và `metadata_verified_B` thành file tổng `master_verified.csv`.
  - [ ] Gộp `metadata_noise_A` và `metadata_noise_B` thành file tổng `master_noise.csv`.
  - [ ] *Lưu ý:* Sử dụng cấu trúc dữ liệu `set()` trong Python để tự động triệt tiêu các dòng trùng lặp (nếu có) trong quá trình gộp sổ Drive.
- [ ] **1.3. Thống kê thông số Dataset đầu vào (Phục vụ viết báo cáo)**
  - [ ] Đếm tổng số câu chuẩn (Ổn) và câu cần lọc nhiễu (Ồn).
  - [ ] Viết đoạn code ngắn dùng thư viện `pydub` tính tổng thời lượng (bao nhiêu giờ, bao nhiêu phút âm thanh) của tập dữ liệu sạch.

---

## 🔊 GIAI ĐOẠN 2: KHỬ NHIỄU DATA BẰNG DEEPFILTERNET (Dự kiến: 2 - 3 ngày)
*Mục tiêu: "Rửa" sạch toàn bộ các file audio trong thư mục `wavs_need_denoise` để đưa về tiêu chuẩn phòng thu.*

- [ ] **2.1. Cài đặt môi trường DeepFilterNet trên Colab**
  - [ ] Cài đặt thư viện: `pip install deepfilternet`.
- [ ] **2.2. Xử lý lọc nhiễu hàng loạt (Batch Processing)**
  - [ ] Viết script quét qua toàn bộ thư mục `wavs_need_denoise`, đẩy từng file qua model `DeepFilterNet` để triệt tiêu tiếng quạt, tiếng gió, tiếng xì xèo.
  - [ ] Xuất file âm thanh đã sạch nhiễu vào một thư mục mới tên là `wavs_denoised`.
- [ ] **2.3. Thu thập minh chứng thực nghiệm (Quan trọng cho chương Đánh giá)**
  - [ ] Chọn ra 2-3 file audio bị ồn điển hình nhất.
  - [ ] Giữ lại bản gốc (Before) và bản sau khi lọc (After) để làm tài liệu đối chiếu phổ âm và làm file demo chạy slide lúc bảo vệ.

---

## 🤖 GIAI ĐOẠN 3: TIỀN XỬ LÝ CHO VITS & CHIA TẬP TRAIN/VAL (Dự kiến: 3 - 4 ngày)
*Mục tiêu: Định dạng lại dữ liệu theo đúng "ngôn ngữ" mạng Nơ-ron VITS yêu cầu.*

- [ ] **3.1. Đồng bộ và Gộp kho Audio**
  - [ ] Di chuyển (hoặc copy) toàn bộ file audio sạch từ `wavs_denoised` vào chung thư mục `wavs` gốc.
  - [ ] Gộp file `master_noise.csv` vào đuôi file `master_verified.csv` để tạo thành file siêu dữ liệu tối cao: `final_metadata.csv`.
- [ ] **3.2. Chia tập dữ liệu (Train/Val Split)**
  - [ ] Viết script chia `final_metadata.csv` theo tỷ lệ **95% cho tập Huấn luyện** và **5% cho tập Xác thực**.
  - [ ] Xuất ra hai tệp cấu trúc dạng:
    - [ ] `train.txt` (Định dạng: `đường_dẫn_audio|nội_dung_text`)
    - [ ] `val.txt` (Định dạng tương tự)
- [ ] **3.3. Tích hợp Module Ngữ âm G2P (Grapheme-to-Phoneme)**
  - [ ] Nghiên cứu sử dụng thư viện `viphoneme` hoặc bộ parser IPA tiếng Việt để chuyển đổi text thành chuỗi ngữ âm ký hiệu (Phonemes), xử lý chuẩn xác hệ thống 6 thanh điệu của tiếng Việt trước khi đẩy vào kiến trúc VITS.

---

## 🚀 GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH VITS (Dự kiến: 7 - 10 ngày)
*Mục tiêu: Tiến hành Fine-tune để AI học giọng đọc đặc trưng từ bộ dataset của nhóm.*

- [ ] **4.1. Thiết lập Mã nguồn và Cấu hình hệ thống**
  - [ ] Clone mã nguồn kiến trúc VITS chuẩn từ GitHub về Google Colab/Drive.
  - [ ] Cấu hình file `config.json`: Điền đúng tần số lấy mẫu (Sample Rate, thường là 22050Hz), Batch size phù hợp với VRAM của GPU, và trỏ đường dẫn đến `train.txt`, `val.txt`.
- [ ] **4.2. Khởi chạy chiến thuật Fine-tuning (Huấn luyện chuyển giao)**
  - [ ] Tải Pre-trained model VITS tiếng Việt nền tảng (Base Model).
  - [ ] Nạp trọng số (Weights) cũ và bắt đầu cho mô hình học bộ dữ liệu sạch của bạn.
  - [ ] Theo dõi biểu đồ suy hao (Loss curve) trên Tensorboard để đảm bảo mô hình hội tụ tốt, không bị hiện tượng học vẹt (Overfitting).
- [ ] **4.3. Kiểm thử giọng đọc (Inference Testing)**
  - [ ] Gõ một vài câu văn mới tinh (AI chưa từng được học) vào file test để xem mô hình tổng hợp ra giọng đọc có mượt mà, ngắt nghỉ tự nhiên và đúng dấu tiếng Việt không.

---

## 📊 GIAI ĐOẠN 5: ĐÁNH GIÁ THỰC NGHIỆM & TRÍCH XUẤT BIỂU ĐỒ (Dự kiến: 3 - 5 ngày)
*Mục tiêu: Tạo ra các bằng chứng khoa học trực quan bằng hình ảnh và số liệu để đưa vào quyển báo cáo.*

- [ ] **5.1. Vẽ biểu đồ phổ âm Spectrogram (Minh chứng cốt lõi)**
  - [ ] Viết đoạn code ngắn sử dụng thư viện `librosa` và `matplotlib` để trích xuất phổ đồ âm thanh.
  - [ ] Vẽ đồ thị đặt cạnh nhau (Side-by-Side):
    - [ ] **Hình 1:** Phổ âm file thô bị nhiễu (Chỉ rõ vệt mờ dải tần thấp của tiếng quạt/máy lạnh).
    - [ ] **Hình 2:** Phổ âm sau khi lọc qua `DeepFilterNet` (Nền đen tĩnh lặng, dải sóng giọng người giữ nguyên độ sắc nét).
- [ ] **5.2. Lập Bảng so sánh Benchmarking (Đánh giá định lượng)**
  - [ ] Thu thập các thông số kỹ thuật để lập bảng đối chiếu giữa các phương pháp: `DeepFilterNet` vs `noisereduce` (DSP truyền thống) vs `RNNoise` theo các tiêu chí: Tốc độ xử lý (RTF), dung lượng mô hình, và độ méo tiếng.
- [ ] **5.3. Khảo sát điểm chất lượng giọng đọc (MOS - Mean Opinion Score)**
  - [ ] Lấy 5 mẫu audio AI sinh ra, gửi cho các bạn cùng lớp nghe thử và chấm điểm từ 1 đến 5 về độ tự nhiên. Tính toán điểm trung bình để đưa vào chương Thực nghiệm.

---

## 📝 GIAI ĐOẠN 6: SOẠN THẢO QUYỂN BÁO CÁO ĐỒ ÁN (Dự kiến: Kéo dài song song đến cuối)**
*Mục tiêu: Đóng gói toàn bộ kiến thức và thành quả thành quyển báo cáo Word chuẩn format học thuật của PTIT.*

- [ ] **6.1. Hoàn thiện Chương 1 & Chương 2 (Cơ sở lý thuyết)**
  - [ ] Viết lý thuyết xử lý tín hiệu số (Sóng âm, tần số, Mel-spectrogram).
  - [ ] Viết kiến trúc mô hình lọc nhiễu và kiến trúc End-to-End của mô hình VITS.
- [ ] **6.2. Hoàn thiện Chương 3 (Thiết kế hệ thống / Data Pipeline)**
  - [ ] Đưa sơ đồ khối luồng xử lý dữ liệu và thuật toán phân chia ca A/B chẵn lẻ vào.
  - [ ] Giải thích chi tiết module tự động dịch số sang chữ (`num2words` + `RegEx`).
- [ ] **6.3. Hoàn thiện Chương 4 & 5 (Thực nghiệm & Kết luận)**
  - [ ] Chèn các biểu đồ Spectrogram Before/After và bảng số liệu so sánh mô hình vào.
  - [ ] Phân tích kết quả, nêu hướng phát triển đề tài (ví dụ: tối ưu mô hình chạy thời gian thực trên thiết bị nhúng).
- [ ] **6.4. Rà soát Format & Thiết kế Slide bảo vệ**
  - [ ] Kiểm tra mục lục tự động, căn lề Word, định dạng font, bảng biểu, danh mục tài liệu tham khảo.
  - [ ] Thiết kế Slide thuyết trình ngắn gọn (15-20 slide), chèn sẵn các file âm thanh demo để bật trực tiếp trước Hội đồng chấm thi.
