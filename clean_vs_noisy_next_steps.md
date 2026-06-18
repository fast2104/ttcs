# Ke hoach tiep theo: So sanh audio truoc va sau khi clean noise

## Muc tieu hien tai

Muc tieu chinh cua de tai khong can tiep tuc train TTS de ra giong noi tot. Huong dung luc nay la danh gia xem viec clean/loai bo am thanh on co cai thien chat luong audio hay khong.

Can so sanh hai nhom audio:

- `noisy_wavs`: audio truoc khi clean, con on/nhac nen/tap am.
- `clean_wavs`: audio sau khi clean, dang nam trong `merged_dataset/wavs` hoac folder clean tuong ung.

Ket qua can thu duoc:

- Bang chi so tung file.
- Bang trung binh toan bo tap.
- Anh spectrogram truoc/sau.
- Diem MOS nghe nguoi that neu co thoi gian.
- Nhan xet dua vao bao cao.

## 1. Dung huong train VITS hien tai

Repo VITS hien tai chi la implementation gian luoc. Khi train from scratch, audio inference hien chi ra tieng re, do:

- Khong tai duoc checkpoint pre-trained vi HuggingFace bao `401 Unauthorized`.
- Model thieu alignment/duration chuan cua VITS.
- KL loss bi lech shape va phai clamp ve 0, nen khong con phan rang buoc latent dung nghia.

Viec can lam:

- Dung train dai neu muc tieu chi la so sanh clean vs noisy.
- Giu lai checkpoint/log hien co nhu mot thu nghiem phu neu can viet phan han che.
- Khong dung loss VITS lam chi so chinh cho de tai clean noise.

## 2. Chuan bi du lieu so sanh

Can co hai folder doc duoc trong Colab:

```text
/content/drive/MyDrive/TTS_Dataset_Project/noisy_wavs
/content/drive/MyDrive/TTS_Dataset_Project/merged_dataset/wavs
```

Trong do:

- `noisy_wavs` la file truoc clean.
- `merged_dataset/wavs` la file sau clean.
- Tot nhat cac file truoc/sau co cung ten, vi script se match theo ten file.

Kiem tra trong Colab:

```bash
!ls "/content/drive/MyDrive/TTS_Dataset_Project/noisy_wavs" | head
!ls "/content/drive/MyDrive/TTS_Dataset_Project/merged_dataset/wavs" | head
```

Neu hai folder khong cung ten file, can tao bang mapping hoac copy/doi ten mot tap mau 20-50 cap de so sanh.

## 3. Chay danh gia chi so audio

Tao output folder:

```text
/content/drive/MyDrive/TTS_Dataset_Project/evaluation_clean_vs_noisy
```

Chay notebook/cell tinh cac chi so:

- `duration_sec`: thoi luong audio.
- `rms`: nang luong trung binh.
- `peak`: bien do cuc dai.
- `noise_floor`: muc nen on uoc luong bang percentile thap cua RMS frame.
- `speech_level`: muc tin hieu giong noi uoc luong.
- `estimated_snr_db`: SNR uoc luong.
- `snr_improvement_db`: muc cai thien SNR sau clean.
- `noise_floor_reduction`: muc giam nen on.

File can tao:

```text
clean_vs_noisy_metrics.csv
summary_metrics.csv
```

Ket qua mong muon:

- `clean_estimated_snr_db` cao hon `noisy_estimated_snr_db`.
- `snr_improvement_db` duong.
- `clean_noise_floor` thap hon `noisy_noise_floor`.

## 4. Ve spectrogram truoc/sau

Chon 5-10 cap audio dai dien:

- Mot cap on nhe.
- Mot cap co nhac nen.
- Mot cap co gio/quat/may lanh.
- Mot cap sau clean tot.
- Mot cap sau clean bi meo tieng neu co.

Xuat anh:

```text
spectrogram_pair_01_*.png
spectrogram_pair_02_*.png
...
```

Dung anh nay cho chuong thuc nghiem:

- Ben trai: audio goc co tap am.
- Ben phai: audio sau clean.
- Nhan xet cac dai tan nhieu nen da giam, vung tieng noi ro hon.

## 5. Lam MOS survey nghe nguoi that

Nen tao folder mau nghe:

```text
mos_samples/noisy
mos_samples/clean
```

Chon 10-20 cap audio. Dat ten an danh de nguoi nghe khong biet file nao la clean:

```text
sample_001_A.wav
sample_001_B.wav
sample_002_A.wav
sample_002_B.wav
```

Nguoi nghe cham 1-5 theo 3 tieu chi:

- Do ro loi noi.
- Muc nhieu nen.
- Chat luong tong the.

Can toi thieu 3-5 nguoi cham neu co the. Ket qua tong hop:

```text
mos_results.csv
```

Bao cao nen co bang:

| Nhom audio | Do ro loi noi | Muc it nhieu | Chat luong tong the |
| --- | ---: | ---: | ---: |
| Truoc clean | ... | ... | ... |
| Sau clean | ... | ... | ... |

## 6. Neu muon co chi so khach quan hon: ASR WER/CER

Neu con thoi gian, dung Whisper hoac ASR tieng Viet de transcript audio truoc/sau:

- Noisy audio -> transcript noisy.
- Clean audio -> transcript clean.
- So voi text goc trong metadata.

Chi so:

- WER: Word Error Rate.
- CER: Character Error Rate.

Ket qua mong muon:

- WER/CER cua audio sau clean thap hon audio truoc clean.

Day la bang chung manh rang clean noise giup loi noi de nhan dien hon.

## 7. Noi dung dua vao bao cao

Chuong dataset/pipeline:

- Mo ta raw data -> slice -> annotate -> classify clean/noisy/trash -> denoise -> merge dataset.
- Thong ke so file sau clean, tong thoi luong, train/val split.

Chuong thuc nghiem:

- Mo ta tap so sanh noisy vs clean.
- Trinh bay bang trung binh SNR/noise floor.
- Chen spectrogram truoc/sau.
- Chen MOS survey neu co.
- Neu co ASR, chen WER/CER.

Phan han che:

- Khong co clean ground truth tuyet doi nen SNR chi la uoc luong.
- Ket qua MOS phu thuoc nguoi nghe va so mau.
- Thu nghiem VITS from scratch chua cho giong tot do thieu pre-trained va implementation gian luoc.

## 8. Checklist hanh dong gan nhat

- [ ] Xac dinh folder audio truoc clean tren Drive.
- [ ] Xac dinh folder audio sau clean tren Drive.
- [ ] Kiem tra hai folder co match ten file hay khong.
- [ ] Chay script tao `clean_vs_noisy_metrics.csv`.
- [ ] Tao `summary_metrics.csv`.
- [ ] Xuat 5-10 anh spectrogram pair.
- [ ] Chon 10-20 cap audio cho MOS.
- [ ] Thu thap diem MOS tu 3-5 nguoi.
- [ ] Tong hop bang ket qua cho bao cao.
- [ ] Viet nhan xet: sau clean SNR tang, noise floor giam, nguoi nghe cham diem tot hon.

