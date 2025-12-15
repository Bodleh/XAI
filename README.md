# XAI untuk Prediksi Keberhasilan Pemasaran Bank

Sistem klasifikasi untuk memprediksi apakah nasabah akan berlangganan deposito berjangka (Bank Marketing, OpenML data_id=1461). Model dilengkapi tiga metode XAI (LIME, SHAP, PFI) agar setiap prediksi bisa dijelaskan secara lokal dan global.

## Tujuan
- Memberi alat bantu bagi tim marketing untuk memprioritaskan prospek dengan alasan yang transparan.
- Menunjukkan bagaimana XAI (post-hoc) memberi konteks pada keputusan model (fitur apa yang mendorong/menahan prediksi).

## Teknik yang dipakai
- Data: OpenML bank marketing (kelas 1 = tidak, kelas 2 = berlangganan).
- Preprocessing: imputasi median (numerik), imputasi most_frequent (kategori), skala numerik, one-hot kategori.
- Model: RandomForestClassifier (300 trees, random_state=42).
- XAI: LIME (local), SHAP (local), Permutation Feature Importance / PFI (global).

## Cara menjalankan
1) Install dependency: `py -3 -m pip install -r requirements.txt`
2) Jalankan satu contoh interaktif (prediksi + LIME/SHAP/PFI untuk 1 sampel):  
   `py -3 main.py`
3) Jalankan batch evaluasi 5–10 sampel + XAI (default 8):  
   `py -3 eval_harness.py`  
   Opsi lain: `py -3 -c "from eval_harness import run_batch_explanations; run_batch_explanations(sample_count=10, top_k=8, random_state=42)"`

## Output yang dihasilkan
- `reports/samples_predictions.csv`: true vs pred + probabilitas per kelas (pilih 5–10 baris untuk slide).
- `reports/explanations.json`: detail per sampel (top fitur LIME & SHAP dengan kontribusi/arah).
- `reports/global_pfi.csv`: fitur paling penting secara global (gunakan nama asli: duration, poutcome, month, contact, pdays, dll.).

## Struktur repositori
- `model/training.py`: load data OpenML, preprocessing, train RandomForest, cetak classification report.
- `xai/explainers.py`: helper LIME/SHAP/PFI untuk pipeline ini.
- `xai/pipeline.py`: alur prediksi + cetak tiga penjelasan untuk satu sampel.
- `eval_harness.py`: generate laporan batch (CSV/JSON) untuk presentasi.
- `main.py`: entry point contoh interaktif satu sampel.
- `reports/`: output otomatis setelah menjalankan batch.

## Ringkasan performa (run terakhir)
- Akurasi ≈ 0.91; macro F1 ≈ 0.73; recall kelas 2 ≈ 0.41 (imbalance → bias ke kelas 1).
- PFI top-5: duration » poutcome > month > contact > pdays.
