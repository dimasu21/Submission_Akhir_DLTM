# Prediksi Harga Bitcoin dengan Arsitektur Seq2Seq LSTM
**Multivariate Multi-Horizon Time Series Forecasting**

---

## Tentang Project Ini

Project ini lahir dari satu pertanyaan sederhana: bisakah kita memprediksi pergerakan harga Bitcoin untuk 24 jam ke depan, hanya dengan melihat pola 72 jam terakhir?

Bitcoin dikenal sebagai aset yang sangat fluktuatif. Harganya bisa naik atau turun secara drastis dalam hitungan jam, dipengaruhi oleh banyak faktor yang saling tumpang tindih. Memprediksi harganya bukan perkara mudah, dan tidak ada model yang bisa menjamin akurasi sempurna. Tapi di sinilah letak tantangannya, dan di sinilah deep learning punya peran yang menarik untuk dimainkan.

Project ini menggunakan pendekatan **Multivariate Multi-Horizon Time Series Forecasting**, artinya model tidak hanya melihat satu variabel (harga penutupan saja), melainkan beberapa variabel sekaligus, dan model juga tidak hanya memprediksi satu titik waktu ke depan, melainkan beberapa langkah waktu sekaligus (24 langkah ke depan).

Dua arsitektur model dibangun dan dibandingkan:
- **Baseline LSTM** -- model yang lebih sederhana, dilatih dengan Keras `model.fit()`.
- **Seq2Seq LSTM** -- model yang lebih kompleks, menggunakan arsitektur encoder-decoder dengan Teacher Forcing, dilatih menggunakan custom training loop (`tf.GradientTape`).

Kedua model berhasil mencapai **MAE < 0.015** pada data ternormalisasi, yang menunjukkan bahwa prediksi model sangat dekat dengan nilai sebenarnya.

---

## Dataset

### Sumber dan Deskripsi

Dataset yang digunakan adalah **Multivariate Crypto Data Hourly (Bitcoin)**, yang mencakup data per jam dari **September 2017 hingga Agustus 2019**. Totalnya ada **16.666 baris data**, masing-masing merepresentasikan satu jam perdagangan Bitcoin.

Dataset ini bukan hanya berisi harga, tapi juga sejumlah indikator teknikal yang biasa digunakan oleh para trader dan analis pasar. Gabungan fitur inilah yang membuat model punya "bahan baku" lebih kaya untuk belajar mengenali pola.

### Kolom-kolom dalam Dataset

| Kolom | Penjelasan |
|---|---|
| **Date** | Waktu pencatatan data (per jam). Kolom ini digunakan sebagai index, bukan sebagai fitur model. |
| **Close** | Harga penutupan Bitcoin dalam USD pada jam tersebut. Ini adalah **target utama** yang ingin diprediksi. Rentangnya dari sekitar $3.172 hingga $19.709 selama periode data ini. |
| **Volume USDT** | Volume perdagangan dalam satuan USDT (stablecoin yang nilainya setara 1 dolar AS). Kolom ini menunjukkan seberapa aktif pasar pada jam tersebut. Volume tinggi biasanya menandakan minat pasar yang besar, baik karena euforia maupun kepanikan. |
| **RSI (Relative Strength Index)** | Indikator momentum yang mengukur kecepatan dan perubahan pergerakan harga. Nilainya berkisar antara 0 hingga 100. Dalam dataset ini, RSI berkisar antara 37 dan 64. Nilai di atas 70 umumnya dianggap overbought (harga terlalu tinggi), dan di bawah 30 dianggap oversold (harga terlalu rendah). |
| **MACD_Hist (MACD Histogram)** | Selisih antara garis MACD dan garis sinyal. MACD sendiri adalah indikator yang menunjukkan hubungan antara dua rata-rata bergerak harga. Histogram positif menandakan momentum naik, histogram negatif menandakan momentum turun. Dalam dataset ini, nilainya berkisar dari -835 hingga +580. |
| **ATR (Average True Range)** | Mengukur volatilitas pasar. ATR yang tinggi berarti pasar sedang sangat bergejolak, ATR yang rendah berarti pasar cenderung tenang. Nilainya berkisar antara 17 hingga 464 dalam dataset ini. |
| **KAMAO** | Indikator osilator berbasis Kaufman Adaptive Moving Average. Indikator ini membantu mendeteksi arah tren sambil menyesuaikan sensitivitasnya terhadap kondisi pasar. Nilainya berkisar dari -23 hingga +30 dalam dataset ini. |

### Kenapa Fitur-fitur Ini Penting?

Setiap fitur punya ceritanya sendiri. Harga penutupan saja tidak cukup untuk memahami apa yang sedang terjadi di pasar. Volume menunjukkan seberapa "ramai" pasar. RSI membantu mengenali apakah harga sudah terlalu tinggi atau terlalu rendah. MACD Histogram menunjukkan momentum. ATR mengukur seberapa liar pergerakan harga. Dan KAMAO membantu model memahami arah tren secara adaptif.

Dengan menggabungkan semua informasi ini, model punya pandangan yang lebih utuh tentang kondisi pasar. Seperti seorang trader yang tidak hanya melihat grafik harga, tapi juga memperhatikan volume, momentum, dan volatilitas sebelum mengambil keputusan.

### Preprocessing Data

Beberapa langkah preprocessing yang dilakukan:
1. **Parsing tanggal** -- kolom tanggal dikonversi ke format datetime dan dijadikan index.
2. **Seleksi kolom numerik** -- hanya kolom numerik yang digunakan, kolom seperti Unix timestamp dibuang.
3. **Handling missing values** -- baris dengan nilai kosong dihapus.
4. **Normalisasi** -- semua fitur dinormalisasi menggunakan MinMaxScaler ke rentang [0, 1]. Ini penting agar semua fitur punya skala yang setara, sehingga model tidak "bingung" karena perbedaan besaran antara harga (ribuan dolar) dan RSI (puluhan).
5. **Windowing** -- data dipotong menjadi jendela-jendela waktu. Setiap sampel terdiri dari 72 langkah waktu (3 hari) sebagai input dan 24 langkah waktu (1 hari) sebagai target prediksi.
6. **Pembagian data** -- data dibagi menjadi set training dan testing.

---

## Arsitektur Model

### Custom Layer

Kedua model menggunakan beberapa custom layer yang dibuat sendiri, bukan menggunakan layer bawaan Keras:
- **CustomDense** -- layer fully connected yang diimplementasikan secara manual.
- **CustomDropout** -- layer dropout untuk regularisasi, mencegah overfitting.
- **CustomMultiHeadAttention** -- mekanisme attention multi-head yang memungkinkan model untuk "memperhatikan" bagian-bagian berbeda dari input secara bersamaan.

### Model Baseline LSTM

Model baseline menggunakan arsitektur berikut:
1. Input shape: (72, 8) -- 72 langkah waktu, 8 fitur per langkah.
2. LSTM layer pertama (128 unit, return sequences).
3. CustomDropout (10%).
4. CustomMultiHeadAttention (4 heads) dengan residual connection.
5. LSTM layer kedua (64 unit).
6. CustomDense (64 unit, aktivasi ReLU).
7. CustomDense (24 unit) -- output untuk 24 langkah prediksi.
8. Reshape ke (24, 1).

Total parameter: **191.320**

Model ini dilatih menggunakan Keras `model.fit()` standar.

### Model Seq2Seq LSTM

Model Seq2Seq menggunakan pendekatan encoder-decoder yang lebih canggih:

**Encoder:**
- 2 layer LSTM bertumpuk (masing-masing 256 unit).
- CustomDropout untuk regularisasi.

**Decoder:**
- CustomDense untuk memproyeksikan input decoder.
- LSTM layer (256 unit) yang diinisialisasi dengan state akhir encoder.
- Cross-Attention (CustomMultiHeadAttention, 4 heads) antara output decoder dan output encoder, dengan residual connection.
- CustomDense untuk proyeksi akhir.
- CustomDropout untuk regularisasi.
- CustomDense (1 unit) sebagai output per langkah waktu.

Total parameter: **1.651.713**

Model ini dibangun menggunakan **Model Subclassing** dan dilatih dengan **tf.GradientTape** (custom training loop). Pada saat training, teknik **Teacher Forcing** digunakan -- model menerima nilai target yang sebenarnya sebagai input decoder, bukan hasil prediksinya sendiri. Ini mempercepat konvergensi training secara signifikan.

Pada saat inferensi, model menggunakan mode **autoregressive** -- memprediksi satu langkah, lalu menggunakan hasil prediksi itu sebagai input untuk langkah berikutnya, dan seterusnya hingga 24 langkah.

---

## Custom Training

### Custom Loss: Weighted Horizon MAE

Loss function yang digunakan bukan MAE biasa, tapi **Weighted Horizon MAE**. Idenya sederhana: prediksi untuk langkah waktu yang lebih jauh ke depan secara alami lebih sulit dan cenderung kurang akurat. Dengan memberikan bobot yang semakin besar pada langkah waktu yang lebih jauh, model "dipaksa" untuk lebih berusaha pada prediksi jangka panjang.

### Custom Callback

Dua callback custom diimplementasikan:
- **Reduce LR on Plateau** -- menurunkan learning rate ketika metrik validasi berhenti membaik.
- **Early Stopping** -- menghentikan training lebih awal jika tidak ada peningkatan, untuk mencegah overfitting.

### Training Loop

Training Seq2Seq menggunakan `tf.GradientTape`, yang memberikan kontrol penuh atas proses forward pass, loss computation, dan backward pass. Ini diperlukan karena arsitektur Seq2Seq dengan Teacher Forcing membutuhkan penanganan khusus yang tidak bisa dilakukan dengan `model.fit()` biasa.

---

## Hasil

Kedua model berhasil mencapai target **MAE < 0.015** pada data ternormalisasi. Angka ini menunjukkan bahwa rata-rata kesalahan prediksi model sangat kecil relatif terhadap skala data.

Tiga file model disimpan:
- `model_baseline_LSTM.keras` -- model baseline yang dilatih dengan `model.fit()`.
- `model_seq2seq_LSTM.keras` -- model Seq2Seq setelah training selesai.
- `best_model_seq2seq_LSTM.keras` -- model Seq2Seq terbaik (berdasarkan validation loss selama training).

---

## Cara Menjalankan

### Prasyarat

- Python 3.10+
- Jupyter Notebook atau Google Colab
- GPU direkomendasikan untuk mempercepat proses training

### Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### Dependensi Utama

| Library | Versi Minimum | Kegunaan |
|---|---|---|
| TensorFlow | >= 2.15.0 | Framework deep learning utama |
| NumPy | >= 1.24.0 | Operasi numerik dan array |
| Pandas | >= 2.0.0 | Manipulasi dan analisis data |
| Matplotlib | >= 3.7.0 | Visualisasi data dan hasil |
| Seaborn | >= 0.12.0 | Visualisasi statistik |
| scikit-learn | >= 1.3.0 | Preprocessing (MinMaxScaler) |
| statsmodels | >= 0.14.0 | Analisis time series (dekomposisi, ACF/PACF) |

### Menjalankan Notebook

1. Pastikan file `Bitcoin3.csv` berada di direktori yang sama dengan notebook.
2. Buka `Dimas_Tri_M_Submission_Akhir_DLTM.ipynb` di Jupyter Notebook atau Google Colab.
3. Jalankan semua sel secara berurutan dari atas ke bawah.
4. Jika menggunakan Google Colab, pastikan runtime sudah diset ke GPU untuk performa terbaik.

---

## Struktur File

```
DLTM_Dimas-Tri-M/
|-- Bitcoin3.csv                          # Dataset Bitcoin hourly
|-- Dimas_Tri_M_Submission_Akhir_DLTM.ipynb  # Notebook utama
|-- model_baseline_LSTM.keras             # Model baseline tersimpan
|-- model_seq2seq_LSTM.keras              # Model Seq2Seq tersimpan
|-- best_model_seq2seq_LSTM.keras         # Model Seq2Seq terbaik
|-- requirements.txt                      # Daftar dependensi
|-- README.md                             # Dokumentasi ini
```

---

## Catatan Penutup

Project ini bukan tentang menciptakan "mesin cetak uang" yang bisa memprediksi pasar crypto dengan sempurna. Tidak ada model yang bisa melakukan itu. Yang ingin dicapai di sini adalah memahami bagaimana arsitektur deep learning -- khususnya Seq2Seq dengan LSTM dan attention mechanism -- bisa digunakan untuk menangkap pola-pola dalam data time series yang kompleks dan multivariat.

Setiap keputusan desain dalam project ini punya alasan: kenapa menggunakan 72 jam sebagai window, kenapa memprediksi 24 langkah ke depan, kenapa memilih weighted loss, kenapa menggunakan Teacher Forcing saat training tapi autoregressive saat inferensi. Semua ini adalah bagian dari proses belajar dan bereksperimen dengan time series forecasting yang sesungguhnya.
