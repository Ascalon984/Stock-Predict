# Dokumentasi Sistem Analisis & Prediksi Saham (IDX Edition)

## 📋 Ringkasan Ekesekutif
Aplikasi ini adalah platform analitik saham canggih yang dirancang khusus untuk pasar saham Indonesia (IDX). Sistem ini menggabungkan ketepatan model statistik klasik (**SARIMA**) dengan kekuatan pembelajaran mesin modern (**LSTM Deep Learning**) untuk memberikan prediksi harga yang akurat, analisis sentimen pasar, dan wawasan teknikal yang mendalam secara real-time.

---

## 🌟 Fitur Utama

### 1. Hybrid AI Prediction Engine (SARIMA + LSTM)
Inti dari sistem ini adalah model hybrid yang bekerja secara tandem:
- **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**: Menangani tren linier dan musiman (seasonality) jangka panjang.
- **LSTM (Long Short-Term Memory)**: Jaringan saraf tiruan yang menangkap pola non-linear kompleks dan volatilitas pasar.
- **Ensemble Cerdas**: Hasil kedua model digabungkan dengan pembobotan dinamis berdasarkan performa terkini masing-masing model.
- **Ketidakpastian Terukur (Uncertainty Quantification)**: Sistem menampilkan "Confidence Interval" (area arsiran) yang menunjukkan rentang kemungkinan harga, dihitung menggunakan *Monte Carlo Dropout* yang distabilkan (Deterministic Seeding) agar konsisten saat dilihat berulang kali.

### 2. Analisis Indikator Teknikal Komprehensif
Sistem secara otomatis menghitung dan menganalisis indikator utama:
- **Trend**: Simple Moving Average (SMA 20/50/200), EMA.
- **Momentum**: RSI (Relative Strength Index) dengan deteksi Overbought/Oversold.
- **Volatilitas**: Bollinger Bands (Upper/Lower/Middle) dan ATR.
- **Oscillator**: MACD (Moving Average Convergence Divergence) dengan deteksi Crossover.

### 3. Smart Market Sentiment Scoring
Mengubah data angka rumit menjadi sinyal yang mudah dibaca:
- Sistem memberikan label sentimen mulai dari **"Sangat Bearish"** hingga **"Sangat Bullish"**.
- Skor dihitung dari gabungan bobot RSI, MACD, posisi MA, Bollinger Bands, dan hasil prediksi AI.
- Menggunakan algoritma *Gradient Smoothing* untuk mencegah perubahan label yang drastis akibat fluktuasi harga minor.

### 4. Analisis Dampak Indikator (Indicator Impact)
Visualisasi canggih (Pie Chart) yang menjawab pertanyaan: *"Mengapa sentimennya Bullish?"*
- Menampilkan persentase kontribusi setiap faktor terhadap keputusan akhir.
- **Contoh**: "Forecast AI menyumbang 40% keputusan, RSI 30%, MACD 20%".
- Membantu trader memahami driver utama pergerakan harga saat ini.

### 5. Data Real-time & WebSocket
- Terhubung langsung dengan data pasar IDX (via Yahoo Finance API).
- **Mode Intraday**: Mendukung grafik 1 menit, 5 menit, dan 1 jam.
- **Konsistensi Waktu**: Semua data dikonversi secara presisi ke Waktu Indonesia Barat (WIB/GMT+7).

---

## 🖥️ Panduan Antarmuka (User Interface)

### Halaman Utama (Dashboard)
1.  **Search Bar**: Cari kode emiten IDX (contoh: `BBCA`, `TLKM`, `GOTO`). Sistem otomatis menambahkan suffix `.JK`.
2.  **Market Summary**: Ticker berjalan (marqee) menampilkan saham-saham *Top Gainers/Losers* hari ini.
3.  **Stock Highlight**: Kartu cepat menampilkan harga terakhir dan perubahan persentase.

### Panel Analisis Saham
1.  **Price Chart**:
    *   Interaktif (Zoom, Pan, Hover).
    *   Beralih Timeframe: 1D, 1W, 1M, 3M, 6M, 1Y.
    *   Beralih Interval: 1m, 5m, 1h, 1d (untuk analisis intraday).
2.  **Prediction Tab**:
    *   Melihat garis prediksi masa depan (garis putus-putus).
    *   Area arsiran menunjukkan tingkat keyakinan AI.
3.  **Sentiment Card**:
    *   Label besar (misal: "BULLISH").
    *   Skor kepercayaan (Confidence Score).
    *   Penjelasan naratif ("RSI dalam kondisi netral, namun AI memprediksi kenaikan").

---

## 🛠️ Arsitektur Teknis

### Backend (Python/FastAPI)
- **Framework**: FastAPI (Async Performance tinggi).
- **ML Libraries**: TensorFlow/Keras (LSTM), Statsmodels/Pmdarima (SARIMA), Scikit-learn.
- **Data Engineering**: Pandas & NumPy untuk manipulasi time-series.
- **Database**: SQLite (untuk caching hasil prediksi dan meminimalkan latensi).

### Frontend (TypeScript/Next.js)
- **Framework**: Next.js 14 (App Router).
- **Styling**: Tailwind CSS (Desain Responsif & Dark Mode).
- **State Management**: React Query (TanStack Query) untuk fetching data & caching.
- **Charting**: Plotly.js (Visualisasi data keuangan standar industri).

---

## 🚀 Panduan Instalasi & Setup

### Prasyarat
- Python 3.9 atau lebih baru.
- Node.js 18 atau lebih baru.

### 1. Menjalankan Backend
```bash
cd backend
# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install dependensi
pip install -r requirements.txt

# Jalankan server
python run.py --dev
```
*Server berjalan di `http://localhost:8000`*

### 2. Menjalankan Frontend
```bash
cd frontend
# Install library
npm install

# Jalankan mode development
npm run dev
```
*Aplikasi dapat diakses di `http://localhost:3000`*

---

## 🧪 Catatan Pengembang
- **Konsistensi AI**: Model LSTM menggunakan *Context-Aware Seeding*. Jika Anda merefresh halaman pada timeframe yang sama, hasil prediksi akan tetap sama (deterministik) untuk menjaga kepercayaan pengguna, namun tetap probabilistik secara matematis.
- **Cache**: Hasil prediksi disimpan (cache) selama 60 menit untuk performa. Gunakan tombol "Refresh Analysis" di UI untuk memaksa prediksi ulang.
