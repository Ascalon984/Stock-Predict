# IDX Stock Analytics & Predictive System

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" alt="License">
</p>

---

## Ringkasan Eksekutif
Aplikasi analitik saham canggih yang dirancang khusus untuk **Pasar Saham Indonesia (IDX)**. Sistem ini menggabungkan ketepatan statistik **SARIMA** dengan kekuatan **LSTM Deep Learning** untuk memberikan prediksi harga, analisis sentimen pasar, dan wawasan teknikal secara *real-time*.

## Fitur Utama

| Fitur | Deskripsi | Teknologi |
| :--- | :--- | :--- |
| **Hybrid AI Engine** | Kombinasi SARIMA (Tren Linier) & LSTM (Non-Linier) | TensorFlow, Statsmodels |
| **Sentiment Scoring** | Label "Bearish" hingga "Bullish" otomatis | Gradient Smoothing Algo |
| **Indicator Impact** | Breakdown kontribusi indikator dalam bentuk Pie Chart | Plotly.js |
| **Real-time Data** | Sinkronisasi data IDX via Yahoo Finance API | WebSocket / Async API |
| **Confidence Area** | Visualisasi ketidakpastian harga (Monte Carlo Dropout) | Deterministic Seeding |

---

## Arsitektur Teknis

### Backend (Intelligence Layer)
* **Core:** FastAPI (High-performance Async).
* **Models:** `LSTM` untuk menangkap volatilitas non-linear & `SARIMA` untuk tren musiman.
* **Data Engineering:** Pandas & NumPy untuk manipulasi *time-series*.
* **Database:** SQLite untuk caching hasil prediksi guna meminimalkan latensi API.

### Frontend (User Experience Layer)
* **Core:** Next.js 15 (App Router).
* **Styling:** Tailwind CSS dengan dukungan *Responsive Dark Mode*.
* **State Management:** TanStack Query (React Query) untuk efisiensi fetching & caching data.
* **Visualization:** Financial charts interaktif menggunakan Plotly.js.

---

## Antarmuka Sistem (User Interface)

### Dashboard Utama
* **Global Search:** Cari emiten (misal: `BBCA`, `TLKM`) otomatis suffix `.JK`.
* **Live Marquee:** Ticker *Top Gainers/Losers* hari ini.
* **Stock Highlight:** Ringkasan cepat harga terakhir dan persentase perubahan.

### Panel Analisis Saham
* **Interaktif Chart:** Zoom, Pan, dan switch timeframe (1D, 1W, 1M, 3M, 6M, 1Y).
* **Prediction Tab:** Garis proyeksi masa depan dengan area arsiran tingkat keyakinan AI.
* **Sentiment Card:** Narasi cerdas mengenai kondisi pasar (misal: "RSI Netral, AI Bullish").

---

Organized by **[Aditya Tri Prasetyo]** • 2026 | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/aditya-tri-prasetyo-7b3a0a396) [![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://instagram.com/aditya.trisetya)

## Cara Menjalankan Proyek (How to Run)

Ikuti langkah-langkah di bawah ini untuk menjalankan sistem di lingkungan lokal Anda.

### 1. Kloning Repositori
```bash
git clone [https://github.com/Ascalon984/Stock-Predict.git](https://github.com/Ascalon984/Stock-Predict.git)
cd Stock-Predict

### 2. Setup Backend (Python)
cd backend
# Membuat environment virtual
python -m venv venv

# Aktivasi Environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Dependensi & Jalankan
pip install -r requirements.txt
python run.py --dev

### 3. Setup Frontend (Next.js)
cd ../frontend
npm install
npm run dev
