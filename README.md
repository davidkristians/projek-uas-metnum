# 🧮 Numerical Methods Pro (Kalkulator Metode Numerik)

Project ini adalah aplikasi kalkulator numerik berbasis **Python (Streamlit)** yang dikembangkan untuk memenuhi Tugas Akhir Mata Kuliah Metode Numerik (Semester 5 Informatika). Aplikasi ini dirancang untuk menyelesaikan berbagai masalah matematika numerik dengan presisi tinggi dan input yang fleksibel (University Grade).

## 🌟 Fitur Utama

Aplikasi ini mencakup 6 modul utama sesuai silabus perkuliahan:

1.  **Akar Persamaan (Roots):**
    * Metode: Bisection, Regula Falsi, Newton-Raphson, Secant.
    * Fitur: Visualisasi grafik f(x), tabel iterasi lengkap, dan deteksi error.
2.  **Sistem Persamaan Linear (SPL):**
    * Metode: Eliminasi Gauss, Gauss-Jordan, LU Decomposition, Jacobi, Gauss-Seidel.
    * Fitur: Input matriks dinamis, deteksi pivot nol, cek konvergensi diagonal dominan.
3.  **Interpolasi:**
    * Metode: Lagrange, Newton Polynomial.
    * Fitur: Visualisasi kurva interpolasi vs titik data asli.
4.  **Integrasi Numerik:**
    * Metode: Trapezium, Simpson 1/3, Simpson 3/8.
    * Fitur: Perbandingan otomatis dengan **Solusi Exact (Analitik)** menggunakan kalkulus simbolik untuk menghitung Error absolut.
5.  **Diferensiasi Numerik:**
    * Metode: Forward, Backward, Central Difference.
    * Fitur: Step size (h) fleksibel (bisa sangat kecil seperti 1e-6) dan perbandingan terhadap nilai exact.
6.  **Persamaan Diferensial (ODE):**
    * Metode: Euler, Heun, Runge-Kutta 4 (RK4).
    * Fitur: Grafik perbandingan multi-metode dalam satu plot untuk analisis akurasi.

### 🧠 Fitur Canggih: Smart Input Parser
Aplikasi ini dilengkapi dengan **Regex Parser** khusus. Anda tidak perlu menulis syntax Python kaku.
* Ketikan `2x` -> otomatis dibaca `2*x`
* Ketikan `e^-x` -> otomatis dibaca `exp(-x)`
* Ketikan `sin^2(x)` -> otomatis dibaca `sin(x)**2`

## 🛠️ Prasyarat (Requirements)

Pastikan di komputer Anda sudah terinstall:
* **Python 3.8** atau lebih baru.
* **Visual Studio Code** (Recommended).

---

## 🚀 Cara Install & Run (Panduan Cepat)

Ikuti langkah ini satu per satu di terminal (Command Prompt / PowerShell / VS Code Terminal):

### 1. Buat Virtual Environment
Agar library tidak bentrok dengan system lain.

python -m venv .venv

### 2. Aktifkan Virtual Environment
Jalankan perintah ini. Jika berhasil, akan muncul tulisan `(.venv)` berwarna hijau di kiri terminal.

**Untuk Windows:**
.\.venv\Scripts\Activate

*(Catatan: Jika muncul error merah "UnauthorizedAccess", ketik perintah ini dulu: `Set-ExecutionPolicy RemoteSigned -Scope Process`, lalu coba lagi)*.

**Untuk Mac/Linux:**
source .venv/bin/activate

### 3. Install Library (Wajib)
Pastikan koneksi internet aktif. Kita butuh `scipy` untuk fitur SPL dan `sympy` untuk kalkulus.

pip install streamlit numpy pandas sympy matplotlib scipy

### 4. Jalankan Aplikasi
Gunakan perintah ini untuk membuka kalkulator di browser:

streamlit run app.py

*(Jika perintah di atas tidak dikenali, gunakan alternatif di bawah ini):*
python -m streamlit run app.py

## 🧪 Contoh Test Case (Untuk Demo/Pengujian)

Gunakan input ini saat presentasi untuk memastikan aplikasi berjalan 100% benar dan akurat:

| Modul | Input Soal | Parameter | Ekspektasi Hasil |
| :--- | :--- | :--- | :--- |
| **Akar (Newton)** | `exp(-x) - x` | Tebakan: `0` | **0.567143** |
| **Integrasi (Simpson)** | `4 / (1+x^2)` | a=`0`, b=`1` | **3.14159** (Pi) |
| **Diferensiasi** | `x^3` | x=`2`, h=`0.0001` | **12.0000** |
| **ODE (RK4)** | `-2*y + x` | x0=`0`, y0=`1`, Target=`1` | Grafik RK4 paling halus/akurat |

---
