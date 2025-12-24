# Kalkulator Metode Numerik (Numerical Methods Calculator)

Project Akhir Mata Kuliah Metode Numerik  
**Semester 5 - Informatika - Petra Christian University**

Aplikasi web berbasis Python (Flask) untuk menyelesaikan masalah matematika menggunakan algoritma numerik. Aplikasi ini dirancang untuk menampilkan hasil perhitungan beserta **tabel iterasi** dan **analisis error** secara transparan.

## Fitur Utama

### 1. Pencarian Akar Persamaan (Root Finding)
Menyelesaikan persamaan non-linear $f(x) = 0$.
* **Metode:**
    * Bisection (Metode Bagi Dua)
    * Newton-Raphson (Metode Terbuka)
* **Output:**
    * Tabel iterasi lengkap ($x$, $f(x)$, $f'(x)$, Error).
    * Deteksi divergensi otomatis.

### 2. Sistem Persamaan Linear (SPL)
Menyelesaikan sistem persamaan linear $Ax = B$.
* **Metode:**
    * Gauss-Jordan Elimination
* **Output:**
    * Langkah-langkah eliminasi matriks (*Matrix Snapshot* per step).
    * Solusi akhir ($x_1, x_2, \dots, x_n$).
    * Fitur *Partial Pivoting* untuk menangani elemen pivot 0.

---

## Requirements

Pastikan komputer Anda sudah terinstall:
1.  **Python 3.x** (Cek dengan `python --version`)
2.  **pip** (Package manager Python)

---

## Cara Instalasi & Menjalankan (Step-by-Step)

Ikuti langkah ini untuk menjalankan program di komputer lokal Anda:

### Langkah 1: Clone atau Download
Download folder project ini, lalu buka terminal/CMD di dalam folder tersebut.

### Langkah 2: Install Library
Jalankan perintah berikut untuk menginstall library yang dibutuhkan (`Flask` dan `Sympy`):

```bash
pip install -r requirements.txt

