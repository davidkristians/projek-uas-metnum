import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import re

# Konfigurasi Halaman
st.set_page_config(page_title="Numerical Methods Pro", layout="wide")

# ==========================================
# FUNGSI BANTUAN (HELPER)
# ==========================================

def preprocess_expression(expr_str):
    """
    Update 1: Membersihkan input agar support user yang mengetik '^' sebagai pangkat
    dan menangani fungsi matematika umum.
    """
    # Ganti ^ dengan **
    expr_str = expr_str.replace("^", "**")
    return expr_str

def get_derivative(func_str):
    """
    Update 2 & 3: Menghitung turunan secara simbolik untuk ditampilkan
    sebagai bukti pemahaman teori.
    """
    x = sp.symbols('x')
    try:
        # Parse string ke sympy expression
        expr = sp.sympify(func_str)
        # Hitung turunan
        diff_expr = sp.diff(expr, x)
        return expr, diff_expr
    except Exception as e:
        return None, None

# ==========================================
# MENU 1: SISTEM PERSAMAAN LINEAR (GAUSS-JORDAN)
# ==========================================
def menu_spl():
    st.header("🧮 Gauss-Jordan Elimination (With Partial Pivoting)")
    st.caption("Solusi Sistem Persamaan Linear (SPL) dengan penanganan Pivot Nol.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        n = st.number_input("Jumlah Variabel (N)", min_value=2, max_value=10, value=3)
    
    st.write("### Masukkan Matriks Augmented")
    st.info("Format: Koefisien variabel diikuti konstanta di kolom terakhir.")

    # Membuat grid input dinamis
    matrix_data = []
    cols = st.columns(n + 1)
    
    # Header label (x1, x2, ..., Konstanta)
    for i, col in enumerate(cols):
        with col:
            if i < n:
                st.markdown(f"**x{i+1}**")
            else:
                st.markdown("**= (Hasil)**")

    # Input values
    for r in range(n):
        row_inputs = []
        cols = st.columns(n + 1)
        for c in range(n + 1):
            with cols[c]:
                # Default value diset 0.0
                val = st.number_input(f"R{r+1}C{c+1}", value=0.0, key=f"mat_{r}_{c}", label_visibility="collapsed")
                row_inputs.append(val)
        matrix_data.append(row_inputs)

    if st.button("Hitung Solusi SPL", type="primary"):
        solve_gauss_jordan(np.array(matrix_data, dtype=float), n)

def solve_gauss_jordan(matrix, n):
    steps = []
    
    # Proses Gauss-Jordan
    for i in range(n):
        # --- PARTIAL PIVOTING (Kunci agar tidak error saat 0) ---
        pivot_row = i
        # Cari nilai absolut terbesar di kolom i mulai dari baris i ke bawah
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[pivot_row][i]):
                pivot_row = k
        
        # Tukar baris jika pivot row bukan baris saat ini
        if pivot_row != i:
            matrix[[i, pivot_row]] = matrix[[pivot_row, i]]
            steps.append(f"🔄 **Tukar Baris:** Baris {i+1} ditukar dengan Baris {pivot_row+1} (Karena pivot awal {matrix[pivot_row][i]:.2f} lebih dominan atau pivot asli 0).")
        else:
            if matrix[i][i] == 0:
                st.error("❌ Sistem tidak memiliki solusi unik (Singular Matrix).")
                return

        # Normalisasi baris pivot (buat elemen diagonal jadi 1)
        pivot_val = matrix[i][i]
        matrix[i] = matrix[i] / pivot_val
        # steps.append(f"Normalisasi Baris {i+1} (Dibagi {pivot_val:.2f})")

        # Eliminasi baris lain (buat jadi 0)
        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                matrix[k] = matrix[k] - factor * matrix[i]

    # Output Logika
    with st.expander("🕵️ Lihat Langkah Pengerjaan (Trace)", expanded=True):
        if len(steps) > 0:
            for s in steps:
                st.markdown(s)
        else:
            st.markdown("✅ Tidak diperlukan penukaran baris (Pivoting).")

    # Output Hasil
    st.divider()
    st.subheader("💡 Solusi Akhir")
    
    res_cols = st.columns(n)
    for i in range(n):
        with res_cols[i]:
            st.metric(label=f"x{i+1}", value=f"{matrix[i][-1]:.4f}")

# ==========================================
# MENU 2: AKAR PERSAMAAN (NEWTON RAPHSON)
# ==========================================
def menu_roots():
    st.header("📉 Akar Persamaan (Newton-Raphson)")
    st.caption("Update 1, 2, 3 diterapkan di sini.")

    # Input Fungsi
    func_input = st.text_input("Masukkan Fungsi f(x)", value="x^2 - 4", help="Gunakan sintaks Python/LaTeX, contoh: x^2 - 4 atau exp(x) - 3*x")
    
    # Preprocessing Input (Update 1)
    clean_func = preprocess_expression(func_input)
    
    # Hitung Turunan Otomatis (Update 2)
    expr, diff_expr = get_derivative(clean_func)

    if expr is not None:
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fungsi Asli $f(x)$:**")
            st.latex(sp.latex(expr))
        with c2:
            st.markdown("**Turunan $f'(x)$ (Otomatis):**")
            # Update 3: Menampilkan turunan sebagai bukti teori
            st.latex(sp.latex(diff_expr))
        st.write("---")
    else:
        st.error("Input fungsi tidak valid.")
        return

    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        x0 = st.number_input("Tebakan Awal (x0)", value=1.0)
    with col_param2:
        tol = st.number_input("Toleransi Error", value=1e-6, format="%.6f")
    with col_param3:
        max_iter = st.number_input("Max Iterasi", value=50, step=1)

    if st.button("Hitung Akar", type="primary"):
        solve_newton_raphson(expr, diff_expr, x0, tol, max_iter)

def solve_newton_raphson(expr, diff_expr, x0, tol, max_iter):
    # Konversi ke fungsi lambda python agar cepat dihitung
    f = sp.lambdify(sp.symbols('x'), expr, 'numpy')
    df = sp.lambdify(sp.symbols('x'), diff_expr, 'numpy')

    data = []
    x_curr = x0
    success = False

    for i in range(1, int(max_iter) + 1):
        try:
            f_val = f(x_curr)
            df_val = df(x_curr)

            if df_val == 0:
                st.error("Turunan bernilai 0. Metode gagal (Division by zero).")
                break

            # Rumus Newton Raphson
            x_next = x_curr - (f_val / df_val)
            error = abs(x_next - x_curr)

            data.append({
                "Iterasi": i,
                "x_curr": f"{x_curr:.6f}",
                "f(x)": f"{f_val:.6f}",
                "f'(x)": f"{df_val:.6f}",
                "Error": f"{error:.6f}"
            })

            x_curr = x_next
            if error < tol:
                success = True
                break
        except Exception as e:
            st.error(f"Terjadi kesalahan hitung: {e}")
            break

    # Tampilkan Tabel
    df_res = pd.DataFrame(data)
    st.table(df_res)

    if success:
        st.success(f"✅ Akar ditemukan di x = {x_curr:.6f}")
    else:
        st.warning("⚠️ Iterasi maksimum tercapai sebelum toleransi terpenuhi.")

# ==========================================
# MAIN APP LOGIC
# ==========================================
def main():
    st.sidebar.title("Navigasi")
    # Pilihan Menu
    menu = st.sidebar.radio("Pilih Metode:", ["Sistem Persamaan Linear", "Akar Persamaan (Newton)"])

    if menu == "Sistem Persamaan Linear":
        menu_spl()
    elif menu == "Akar Persamaan (Newton)":
        menu_roots()

if __name__ == "__main__":
    main()