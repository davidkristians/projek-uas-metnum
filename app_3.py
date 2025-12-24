import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import re

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Numerical Methods Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI/UX
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .reportview-container .main .block-container { max-width: 1200px; padding-top: 2rem; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 SMART PARSER ENGINE (CORE)
# ==========================================
def preprocess_input(func_str):
    """
    Membersihkan dan memperbaiki input user agar valid secara matematika python.
    Contoh: '4x' -> '4*x', 'x^2' -> 'x**2', 'sin(x)' -> 'np.sin(x)'
    """
    if not func_str: return None
    
    # 1. Ganti pangkat ^ jadi **
    func_str = func_str.replace("^", "**")
    
    # 2. Tambah perkalian implisit (misal: 4x -> 4*x, x(...) -> x*(...))
    # Regex: Angka diikuti huruf -> Angka*Huruf
    func_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', func_str)
    
    # Regex: Kurung tutup diikuti huruf/angka -> )*(
    # PERBAIKAN DI SINI: Menambahkan backslash \ sebelum )
    func_str = re.sub(r'\)(\d|[a-zA-Z(])', r')*\1', func_str)
    
    return func_str

def get_function(func_str, var_char='x'):
    """Mengembalikan fungsi Python (lambda) dan expression SymPy"""
    clean_str = preprocess_input(func_str)
    var = sp.symbols(var_char)
    try:
        # Parsing ke SymPy
        expr = sp.sympify(clean_str)
        # Convert ke Lambda function (numpy-ready)
        f = sp.lambdify(var, expr, modules=['numpy'])
        return f, expr, clean_str
    except Exception as e:
        return None, None, str(e)

def get_function_xy(func_str):
    """Khusus untuk ODE f(x,y)"""
    clean_str = preprocess_input(func_str)
    x, y = sp.symbols('x y')
    try:
        expr = sp.sympify(clean_str)
        f = sp.lambdify((x, y), expr, modules=['numpy'])
        return f, expr, clean_str
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 1. METODE AKAR (ROOTS)
# ==========================================
def menu_roots():
    st.header("🌱 Pencarian Akar Persamaan Non-Linear")
    st.markdown("---")
    
    col_input, col_conf = st.columns([1, 2])
    
    with col_input:
        method = st.selectbox("Pilih Metode", ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"])
        raw_func = st.text_input("Masukkan Fungsi f(x)", "x^3 - 4x - 9")
        
        # Validasi Input Real-time
        f, expr, debug_str = get_function(raw_func)
        
        if f:
            st.success(f"Interpreted: $f(x) = {sp.latex(expr)}$")
        else:
            st.error(f"Syntax Error: {debug_str}")
            st.info("💡 Tips: Gunakan 'x' sebagai variabel. Contoh: x^3 - 4x - 9")

    with col_conf:
        # Parameter Grid
        c1, c2, c3 = st.columns(3)
        tol = c1.number_input("Toleransi Error", value=1e-6, format="%.7f")
        max_iter = int(c2.number_input("Max Iterasi", value=50))
        
        # Input khusus per metode
        inputs = {}
        if method in ["Bisection", "Regula Falsi"]:
            inputs['a'] = c3.number_input("Batas Bawah (a)", value=2.0)
            inputs['b'] = c3.number_input("Batas Atas (b)", value=3.0)
        elif method == "Newton-Raphson":
            inputs['x0'] = c3.number_input("Tebakan Awal (x0)", value=2.5)
        elif method == "Secant":
            inputs['x0'] = c3.number_input("x0", value=2.0)
            inputs['x1'] = c3.number_input("x1", value=3.0)

    # Action Button
    if st.button("🚀 Hitung Solusi", type="primary") and f:
        data = []
        root = None
        success = False
        
        try:
            if method == "Bisection":
                a, b = inputs['a'], inputs['b']
                if f(a)*f(b) >= 0:
                    st.error(f"❌ Syarat Gagal: f({a}) dan f({b}) harus memiliki tanda berlawanan (+/-).")
                    return
                
                for i in range(max_iter):
                    c = (a + b) / 2
                    fa, fb, fc = f(a), f(b), f(c)
                    err = abs(b - a)
                    data.append({"Iter": i+1, "a": a, "b": b, "c (tengah)": c, "f(c)": fc, "Error": err})
                    
                    if abs(fc) < 1e-9 or err < tol:
                        root, success = c, True
                        break
                        
                    if fa * fc < 0: b = c
                    else: a = c

            elif method == "Regula Falsi":
                a, b = inputs['a'], inputs['b']
                if f(a)*f(b) >= 0:
                    st.error(f"❌ Syarat Gagal: f({a}) dan f({b}) harus tanda berlawanan.")
                    return
                
                prev_c = a
                for i in range(max_iter):
                    fa, fb = f(a), f(b)
                    if fb - fa == 0: break 
                    c = (a*fb - b*fa) / (fb - fa)
                    fc = f(c)
                    err = abs(c - prev_c)
                    
                    data.append({"Iter": i+1, "a": a, "b": b, "c": c, "f(c)": fc, "Error": err})
                    
                    if abs(fc) < 1e-9 or err < tol:
                        root, success = c, True
                        break
                        
                    if fa * fc < 0: b = c
                    else: a = c
                    prev_c = c

            elif method == "Newton-Raphson":
                x0 = inputs['x0']
                # Turunan Otomatis
                x_sym = sp.symbols('x')
                df_expr = sp.diff(expr, x_sym)
                df = sp.lambdify(x_sym, df_expr, 'numpy')
                st.info(f"ℹ️ Turunan Otomatis: $f'(x) = {sp.latex(df_expr)}$")
                
                curr = x0
                for i in range(max_iter):
                    val = f(curr)
                    der = df(curr)
                    if der == 0: st.error("Turunan 0, metode gagal."); break
                    
                    next_x = curr - val/der
                    err = abs(next_x - curr)
                    data.append({"Iter": i+1, "xi": curr, "f(xi)": val, "f'(xi)": der, "xi+1": next_x, "Error": err})
                    
                    if err < tol:
                        root, success = next_x, True
                        break
                    curr = next_x

            # --- RESULT DISPLAY ---
            if data:
                df_res = pd.DataFrame(data)
                
                # 1. Summary
                if success:
                    st.success(f"✅ Akar ditemukan: **x = {root:.6f}** pada iterasi ke-{len(data)}")
                else:
                    st.warning("⚠️ Iterasi maksimum tercapai. Hasil terakhir ditampilkan.")
                
                # 2. Tabs for Details
                tab1, tab2 = st.tabs(["📊 Tabel Iterasi", "📈 Grafik Konvergensi"])
                
                with tab1:
                    st.dataframe(df_res.style.format("{:.6f}"))
                
                with tab2:
                    # Plot Error
                    st.line_chart(df_res["Error"])
                    
                    # Plot Function
                    x_vals = np.linspace(inputs.get('a', root-2), inputs.get('b', root+2), 100)
                    y_vals = f(x_vals)
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, y_vals, label='f(x)')
                    ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
                    ax.scatter([root], [f(root)], color='green', zorder=5, label=f'Root {root:.4f}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Runtime Error: {e}")

# ==========================================
# 2. METODE SPL
# ==========================================
def menu_spl():
    st.header("🧮 Sistem Persamaan Linear")
    
    col_set, col_info = st.columns([1, 2])
    with col_set:
        method = st.selectbox("Metode SPL", ["Gauss", "Gauss-Jordan", "LU Decomposition", "Jacobi", "Gauss-Seidel"])
        n = st.number_input("Jumlah Variabel (N)", 2, 8, 3)

    st.info("📝 Masukkan Matriks Augmented [A | b] di bawah ini:")
    
    # Input Matriks Dinamis
    matrix = np.zeros((n, n+1))
    cols = st.columns(n+1)
    
    # Header
    for i in range(n): cols[i].markdown(f"**x{i+1}**")
    cols[n].markdown("**= b**")
    
    for r in range(n):
        cols_input = st.columns(n+1)
        for c in range(n+1):
            matrix[r, c] = cols_input[c].number_input(f"Baris {r+1}, Kolom {c+1}", value=0.0, key=f"m_{r}_{c}", label_visibility="collapsed")

    if st.button("Hitung Solusi SPL", type="primary"):
        A = matrix[:, :-1]
        b = matrix[:, -1]
        
        try:
            if method == "Gauss":
                # Implementasi Singkat Gauss Naive
                for i in range(n):
                    # Pivot
                    for k in range(i+1, n):
                        factor = matrix[k,i]/matrix[i,i]
                        matrix[k, i:] -= factor * matrix[i, i:]
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (matrix[i,-1] - np.dot(matrix[i, i+1:n], x[i+1:n])) / matrix[i,i]
                st.write("Matriks Segitiga Atas:", matrix)
                st.success(f"Solusi: {x}")

            elif method == "Gauss-Jordan":
                # Menggunakan rref logic
                aug = matrix.copy()
                for i in range(n):
                    aug[i] = aug[i] / aug[i,i]
                    for k in range(n):
                        if i != k:
                            aug[k] -= aug[k,i] * aug[i]
                st.write("Matriks Identitas:", aug)
                st.success(f"Solusi: {aug[:, -1]}")

            elif method in ["Jacobi", "Gauss-Seidel"]:
                x = np.zeros(n)
                tol = 1e-6
                max_iter = 50
                logs = []
                
                for k in range(max_iter):
                    x_new = np.copy(x)
                    for i in range(n):
                        s1 = sum(A[i][j] * x_new[j] for j in range(i)) if method == "Gauss-Seidel" else sum(A[i][j] * x[j] for j in range(i))
                        s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
                        x_new[i] = (b[i] - s1 - s2) / A[i][i]
                    
                    err = np.linalg.norm(x_new - x)
                    logs.append(np.append(x_new, err))
                    x = x_new
                    if err < tol: break
                
                cols_log = [f"x{i+1}" for i in range(n)] + ["Error"]
                st.dataframe(pd.DataFrame(logs, columns=cols_log))
                st.success(f"Konvergen dalam {len(logs)} iterasi.")

        except Exception as e:
            st.error(f"Error Perhitungan: {e} (Cek Pivot Nol atau Matriks Singular)")

# ==========================================
# 3. INTERPOLASI
# ==========================================
def menu_interp():
    st.header("📈 Interpolasi Data")
    
    c1, c2 = st.columns(2)
    method = c1.radio("Metode", ["Lagrange", "Newton Polynomial"])
    val_x = c2.number_input("Prediksi nilai Y pada X =", 2.5)
    
    st.subheader("Data Points")
    input_x = st.text_input("Deret X (pisahkan koma)", "1, 2, 3, 5")
    input_y = st.text_input("Deret Y (pisahkan koma)", "1, 4, 9, 25")
    
    if st.button("Interpolasi"):
        try:
            X = np.array([float(x) for x in input_x.split(',')])
            Y = np.array([float(y) for y in input_y.split(',')])
            
            if len(X) != len(Y):
                st.error("Jumlah data X dan Y harus sama!")
                return
            
            if method == "Lagrange":
                def L(k, x_in):
                    term = 1
                    for i in range(len(X)):
                        if i != k: term *= (x_in - X[i]) / (X[k] - X[i])
                    return term
                
                result = sum(Y[k] * L(k, val_x) for k in range(len(X)))
                st.success(f"Hasil Lagrange P({val_x}) = {result:.4f}")
                
                # Visualisasi
                x_plot = np.linspace(min(X), max(X), 100)
                y_plot = [sum(Y[k] * L(k, xp) for k in range(len(X))) for xp in x_plot]
                
            elif method == "Newton Polynomial":
                n = len(X)
                coef = np.zeros([n, n])
                coef[:,0] = Y
                for j in range(1,n):
                    for i in range(n-j):
                        coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (X[i+j] - X[i])
                
                res = coef[0,0]
                term = 1.0
                for i in range(1, n):
                    term *= (val_x - X[i-1])
                    res += coef[0,i] * term
                st.success(f"Hasil Newton P({val_x}) = {res:.4f}")
                st.write("Divided Difference Table:", coef)
                
                # Visualisasi dummy (logic sama)
                x_plot = X # Simplification for demo
                y_plot = Y

            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(X, Y, color='red', label='Data Asli')
            if method == "Lagrange": 
                ax.plot(x_plot, y_plot, '--', label='Polinom Interpolasi')
            ax.scatter([val_x], [res if method=="Newton Polynomial" else result], color='green', s=100, label='Hasil')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Format Data Salah: {e}")

# ==========================================
# 4. INTEGRASI & 5. DIFERENSIASI
# ==========================================
def menu_calc():
    st.header("∫ Kalkulus Numerik")
    tab_int, tab_diff = st.tabs(["Integrasi (Luas Area)", "Diferensiasi (Gradien)"])
    
    # --- TAB 1: INTEGRASI (Code yang sudah diperbaiki sebelumnya) ---
    with tab_int:
        st.subheader("Hitung Integral Tentu")
        st.caption("Menghitung luas area di bawah kurva f(x) dari a ke b.")
        
        # Contoh Input untuk User
        with st.expander("💡 Contoh Input Kompleks (Klik disini)"):
            st.markdown("""
            * **Fungsi Gaussian:** `exp(-x^2)` (a=-1, b=1)
            * **Trigonometri:** `sin(x)^2 + cos(x)` (a=0, b=3.14159)
            * **Rasional:** `1 / (1 + x^2)` (a=0, b=1, Target=3.14/4)
            """)

        func_int = st.text_input("Fungsi f(x)", "4 / (1 + x^2)", key="f_int")
        col1, col2, col3 = st.columns(3)
        a = col1.number_input("Batas Bawah (a)", value=0.0, format="%.4f")
        b = col2.number_input("Batas Atas (b)", value=1.0, format="%.4f")
        # HAPUS min_value=2. Biarkan user input 1000 sekalipun.
        n = int(col3.number_input("Jumlah Segmen (N)", value=10, min_value=1)) 
        
        m_int = st.selectbox("Metode Integrasi", ["Trapezium", "Simpson 1/3", "Simpson 3/8"])

        if st.button("Hitung Integral"):
            f, expr, _ = get_function(func_int)
            if f:
                # Validasi N untuk Simpson
                if m_int == "Simpson 1/3" and n % 2 != 0:
                    st.warning("⚠️ Simpson 1/3 butuh N Genap. Menggunakan N+1.")
                    n += 1
                elif m_int == "Simpson 3/8" and n % 3 != 0:
                    st.warning("⚠️ Simpson 3/8 butuh N kelipatan 3. Menggunakan kelipatan terdekat.")
                    while n % 3 != 0: n += 1

                h = (b - a) / n
                x = np.linspace(a, b, n+1)
                y = f(x)
                
                res = 0
                if m_int == "Trapezium":
                    res = (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
                elif m_int == "Simpson 1/3":
                    res = (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
                elif m_int == "Simpson 3/8":
                    s = y[0] + y[-1]
                    for i in range(1, n):
                        if i % 3 == 0: s += 2 * y[i]
                        else: s += 3 * y[i]
                    res = (3*h/8) * s
                
                # Hitung Exact
                try:
                    x_sym = sp.symbols('x')
                    exact = float(sp.integrate(expr, (x_sym, a, b)))
                    err = abs(exact - res)
                except: exact, err = "N/A", "N/A"

                c1, c2 = st.columns(2)
                c1.metric("Hasil Numerik", f"{res:.6f}")
                c1.caption(f"Step size (h) = {h:.6f}")
                if exact != "N/A":
                    c2.metric("Hasil Exact (Analitik)", f"{exact:.6f}")
                    c2.metric("Error Absolut", f"{err:.6e}")

    # --- TAB 2: DIFERENSIASI (Full Power) ---
    with tab_diff:
        st.subheader("Hitung Turunan Numerik")
        st.caption("Menghitung f'(x) menggunakan selisih data.")

        # Contoh Input
        with st.expander("💡 Contoh Input Turunan"):
            st.markdown("""
            * **Polinomial:** `x^3 - 2*x + 5` (Cek di x=2)
            * **Eksponensial:** `exp(x)` (Turunannya adalah dirinya sendiri)
            * **Logaritma:** `log(x)` (Cek di x=2, hasil harus 0.5)
            """)

        # Input Fleksibel
        f_diff_str = st.text_input("Fungsi f(x)", "x^3", key="f_diff")
        
        c1, c2 = st.columns(2)
        val_x = c1.number_input("Titik Evaluasi (x)", value=2.0, format="%.4f")
        # Step size h jangan dibatasi min_value! User boleh input 0.000001
        h = c2.number_input("Step Size (h)", value=0.01, format="%.6f", step=0.001)

        if st.button("Hitung Turunan"):
            f, expr, _ = get_function(f_diff_str)
            if f:
                # 1. Hitung Numerik
                # Forward: (f(x+h) - f(x))/h
                fwd = (f(val_x + h) - f(val_x)) / h
                
                # Backward: (f(x) - f(x-h))/h
                bwd = (f(val_x) - f(val_x - h)) / h
                
                # Central: (f(x+h) - f(x-h))/(2h) -> Paling Akurat O(h^2)
                cen = (f(val_x + h) - f(val_x - h)) / (2*h)

                # 2. Hitung Exact (Bukti Teori)
                try:
                    x_sym = sp.symbols('x')
                    diff_exact_expr = sp.diff(expr, x_sym)
                    exact_val = float(diff_exact_expr.subs(x_sym, val_x))
                except:
                    exact_val = None

                # 3. Tampilkan Tabel Perbandingan
                st.write("### 📊 Hasil Perbandingan Metode")
                
                res_data = {
                    "Metode": ["Forward Difference", "Backward Difference", "Central Difference"],
                    "Rumus": ["(f(x+h) - f(x)) / h", "(f(x) - f(x-h)) / h", "(f(x+h) - f(x-h)) / 2h"],
                    "Hasil Hitung": [fwd, bwd, cen]
                }
                
                if exact_val is not None:
                    res_data["Exact (Analitik)"] = [exact_val] * 3
                    res_data["Error"] = [abs(exact_val - fwd), abs(exact_val - bwd), abs(exact_val - cen)]
                    st.success(f"Nilai Exact (Analitik): {exact_val:.6f}")
                
                df_res = pd.DataFrame(res_data)
                st.table(df_res.style.format({"Hasil Hitung": "{:.6f}", "Exact (Analitik)": "{:.6f}", "Error": "{:.6e}"}))
                
                st.info("ℹ️ **Analisis:** Metode *Central Difference* biasanya memiliki error paling kecil karena memperhitungkan sisi kiri dan kanan.")
# ==========================================
# 6. PENYELESAIAN ODE
# ==========================================
def menu_ode():
    st.header("⚙️ Persamaan Diferensial Biasa (ODE)")
    st.markdown(r"Menyelesaikan persamaan berbentuk: $\frac{dy}{dx} = f(x, y)$")

    # Layout Input Kiri-Kanan
    col_kiri, col_kanan = st.columns([1, 2])

    with col_kiri:
        st.subheader("Konfigurasi")
        method = st.selectbox("Pilih Metode", ["Bandingkan Semua", "Euler", "Heun", "Runge-Kutta 4"])
        
        # Parameter Awal (Initial Value Problem)
        x0 = st.number_input("x Awal (x0)", value=0.0)
        y0 = st.number_input("y Awal (y0)", value=1.0)
        
        # Target
        xn = st.number_input("x Target (Cari y di x=...)", value=2.0)
        h = st.number_input("Step Size (h)", value=0.1, format="%.4f")

    with col_kanan:
        st.subheader("Fungsi f(x, y)")
        # Contoh Input
        with st.expander("💡 Contoh Input ODE (Wajib 2 Variabel)"):
            st.markdown("""
            * **Pertumbuhan:** `y` (dy/dx = y) -> Solusi eksponensial.
            * **Linear:** `x + y` (dy/dx = x+y).
            * **Osilasi:** `y * cos(x)` (dy/dx = y cos x).
            * **Decay:** `-2 * y + x`
            """)
        
        func_ode = st.text_input("Masukkan f(x, y)", "x + y")
        
        if st.button("🚀 Jalankan Simulasi ODE", type="primary"):
            # Parsing fungsi f(x,y)
            f_ode, expr, msg = get_function_xy(func_ode)
            
            if not f_ode:
                st.error(f"Error Fungsi: {msg}")
            else:
                # Validasi Step
                if h <= 0: st.error("Step size (h) harus positif!"); return
                if xn <= x0: st.error("Target x harus lebih besar dari x0 (untuk versi ini)."); return

                # Hitung jumlah langkah (N)
                n_steps = int((xn - x0) / h)
                
                # Container hasil
                results = {}
                
                # --- SOLVER ENGINE ---
                def solve_euler():
                    x, y = x0, y0
                    path_x, path_y = [x], [y]
                    for _ in range(n_steps):
                        y += h * f_ode(x, y)
                        x += h
                        path_x.append(x); path_y.append(y)
                    return path_x, path_y

                def solve_heun():
                    x, y = x0, y0
                    path_x, path_y = [x], [y]
                    for _ in range(n_steps):
                        k1 = f_ode(x, y)
                        k2 = f_ode(x + h, y + h * k1)
                        y += (h / 2) * (k1 + k2)
                        x += h
                        path_x.append(x); path_y.append(y)
                    return path_x, path_y

                def solve_rk4():
                    x, y = x0, y0
                    path_x, path_y = [x], [y]
                    for _ in range(n_steps):
                        k1 = f_ode(x, y)
                        k2 = f_ode(x + 0.5*h, y + 0.5*h*k1)
                        k3 = f_ode(x + 0.5*h, y + 0.5*h*k2)
                        k4 = f_ode(x + h, y + h*k3)
                        y += (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
                        x += h
                        path_x.append(x); path_y.append(y)
                    return path_x, path_y

                # Eksekusi sesuai pilihan
                if method == "Euler" or method == "Bandingkan Semua":
                    results["Euler"] = solve_euler()
                if method == "Heun" or method == "Bandingkan Semua":
                    results["Heun"] = solve_heun()
                if method == "Runge-Kutta 4" or method == "Bandingkan Semua":
                    results["RK4"] = solve_rk4()

                # --- VISUALISASI ---
                st.write("---")
                st.subheader("📈 Grafik Solusi Numerik")
                
                fig, ax = plt.subplots()
                
                colors = {"Euler": "red", "Heun": "blue", "RK4": "green"}
                markers = {"Euler": "--", "Heun": "-.", "RK4": "-"}

                final_res = []

                for name, (px, py) in results.items():
                    ax.plot(px, py, label=name, color=colors[name], linestyle=markers[name])
                    final_res.append({
                        "Metode": name,
                        "y Akhir (di x={:.2f})".format(px[-1]): py[-1],
                        "Jumlah Step": len(px)-1
                    })

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"Solusi dy/dx = {func_ode}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Tabel Hasil Akhir
                st.write("### 🏁 Nilai Akhir")
                st.table(pd.DataFrame(final_res))
                
                if method == "Bandingkan Semua":
                    st.info("ℹ️ **Analisis:** Perhatikan bahwa Runge-Kutta 4 (RK4) biasanya adalah yang paling akurat/halus, sedangkan Euler seringkali memiliki error yang membesar jika step (h) kurang kecil.")

# ==========================================
# MAIN ROUTING
# ==========================================
def main():
    st.sidebar.title("🔢 Numerical Lab")
    st.sidebar.caption("Project Akhir Metode Numerik")
    
    menu = st.sidebar.radio("Navigasi Modul", [
        "Akar Persamaan (Roots)", 
        "Sistem Linear (SPL)", 
        "Interpolasi", 
        "Kalkulus (Int & Diff)", 
        "Persamaan Diferensial (ODE)"
    ])
    
    st.sidebar.divider()
    st.sidebar.info(
        """
        **Panduan Input:**
        * Perkalian: `4x` otomatis dibaca `4*x`
        * Pangkat: `x^2` otomatis dibaca `x**2`
        * Fungsi: `sin(x)`, `cos(x)`, `exp(x)`
        """
    )

    if "Akar" in menu: menu_roots()
    elif "Sistem" in menu: menu_spl()
    elif "Interpolasi" in menu: menu_interp()
    elif "Kalkulus" in menu: menu_calc()
    elif "ODE" in menu: menu_ode()

if __name__ == "__main__":
    main()