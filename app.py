import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import scipy.linalg
import re

# ==========================================
# 0. CONFIG & UTILS (CORE ENGINE)
# ==========================================
st.set_page_config(page_title="Numerical Methods Pro", layout="wide", page_icon="🧮")

# CSS untuk Tampilan Professional
st.markdown("""
<style>
    .stTable { font-size: 0.9rem; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stAlert { border-left: 5px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

def preprocess_input(func_str):
    """
    Smart Parser: Mengubah input 'malas' user menjadi syntax Python yang valid.
    Contoh: '2x' -> '2*x', 'x^2' -> 'x**2', 'e^-x' -> 'exp(-x)'
    """
    if not func_str: return None
    func_str = func_str.replace("^", "**")
    func_str = func_str.replace("e^", "exp")
    # Regex: Angka ketemu Huruf/KurungBuka -> Angka * Huruf
    func_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', func_str)
    # Regex: KurungTutup ketemu Angka/Huruf -> ) * ...
    func_str = re.sub(r'\)(\d|[a-zA-Z(])', r')*\1', func_str)
    return func_str

def get_function(func_str, var_char='x'):
    """Parser untuk f(x)"""
    clean_str = preprocess_input(func_str)
    var = sp.symbols(var_char)
    try:
        expr = sp.sympify(clean_str)
        f = sp.lambdify(var, expr, modules=['numpy', 'scipy'])
        return f, expr, clean_str
    except Exception as e:
        return None, None, str(e)

def get_function_xy(func_str):
    """Parser untuk f(x, y) - ODE"""
    clean_str = preprocess_input(func_str)
    x, y = sp.symbols('x y')
    try:
        expr = sp.sympify(clean_str)
        f = sp.lambdify((x, y), expr, modules=['numpy'])
        return f, expr, clean_str
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 1. MODUL AKAR (ROOTS)
# ==========================================
def menu_roots():
    st.header("🌱 Akar Persamaan Non-Linear")
    st.info("Mencari x dimana f(x) = 0")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.selectbox("Metode", ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"])
        func_str = st.text_input("f(x)", "x^3 - 2x - 5")
        
        # Parameter input dinamis
        params = {}
        if method in ["Bisection", "Regula Falsi"]:
            params['a'] = st.number_input("Batas Bawah (a)", value=2.0)
            params['b'] = st.number_input("Batas Atas (b)", value=3.0)
        elif method == "Newton-Raphson":
            params['x0'] = st.number_input("Tebakan Awal (x0)", value=2.0)
        elif method == "Secant":
            params['x0'] = st.number_input("x0", value=2.0)
            params['x1'] = st.number_input("x1", value=3.0)
            
        tol = st.number_input("Toleransi Error", value=1e-6, format="%.7f")
        max_iter = int(st.number_input("Max Iterasi", value=50))

    with c2:
        if st.button("Hitung Akar", type="primary"):
            f, expr, msg = get_function(func_str)
            if not f:
                st.error(f"Syntax Error: {msg}"); return

            data = []
            root = None
            success = False

            try:
                # --- LOGIKA METODE ---
                if method == "Bisection":
                    a, b = params['a'], params['b']
                    if f(a)*f(b) >= 0: st.error("❌ Syarat Gagal: f(a)*f(b) harus < 0"); return
                    
                    for i in range(max_iter):
                        c = (a + b) / 2
                        err = abs(b - a)
                        data.append({"Iter": i+1, "a": a, "b": b, "c": c, "f(c)": f(c), "Error": err})
                        if abs(f(c)) < 1e-9 or err < tol: root=c; success=True; break
                        if f(a)*f(c) < 0: b = c
                        else: a = c

                elif method == "Regula Falsi":
                    a, b = params['a'], params['b']
                    if f(a)*f(b) >= 0: st.error("❌ Syarat Gagal: f(a)*f(b) harus < 0"); return
                    prev_c = a
                    for i in range(max_iter):
                        fa, fb = f(a), f(b)
                        c = (a*fb - b*fa)/(fb - fa)
                        err = abs(c - prev_c)
                        data.append({"Iter": i+1, "a": a, "b": b, "c": c, "f(c)": f(c), "Error": err})
                        if abs(f(c)) < 1e-9 or err < tol: root=c; success=True; break
                        if f(a)*f(c) < 0: b = c
                        else: a = c
                        prev_c = c

                elif method == "Newton-Raphson":
                    x0 = params['x0']
                    # Auto Derivative
                    x_sym = sp.symbols('x')
                    df_expr = sp.diff(expr, x_sym)
                    df = sp.lambdify(x_sym, df_expr, 'numpy')
                    st.latex(f"f'(x) = {sp.latex(df_expr)}")
                    
                    curr = x0
                    for i in range(max_iter):
                        val, der = f(curr), df(curr)
                        if der == 0: st.error("Turunan 0 (Division by Zero)"); break
                        next_x = curr - val/der
                        err = abs(next_x - curr)
                        data.append({"Iter": i+1, "xi": curr, "f(xi)": val, "f'(xi)": der, "Error": err})
                        if err < tol: root=next_x; success=True; break
                        curr = next_x

                elif method == "Secant":
                    x0, x1 = params['x0'], params['x1']
                    for i in range(max_iter):
                        fx0, fx1 = f(x0), f(x1)
                        if fx1 - fx0 == 0: break
                        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                        err = abs(x2 - x1)
                        data.append({"Iter": i+1, "x(i-1)": x0, "x(i)": x1, "x(i+1)": x2, "Error": err})
                        if err < tol: root=x2; success=True; break
                        x0, x1 = x1, x2

                # --- OUTPUT ---
                if success: st.success(f"✅ Akar Konvergen: x = {root:.6f}")
                else: st.warning("⚠️ Iterasi Max Tercapai")
                
                st.dataframe(pd.DataFrame(data).style.format("{:.6f}"))
                
                # Plotting
                fig, ax = plt.subplots(figsize=(8,3))
                x_vals = np.linspace(root-2, root+2, 100)
                ax.plot(x_vals, f(x_vals), label="f(x)")
                ax.axhline(0, color='red', linestyle='--')
                ax.scatter([root], [f(root)], color='green', zorder=5)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            except Exception as e: st.error(f"Runtime Error: {e}")

# ==========================================
# 2. MODUL SPL (LINEAR SYSTEMS)
# ==========================================
def menu_spl():
    st.header("🧮 Sistem Persamaan Linear (SPL)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.selectbox("Metode", ["Gauss", "Gauss-Jordan", "LU Decomposition", "Jacobi", "Gauss-Seidel"])
        n = int(st.number_input("Jumlah Variabel (N)", 2, 10, 3))
    
    st.caption("Masukkan Matriks Augmented [A | b]")
    # Input Matriks
    mat_data = np.zeros((n, n+1))
    cols = st.columns(n+1)
    for i in range(n): cols[i].markdown(f"**x{i+1}**")
    cols[n].markdown("**= b**")
    
    for r in range(n):
        c_in = st.columns(n+1)
        for c in range(n+1):
            mat_data[r,c] = c_in[c].number_input(f"R{r}C{c}", value=0.0, key=f"spl_{r}_{c}", label_visibility="collapsed")

    if st.button("Hitung SPL", type="primary"):
        A = mat_data[:, :-1]
        b = mat_data[:, -1]
        
        try:
            if method == "Gauss":
                # Forward Elimination
                aug = mat_data.copy()
                steps = []
                for i in range(n):
                    # Pivot strategy simple
                    pivot = aug[i,i]
                    if pivot == 0: st.error("Pivot 0 detected!"); return
                    for k in range(i+1, n):
                        factor = aug[k,i]/pivot
                        aug[k, i:] -= factor * aug[i, i:]
                
                # Back Subst
                x = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x[i] = (aug[i,-1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i,i]
                
                st.write("Matriks Segitiga Atas:", aug)
                st.success(f"Solusi: {x}")

            elif method == "Gauss-Jordan":
                aug = mat_data.copy()
                for i in range(n):
                    aug[i] /= aug[i,i] # Make pivot 1
                    for k in range(n):
                        if k != i:
                            aug[k] -= aug[k,i] * aug[i] # Make others 0
                st.success(f"Solusi: {aug[:, -1]}")

            elif method == "LU Decomposition":
                P, L, U = scipy.linalg.lu(A)
                y = scipy.linalg.solve(L, P.dot(b)) # Ly = Pb
                x = scipy.linalg.solve(U, y)        # Ux = y
                
                c1, c2 = st.columns(2)
                c1.write("Matriks L", L)
                c2.write("Matriks U", U)
                st.success(f"Solusi: {x}")

            elif method in ["Jacobi", "Gauss-Seidel"]:
                x = np.zeros(n)
                tol = 1e-6
                max_it = 50
                logs = []
                
                # Cek Diagonal Dominan
                diag = np.abs(np.diag(A))
                off_diag = np.sum(np.abs(A), axis=1) - diag
                if not np.all(diag > off_diag):
                    st.warning("⚠️ Matriks TIDAK Diagonally Dominant. Metode iteratif mungkin tidak konvergen.")

                for k in range(max_it):
                    x_new = np.copy(x)
                    for i in range(n):
                        s1 = sum(A[i][j] * x_new[j] for j in range(i)) if method == "Gauss-Seidel" else sum(A[i][j] * x[j] for j in range(i))
                        s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
                        x_new[i] = (b[i] - s1 - s2) / A[i][i]
                    
                    err = np.linalg.norm(x_new - x)
                    logs.append(np.append(x_new, err))
                    x = x_new
                    if err < tol: break
                
                st.dataframe(pd.DataFrame(logs, columns=[f"x{i+1}" for i in range(n)] + ["Error"]))
                st.success(f"Konvergen di iterasi {len(logs)}")

        except Exception as e:
            st.error(f"Error Perhitungan: {e}")

# ==========================================
# 3. MODUL INTERPOLASI
# ==========================================
def menu_interp():
    st.header("📈 Interpolasi")
    
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Metode", ["Newton Polynomial", "Lagrange"])
        x_in = st.text_input("Data X (koma)", "1, 2, 3, 4")
        y_in = st.text_input("Data Y (koma)", "1, 4, 9, 16")
    with col2:
        val = st.number_input("Cari Nilai Y pada X =", value=2.5)

    if st.button("Hitung Interpolasi"):
        try:
            X = np.array([float(k) for k in x_in.split(',')])
            Y = np.array([float(k) for k in y_in.split(',')])
            if len(X) != len(Y): st.error("Jumlah X dan Y beda!"); return
            
            res = 0
            if method == "Lagrange":
                for k in range(len(X)):
                    term = 1
                    for i in range(len(X)):
                        if i != k: term *= (val - X[i]) / (X[k] - X[i])
                    res += Y[k] * term
                st.success(f"Hasil Lagrange P({val}) = {res:.5f}")

            elif method == "Newton Polynomial":
                n = len(X)
                coef = np.zeros([n, n])
                coef[:,0] = Y
                for j in range(1,n):
                    for i in range(n-j):
                        coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (X[i+j] - X[i])
                
                st.write("Tabel Selisih Terbagi (Divided Difference):", coef)
                
                res = coef[0,0]
                term = 1.0
                for i in range(1, n):
                    term *= (val - X[i-1])
                    res += coef[0,i] * term
                st.success(f"Hasil Newton P({val}) = {res:.5f}")

            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(X, Y, color='red', label='Data')
            ax.scatter([val], [res], color='blue', marker='x', s=100, label='Prediksi')
            
            # Smooth Curve
            x_smooth = np.linspace(min(X), max(X), 100)
            if method == "Lagrange":
                y_smooth = []
                for xs in x_smooth:
                    ys = 0
                    for k in range(len(X)):
                        term = 1
                        for i in range(len(X)):
                            if i != k: term *= (xs - X[i]) / (X[k] - X[i])
                        ys += Y[k] * term
                    y_smooth.append(ys)
                ax.plot(x_smooth, y_smooth, '--', alpha=0.5)

            ax.legend()
            st.pyplot(fig)

        except Exception as e: st.error(f"Input Error: {e}")

# ==========================================
# 4. MODUL INTEGRASI
# ==========================================
def menu_integ():
    st.header("∫ Integrasi Numerik")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        method = st.selectbox("Metode", ["Trapezium", "Simpson 1/3", "Simpson 3/8"])
        func = st.text_input("f(x)", "4/(1+x^2)")
        a = st.number_input("Batas Bawah (a)", value=0.0)
        b = st.number_input("Batas Atas (b)", value=1.0)
        n = int(st.number_input("Segmen (N)", value=10, min_value=1))

    with col2:
        if st.button("Hitung Integral", type="primary"):
            f, expr, msg = get_function(func)
            if not f: st.error(msg); return
            
            # Logic Check
            if method == "Simpson 1/3" and n % 2 != 0:
                st.warning("Info: N diubah ke Genap (N+1) untuk Simpson 1/3")
                n += 1
            if method == "Simpson 3/8" and n % 3 != 0:
                st.warning("Info: N disesuaikan kelipatan 3 untuk Simpson 3/8")
                while n % 3 != 0: n += 1
            
            h = (b - a) / n
            x = np.linspace(a, b, n+1)
            y = f(x)
            
            res = 0
            if method == "Trapezium":
                res = (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
            elif method == "Simpson 1/3":
                res = (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
            elif method == "Simpson 3/8":
                s = y[0] + y[-1]
                for i in range(1, n):
                    if i%3 == 0: s += 2*y[i]
                    else: s += 3*y[i]
                res = (3*h/8) * s
            
            # Exact
            try:
                exact = float(sp.integrate(expr, (sp.symbols('x'), a, b)))
                err = abs(exact - res)
            except: exact = "N/A"; err = "N/A"

            st.metric("Hasil Numerik", f"{res:.6f}")
            if exact != "N/A":
                c1, c2 = st.columns(2)
                c1.metric("Hasil Exact", f"{exact:.6f}")
                c2.metric("Error Absolut", f"{err:.6e}")
            
            # Visualisasi
            fig, ax = plt.subplots(figsize=(8,3))
            x_plot = np.linspace(a, b, 100)
            ax.plot(x_plot, f(x_plot), label="f(x)")
            ax.fill_between(x_plot, f(x_plot), alpha=0.2)
            ax.legend()
            st.pyplot(fig)

# ==========================================
# 5. MODUL DIFERENSIASI
# ==========================================
def menu_diff():
    st.header("dy/dx Diferensiasi Numerik")
    
    col1, col2 = st.columns(2)
    with col1:
        func_str = st.text_input("f(x)", "x^3 - 2x + 1", key="diff_func")
        x_val = st.number_input("Titik x", value=2.0)
        h = st.number_input("Step (h)", value=0.01, format="%.5f")

    if st.button("Hitung Turunan"):
        f, expr, msg = get_function(func_str)
        if not f: st.error(msg); return
        
        # Formulas
        fwd = (f(x_val + h) - f(x_val)) / h
        bwd = (f(x_val) - f(x_val - h)) / h
        cen = (f(x_val + h) - f(x_val - h)) / (2*h)
        
        # Exact
        try:
            diff_ex = sp.diff(expr, sp.symbols('x'))
            exact = float(diff_ex.subs(sp.symbols('x'), x_val))
        except: exact = None
        
        data = {
            "Metode": ["Forward", "Backward", "Central"],
            "Rumus": ["(f(x+h)-f(x))/h", "(f(x)-f(x-h))/h", "(f(x+h)-f(x-h))/2h"],
            "Hasil": [fwd, bwd, cen]
        }
        if exact is not None:
            data["Exact"] = [exact]*3
            data["Error"] = [abs(exact-fwd), abs(exact-bwd), abs(exact-cen)]
            st.latex(f"f'(x)_{{exact}} = {sp.latex(diff_ex)} \quad | \quad f'({x_val}) = {exact:.5f}")
            
        st.table(pd.DataFrame(data).style.format({"Hasil": "{:.6f}", "Error": "{:.6e}"}))

# ==========================================
# 6. MODUL ODE
# ==========================================
def menu_ode():
    st.header("⚙️ Penyelesaian ODE")
    st.caption("dy/dx = f(x, y)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.selectbox("Metode", ["Euler", "Heun", "Runge-Kutta 4", "Bandingkan Semua"])
        x0 = st.number_input("x0", 0.0)
        y0 = st.number_input("y0", 1.0)
        xn = st.number_input("Target x", 1.0)
        h = st.number_input("Step h", 0.1)

    with c2:
        func_ode = st.text_input("f(x, y)", "x + y")
        
        if st.button("Simulasi ODE", type="primary"):
            f, expr, msg = get_function_xy(func_ode)
            if not f: st.error(msg); return
            
            steps = int((xn - x0)/h)
            if steps <= 0: st.error("Target x harus > x0"); return
            
            results = {}
            
            # Solver Functions
            def solve(algo):
                x, y = x0, y0
                path_x, path_y = [x], [y]
                for _ in range(steps):
                    if algo == "Euler":
                        y += h * f(x, y)
                    elif algo == "Heun":
                        k1 = f(x, y)
                        k2 = f(x+h, y + h*k1)
                        y += (h/2)*(k1+k2)
                    elif algo == "RK4":
                        k1 = f(x, y)
                        k2 = f(x + 0.5*h, y + 0.5*h*k1)
                        k3 = f(x + 0.5*h, y + 0.5*h*k2)
                        k4 = f(x + h, y + h*k3)
                        y += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
                    x += h
                    path_x.append(x); path_y.append(y)
                return path_x, path_y

            algos = ["Euler", "Heun", "RK4"] if method == "Bandingkan Semua" else [method.replace("Runge-Kutta 4", "RK4")]
            
            fig, ax = plt.subplots()
            summary = []
            
            for algo in algos:
                px, py = solve(algo)
                ax.plot(px, py, label=algo, marker='.', markersize=4)
                summary.append({"Metode": algo, "y Akhir": py[-1], "Step": steps})
                
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            st.table(pd.DataFrame(summary))

# ==========================================
# MAIN APP NAV
# ==========================================
def main():
    with st.sidebar:
        st.title("Metode Numerik")
        st.markdown("---")
        menu = st.radio("Pilih Topik", [
            "Akar Persamaan", 
            "Sistem Linear (SPL)", 
            "Interpolasi", 
            "Integrasi", 
            "Diferensiasi", 
            "ODE"
        ])
        st.info("💡 **Tips Input:**\n- `2x` -> `2*x`\n- `e^-x` -> `exp(-x)`\n- `sin^2(x)` -> `sin(x)**2`")

    if menu == "Akar Persamaan": menu_roots()
    elif menu == "Sistem Linear (SPL)": menu_spl()
    elif menu == "Interpolasi": menu_interp()
    elif menu == "Integrasi": menu_integ()
    elif menu == "Diferensiasi": menu_diff()
    elif menu == "ODE": menu_ode()

if __name__ == "__main__":
    main()