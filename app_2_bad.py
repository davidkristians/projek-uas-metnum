import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

# ==========================================
# CONFIG & UTILS
# ==========================================
st.set_page_config(page_title="Ultimate Numerical Methods", layout="wide")

def parse_func(func_str):
    """Mengubah string fungsi menjadi fungsi Python yang bisa dieksekusi"""
    x, y = sp.symbols('x y')
    try:
        expr = sp.sympify(func_str.replace("^", "**"))
        f = sp.lambdify((x, y), expr, "numpy")
        return f, expr
    except:
        return None, None

def parse_func_x_only(func_str):
    """Versi 1 variabel (x)"""
    x = sp.symbols('x')
    try:
        expr = sp.sympify(func_str.replace("^", "**"))
        f = sp.lambdify(x, expr, "numpy")
        return f, expr
    except:
        return None, None

# ==========================================
# 1. METODE AKAR (ROOTS)
# ==========================================
def menu_roots():
    st.header("1. Metode Akar Persamaan")
    method = st.selectbox("Pilih Metode", ["Bisection", "Regula Falsi", "Newton-Raphson", "Secant"])
    
    func_str = st.text_input("Fungsi f(x)", "x^2 - 4")
    f, expr = parse_func_x_only(func_str)
    
    if not f:
        st.error("Fungsi tidak valid.")
        return

    col1, col2, col3 = st.columns(3)
    tol = col3.number_input("Toleransi", value=1e-5, format="%.6f")
    max_iter = int(col3.number_input("Max Iter", value=50))
    
    data = []
    
    if st.button("Hitung Akar"):
        if method == "Bisection":
            a = col1.number_input("Batas Bawah (a)", value=0.0)
            b = col2.number_input("Batas Atas (b)", value=3.0)
            if f(a)*f(b) >= 0:
                st.error("f(a) * f(b) harus < 0 (Tanda berbeda)")
                return
            for i in range(max_iter):
                c = (a + b) / 2
                err = abs(b - a)
                data.append({"Iter": i+1, "a": a, "b": b, "c": c, "f(c)": f(c), "Error": err})
                if f(c) == 0 or err < tol: break
                if f(a)*f(c) < 0: b = c
                else: a = c
                
        elif method == "Regula Falsi":
            a = col1.number_input("Batas Bawah (a)", value=0.0)
            b = col2.number_input("Batas Atas (b)", value=3.0)
            if f(a)*f(b) >= 0:
                st.error("f(a) * f(b) harus < 0")
                return
            prev_c = a
            for i in range(max_iter):
                fa, fb = f(a), f(b)
                c = (a*fb - b*fa) / (fb - fa)
                err = abs(c - prev_c)
                data.append({"Iter": i+1, "a": a, "b": b, "c": c, "f(c)": f(c), "Error": err})
                if abs(f(c)) < tol: break
                if f(a)*f(c) < 0: b = c
                else: a = c
                prev_c = c

        elif method == "Newton-Raphson":
            x0 = col1.number_input("Tebakan Awal (x0)", value=1.0)
            x = sp.symbols('x')
            df_expr = sp.diff(expr, x)
            df = sp.lambdify(x, df_expr, "numpy")
            st.latex(f"f'(x) = {sp.latex(df_expr)}")
            
            curr = x0
            for i in range(max_iter):
                val = f(curr)
                der = df(curr)
                if der == 0: break
                next_x = curr - val/der
                err = abs(next_x - curr)
                data.append({"Iter": i+1, "xi": curr, "f(xi)": val, "f'(xi)": der, "xi+1": next_x, "Error": err})
                if err < tol: break
                curr = next_x

        elif method == "Secant":
            x0 = col1.number_input("x0", value=0.0)
            x1 = col2.number_input("x1", value=1.0)
            for i in range(max_iter):
                fx0, fx1 = f(x0), f(x1)
                if fx1 - fx0 == 0: break
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                err = abs(x2 - x1)
                data.append({"Iter": i+1, "x(i-1)": x0, "x(i)": x1, "x(i+1)": x2, "Error": err})
                if err < tol: break
                x0, x1 = x1, x2

        st.table(pd.DataFrame(data))

# ==========================================
# 2. METODE SPL
# ==========================================
def menu_spl():
    st.header("2. Sistem Persamaan Linear (SPL)")
    method = st.selectbox("Metode", ["Gauss Elimination", "Gauss-Jordan", "LU Decomposition", "Jacobi", "Gauss-Seidel"])
    n = st.number_input("Jumlah Variabel (N)", 2, 5, 3)
    
    st.write("Input Matriks Augmented [A | b]")
    matrix = np.zeros((n, n+1))
    cols = st.columns(n+1)
    for r in range(n):
        cols_input = st.columns(n+1)
        for c in range(n+1):
            matrix[r, c] = cols_input[c].number_input(f"R{r}C{c}", value=0.0, key=f"spl_{r}_{c}")

    if st.button("Hitung SPL"):
        A = matrix[:, :-1]
        b = matrix[:, -1]
        
        if method == "Gauss Elimination":
            # Forward Elimination
            for i in range(n):
                for k in range(i+1, n):
                    factor = matrix[k,i] / matrix[i,i]
                    matrix[k, i:] -= factor * matrix[i, i:]
            # Back Subst
            x = np.zeros(n)
            for i in range(n-1, -1, -1):
                x[i] = (matrix[i, -1] - np.dot(matrix[i, i+1:n], x[i+1:n])) / matrix[i,i]
            st.write("Hasil Eliminasi Gauss:", matrix)
            st.success(f"Solusi: {x}")

        elif method == "Gauss-Jordan":
            # Simplified version using rref logic logic implemented previously
            # Reusing basic logic for brevity
            aug = matrix.copy()
            for i in range(n):
                aug[i] = aug[i] / aug[i,i]
                for k in range(n):
                    if k != i:
                        aug[k] -= aug[k,i] * aug[i]
            st.success(f"Solusi: {aug[:, -1]}")

        elif method == "LU Decomposition":
            import scipy.linalg
            try:
                P, L, U = scipy.linalg.lu(A)
                st.write("L Matrix:", L)
                st.write("U Matrix:", U)
                y = scipy.linalg.solve_triangular(L, P.dot(b), lower=True)
                x = scipy.linalg.solve_triangular(U, y)
                st.success(f"Solusi: {x}")
            except:
                st.error("Singular Matrix")

        elif method in ["Jacobi", "Gauss-Seidel"]:
            x = np.zeros(n)
            tol = 1e-6
            max_it = 50
            results = []
            
            for k in range(max_it):
                x_new = np.copy(x)
                for i in range(n):
                    s1 = sum(A[i][j] * x_new[j] for j in range(i)) if method == "Gauss-Seidel" else sum(A[i][j] * x[j] for j in range(i))
                    s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
                    x_new[i] = (b[i] - s1 - s2) / A[i][i]
                
                diff = np.linalg.norm(x_new - x)
                results.append(np.append(x_new, diff))
                if diff < tol: break
                x = x_new
                
            cols_df = [f"x{i+1}" for i in range(n)] + ["Error"]
            st.dataframe(pd.DataFrame(results, columns=cols_df))

# ==========================================
# 3. INTERPOLASI
# ==========================================
def menu_interp():
    st.header("3. Interpolasi")
    method = st.radio("Metode", ["Newton Polynomial", "Lagrange"])
    
    input_str_x = st.text_input("Data X (pisahkan koma)", "1, 2, 3")
    input_str_y = st.text_input("Data Y (pisahkan koma)", "1, 4, 9")
    val_x = st.number_input("Cari nilai Y pada X =", 2.5)

    try:
        X = np.array([float(k) for k in input_str_x.split(',')])
        Y = np.array([float(k) for k in input_str_y.split(',')])
    except:
        return

    if st.button("Hitung Interpolasi"):
        if method == "Lagrange":
            def L(k, x):
                term = 1
                for i in range(len(X)):
                    if i != k:
                        term *= (x - X[i]) / (X[k] - X[i])
                return term
            
            result = sum(Y[k] * L(k, val_x) for k in range(len(X)))
            st.info(f"Hasil Lagrange di x={val_x} adalah y={result}")

        elif method == "Newton Polynomial":
            n = len(X)
            coef = np.zeros([n, n])
            coef[:,0] = Y
            for j in range(1,n):
                for i in range(n-j):
                    coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (X[i+j] - X[i])
            
            st.write("Tabel Selisih Terbagi (Divided Difference):")
            st.write(coef)
            
            # Hitung nilai
            res = coef[0,0]
            term = 1.0
            for i in range(1, n):
                term *= (val_x - X[i-1])
                res += coef[0,i] * term
            st.info(f"Hasil Newton Polynomial di x={val_x} adalah y={res}")

# ==========================================
# 4. INTEGRASI NUMERIK
# ==========================================
def menu_integration():
    st.header("4. Integrasi Numerik")
    method = st.selectbox("Metode", ["Trapezium", "Simpson 1/3", "Simpson 3/8"])
    func_str = st.text_input("Fungsi f(x)", "x^2")
    f, _ = parse_func_x_only(func_str)
    
    c1, c2, c3 = st.columns(3)
    a = c1.number_input("Batas Bawah (a)", 0.0)
    b = c2.number_input("Batas Atas (b)", 2.0)
    n = int(c3.number_input("Jumlah Segmen (n)", 10))

    if st.button("Hitung Integral"):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        
        result = 0
        if method == "Trapezium":
            result = (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
        elif method == "Simpson 1/3":
            if n % 2 != 0: st.error("n harus genap untuk Simpson 1/3"); return
            result = (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
        elif method == "Simpson 3/8":
            if n % 3 != 0: st.error("n harus kelipatan 3 untuk Simpson 3/8"); return
            result = (3*h/8) * (y[0] + 3*np.sum(y[1:-1]) - 3*np.sum(y[3:-1:3]) + y[-1]) # Simplified logic needed here, using standard formula better
            # Manual loop for safety in 3/8
            s = y[0] + y[-1]
            for i in range(1, n):
                if i % 3 == 0: s += 2 * y[i]
                else: s += 3 * y[i]
            result = (3*h/8) * s

        st.success(f"Hasil Integrasi ({method}): {result}")

# ==========================================
# 5. DIFERENSIASI NUMERIK
# ==========================================
def menu_diff():
    st.header("5. Diferensiasi Numerik")
    func_str = st.text_input("Fungsi f(x)", "x^2")
    f, expr = parse_func_x_only(func_str)
    
    val_x = st.number_input("Titik x", 2.0)
    h = st.number_input("Step size (h)", 0.1)
    
    if st.button("Hitung Turunan"):
        forward = (f(val_x + h) - f(val_x)) / h
        backward = (f(val_x) - f(val_x - h)) / h
        central = (f(val_x + h) - f(val_x - h)) / (2*h)
        
        exact = float(sp.diff(expr, sp.symbols('x')).subs(sp.symbols('x'), val_x))
        
        res_df = pd.DataFrame({
            "Metode": ["Forward", "Backward", "Central", "Exact (Analytical)"],
            "Hasil": [forward, backward, central, exact],
            "Error": [abs(exact-forward), abs(exact-backward), abs(exact-central), 0]
        })
        st.table(res_df)

# ==========================================
# 6. PENYELESAIAN ODE
# ==========================================
def menu_ode():
    st.header("6. Persamaan Diferensial (ODE)")
    st.caption("dy/dx = f(x, y)")
    method = st.selectbox("Metode", ["Euler", "Heun", "Runge-Kutta 4"])
    
    func_str = st.text_input("f(x, y) = ", "x + y")
    f, _ = parse_func(func_str) # Perlu f(x,y)
    
    if not f: return

    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.number_input("x0", 0.0)
    y0 = c2.number_input("y0", 1.0)
    xh = c3.number_input("Target x", 1.0)
    h = c4.number_input("Step (h)", 0.2)
    
    if st.button("Selesaikan ODE"):
        n = int((xh - x0)/h)
        x = x0
        y = y0
        res = [{"iter": 0, "x": x, "y": y}]
        
        for i in range(n):
            if method == "Euler":
                y = y + h * f(x, y)
            elif method == "Heun":
                k1 = f(x, y)
                k2 = f(x + h, y + h*k1)
                y = y + (h/2)*(k1 + k2)
            elif method == "Runge-Kutta 4":
                k1 = f(x, y)
                k2 = f(x + 0.5*h, y + 0.5*h*k1)
                k3 = f(x + 0.5*h, y + 0.5*h*k2)
                k4 = f(x + h, y + h*k3)
                y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            
            x += h
            res.append({"iter": i+1, "x": x, "y": y})
            
        st.dataframe(pd.DataFrame(res))

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.sidebar.title("Numerical Methods Suite")
    menu = st.sidebar.radio("Pilih Modul", [
        "Roots (Akar)", 
        "Linear Systems (SPL)", 
        "Interpolation", 
        "Integration", 
        "Differentiation", 
        "ODE"
    ])
    
    if menu == "Roots (Akar)": menu_roots()
    elif menu == "Linear Systems (SPL)": menu_spl()
    elif menu == "Interpolation": menu_interp()
    elif menu == "Integration": menu_integration()
    elif menu == "Differentiation": menu_diff()
    elif menu == "ODE": menu_ode()

if __name__ == "__main__":
    main()