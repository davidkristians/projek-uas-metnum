from flask import Flask, render_template, request, jsonify
import sympy as sp
import copy

app = Flask(__name__)

# --- ROUTES (Jalur Navigasi) ---

@app.route('/')
def home():
    """Halaman Utama (Menu Pilihan)"""
    return render_template('index.html')

@app.route('/roots')
def page_roots():
    """Halaman Kalkulator Akar (Bisection/Newton)"""
    return render_template('roots.html')

@app.route('/spl')
def page_spl():
    """Halaman Kalkulator SPL (Gauss-Jordan)"""
    return render_template('spl.html')

# --- LOGIKA BACKEND ---

def evaluate_function(func_str, val):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(func_str)
        return float(expr.subs(x, val))
    except:
        return None

@app.route('/calculate_roots', methods=['POST'])
def calculate_roots():
    """Logika Perhitungan Akar (Kode Lama)"""
    data = request.json
    method = data.get('method')
    func_str = data.get('func')
    tol = float(data.get('tol'))
    max_iter = int(data.get('max_iter'))
    
    results = []
    root = None
    status = "Gagal"
    msg = ""

    try:
        if method == 'bisection':
            a = float(data.get('a'))
            b = float(data.get('b'))
            if evaluate_function(func_str, a) * evaluate_function(func_str, b) >= 0:
                return jsonify({'error': 'Syarat f(a)*f(b) < 0 tidak terpenuhi.'})
            
            c_old = a
            for i in range(1, max_iter + 1):
                c = (a + b) / 2
                fc = evaluate_function(func_str, c)
                error = abs(c - c_old) if i > 1 else abs(b - a)
                results.append({'iter': i, 'a': round(a, 5), 'b': round(b, 5), 'x': round(c, 5), 'fx': round(fc, 5), 'error': round(error, 5)})
                
                if abs(fc) < tol or error < tol:
                    root = c; status = "Sukses"; msg = f"Konvergen di iterasi {i}"; break
                if evaluate_function(func_str, a) * fc < 0: b = c
                else: a = c
                c_old = c

        elif method == 'newton':
            x0 = float(data.get('x0'))
            x = sp.symbols('x')
            expr = sp.sympify(func_str)
            deriv = sp.diff(expr, x)
            x_curr = x0
            for i in range(1, max_iter + 1):
                fx = float(expr.subs(x, x_curr))
                dfx = float(deriv.subs(x, x_curr))
                if dfx == 0: return jsonify({'error': 'Turunan nol.'})
                x_next = x_curr - (fx / dfx)
                error = abs(x_next - x_curr)
                results.append({'iter': i, 'x_old': round(x_curr, 5), 'fx': round(fx, 5), 'dfx': round(dfx, 5), 'x': round(x_next, 5), 'error': round(error, 5)})
                if error < tol: root = x_next; status = "Sukses"; msg = f"Konvergen di iterasi {i}"; break
                x_curr = x_next
                
        return jsonify({'status': status, 'message': msg, 'root': round(root, 6) if root else '-', 'data': results, 'derivative': str(sp.diff(sp.sympify(func_str), sp.symbols('x'))) if method == 'newton' else '-'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/calculate_spl', methods=['POST'])
def calculate_spl():
    """Logika Baru: Eliminasi Gauss-Jordan"""
    try:
        data = request.json
        matrix = data.get('matrix') # Matrix Augmented (N x N+1)
        n = len(matrix)
        steps = [] # Menyimpan snapshot matrix setiap langkah

        # Algoritma Gauss-Jordan
        for i in range(n):
            # 1. Pivot (Jadikan diagonal utama menjadi 1)
            pivot = matrix[i][i]
            if pivot == 0:
                return jsonify({'error': 'Pivot 0 terdeteksi. Sistem mungkin tidak memiliki solusi unik.'})
            
            for j in range(n + 1):
                matrix[i][j] /= pivot
            
            steps.append({'step': f"Normalisasi Baris {i+1} (Bagi dengan {round(pivot, 3)})", 'matrix': copy.deepcopy(matrix)})

            # 2. Eliminasi (Jadikan elemen lain di kolom yang sama menjadi 0)
            for k in range(n):
                if k != i:
                    factor = matrix[k][i]
                    for j in range(n + 1):
                        matrix[k][j] -= factor * matrix[i][j]
                    
                    if factor != 0:
                        steps.append({'step': f"Eliminasi Baris {k+1} menggunakan Baris {i+1}", 'matrix': copy.deepcopy(matrix)})

        # Hasil akhir ada di kolom terakhir
        solution = [row[n] for row in matrix]
        
        return jsonify({
            'status': 'Sukses',
            'solution': solution,
            'steps': steps
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)