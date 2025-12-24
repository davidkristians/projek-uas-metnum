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
    """
    Menghitung nilai fungsi f(x) dengan penanganan error yang lebih baik.
    """
    x = sp.symbols('x')
    try:
        # Ganti simbol pangkat '^' menjadi '**' agar sesuai sintaks Python
        func_str = func_str.replace('^', '**')
        
        # Parsing string menjadi ekspresi matematika
        expr = sp.sympify(func_str)
        
        # Substitusi nilai x dan hitung hasil float
        result = float(expr.subs(x, val))
        
        # Cek jika hasilnya bilangan imajiner (tidak valid untuk metode ini)
        if isinstance(result, complex):
            return None
            
        return result
    except Exception as e:
        print(f"Error evaluating function: {e}")
        return None

@app.route('/calculate_roots', methods=['POST'])
def calculate_roots():
    """Logika Perhitungan Akar (Bisection & Newton Raphson)"""
    data = request.json
    method = data.get('method')
    func_str = data.get('func')
    tol = float(data.get('tol'))
    max_iter = int(data.get('max_iter'))
    
    results = []
    
    try:
        # --- METODE BISECTION ---
        if method == 'bisection':
            a = float(data.get('a'))
            b = float(data.get('b'))
            
            fa = evaluate_function(func_str, a)
            fb = evaluate_function(func_str, b)

            if fa is None or fb is None:
                return jsonify({'error': 'Fungsi tidak valid atau mengandung sintaks yang salah.'})

            if fa * fb > 0:
                return jsonify({'error': 'Tebakan awal a dan b tidak mengurung akar (f(a)*f(b) > 0).'})

            for i in range(max_iter):
                c = (a + b) / 2
                fc = evaluate_function(func_str, c)
                
                if fc is None: break

                error = abs(b - a)
                results.append({
                    'iter': i+1,
                    'a': round(a, 6),
                    'b': round(b, 6),
                    'x': round(c, 6),
                    'fx': round(fc, 6),
                    'error': round(error, 6)
                })

                if abs(fc) < tol or error < tol:
                    return jsonify({'status': 'Sukses', 'root': c, 'data': results})

                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc

            return jsonify({'status': 'Gagal (Max Iter)', 'root': c, 'data': results})

        # --- METODE NEWTON RAPHSON ---
        elif method == 'newton':
            x_curr = float(data.get('x0'))
            
            # Cari turunan fungsi secara otomatis
            x_sym = sp.symbols('x')
            func_expr = sp.sympify(func_str.replace('^', '**'))
            deriv_expr = sp.diff(func_expr, x_sym) # Turunan pertama

            for i in range(max_iter):
                f_val = float(func_expr.subs(x_sym, x_curr))
                f_prime = float(deriv_expr.subs(x_sym, x_curr))

                if f_prime == 0:
                    return jsonify({'error': 'Turunan nol ditemukan. Metode Newton gagal.'})

                x_next = x_curr - (f_val / f_prime)
                error = abs(x_next - x_curr)

                results.append({
                    'iter': i+1,
                    'x_old': round(x_curr, 6),
                    'fx': round(f_val, 6),
                    'dfx': round(f_prime, 6),
                    'x': round(x_next, 6),
                    'error': round(error, 6)
                })

                if error < tol:
                    return jsonify({'status': 'Sukses', 'root': x_next, 'data': results})

                x_curr = x_next

            return jsonify({'status': 'Gagal (Max Iter)', 'root': x_curr, 'data': results})

    except Exception as e:
        return jsonify({'error': f"Terjadi kesalahan sistem: {str(e)}"})


@app.route('/calculate_spl', methods=['POST'])
def calculate_spl():
    """Logika Perhitungan SPL (Gauss-Jordan)"""
    try:
        data = request.json
        matrix = data.get('matrix') # Matrix Augmented (N x N+1)
        n = len(matrix)
        steps = [] # Menyimpan snapshot matrix setiap langkah

        # --- MULAI BAGIAN YG DI-IMPROVE (Gauss-Jordan dengan Partial Pivoting) ---
        for i in range(n):
            # 1. PARTIAL PIVOTING (Cari baris dengan nilai absolut terbesar di kolom i)
            # Tujuannya menghindari pembagian dengan 0 dan mengurangi error pembulatan
            max_row = i
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k
            
            # Tukar baris jika pivot terbesar bukan di baris saat ini
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                steps.append({
                    'step': f"Tukar Baris {i+1} dengan Baris {max_row+1} (Strategi Pivoting)", 
                    'matrix': copy.deepcopy(matrix)
                })

            # 2. Cek Pivot 0 (Setelah ditukar, jika masih 0 berarti matriks singular)
            pivot = matrix[i][i]
            if abs(pivot) < 1e-10: # Toleransi angka sangat kecil mendekati 0
                return jsonify({'error': 'Sistem tidak memiliki solusi unik (Matriks Singular).'})
            
            # 3. Normalisasi (Jadikan diagonal utama menjadi 1)
            for j in range(n + 1):
                matrix[i][j] /= pivot
            
            steps.append({'step': f"Normalisasi Baris {i+1} (Bagi dengan {round(pivot, 4)})", 'matrix': copy.deepcopy(matrix)})

            # 4. Eliminasi (Jadikan elemen lain di kolom yang sama menjadi 0)
            for k in range(n):
                if k != i:
                    factor = matrix[k][i]
                    for j in range(n + 1):
                        matrix[k][j] -= factor * matrix[i][j]
                    
                    if factor != 0: # Hanya catat step jika ada perubahan
                        steps.append({'step': f"Eliminasi Baris {k+1} - ({round(factor, 4)} * Baris {i+1})", 'matrix': copy.deepcopy(matrix)})
        # --- SELESAI BAGIAN YG DI-IMPROVE ---

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