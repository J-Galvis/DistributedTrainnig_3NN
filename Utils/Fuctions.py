
# FUNCIONES DE ACTIVACIÓN, PÉRDIDA Y PREDICCIÓN

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE ACTIVACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def relu(Z):
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)

    ¿Por qué necesitamos activaciones?
    Sin ellas, la red sería equivalente a UNA sola multiplicación de matrices
    sin importar cuántas capas tenga (composición de funciones lineales = lineal).
    ReLU introduce no-linealidad, permitiendo a la red aprender patrones complejos.

    Ejemplo con una fila: [-2.5, 0.0, 3.1, -0.4, 1.7]
    Después de ReLU:       [ 0.0, 0.0, 3.1,  0.0, 1.7]
    """
    return np.maximum(0, Z)


def derivada_relu(Z):
    """
    Derivada de ReLU:
      ReLU'(x) = 1  si x > 0
      ReLU'(x) = 0  si x ≤ 0

    En NumPy: (Z > 0) retorna True/False, que se comporta como 1/0
    en operaciones aritméticas.

    Usamos esto en el backward pass para saber qué neuronas "se activaron"
    y cuáles no contribuyeron al error.
    """
    return (Z > 0).astype(np.float64)


def softmax(Z):
    """
    Softmax: convierte un vector de números reales en probabilidades que suman 1.

      Softmax(z_i) = exp(z_i) / Σ exp(z_j)

    Por qué restamos el máximo (truco de estabilidad numérica):
      exp(700) desborda a infinito en float64. Si restamos el máximo primero,
      los valores quedan en rango manejable sin cambiar el resultado matemático
      porque exp(a-c) / Σ exp(b-c) = exp(a) / Σ exp(b).

    Z tiene forma (N, 10). Procesamos todas las muestras a la vez.
    keepdims=True mantiene las dimensiones para que la resta funcione bien.
    """
    Z_estable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_estable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCION DE PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy(A2, Y):
    """
    Entropía Cruzada: L = -1/N · Σ Σ Y_ij · log(A2_ij)

    Mide la diferencia entre la distribución predicha (A2) y la real (Y).
    Cuando la predicción es perfecta (A2[clase_correcta] ≈ 1), la pérdida → 0.
    Cuando la predicción es terrible (A2[clase_correcta] ≈ 0), la pérdida → ∞.

    np.clip evita log(0) = -∞ que haría explotar los cálculos.
    """
    N = Y.shape[0]
    A2_seguro = np.clip(A2, 1e-15, 1 - 1e-15)
    return -np.sum(Y * np.log(A2_seguro)) / N

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PRECISIÓN
# ─────────────────────────────────────────────────────────────────────────────

def precision(y_pred, y_real):
    """
    Porcentaje de predicciones correctas.
    np.mean(y_pred == y_real) compara elemento a elemento y promedia los True/False.
    """
    return np.mean(y_pred == y_real) * 100
  
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predecir(X, W1, b1, W2, b2):
    """
    Realiza el forward pass y toma la clase con mayor probabilidad.
    np.argmax devuelve el índice del valor máximo en cada fila.
    Si A2[i] = [0.01, 0.02, 0.05, 0.85, ...], argmax = 3 → predicción: dígito 3
    """
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def forward(X, W1, b1, W2, b2):
    """
    Propaga la entrada X a través de la red para obtener predicciones.

    Álgebra lineal involucrada:
    ───────────────────────────
    Z1 = X · W1 + b1
      X  es (N, 784)
      W1 es (784, 128)
      X·W1 es (N, 128)   ← regla de multiplicación de matrices: (m,n)·(n,p)=(m,p)
      b1 es (1, 128)     ← se suma con broadcasting: NumPy expande b1 a (N,128)

    Cada fila de Z1 corresponde a una muestra. El elemento Z1[i,j] es la
    suma ponderada de todos los píxeles de la muestra i hacia la neurona j.
    En otras palabras: Z1[i,j] = Σ_k X[i,k] * W1[k,j] + b1[0,j]

    Retornamos Z1 y Z2 porque los necesitamos en el backward pass
    para calcular las derivadas de ReLU.
    """
    # ── Capa 1 ───────────────────────────────────────────────────────────────
    Z1 = X @ W1 + b1        # (N, 784) @ (784, 128) + (1, 128) = (N, 128)
    A1 = relu(Z1)           # (N, 128)  — aplicamos ReLU elemento a elemento

    # ── Capa 2 ───────────────────────────────────────────────────────────────
    Z2 = A1 @ W2 + b2       # (N, 128) @ (128, 10)  + (1, 10)  = (N, 10)
    A2 = softmax(Z2)        # (N, 10)  — convertimos a probabilidades

    return Z1, A1, Z2, A2


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD PASS (Backpropagation)
# ─────────────────────────────────────────────────────────────────────────────

def backward(X, Y, Z1, A1, A2, W2):
    """
    Calcula los gradientes usando la regla de la cadena (chain rule).

    Idea intuitiva:
    ───────────────
    Queremos saber: "si aumento un poco W2[i,j], ¿sube o baja la pérdida?"
    Eso es la derivada parcial ∂L/∂W2[i,j], o sea el gradiente dW2.

    Con la regla de la cadena:
      ∂L/∂W2 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂W2

    El truco bonito de Softmax + Cross-Entropy es que sus derivadas
    combinadas simplifican a:
      dZ2 = A2 - Y

    Esto significa: el error es simplemente "qué tan diferente fue la
    probabilidad predicha de la etiqueta real". Elegante y eficiente.

    Dimensiones (verificación de álgebra):
    ───────────────────────────────────────
    dZ2  : (N, 10)
    dW2  : (128, N) @ (N, 10)  =  (128, 10)  ← misma forma que W2 ✓
    db2  : promedio de (N, 10) sobre eje 0    =  (1, 10)   ← misma forma que b2 ✓
    dA1  : (N, 10) @ (10, 128) =  (N, 128)
    dZ1  : (N, 128) * (N, 128) =  (N, 128)   ← multiplicación elemento a elemento
    dW1  : (784, N) @ (N, 128) =  (784, 128) ← misma forma que W1 ✓
    db1  : promedio de (N, 128) sobre eje 0   =  (1, 128)  ← misma forma que b1 ✓
    """
    N = X.shape[0]

    # ── Gradientes de la capa de salida ───────────────────────────────────────
    dZ2 = A2 - Y                              # (N, 10)  — error de predicción
    dW2 = A1.T @ dZ2 / N                     # (128, 10) — promedio del gradiente
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N  # (1, 10)

    # ── Propagamos el error hacia atrás a través de W2 ────────────────────────
    dA1 = dZ2 @ W2.T                          # (N, 128) — error en capa oculta

    # ── Pasamos a través de la derivada de ReLU ───────────────────────────────
    # Solo las neuronas que estaban "activas" (Z1 > 0) reciben gradiente
    dZ1 = dA1 * derivada_relu(Z1)            # (N, 128)

    # ── Gradientes de la capa de entrada ──────────────────────────────────────
    dW1 = X.T @ dZ1 / N                      # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N  # (1, 128)

    return dW1, db1, dW2, db2
