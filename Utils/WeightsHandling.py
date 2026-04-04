
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

def inicializar_pesos(semilla=42):
    """
    ¿Por qué no inicializar todos los pesos en 0?
    Si todos los pesos son iguales, todas las neuronas de una capa aprenden
    exactamente lo mismo (problema de simetría). Necesitamos valores aleatorios
    para que cada neurona aprenda características distintas.

    Inicialización de He (para ReLU):
      W = N(0, 1) * sqrt(2 / n_entradas)
    Escala los pesos para que la varianza de las activaciones sea estable
    durante el forward pass. Sin esto, los gradientes pueden explotar o
    desvanecerse con redes profundas.
    """
    np.random.seed(semilla)

    # W1: forma (784, 128) — conecta entrada con capa oculta
    # Cada columna j contiene los pesos que van HACIA la neurona j de la capa oculta
    W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)

    # b1: forma (1, 128) — un sesgo por neurona en la capa oculta
    # Los sesgos los iniciamos en 0 (no tienen el problema de simetría)
    b1 = np.zeros((1, 128))

    # W2: forma (128, 10) — conecta capa oculta con capa de salida
    W2 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)

    # b2: forma (1, 10) — un sesgo por clase de salida
    b2 = np.zeros((1, 10))

    print(f"\n  W1: {W1.shape}  (pesos entrada → oculta)")
    print(f"  b1: {b1.shape}  (sesgos capa oculta)")
    print(f"  W2: {W2.shape}  (pesos oculta → salida)")
    print(f"  b2: {b2.shape}  (sesgos capa salida)")

    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# ACTUALIZACIÓN DE PESOS (Gradient Descent)
# ─────────────────────────────────────────────────────────────────────────────

def actualizar_pesos(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Gradient Descent (Descenso del Gradiente):
      W_nuevo = W_viejo - lr · dW

    Intuición:
      El gradiente apunta hacia la dirección de mayor aumento de la pérdida.
      Restándolo, vamos en la dirección opuesta: hacia donde la pérdida baja.
      lr (learning rate) controla qué tan grande es cada paso.

    ¿Por qué lr=0.4?
      - Si lr es muy grande: los pasos son enormes y el modelo "rebota" sin converger.
      - Si lr es muy pequeño: el aprendizaje es muy lento.
      - 0.4 es un valor razonable para este problema con full-batch GD.
    """
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

