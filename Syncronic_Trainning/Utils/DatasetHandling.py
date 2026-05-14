import numpy as np
from torchvision import datasets
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA Y PREPARACIÓN DEL DATASET MNIST
# ─────────────────────────────────────────────────────────────────────────────
# Usamos torchvision SOLO para descargar el dataset. Inmediatamente
# convertimos todo a arrays de NumPy y no volvemos a usar PyTorch.

def cargar_mnist(data_dir='./data'):
    """
    Descarga MNIST con torchvision y lo convierte completamente a NumPy.
    Retorna X (imágenes) e y (etiquetas) como arrays de NumPy.
    """
    print("Descargando MNIST con torchvision...")

    # Descargamos train y test por separado (estándar MNIST)
    transform = transforms.ToTensor()  # solo para poder acceder a los datos
    mnist_train = datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform)
    mnist_test  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Convertimos a numpy de inmediato
    # .data son los píxeles: tensor de forma (60000, 28, 28) con valores 0-255
    # .targets son las etiquetas: tensor de forma (60000,) con valores 0-9
    X_train_raw = mnist_train.data.numpy().astype(np.float64)   # (60000, 28, 28)
    y_train_raw = mnist_train.targets.numpy()                    # (60000,)
    X_test_raw  = mnist_test.data.numpy().astype(np.float64)    # (10000, 28, 28)
    y_test_raw  = mnist_test.targets.numpy()                     # (10000,)

    # Unimos todo en un solo bloque (70000 muestras totales)
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)   # (70000, 28, 28)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)   # (70000,)

    print(f"  Dataset completo: {X_all.shape[0]} imágenes de 28×28 píxeles")

    return X_all, y_all


def preprocesar(X_all, y_all, fraccion_entrenamiento=0.7):
    """
    Aplana, normaliza y divide el dataset.

    Aplanar: (N, 28, 28) → (N, 784)
      Cada imagen 2D se convierte en un vector 1D de 784 elementos.
      Esto nos da la matriz X que necesitamos para multiplicar con W1.

    Normalizar: dividir por 255
      Los píxeles van de 0 a 255. Dividimos por 255 para que queden entre
      0 y 1. Esto ayuda al gradiente descendente a converger más rápido
      porque los valores no son demasiado grandes.

    One-Hot Encoding:
      Si la etiqueta es el dígito 3, la convertimos al vector:
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
      Esto es necesario para calcular la pérdida con Softmax.
    """
    N = X_all.shape[0]

    # ── Aplanar ──────────────────────────────────────────────────────────────
    # reshape(-1, 784): -1 significa "NumPy calcula este eje automáticamente"
    # Básicamente decimos: "convierte cada imagen 28×28 en una fila de 784"
    X = X_all.reshape(-1, 784)          # (70000, 784)

    # ── Normalizar ───────────────────────────────────────────────────────────
    X = X / 255.0                       # valores entre 0 y 1

    # ── One-Hot Encoding ─────────────────────────────────────────────────────
    # np.eye(10) crea la matriz identidad 10×10
    # Si y[i] = 3, entonces np.eye(10)[3] = [0,0,0,1,0,0,0,0,0,0]
    Y = np.eye(10)[y_all]               # (70000, 10)

    # ── División 70% entrenamiento / 30% prueba ───────────────────────────────
    n_train = int(N * fraccion_entrenamiento)
    # Mezclamos aleatoriamente los índices para que el split sea representativo
    indices = np.random.permutation(N)
    idx_train = indices[:n_train]
    idx_test  = indices[n_train:]

    X_train, Y_train, y_train = X[idx_train], Y[idx_train], y_all[idx_train]
    X_test,  Y_test,  y_test  = X[idx_test],  Y[idx_test],  y_all[idx_test]

    print(f"\n  Muestras de entrenamiento : {X_train.shape[0]}")
    print(f"  Muestras de prueba        : {X_test.shape[0]}")
    print(f"  Forma de X_train          : {X_train.shape}  ← (muestras, píxeles)")
    print(f"  Forma de Y_train          : {Y_train.shape}  ← (muestras, clases)")

    return X_train, Y_train, y_train, X_test, Y_test, y_test


def particionar_dataset(X_train, Y_train, y_train, num_particiones, random_seed):
    """
    Divide el dataset de entrenamiento en K particiones iguales.
    
    IMPORTANTE: Usa RANDOM_SEED para garantizar que el servidor y workers
    particionen el dataset exactamente de la misma manera.
    
    Retorna lista de tuplas (X_k, Y_k, y_k)
    """
    N = X_train.shape[0]
    
    # ┌─ SEMILLA FIJA PARA SINCRONIZACIÓN ─┐
    np.random.seed(random_seed)
    # └────────────────────────────────────┘
    
    indices = np.random.permutation(N)
    
    X_mezclado = X_train[indices]
    Y_mezclado = Y_train[indices]
    y_mezclado = y_train[indices]
    
    X_partes = np.array_split(X_mezclado, num_particiones)
    Y_partes = np.array_split(Y_mezclado, num_particiones)
    y_partes = np.array_split(y_mezclado, num_particiones)
    
    particiones = []
    for k in range(num_particiones):
        particiones.append((X_partes[k], Y_partes[k], y_partes[k]))
    
    print(f"\n  Dataset dividido en {num_particiones} particiones (SEED={random_seed}):")
    for k, (X_k, Y_k, y_k) in enumerate(particiones):
        digitos_unicos = np.unique(y_k)
        print(f"    Partición {k+1}: {X_k.shape[0]:5d} muestras  │  "
              f"Dígitos presentes: {digitos_unicos}")
    
    return particiones
