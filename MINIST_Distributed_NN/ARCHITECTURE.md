# DISEÑO ARQUITECTÓNICO — Entrenamiento Distribuido con Sockets

## Resumen Ejecutivo

Se ha transformado un sistema de entrenamiento neural basado en **multiprocessing** (local) a una arquitectura **distribuida verdadera** usando **sockets** y **pickle** para comunicación entre procesos potencialmente en diferentes máquinas.

### Cambios Principales

| Aspecto | Antes (Multiprocessing) | Ahora (Sockets) |
|---------|------------------------|-----------------|
| **Comunicación** | Memoria compartida | Sockets TCP/IP + Pickle |
| **Distribución** | Solo local (1 máquina) | Red (múltiples máquinas) |
| **Pesos** | Pasan directamente en Pool.map | Serializados en mensajes |
| **Gradientes** | Retornados en tuplas | Empaquetados en MessageFromWorker |
| **Sincronización** | Pool.map (bloqueante) | Protocolos manuales |
| **Escalabilidad** | ~4-8 procesos por máquina | Teóricamente ilimitado (N workers) |

---

## Arquitectura Detallada

### 1. Inicialización

```
┌──────────────────────────────────────────────────────────┐
│                    SERVER STARTUP                        │
├──────────────────────────────────────────────────────────┤
│ 1. Cargar MNIST (49,000 imágenes)                        │
│ 2. Crear pesos iniciales W1, b1, W2, b2                  │
│ 3. Abrir puerto 9999                                     │
│ 4. Esperar 4 conexiones de workers                       │
│ 5. Imprimir estado de espera                             │
└──────────────────────────────────────────────────────────┘

                         ↓

┌──────────────────────────────────────────────────────────┐
│                  WORKER STARTUP (x4)                     │
├──────────────────────────────────────────────────────────┤
│ 1. Conectar a SERVER en localhost:9999                   │
│ 2. Cargar MNIST localmente (49,000 imágenes)            │
│ 3. Particionar dataset igual que server                  │
│    (IMPORTANTE: Mismo shuffle → Misma partición)         │
│ 4. Esperar instrucciones del server                      │
└──────────────────────────────────────────────────────────┘
```

### 2. Distribución de Datos

#### Problema Original

En multiprocessing, los datos se pasaban directamente a través de Pool.map:

```python
for idx, (X_k, Y_k, y_k) in enumerate(particiones):
    args = (idx, X_k, Y_k, y_k, W1_global, ...)
    # El pool picklea estos datos automáticamente
```

#### Solución Distribuida

Ambos (server y workers) cargan MNIST localmente y aplican **la misma partición**:

```
SERVIDOR
├─ Cargar MNIST
├─ Shuffle con random seed (o mismo shuffle)
└─ Particionar: [Batch0, Batch1, Batch2, Batch3]

WORKER 0  ── Cargar MNIST ──┐
WORKER 1  ── Cargar MNIST ──┼─ Shuffle Idéntico (¡CLAVE!)
WORKER 2  ── Cargar MNIST ──┤
WORKER 3  ── Cargar MNIST ──┘
             └─ Particionar: [Batch0, Batch1, Batch2, Batch3]
```

**Crítico**: `np.random.permutation()` debe producir el MISMO índice shuffle en todas partes.

⚠️ **Actualmente**: Cada proceso obtiene un shuffle diferente. Solución: Establecer un seed global.

```python
# En ambos archivos (Server.py y Worker.py)
np.random.seed(42)  # ← Agregar esto
indices = np.random.permutation(N)
```

### 3. Protocolo de Mensajes

#### Estructura

```
Mensaje Binario sobre Socket:
┌─────────────────────────────────────┐
│  4 bytes: LENGTH (big-endian)       │  ← Prefijo de longitud
├─────────────────────────────────────┤
│  Pickle serializado                 │
│  ├─ MessageFromServer o             │
│  └─ MessageFromWorker               │
└─────────────────────────────────────┘
```

#### MessageFromServer (Server → Worker)

```python
@dataclass
class MessageFromServer:
    batch_id: int              # ¿Qué partición? 0,1,2,3
    epoch: int                 # ¿Qué época? 1...100
    init_signal: bool          # Primera época?
    stop_signal: bool          # Última época?
    learning_rate: float       # Tasa: 0.1
    W1, b1, W2, b2: ndarray   # Pesos globales actuales
```

Tamaño: ~50KB (pesos vectores) + overhead

#### MessageFromWorker (Worker → Server)

```python
@dataclass
class MessageFromWorker:
    worker_id: int             # Identificador del worker
    batch_id: int              # Qué partición entrenó
    epoch: int                 # Qué época entrenó
    dW1, db1, dW2, db2: ndarray  # GRADIENTES (no pesos)
    loss: float                # Pérdida del batch
    accuracy: float            # Precisión del batch
    training_time: float       # Cuánto tardó
```

Tamaño: ~50KB (gradientes vectores) + overhead

### 4. Ciclo de Entrenamiento por Época

```
ÉPOCA N
═══════════════════════════════════════════════════════════════════

1. SERVER: Distribuir Trabajo
   ├─ Para cada worker (0,1,2,3):
   │  └─ Enviar MessageFromServer(
   │        batch_id=0/1/2/3,
   │        epoch=N,
   │        W1=copia de W1_global,
   │        b1=copia de b1_global,
   │        W2=copia de W2_global,
   │        b2=copia de b2_global
   │     )
   └─ Esperar respuestas

2. WORKERS: Entrenar (EN PARALELO)
   ├─ Worker 0:
   │  ├─ Recibir MessageFromServer
   │  ├─ Desempaquetar mensaje
   │  ├─ Forward pass en Batch 0 (12,250 muestras)
   │  ├─ Backward pass → Calcular gradientes
   │  ├─ Empaquetar gradientes en MessageFromWorker
   │  └─ Enviar respuesta
   │
   ├─ Worker 1: (Igual pero con Batch 1)
   ├─ Worker 2: (Igual pero con Batch 2)
   ├─ Worker 3: (Igual pero con Batch 3)
   └─ ⏱️  Tiempo: max(tiempo_worker_0, ..., tiempo_worker_3)

3. SERVER: Recolectar Resultados
   ├─ Recibir 4 MessageFromWorker (con gradientes)
   ├─ Desempaquetar:
   │  ├─ dW1_0, db1_0, dW2_0, db2_0  (de Worker 0)
   │  ├─ dW1_1, db1_1, dW2_1, db2_1  (de Worker 1)
   │  ├─ dW1_2, db1_2, dW2_2, db2_2  (de Worker 2)
   │  └─ dW1_3, db1_3, dW2_3, db2_3  (de Worker 3)
   └─ Registrar métricas (loss, acc)

4. SERVER: Actualizar Pesos Globales
   ├─ Promediar gradientes:
   │  ├─ dW1_prom = (dW1_0 + dW1_1 + dW1_2 + dW1_3) / 4
   │  ├─ db1_prom = (db1_0 + db1_1 + db1_2 + db1_3) / 4
   │  ├─ dW2_prom = (dW2_0 + dW2_1 + dW2_2 + dW2_3) / 4
   │  └─ db2_prom = (db2_0 + db2_1 + db2_2 + db2_3) / 4
   │
   ├─ Actualizar con descenso de gradiente:
   │  ├─ W1 = W1 - lr * dW1_prom
   │  ├─ b1 = b1 - lr * db1_prom
   │  ├─ W2 = W2 - lr * dW2_prom
   │  └─ b2 = b2 - lr * db2_prom
   │
   └─ W1, b1, W2, b2 ahora son W_global(N) para época N+1

5. SERVER: Evaluar
   ├─ Forward pass en TODO el dataset de entrenamiento (49,000)
   ├─ Calcular loss global y accuracy
   ├─ Forward pass en todos los datos de test
   ├─ Calcular test accuracy
   └─ Mostrar progreso

═══════════════════════════════════════════════════════════════════
```

### 5. Comparación: Multiprocessing vs Sockets

#### Multiprocessing (código antiguo)

```python
# Server.py o in-process
with Pool(processes=4) as pool:
    resultados = pool.map(worker_entrenar_particion, args_lista)
    # Automático:
    # - Pickle args
    # - Fork procesos
    # - Ejecutar worker_entrenar_particion
    # - Pickle resultados
    # - Enviar de vuelta
    # - Unpickle
    # - Retornar como lista
```

**Ventajas:**
- ✓ Rápido (compartir memoria)
- ✓ Sincronización automática
- ✓ Simple de codificar

**Desventajas:**
- ✗ Solo funciona en 1 máquina
- ✗ Propietario de Python
- ✗ No es escalable a red

#### Sockets (código nuevo)

```python
# Server.py
for batch_id in range(4):
    msg = MessageFromServer(...)
    send_message(sock[batch_id], msg)

for batch_id in range(4):
    result_msg = receive_message(sock[batch_id])
    procesar_gradientes(result_msg)

# Worker.py
while True:
    msg = receive_message(server_socket)
    train_epoch(...)
    result = MessageFromWorker(...)
    send_message(server_socket, result)
```

**Ventajas:**
- ✓ Funciona entre máquinas (verdadera distribución)
- ✓ Estándar (sockets TCP/IP)
- ✓ Escalable a N workers
- ✓ Agnóstico de lenguaje (podría usar Java, Go, etc.)

**Desventajas:**
- ✗ Más lento (serialización)
- ✗ Manejo manual de sincronización
- ✗ Debugging más difícil

---

## Plano de Ejecución

### Terminal 1: Server

```bash
$ cd Distributed_NN && python Server.py

[Server inicia]
├─ Carga MNIST
├─ Abre socket en localhost:9999
├─ Espera 4 conexiones
└─ [Esperando] Worker batch_id=0...
```

### Terminal 2: Worker 0

```bash
$ cd Distributed_NN && python Worker.py

[Worker 0]
├─ Se conecta a localhost:9999
├─ [Server] ✓ Worker 0 conectado
├─ Carga MNIST localmente
├─ Particiona
├─ Espera instrucciones
└─ [Esperando] Worker batch_id=1...
```

### Terminal 3: Worker 1

```bash
$ cd Distributed_NN && python Worker.py

[Worker 1]
├─ Se conecta a localhost:9999
├─ [Server] ✓ Worker 1 conectado
├─ Carga MNIST
├─ Particiona
└─ [Esperando] Worker batch_id=2...
```

### Terminal 4: Worker 2

```bash
$ cd Distributed_NN && python Worker.py

[Worker 2]
├─ Se conecta
├─ [Server] ✓ Worker 2 conectado
└─ [Esperando] Worker batch_id=3...
```

### Terminal 5: Worker 3

```bash
$ cd Distributed_NN && python Worker.py

[Worker 3]
├─ Se conecta
├─ [Server] ✓ Worker 3 conectado ← ¡El último!
└─ [Server] TODOS CONECTADOS → Comienza entrenamiento
```

---

## Implementación Técnica

### Funciones Clave

#### send_message() / receive_message()

```python
def send_message(sock, message):
    """
    Serializa y envía con longitud prefijada.
    
    Formato:
    [4 bytes: long] [pickled data]
    """
    data = pickle.dumps(message)
    header = struct.pack('!I', len(data))  # Big-endian
    sock.sendall(header + data)

def receive_message(sock):
    """
    Recibe mensajes con longitud prefijada.
    """
    header = sock.recv(4)
    length = struct.unpack('!I', header)[0]
    data = b''
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        data += chunk
    return pickle.loads(data)
```

**Por qué longitud prefijada?**
- Los sockets envían bytes, no mensajes estructurados
- Sin conocer la longitud, no sabríamos dónde termina un mensaje
- El prefijo permite unpickle exacto

#### Iteración de Gradientes

```python
# Server.py - Actualizar pesos
def update_global_weights(self, gradientes_epoca):
    dW1_prom, db1_prom, dW2_prom, db2_prom = promediar_gradientes(gradientes_epoca)
    
    # Descenso de gradiente
    W1 = W1 - lr * dW1_prom
    b1 = b1 - lr * db1_prom
    # ... etc
```

**Nota**: En vez de promediar PESOS (como en Arnovi/Diego), 
promediamos GRADIENTES, que es más principled para federated learning.

---

## Posibles Mejoras

### 1. Determinismo en Shuffle

**Problema Actual**: Cada proceso obtiene shuffle diferente

**Solución**:
```python
# Ambos archivos
np.random.seed(42)
indices = np.random.permutation(N)
```

### 2. Compresión de Gradientes

Los gradientes son vectores grandes (50KB cada uno).

**Opción**: Comprimir antes de enviar

```python
import gzip

data = pickle.dumps(message)
compressed = gzip.compress(data)
# Enviar compressed...
```

### 3. Canal Seguro (SSL/TLS)

```python
import ssl

context = ssl.create_default_context()
with socket.socket() as sock:
    with context.wrap_socket(sock) as ssocked:
        ssocked.connect((host, port))
```

### 4. Recuperación de Fallos

Si un worker se desconecta:
- Server detecta timeout
- Server continúa con otros workers
- Re-intenta conexión

### 5. Escalado Automático

Detectar # de workers y adaptarse automáticamente

```python
# Servidor
connected = 0
max_workers = 4
while connected < max_workers:
    accept worker, connected += 1
    # Pero si solo se conectan 3, continuar de todas formas
```

---

## Resumen

La arquitectura distribuida con sockets transforma un sistema local en uno potencialmente escalable globalmente, manteniendo la lógica de entrenamiento igual.

**Costo**: Overhead de serialización y latencia de red.
**Beneficio**: Verdadera distribución, escalabilidad, estándar abierto.

Este es el primer paso hacia **Federated Learning** en producción.
