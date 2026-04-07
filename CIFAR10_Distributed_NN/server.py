"""
=============================================================================
  SERVIDOR — ENTRENAMIENTO NEURONAL DISTRIBUIDO CON SOCKETS
=============================================================================

El servidor:
1. Carga y particiona el dataset CIFAR10 en K particiones
2. Abre un socket servidor esperando conexiones de workers
3. Para cada época:
   - Envía a cada worker: epoch, batch_ids, pesos globales, learning_rate, init/stop signal
   - Recibe de cada worker: gradientes calculados
   - Promedia los gradientes
   - Actualiza los pesos globales
4. Al final, evaluación en test
=============================================================================
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import socket
import time
import json
from datetime import datetime
from typing import Dict, List
import argparse

# Agregar el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defineNetwork import Net
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from messageHandling import send_message, receive_message

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL SERVIDOR
# ─────────────────────────────────────────────────────────────────────────────

# Importar constantes desde TrainingConfig
NUM_WORKERS = TrainingConfig.num_workers
LEARNING_RATE = TrainingConfig.learning_rate
INTERVALO_LOG = TrainingConfig.intervalo_log
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
BATCH_SIZE = TrainingConfig.batch_size
SAVE_FILE = TrainingConfig.save_file
NUM_EPOCHS = TrainingConfig.epocas

def testingNetwork(testloader, net):
    """Evalúa el modelo en el dataset de prueba"""
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)

def accuracyTest(net, transform, num_workers):
    """Carga el dataset de prueba y evalúa"""
    print("Starting testing...")
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=num_workers)
    return testingNetwork(testloader, net)

class DistributedTrainingServer:
    """
    Servidor de Entrenamiento Distribuido CIFAR10.
    
    Maneja conexiones de múltiples workers y coordina el entrenamiento federado.
    """
    
    def __init__(self, host, port, num_workers, epocas, learning_rate):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epocas = epocas
        self.learning_rate = learning_rate
        
        # Modelo
        self.net = Net()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=1e-2,
                                     betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=epocas,
            steps_per_epoch=len(TRAINLOADER),
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )
        
        # Conexiones de workers
        self.worker_sockets: Dict[int, socket.socket] = {}
        self.worker_connected = {}
        
        # Datos
        self.total_batches = len(TRAINLOADER)
        
        # Historial de checkpoints por INTERVALO_LOG
        self.historial_intervalo_epochs = []      # Épocas en las que se guardó
        self.historial_intervalo_times = []       # Tiempos acumulados
        self.historial_intervalo_acc_test = []    # Precisión en test
        self.historial_intervalo_loss = []        # Loss promedio
    
    def setup_socket_server(self):
        """Configura el socket servidor."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        self.server_socket.settimeout(SOCKET_TIMEOUT)
        
        print(f"\n{'='*70}")
        print(f"  SERVIDOR DISTRIBUIDO — ESCUCHANDO EN {self.host}:{self.port}")
        print(f"{'='*70}")
        print(f"  Esperando {self.num_workers} conexiones de workers...")
    
    def wait_for_workers(self):
        """
        Espera a que se conecten todos los workers.
        Asigna worker_id basado en el orden de conexión.
        Envía mensaje de sincronización inicial a cada worker.
        """
        # FASE 1: Aceptar todas las conexiones
        for worker_id in range(self.num_workers):
            try:
                print(f"\n  [Esperando] Worker {worker_id}...")
                client_socket, client_address = self.server_socket.accept()
                client_socket.settimeout(SOCKET_TIMEOUT)
                
                self.worker_sockets[worker_id] = client_socket
                self.worker_connected[worker_id] = True
                
                print(f"  ✓ Worker {worker_id} conectado desde {client_address}")
                
            except socket.timeout:
                print(f"\n  ✗ Timeout esperando worker {worker_id}")
                raise
            except Exception as e:
                print(f"\n  ✗ Error aceptando conexión: {e}")
                raise
        
        # FASE 2: Enviar mensaje de sincronización a todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE SINCRONIZACIÓN — Enviando señales de inicio a workers")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                # Crear mensaje de sincronización (epoch=0, init_signal=True)
                params = {name: param.data.cpu().numpy() for name, param in self.net.named_parameters()}
                message = MessageFromServer(
                    batch_ids=[],
                    epoch=0,
                    init_signal=True,
                    stop_signal=False,
                    learning_rate=self.learning_rate,
                    params=params
                )
                
                # Enviar mensaje de sincronización
                sock = self.worker_sockets[worker_id]
                send_message(sock, message)
                
                print(f"    → Sincronización enviada a worker {worker_id}")
                
            except Exception as e:
                print(f"    ✗ Error sincronizando worker {worker_id}: {e}")
                raise
        
        # FASE 3: Esperar confirmación (handshake) de todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE HANDSHAKE — Esperando confirmación de workers")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                sock = self.worker_sockets[worker_id]
                ready_msg = receive_message(sock)
                
                print(f"    ✓ Worker {worker_id} listo (dataset_size={ready_msg.dataset_size})")
                
            except Exception as e:
                print(f"    ✗ Error esperando confirmación de worker {worker_id}: {e}")
                raise
        
        print(f"  ✓ Todos los workers sincronizados y listos para entrenar")
    
    def distribute_work(self, epoch):
        """
        Distribuye trabajo a todos los workers para una época.
        
        Envía a cada worker: epoch, batch_ids, pesos globales, learning_rate, etc.
        """
        print(f"\n  {'─'*68}")
        print(f"  ÉPOCA {epoch}/{self.epocas} — DISTRIBUYENDO TRABAJO A WORKERS")
        print(f"  {'─'*68}")
        
        # Calcular distribución de batches entre workers
        batches_per_worker = self.total_batches // self.num_workers
        remaining_batches = self.total_batches % self.num_workers
        
        for worker_id in range(self.num_workers):
            try:
                # Calcular batches para este worker
                batch_count = batches_per_worker + (1 if worker_id < remaining_batches else 0)
                batch_start = worker_id * batches_per_worker + min(worker_id, remaining_batches)
                batch_ids = list(range(batch_start, batch_start + batch_count))
                
                # Obtener parámetros actuales del modelo
                params = {name: param.data.cpu().numpy() for name, param in self.net.named_parameters()}
                
                # Crear mensaje para el worker
                message = MessageFromServer(
                    batch_ids=batch_ids,
                    epoch=epoch,
                    init_signal=(epoch == 1),
                    stop_signal=(epoch == self.epocas),
                    learning_rate=self.learning_rate,
                    params=params
                )
                
                # Enviar al worker
                sock = self.worker_sockets[worker_id]
                send_message(sock, message)
                
                print(f"    ✓ Enviado a worker {worker_id}: epoch={epoch}, batches={len(batch_ids)}")
                
            except Exception as e:
                print(f"    ✗ Error enviando a worker {worker_id}: {e}")
                raise
    
    def collect_results(self):
        """
        Recolecta resultados de todos los workers para la época actual.
        
        Recibe gradientes y métricas de cada worker.
        """
        print(f"\n  {'─'*68}")
        print(f"  RECOLECTANDO RESULTADOS DE WORKERS")
        print(f"  {'─'*68}")
        
        all_messages = []
        
        for worker_id in range(self.num_workers):
            try:
                sock = self.worker_sockets[worker_id]
                message = receive_message(sock)
                
                all_messages.append(message)
                print(f"    ✓ Worker {worker_id}: {message}")
                
            except Exception as e:
                print(f"    ✗ Error recibiendo de worker {worker_id}: {e}")
                raise
        
        return all_messages
    
    def average_gradients(self, messages_list):
        """
        Promedia los gradientes de todos los workers.
        
        Retorna:
            Dict con gradientes promediados para cada parámetro
        """
        num_workers = len(messages_list)
        
        # Inicializar diccionario de gradientes promediados
        avg_grads = {}
        
        # Iterar sobre todas las claves de parámetros del primer worker
        if num_workers > 0:
            for param_name in messages_list[0].gradients.keys():
                # Promediar este parámetro de todos los workers
                grads_list = [msg.gradients[param_name] for msg in messages_list]
                avg_grads[param_name] = sum(grads_list) / num_workers
        
        return avg_grads
    
    def update_model(self, avg_grads):
        """
        Actualiza los pesos del modelo usando los gradientes promediados.
        """
        self.optimizer.zero_grad()
        
        # Asignar gradientes a los parámetros
        for name, param in self.net.named_parameters():
            if name in avg_grads:
                param.grad = torch.tensor(avg_grads[name], dtype=param.dtype, device=param.device)
        
        # Aplicar clipping
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        
        # Actualizar pesos
        self.optimizer.step()
        self.scheduler.step()
    
    def evaluate_global_model(self, epoch, tiempo_actual, test_acc, avg_loss):
        """
        Evalúa el modelo global y guarda métricas en historial.
        
        Parámetros:
            epoch: int, número de época actual
            tiempo_actual: float, tiempo transcurrido desde el inicio del entrenamiento
            test_acc: float, precisión en test
            avg_loss: float, pérdida promedio de la época
        """
        if epoch % INTERVALO_LOG == 0 or epoch == 1:
            self.historial_intervalo_epochs.append(epoch)
            self.historial_intervalo_times.append(round(tiempo_actual, 6))
            self.historial_intervalo_acc_test.append(round(test_acc, 2))
            self.historial_intervalo_loss.append(round(avg_loss, 6))
            
            print(f"\n  {'─'*68}")
            print(f"  EVALUACIÓN GLOBAL — ÉPOCA {epoch}/{self.epocas}")
            print(f"  {'─'*68}")
            print(f"    ✓ GLOBAL → Loss: {avg_loss:.4f} │ Acc Test: {test_acc:.1f}%")
            print(f"    ⏱ Tiempo acumulado: {tiempo_actual:.2f}s")
    
    def training_loop(self):
        """
        Bucle principal de entrenamiento.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO ENTRENAMIENTO DISTRIBUIDO")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        
        try:
            for epoch in range(1, self.epocas + 1):
                epoch_start = time.time()
                
                # Distribuir trabajo
                self.distribute_work(epoch)
                
                # Recolectar resultados
                messages = self.collect_results()
                
                # Promediar gradientes y calcular pérdida promedio
                avg_grads = self.average_gradients(messages)
                avg_loss = sum(msg.loss for msg in messages) / len(messages) if messages else 0.0
                
                # Actualizar modelo
                self.update_model(avg_grads)
                
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                
                # Evaluar en test (cada INTERVALO_LOG épocas)
                if epoch % INTERVALO_LOG == 0 or epoch == 1:
                    test_acc = accuracyTest(self.net, TRANSFORM, 0)
                else:
                    test_acc = 0.0
                
                # Registrar métricas en historial
                self.evaluate_global_model(epoch, total_time, test_acc, avg_loss)
                
                if epoch % INTERVALO_LOG == 0 or epoch == 1:
                    print(f"  Epoch {epoch} completada en {epoch_time:.4f}s (Total: {total_time:.4f}s)\n")
            
            print(f"\n{'='*70}")
            print(f"  ENTRENAMIENTO COMPLETADO")
            print(f"{'='*70}\n")
            
            # Guardar modelo y metadatos
            self.save_model_with_metadata(training_start)
        
        except Exception as e:
            print(f"\n✗ Error durante entrenamiento: {e}")
            raise
        finally:
            # Cerrar conexiones
            for worker_id, sock in self.worker_sockets.items():
                try:
                    sock.close()
                except:
                    pass
            self.server_socket.close()
    
    def save_model_with_metadata(self, training_start):
        """
        Guarda el modelo y sus metadatos en formato JSON.
        """
        # Crear directorio para stats si no existe
        stats_dir = './stats'
        os.makedirs(stats_dir, exist_ok=True)
        
        # Calcular tiempo total
        total_time = time.time() - training_start
        
        # Solicitar nombre del modelo
        model_name = input("\n  Ingrese un nombre para guardar el modelo: ").strip()
        if not model_name:
            model_name = f"cifar10_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Guardar pesos del modelo
        model_path = os.path.join('./modelos_guardados', f"{model_name}.pth")
        os.makedirs('./modelos_guardados', exist_ok=True)
        torch.save(self.net.state_dict(), model_path)
        print(f"\n  ✓ Modelo guardado en: {model_path}")
        
        # Preparar metadatos
        metadata = {
            'nombre_modelo': model_name,
            'fecha_guardado': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'arquitectura': 'Distributed CNN with Sockets',
            'epocas': self.epocas,
            'learning_rate': self.learning_rate,
            'num_workers': self.num_workers,
            'training_time_seconds': round(total_time, 2),
            'server_host': self.host,
            'server_port': self.port,
            'batch_size': BATCH_SIZE,
            'historial_intervalo_epochs': self.historial_intervalo_epochs,
            'historial_intervalo_times': self.historial_intervalo_times,
            'historial_intervalo_acc_test': self.historial_intervalo_acc_test,
            'historial_intervalo_loss': self.historial_intervalo_loss,
        }
        
        # Guardar metadatos en JSON
        metadata_path = os.path.join(stats_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Metadatos guardados en: {metadata_path}")

def start_server():
    """Inicia el servidor de entrenamiento distribuido"""
    server = DistributedTrainingServer(SERVER_HOST, SERVER_PORT, NUM_WORKERS, NUM_EPOCHS, LEARNING_RATE)
    server.setup_socket_server()
    server.wait_for_workers()
    server.training_loop()

if __name__ == '__main__':
        # permitir pasar parámetros por línea de comandos para el servidor
    parser = argparse.ArgumentParser(
        description="Servidor para entrenamiento distribuido."
    )

    parser.add_argument(
        "--host",
        "-H",
        default=SERVER_HOST,
        help=f"Host en el que el servidor escuchará (por defecto: {SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=SERVER_PORT,
        help=f"Puerto en el que el servidor escuchará (por defecto: {SERVER_PORT})",
    )
    parser.add_argument(
        "--particiones",
        "-n",
        type=int,
        default=NUM_WORKERS,
        help=f"Número de particiones/datos (por defecto: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--epocas",
        "-e",
        type=int,
        default=NUM_EPOCHS,
        help=f"Cantidad de épocas para entrenar (por defecto: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Tasa de aprendizaje (por defecto: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--intervalo-log",
        type=int,
        default=INTERVALO_LOG,
        help=f"Intervalo de logging de métricas (por defecto: {INTERVALO_LOG})",
    )


    args = parser.parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    NUM_WORKERS = args.particiones
    NUM_EPOCHS = args.epocas
    LEARNING_RATE = args.lr
    INTERVALO_LOG = args.intervalo_log

    # Definir TRANSFORM localmente
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Crear dataset y dataloader
    TRAINSET = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
    TRAINLOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=(NUM_WORKERS > 0))


    start_server()