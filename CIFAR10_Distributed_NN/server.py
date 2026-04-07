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
import csv
from typing import Dict, List

# Agregar el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defineNetwork import Net, TRANSFORM, NUM_EPOCHS, SAVE_FILE, TRAINLOADER, PORT, HOST
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from messageHandling import send_message, receive_message

# Configuración
NUM_WORKERS = TrainingConfig.num_workers
LEARNING_RATE = TrainingConfig.learning_rate
INTERVALO_LOG = TrainingConfig.intervalo_log
SOCKET_TIMEOUT = TrainingConfig.socket_timeout

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
        
        # Historial
        self.historial_epochs = []
        self.historial_times = []
        self.historial_accuracies = []
    
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
    
    def training_loop(self):
        """
        Bucle principal de entrenamiento.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO ENTRENAMIENTO DISTRIBUIDO")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        
        # Crear directorio de resultados
        results_dir = './Results'
        os.makedirs(results_dir, exist_ok=True)
        
        server_time_file = os.path.join(results_dir, 'Server_time.csv')
        
        # Escribir encabezado del CSV
        with open(server_time_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Epoch_Time', 'Total_Time', 'Num_Workers', 'Test_Accuracy'])
        
        try:
            for epoch in range(1, self.epocas + 1):
                epoch_start = time.time()
                
                # Distribuir trabajo
                self.distribute_work(epoch)
                
                # Recolectar resultados
                messages = self.collect_results()
                
                # Promediar gradientes
                avg_grads = self.average_gradients(messages)
                
                # Actualizar modelo
                self.update_model(avg_grads)
                
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                
                # Evaluar en test (cada INTERVALO_LOG épocas)
                if epoch % INTERVALO_LOG == 0:
                    test_acc = accuracyTest(self.net, TRANSFORM, 0)
                    print(f"\n  Epoch {epoch}: Test Accuracy = {test_acc:.4f}%")
                else:
                    test_acc = 0.0
                
                # Guardar en CSV
                with open(server_time_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f"{epoch_time:.4f}", f"{total_time:.4f}", self.num_workers, f"{test_acc:.4f}"])
                
                print(f"  Epoch {epoch} completada en {epoch_time:.4f}s (Total: {total_time:.4f}s)\n")
            
            print(f"\n{'='*70}")
            print(f"  ENTRENAMIENTO COMPLETADO")
            print(f"{'='*70}\n")
            
            # Guardar modelo
            torch.save(self.net.state_dict(), SAVE_FILE)
            print(f"Modelo guardado en {SAVE_FILE}")
        
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

def start_server():
    """Inicia el servidor de entrenamiento distribuido"""
    server = DistributedTrainingServer(HOST, PORT, NUM_WORKERS, NUM_EPOCHS, LEARNING_RATE)
    server.setup_socket_server()
    server.wait_for_workers()
    server.training_loop()

if __name__ == '__main__':
    start_server()