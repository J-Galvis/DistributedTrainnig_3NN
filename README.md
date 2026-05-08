# Distributed Neural Network Training Framework

A comprehensive framework for training neural networks under different datasets and comparing **monolithic** (single-machine sequential) versus **synchronous parallelism** (local multiprocessing and distributed socket-based) approaches. This project implements distributed training across MNIST, CIFAR-10, and ImageNet datasets with detailed performance analysis.

## Overview of Training Approaches

This framework compares three distinct training paradigms:

### 1. **Monolithic Training**
Single-machine, sequential training serving as the baseline for performance comparison.

### 2. **Synchronous Parallelism — Local**
Multi-process training on a single machine using Python's `multiprocessing` module. Dataset is partitioned across worker processes, gradients are computed in parallel, and weights are synchronized after each epoch.

### 3. **Synchronous Parallelism — Distributed**
True distributed training using **sockets** and **pickle** for inter-process communication. Enables training across multiple machines:
- **Server**: Coordinates training, aggregates gradients, updates global model
- **Workers**: Load partitions of data locally, train independently, send gradient updates
- **Protocol**: Defined message structures for Server ↔ Worker communication

---

## Project Structure

```
.
├── MINIST_Distributed_NN/          # MNIST distributed implementation
│   ├── Server.py                   # Server coordinator
│   ├── Worker.py                   # Worker agent
│   ├── Protocol.py                 # Message definitions
│   ├── messageHandling.py           # Communication utilities
│   └── ARCHITECTURE.md              # Detailed distributed architecture
│
├── CIFAR10_Distributed_NN/         # CIFAR-10 distributed implementation
│   ├── Server.py
│   ├── Worker.py
│   ├── Protocol.py
│   ├── messageHandling.py
│   └── defineNetwork.py             # PyTorch network definition
│
├── ImageNet_NN/                     # ImageNet implementation
│   ├── Server.py
│   ├── Worker.py
│   ├── Protocol.py
│   ├── messageHandling.py
│   └── defineNetwork.py
│
├── Notebooks/                       # Analysis and visualization
│   ├── MNIST_visualization.ipynb    # Dataset exploration & metrics
│   ├── CIFAR_visualization.ipynb
│   ├── ImageNet_visualization.ipynb
│   ├── Training_Comparison.ipynb    # Core comparison: monolithic vs. approaches
│   ├── Multiprocessing_Comparison.ipynb  # Local parallelism analysis
│   └── Vm_Configuration.md          # Cloud VM setup guide
│
├── Utils/                           # Shared utilities
│   ├── DatasetHandling.py           # Dataset loading and preprocessing
│   ├── Fuctions.py                  # Forward pass, backprop, loss, metrics
│   ├── WeightsHandling.py           # Weight initialization and updates
│   ├── TimeMeasurement.py           # Performance timing utilities
│   ├── Graphics.py                  # Visualization helpers
│   ├── ModelPersistence.py          # Model save/load
│   └── loadImageNet.py              # ImageNet-specific utilities
│
├── stats/                           # Performance metrics (JSON format)
│   ├── MNIST/
│   ├── CIFAR_10/
│   └── ImageNet/
│
├── data/                            # Datasets (CIFAR-10 included)
│   └── cifar-10-batches-py/
│
└── requirements.txt                 # Python dependencies
```

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip and venv (virtual environment support)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/J-Galvis/DistributedTrainnig_3NN.git
cd DistributedTrainnig_3NN
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy — numerical computations
- pandas — data handling
- torch / torchvision — deep learning framework
- altair — interactive visualizations
- datasets — dataset utilities
---

## Distributed Architecture

### Communication Protocol

**Server → Worker** (via `MessageFromServer`):
- Batch IDs to process
- Current epoch
- Model parameters (PyTorch state_dict)
- Learning rate, init/stop signals

**Worker → Server** (via `MessageFromWorker`):
- Computed gradients
- Loss and accuracy on assigned batches
- Training time elapsed

**Worker → Server** (via `WorkerReadyMessage`):
- Synchronization confirmation after receiving model params

### Training Loop

1. **Initialization**: Server loads dataset, creates initial weights, listens for worker connections
2. **Distribution**: Server sends batch IDs and current model params to all workers
3. **Local Training**: Each worker trains on its partition independently, computes gradients
4. **Collection**: Server waits for all workers to send back gradients, loss, accuracy
5. **Aggregation**: Gradients are averaged across workers
6. **Update**: Global model weights updated with averaged gradients
7. **Repeat**: Next epoch starts with updated weights

### Key Files

- **`Protocol.py`** — Dataclass definitions for `MessageFromServer`, `MessageFromWorker`, `WorkerReadyMessage`
- **`messageHandling.py`** — Socket serialization/deserialization utilities (pickle-based)
- **`Server.py`** — `DistributedTrainingServer` class, orchestration logic
- **`Worker.py`** — Worker agent, dataset partition handling, gradient computation

For deeper architectural details, see [MINIST_Distributed_NN/ARCHITECTURE.md](MINIST_Distributed_NN/ARCHITECTURE.md).

---

## Analysis & Results

Detailed performance comparisons and visualizations are available in the Jupyter notebooks:

### Training Comparison
**[Notebooks/Training_Comparison.ipynb](Notebooks/Training_Comparison.ipynb)**
- Trains three architectures on MNIST:
  - BasicNeuralNetwork (monolithic baseline)
  - ArnoviNeuralNetwork (post-training weight averaging)
  - DiegoNeuralNetwork (federated learning with epoch-level sync)
- Plots: Loss curves, training/test accuracy, convergence behavior
- Hyperparam control: shared learning rate, epochs, batch sizes

### Multiprocessing Comparison
**[Notebooks/Multiprocessing_Comparison.ipynb](Notebooks/Multiprocessing_Comparison.ipynb)**
- Compares local multiprocessing performance
- Measures training time, speedup, communication overhead
- Scales across 1, 2, 3, 4+ workers

### Dataset Visualizations
- **[MNIST_visualization.ipynb](Notebooks/MNIST_visualization.ipynb)** — Sample images, class distribution, preprocessing steps
- **[CIFAR_visualization.ipynb](Notebooks/CIFAR_visualization.ipynb)** — CIFAR-10 dataset overview
- **[ImageNet_visualization.ipynb](Notebooks/ImageNet_visualization.ipynb)** — ImageNet subset exploration

### Performance Metrics
Raw results (timing, accuracy) are stored as JSON in:
- `stats/MNIST/` — Monolithic and distributed training metrics
- `stats/CIFAR_10/` — CIFAR-10 experiments
- `stats/ImageNet/` — ImageNet experiments

---

## Key Files & Modules Reference

| File | Purpose |
|------|---------|
| `Utils/DatasetHandling.py` | Load MNIST, CIFAR-10; preprocessing (flatten, normalize, one-hot encoding) |
| `Utils/Fuctions.py` | Forward pass, backpropagation, cross-entropy loss, accuracy |
| `Utils/WeightsHandling.py` | Weight initialization, gradient updates |
| `Utils/TimeMeasurement.py` | Measure and log training duration per epoch |
| `Utils/Graphics.py` | Generate comparison plots (loss, accuracy curves) |
| `Utils/ModelPersistence.py` | Save/load trained models |
| `MINIST_Distributed_NN/defineNetwork.py` | Simple 2-layer feedforward network |
| `CIFAR10_Distributed_NN/defineNetwork.py` | PyTorch CNN for CIFAR-10 |
| `ImageNet_NN/defineNetwork.py` | PyTorch model for ImageNet |

---

## System Requirements

### CPU & Memory
- **Single Machine (Monolithic / Multiprocessing)**: 
  - 2+ cores recommended for multiprocessing speedup
  - 4 GB RAM minimum (8 GB+ for ImageNet)
  
- **Distributed Training**:
  - Server: 2+ cores, 4–8 GB RAM
  - Each Worker: 1+ core, 2–4 GB RAM per worker
  - Network: Low-latency LAN recommended; works over internet with higher overhead

### Network (for Distributed Mode)
- Server and workers must be reachable via TCP sockets
- Default port: 9999 (configurable)
- Firewall: Ensure inbound/outbound on chosen port is open

### Cloud VM Setup
For multi-machine distributed setup on Google Cloud, AWS, or similar, see [Notebooks/Vm_Configuration.md](Notebooks/Vm_Configuration.md) for detailed VM configuration steps.

