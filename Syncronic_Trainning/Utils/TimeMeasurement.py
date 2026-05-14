"""
Módulo de Medición de Tiempos
==============================
Proporciona funciones para medir y rastrear tiempos de ejecución por época.
"""

import time
from typing import Dict, List
import numpy as np


class TimeMeasurement:
    """
    Clase para rastrear tiempos de entrenamiento por época.
    """
    
    def __init__(self, network_name: str):
        """
        Inicializa el medidor de tiempos.
        
        Args:
            network_name: Nombre de la red neuronal
        """
        self.network_name = network_name
        self.epoch_times: List[float] = []
        self.current_epoch_start: float = None
        self.total_time: float = 0.0
    
    def start_epoch(self):
        """Marca el inicio de una época."""
        self.current_epoch_start = time.time()
    
    def end_epoch(self) -> float:
        """
        Marca el fin de una época y retorna el tiempo transcurrido.
        
        Returns:
            Tiempo en segundos de la época actual
        """
        if self.current_epoch_start is None:
            return 0.0
        
        elapsed = time.time() - self.current_epoch_start
        self.epoch_times.append(elapsed)
        self.total_time += elapsed
        return elapsed
    
    def get_epoch_times(self) -> List[float]:
        """Retorna lista de tiempos por época."""
        return self.epoch_times
    
    def get_total_time(self) -> float:
        """Retorna tiempo total de entrenamiento."""
        return self.total_time
    
    def get_average_time(self) -> float:
        """Retorna tiempo promedio por época."""
        if len(self.epoch_times) == 0:
            return 0.0
        return np.mean(self.epoch_times)
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas completas de tiempo."""
        if len(self.epoch_times) == 0:
            return {
                'network': self.network_name,
                'total_time': 0.0,
                'average_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'num_epochs': 0
            }
        
        return {
            'network': self.network_name,
            'total_time': self.total_time,
            'average_time': self.get_average_time(),
            'min_time': np.min(self.epoch_times),
            'max_time': np.max(self.epoch_times),
            'num_epochs': len(self.epoch_times)
        }
    
    def print_stats(self):
        """Imprime estadísticas de tiempo."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  ESTADÍSTICAS DE TIEMPO - {stats['network']}")
        print(f"{'='*60}")
        print(f"  Número de épocas      : {stats['num_epochs']}")
        print(f"  Tiempo total          : {stats['total_time']:.2f}s")
        print(f"  Tiempo promedio/época : {stats['average_time']:.4f}s")
        print(f"  Tiempo mínimo/época   : {stats['min_time']:.4f}s")
        print(f"  Tiempo máximo/época   : {stats['max_time']:.4f}s")
        print(f"{'='*60}")
