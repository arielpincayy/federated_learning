"""
Módulo para capturar métricas de red del sistema y transmisión de datos.
Utiliza psutil para obtener estadísticas de interfaz de red y calcula bandwidth/throughput.
"""

import time
import psutil
from typing import Dict, Optional, Tuple
from logging_config import get_logger

logger = get_logger(__name__)


class NetworkMetricsCollector:
    """
    Captura métricas de red antes y después de operaciones de comunicación.
    Mide bytes TX/RX del sistema, paquetes, errores, drops y calcula bandwidth.
    """

    def __init__(self, interface: str = None):
        """
        Inicializa el colector de métricas.
        
        Args:
            interface: Nombre de la interfaz de red (ej: 'eth0'). 
                      Si es None, usa la interfaz por defecto.
        """
        self.interface = interface or self._get_default_interface()
        self.start_time = None
        self.end_time = None
        self.start_stats = None
        self.end_stats = None
        self.bytes_model_tx = 0  # Bytes del modelo transmitidos
        self.bytes_model_rx = 0  # Bytes del modelo recibidos

    @staticmethod
    def _get_default_interface() -> str:
        """Detecta la interfaz de red por defecto (no loopback)."""
        try:
            stats = psutil.net_if_stats()
            for iface in stats:
                if iface != "lo" and stats[iface].isup:
                    return iface
        except Exception as e:
            logger.warning(f"No se pudo detectar interfaz de red: {e}. Usando 'eth0'")
        return "eth0"

    def _get_net_stats(self) -> Dict:
        """Obtiene estadísticas de la interfaz de red actual."""
        try:
            stats = psutil.net_if_stats()
            io_counters = psutil.net_io_counters(pernic=True)
            
            if self.interface not in io_counters:
                logger.warning(f"Interfaz {self.interface} no encontrada. Usando la primera disponible.")
                self.interface = list(io_counters.keys())[0]
            
            counters = io_counters[self.interface]
            
            return {
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_recv": counters.packets_recv,
                "errin": counters.errin,
                "errout": counters.errout,
                "dropin": counters.dropin,
                "dropout": counters.dropout,
            }
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de red: {e}")
            return {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0,
                "errin": 0,
                "errout": 0,
                "dropin": 0,
                "dropout": 0,
            }

    def start_monitoring(self):
        """Inicia el monitoreo de red."""
        self.start_time = time.time()
        self.start_stats = self._get_net_stats()
        logger.debug(f"[NET] Monitoreo iniciado en interfaz {self.interface}")

    def end_monitoring(self) -> Dict:
        """
        Termina el monitoreo y calcula las métricas.
        
        Returns:
            Dict con métricas de red calculadas.
        """
        self.end_time = time.time()
        self.end_stats = self._get_net_stats()
        
        if not self.start_stats or not self.end_stats:
            logger.error("[NET] No se pudieron recopilar estadísticas de red")
            return self._empty_metrics()
        
        elapsed_time = self.end_time - self.start_time
        if elapsed_time == 0:
            elapsed_time = 0.001  # Evitar división por cero
        
        # Diferencias en bytes y paquetes
        bytes_tx = self.end_stats["bytes_sent"] - self.start_stats["bytes_sent"]
        bytes_rx = self.end_stats["bytes_recv"] - self.start_stats["bytes_recv"]
        packets_sent = self.end_stats["packets_sent"] - self.start_stats["packets_sent"]
        packets_recv = self.end_stats["packets_recv"] - self.start_stats["packets_recv"]
        
        # Errores y drops
        errors_in = self.end_stats["errin"] - self.start_stats["errin"]
        errors_out = self.end_stats["errout"] - self.start_stats["errout"]
        drops_in = self.end_stats["dropin"] - self.start_stats["dropin"]
        drops_out = self.end_stats["dropout"] - self.start_stats["dropout"]
        
        # Bandwidth en kbps (kilobits por segundo)
        bandwidth_tx_kbps = (bytes_tx * 8) / (elapsed_time * 1000) if elapsed_time > 0 else 0
        bandwidth_rx_kbps = (bytes_rx * 8) / (elapsed_time * 1000) if elapsed_time > 0 else 0
        
        # Throughput total (TX + RX) en kbps
        total_bytes = bytes_tx + bytes_rx
        throughput_kbps = (total_bytes * 8) / (elapsed_time * 1000) if elapsed_time > 0 else 0
        
        metrics = {
            "net_bytes_tx_system": bytes_tx,
            "net_bytes_rx_system": bytes_rx,
            "net_bytes_tx_model": self.bytes_model_tx,
            "net_bytes_rx_model": self.bytes_model_rx,
            "net_packets_sent": packets_sent,
            "net_packets_recv": packets_recv,
            "net_errors_in": errors_in,
            "net_errors_out": errors_out,
            "net_drops_in": drops_in,
            "net_drops_out": drops_out,
            "net_bandwidth_tx_kbps": round(bandwidth_tx_kbps, 2),
            "net_bandwidth_rx_kbps": round(bandwidth_rx_kbps, 2),
            "net_throughput_kbps": round(throughput_kbps, 2),
            "net_transmission_time_s": round(elapsed_time, 3),
        }
        
        logger.debug(f"[NET] Métricas recopiladas: TX={bytes_tx}B, RX={bytes_rx}B, "
                    f"BW_TX={bandwidth_tx_kbps:.2f}kbps, BW_RX={bandwidth_rx_kbps:.2f}kbps")
        
        return metrics

    def set_model_bytes_transferred(self, bytes_tx: int = 0, bytes_rx: int = 0):
        """
        Registra los bytes reales del modelo que se transfirieron.
        
        Args:
            bytes_tx: Bytes del modelo transmitidos
            bytes_rx: Bytes del modelo recibidos
        """
        self.bytes_model_tx = bytes_tx
        self.bytes_model_rx = bytes_rx

    @staticmethod
    def _empty_metrics() -> Dict:
        """Retorna un dict con métricas de red vacías (ceros)."""
        return {
            "net_bytes_tx_system": 0,
            "net_bytes_rx_system": 0,
            "net_bytes_tx_model": 0,
            "net_bytes_rx_model": 0,
            "net_packets_sent": 0,
            "net_packets_recv": 0,
            "net_errors_in": 0,
            "net_errors_out": 0,
            "net_drops_in": 0,
            "net_drops_out": 0,
            "net_bandwidth_tx_kbps": 0,
            "net_bandwidth_rx_kbps": 0,
            "net_throughput_kbps": 0,
            "net_transmission_time_s": 0,
        }


def collect_system_metrics() -> dict:
    """Recopila métricas del sistema del nodo.

    Devuelve:
        cpu_percent, ram_percent, cpu_freq_mhz y open_sockets.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = round(cpu_freq.current, 2) if cpu_freq else 0.0
        open_sockets = len(psutil.net_connections(kind="inet"))

        return {
            "cpu_percent": round(cpu_percent, 2),
            "ram_percent": round(ram_percent, 2),
            "cpu_freq_mhz": cpu_freq_mhz,
            "open_sockets": open_sockets,
        }
    except Exception as e:
        logger.error(f"Error al recopilar métricas del sistema: {e}")
        return {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "cpu_freq_mhz": 0.0,
            "open_sockets": 0,
        }


class NetworkMetricsContext:
    """
    Context manager para capturar automáticamente métricas de red antes y después
    de una operación. Uso:
    
    with NetworkMetricsContext() as metrics:
        # realizar operación de red
        metrics.set_model_bytes_transferred(bytes_tx=1000)
    metrics_dict = metrics.end_monitoring()
    """

    def __init__(self, interface: str = None):
        self.collector = NetworkMetricsCollector(interface)

    def __enter__(self) -> NetworkMetricsCollector:
        self.collector.start_monitoring()
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Excepción durante monitoreo de red: {exc_type.__name__}: {exc_val}")
        return False
