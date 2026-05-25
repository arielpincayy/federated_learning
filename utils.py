import json
import csv
import os

import pandas as pd
from config import NODES_JSON, METRICS_CSV, RECEIVED_MODEL_FILENAME
from logging_config import get_logger

logger = get_logger(__name__)

def get_ipport(addr: str) -> tuple[str, int]:
    """
    Entrada: dirección en formato "host:port" (soporta IPv6 como "::1:8765").
    Salida: tupla (host, port) donde port es entero.
    """
    host, port = addr.rsplit(":", 1)
    return host, int(port)

def save_nodes(ips: list[str], path: str = NODES_JSON) -> dict[str, str]:
    """
    Entrada: lista de direcciones (host:port), ruta del archivo JSON de salida.
    Salida: dict guardado en disco con formato {"n_nodes": N, "Nodo_1": "addr", ...}.
    """
    nodes = {f"Nodo_{i + 1}": addr for i, addr in enumerate(ips)}
    nodes["n_nodes"] = len(ips)
    with open(path, "w") as f:
        json.dump(nodes, f, indent=2)
    logger.info(f"[CENTRAL] Nodos guardados en {path}: {nodes}")
    return nodes


def load_nodes(path: str = NODES_JSON) -> dict[str, str]:
    """
    Entrada: ruta del archivo JSON de nodos.
    Salida: dict con los nodos registrados incluyendo n_nodes.
    """
    with open(path) as f:
        return json.load(f)
    

def append_metrics(metrics_list: list[dict], round_n: int, K: int = 5, tol: float = 1e-4, path: str = METRICS_CSV) -> bool:
    """
    Entrada: 
        - metrics_list: Lista de dicts de métricas por nodo (incluye métricas de entrenamiento y red).
        - round_n: Número de la ronda actual.
        - K: Ventana de épocas pasadas para evaluar convergencia.
        - tol: Tolerancia para determinar si el loss ha dejado de disminuir.
        - path: Ruta del CSV.
    """
    # 1. Convertir la lista de entrada en un DataFrame de Pandas
    df_new = pd.DataFrame(metrics_list)
    df_new["round"] = round_n
    
    # Fieldnames: métricas de entrenamiento + métricas de red
    fieldnames = [
        "round", "node", 
        "accuracy", "precision", "recall", "f1_score", "trainning_time", "loss",
        # Métricas de red
        "net_bytes_tx_system", "net_bytes_rx_system", 
        "net_bytes_tx_model", "net_bytes_rx_model",
        "net_packets_sent", "net_packets_recv",
        "net_errors_in", "net_errors_out",
        "net_drops_in", "net_drops_out",
        "net_bandwidth_tx_kbps", "net_bandwidth_rx_kbps",
        "net_throughput_kbps", "net_transmission_time_s"
    ]
    df_new = df_new.reindex(columns=fieldnames)
    
    # 2. Guardar/Añadir al archivo CSV
    write_header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", index=False, header=write_header)
    logger.info(f"[CENTRAL] Métricas de ronda {round_n} guardadas en {path}")
    
    # 3. Leer el histórico y forzar tipos de datos numéricos
    try:
        df_history = pd.read_csv(path)
        
        # Convertimos 'round' y 'loss' a números. 'errors="coerce"' transformará cualquier 
        # texto inválido en NaN, y luego eliminamos esos NaN.
        df_history["round"] = pd.to_numeric(df_history["round"], errors="coerce")
        df_history["loss"] = pd.to_numeric(df_history["loss"], errors="coerce")
        df_history = df_history.dropna(subset=["round", "loss"])
        
        # Aseguramos que las rondas queden como enteros para el ordenamiento y agrupación
        df_history["round"] = df_history["round"].astype(int)
        # --------------------------
        
        # Agrupar por ronda y calcular el promedio del loss
        df_rounds = df_history.groupby("round")["loss"].mean().sort_index()
        
        # Verificar si tenemos suficientes datos para la ventana K
        if len(df_rounds) >= K:
            recent_losses = df_rounds.tail(K).values
            
            # Evaluar la diferencia absoluta consecutiva en la ventana K
            diffs = abs(pd.Series(recent_losses).diff().dropna())
            
            if (diffs < tol).all():
                logger.info(f"[CONVERGENCIA] ¡El modelo ha convergido! El cambio en las últimas {K} rondas es menor a {tol}.")
                return True
            else:
                max_diff = diffs.max()
                logger.info(f"[ENTRENAMIENTO] Sin convergencia aún. Cambio máximo reciente: {max_diff:.6f} (tol: {tol}).")
                return False
        else:
            logger.info(f"[INFO] Ventana insuficiente para evaluar convergencia. Rondas calculadas: {len(df_rounds)}/{K}")
            return False
            
    except Exception as e:
        logger.error(f"Error al calcular la convergencia: {e}")


async def _save_file(data: bytes, save_path: str, filename: str = RECEIVED_MODEL_FILENAME) -> str:
    """
    Entrada: datos binarios, directorio destino, nombre de archivo opcional.
    Salida: ruta donde se guardó el archivo.
    """
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    logger.info(f"[FILE SAVED] {filepath}")
    return filepath
