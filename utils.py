import json
import csv
import os

import pandas as pd
import numpy as np
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

def load_nodes2dict(path: str = NODES_JSON) -> dict[str, str]:
    """
    Entrada: ruta del archivo JSON de nodos.
    Salida: dict con formato {"Nodo_1": "addr", ...} sin la clave "n_nodes".
    """
    with open(path, "r") as f:
        data = json.load(f)
    return {addr: nid for nid, addr in data.items() if nid != "n_nodes"}


def load_nodes(path: str = NODES_JSON) -> dict[str, str]:
    """
    Entrada: ruta del archivo JSON de nodos.
    Salida: dict con los nodos registrados incluyendo n_nodes.
    """
    with open(path) as f:
        return json.load(f)
    

def append_metrics(
    metrics_list: list[dict],
    round_n: int,
    K: int = 5,
    tol: float = 1e-4,
    path: str = METRICS_CSV,
    convergence_time_s: float | None = None,
    extra_path: str | None = None,
) -> bool:
    """
    Entrada:
        - metrics_list: Lista de dicts de métricas por nodo (incluye métricas de entrenamiento y red).
        - round_n: Número de la ronda actual.
        - K: Ventana de rondas pasadas para evaluar convergencia.
        - tol: Tolerancia para determinar si el loss ha dejado de disminuir.
        - path: Ruta del CSV principal.
        - convergence_time_s: Tiempo de convergencia global en segundos, si aplica.
        - extra_path: Ruta del CSV de métricas extra si hay columnas fuera del esquema.
    """
    # 1. Convertir la lista de entrada en un DataFrame de Pandas
    df_new = pd.DataFrame(metrics_list)
    df_new["round"] = round_n
    if "node" not in df_new.columns:
        df_new["node"] = "unknown"
    if "h_ronda" not in df_new.columns:
        df_new["h_ronda"] = None

    # Fieldnames: métricas de entrenamiento, métricas del modelo, métricas de red y métricas adicionales
    fieldnames = [
        "round", "h_ronda", "node",
        "accuracy", "precision", "recall", "f1_score", "specificity", "sensitivity", "trainning_time", "loss",
        "latency_model_download_s", "latency_model_upload_s",
        "net_bytes_tx_system", "net_bytes_rx_system",
        "net_bytes_tx_model", "net_bytes_rx_model",
        "net_packets_sent", "net_packets_recv",
        "net_errors_in", "net_errors_out",
        "net_drops_in", "net_drops_out",
        "net_bandwidth_tx_kbps", "net_bandwidth_rx_kbps",
        "net_throughput_kbps", "net_transmission_time_s",
        "comm_overhead_bytes", "aggregation_time_s", "inter_silo_variance",
        "converged_round", "convergence_time_s",
        "cpu_percent", "ram_percent", "cpu_freq_mhz", "open_sockets",
    ]

    # 2. Buscar columnas extra y guardarlas en un CSV adicional si hay alguna
    extra_path = extra_path or path.replace(".csv", "_extra.csv")
    extra_cols = [col for col in df_new.columns if col not in fieldnames]
    if extra_cols:
        extra_cols_ordered = [c for c in ["round", "node", "h_round"] if c in df_new.columns] + extra_cols
        extra_df = df_new[extra_cols_ordered].copy()
        write_header_extra = not os.path.exists(extra_path)
        extra_df.to_csv(extra_path, mode="a", index=False, header=write_header_extra)
        logger.info(f"[METRICS] Columnas extra guardadas en {extra_path}: {extra_cols}")

    # 3. Evaluar convergencia antes de guardar las métricas actuales
    existing = pd.DataFrame()
    if os.path.exists(path):
        existing = pd.read_csv(path)
    df_combined = pd.concat([existing, df_new], ignore_index=True, sort=False) if not existing.empty else df_new.copy()

    try:
        df_combined["round"] = pd.to_numeric(df_combined["round"], errors="coerce")
        df_combined["loss"] = pd.to_numeric(df_combined["loss"], errors="coerce")
        df_combined = df_combined.dropna(subset=["round", "loss"])
        df_combined["round"] = df_combined["round"].astype(int)
        df_rounds = df_combined.groupby("round")["loss"].mean().sort_index()

        if len(df_rounds) >= K:
            recent_losses = df_rounds.tail(K).values
            diffs = abs(pd.Series(recent_losses).diff().dropna())
            converged = (diffs < tol).all()
        else:
            converged = False
    except Exception as e:
        logger.error(f"Error al calcular la convergencia: {e}")
        converged = False

    df_new["converged_round"] = round_n if converged else None
    df_new["convergence_time_s"] = round(convergence_time_s, 3) if converged and convergence_time_s is not None else None

    # 4. Guardar/Añadir al archivo CSV principal con el esquema conocido
    df_main = df_new.reindex(columns=fieldnames)
    write_header = not os.path.exists(path)
    df_main.to_csv(path, mode="a", index=False, header=write_header)
    logger.info(f"[CENTRAL] Métricas de ronda {round_n} guardadas en {path}")

    if converged:
        logger.info(f"[CONVERGENCIA] ¡El modelo ha convergido en la ronda {round_n}! Tiempo de convergencia: {convergence_time_s:.3f}s")
    else:
        logger.info(f"[ENTRENAMIENTO] Sin convergencia aún en la ronda {round_n}.")

    return converged

def hierarchy_convergence(ruta_csv, k=3, tolerancia=1e-4):
    """
    Procesa el CSV de entrenamiento federado y determina si el sistema
    ha convergido basándose en los últimos 'k' h_rondas.
    
    Parameters:
    -----------
    ruta_csv : str
        Ruta al archivo CSV con los datos de los nodos.
    k : int
        Tamaño de la ventana de tiempo (últimas h_rondas) a evaluar.
    tolerancia : float, default 1e-4
        El umbral de cambio máximo permitido en el loss para considerar convergencia.
        
    Returns:
    --------
    bool
        True si convergió, False en caso contrario.
    pd.DataFrame
        El DataFrame con el histórico del loss promedio por h_ronda para análisis.
    """
    # 1. Cargar el dataset
    # Nota: Si tus columnas están juntas en el archivo real, asegúrate de ajustar los nombres.
    df = pd.read_csv(ruta_csv)
    
    # 2. Extraer el último 'round' de cada nodo en cada 'h_ronda'
    df_ultimos = df.loc[df.groupby(['h_ronda', 'node'])['round'].idxmax()]
    
    # 3. Calcular el loss promedio entre los nodos para cada h_ronda
    df_promedio = df_ultimos.groupby('h_ronda')['loss'].mean().reset_index()
    df_promedio = df_promedio.sort_values('h_ronda').reset_index(drop=True)
    
    # 4. Verificar si tenemos suficientes datos para la ventana k
    if len(df_promedio) < k:
        print(f"Advertencia: No hay suficientes h_rondas ({len(df_promedio)}) para cubrir la ventana k={k}.")
        return False

    # 5. Evaluar la convergencia en la ventana k (los últimos k h_rondas)
    ultimos_loss = df_promedio['loss'].iloc[-k:].values
    
    # Criterio: Estabilidad (Rango absoluto entre el máximo y mínimo de la ventana)
    variacion = np.max(ultimos_loss) - np.min(ultimos_loss)
    
    # Si la variación en las últimas k rondas es menor al umbral, hay convergencia
    ha_convergido = variacion < tolerancia
    
    return bool(ha_convergido)



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
