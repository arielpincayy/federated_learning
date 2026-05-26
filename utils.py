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
        - metrics_list: Lista de dicts de métricas por nodo.
        - round_n: Número de la ronda actual.
        - K: Ventana de rondas pasadas para evaluar convergencia.
        - tol: Tolerancia para determinar si el loss ha dejado de disminuir.
        - path: Ruta del CSV principal.
        - convergence_time_s: Tiempo de convergencia global en segundos, si aplica.
        - extra_path: Ruta del CSV de métricas extra si hay columnas fuera del esquema.

    Salida: bool indicando si el modelo convergió en esta ronda.

    Notas:
        - La columna 'converged' siempre se escribe: True si convergió, False si no.
        - 'converged_round' contiene el número de ronda en que convergió, o None.
        - 'convergence_time_s' contiene el tiempo de convergencia, o None.
        - Las métricas de red del central (central_net_*) se incluyen en el esquema
          principal y van al CSV directamente.
    """
    df_new = pd.DataFrame(metrics_list)
    df_new["round"] = round_n
    if "node" not in df_new.columns:
        df_new["node"] = "unknown"
    if "h_ronda" not in df_new.columns:
        df_new["h_ronda"] = None

    # Esquema principal del CSV.
    # Las columnas central_net_* capturan las métricas de red vistas desde el servidor
    # central (TX al distribuir el modelo global, RX al recibir modelos de los nodos).
    fieldnames = [
        # ── Identificación ─────────────────────────────────────────────────────
        "round", "h_ronda", "node",

        # ── Métricas de entrenamiento ──────────────────────────────────────────
        "accuracy", "precision", "recall", "f1_score",
        "specificity", "sensitivity",
        "trainning_time", "loss",

        # ── Latencias de comunicación del nodo ────────────────────────────────
        "latency_model_download_s",   # Tiempo que tardó el nodo en descargar el modelo global
        "latency_model_upload_s",     # Tiempo que tardó el nodo en subir su modelo entrenado

        # ── Red: métricas del nodo (TX = subida modelo, RX = descarga modelo) ─
        "net_bytes_tx_model",         # Bytes del modelo enviados por el nodo
        "net_bytes_tx_system",        # Bytes totales TX del sistema durante la subida
        "net_bandwidth_tx_kbps",      # Ancho de banda TX del nodo al subir
        "net_packets_sent",
        "net_errors_out",
        "net_drops_out",
        "net_bytes_rx_model",         # Bytes del modelo recibidos por el nodo
        "net_bytes_rx_system",        # Bytes totales RX del sistema durante la descarga
        "net_bandwidth_rx_kbps",      # Ancho de banda RX del nodo al descargar
        "net_packets_recv",
        "net_errors_in",
        "net_drops_in",
        "net_throughput_kbps",        # Throughput total TX+RX del nodo
        "net_transmission_time_s",    # Tiempo de transmisión de la subida

        # ── Red: métricas del servidor central ────────────────────────────────
        "central_net_bytes_rx_model",     # Bytes del modelo recibidos por el central
        "central_net_bytes_rx_system",    # Bytes totales RX del sistema del central
        "central_net_bandwidth_rx_kbps",  # BW RX del central al recibir modelo
        "central_net_throughput_kbps",    # Throughput del central al recibir
        "central_net_transmission_time_s",# Tiempo de recepción en el central
        "central_net_bytes_tx_model",     # Bytes del modelo distribuido por el central
        "central_net_bytes_tx_system",    # Bytes TX del sistema del central al distribuir
        "central_net_bandwidth_tx_kbps",  # BW TX del central al distribuir
        "central_latency_model_dist_s",   # Latencia de distribución del modelo global

        # ── Métricas de hardware del nodo ─────────────────────────────────────
        "cpu_percent",
        "ram_percent",
        "cpu_freq_mhz",
        "open_sockets",

        # ── Métricas de agregación y convergencia ─────────────────────────────
        "comm_overhead_bytes",        # Bytes extra de comunicación (modelo × (n-1))
        "aggregation_time_s",         # Tiempo de FedAvg en el central
        "inter_silo_variance",        # Varianza del loss entre nodos
        "converged",                  # True/False — siempre presente
        "converged_round",            # Número de ronda en que convergió, o None
        "convergence_time_s",         # Tiempo hasta convergencia, o None
    ]

    # ── Columnas extra (fuera del esquema): CSV separado ──────────────────────
    extra_path = extra_path or path.replace(".csv", "_extra.csv")
    extra_cols = [col for col in df_new.columns if col not in fieldnames]
    if extra_cols:
        extra_cols_ordered = [c for c in ["round", "node", "h_ronda"] if c in df_new.columns] + extra_cols
        extra_df = df_new[extra_cols_ordered].copy()
        write_header_extra = not os.path.exists(extra_path)
        extra_df.to_csv(extra_path, mode="a", index=False, header=write_header_extra)
        logger.info(f"[METRICS] Columnas extra guardadas en {extra_path}: {extra_cols}")

    # ── Evaluación de convergencia ─────────────────────────────────────────────
    existing = pd.DataFrame()
    if os.path.exists(path):
        existing = pd.read_csv(path)

    df_combined = (
        pd.concat([existing, df_new], ignore_index=True, sort=False)
        if not existing.empty
        else df_new.copy()
    )

    converged = False
    try:
        df_combined["round"] = pd.to_numeric(df_combined["round"], errors="coerce")
        df_combined["loss"] = pd.to_numeric(df_combined["loss"], errors="coerce")
        df_combined = df_combined.dropna(subset=["round", "loss"])
        df_combined["round"] = df_combined["round"].astype(int)
        df_rounds = df_combined.groupby("round")["loss"].mean().sort_index()

        if len(df_rounds) >= K:
            recent_losses = df_rounds.tail(K).values
            diffs = abs(pd.Series(recent_losses).diff().dropna())
            converged = bool((diffs < tol).all())
        else:
            converged = False
    except Exception as e:
        logger.error(f"Error al calcular la convergencia: {e}")
        converged = False

    # ── Escribir columnas de convergencia — siempre presentes ─────────────────
    df_new["converged"] = converged                                          # True o False, nunca None
    df_new["converged_round"] = round_n if converged else None
    df_new["convergence_time_s"] = (
        round(convergence_time_s, 3) if converged and convergence_time_s is not None else None
    )

    # ── Guardar en CSV principal ───────────────────────────────────────────────
    df_main = df_new.reindex(columns=fieldnames)
    write_header = not os.path.exists(path)
    df_main.to_csv(path, mode="a", index=False, header=write_header)
    logger.info(f"[CENTRAL] Métricas de ronda {round_n} guardadas en {path}")

    if converged:
        logger.info(
            f"[CONVERGENCIA] ¡El modelo ha convergido en la ronda {round_n}! "
            f"Tiempo de convergencia: {convergence_time_s:.3f}s"
        )
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
    """
    df = pd.read_csv(ruta_csv)
    
    df_ultimos = df.loc[df.groupby(['h_ronda', 'node'])['round'].idxmax()]
    
    df_promedio = df_ultimos.groupby('h_ronda')['loss'].mean().reset_index()
    df_promedio = df_promedio.sort_values('h_ronda').reset_index(drop=True)
    
    if len(df_promedio) < k:
        print(f"Advertencia: No hay suficientes h_rondas ({len(df_promedio)}) para cubrir la ventana k={k}.")
        return False

    ultimos_loss = df_promedio['loss'].iloc[-k:].values
    
    variacion = np.max(ultimos_loss) - np.min(ultimos_loss)
    
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