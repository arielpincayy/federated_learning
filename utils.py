import json
import csv
import os
from config import NODES_JSON, METRICS_CSV

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
    print(f"[CENTRAL] Nodos guardados en {path}: {nodes}")
    return nodes


def load_nodes(path: str = NODES_JSON) -> dict[str, str]:
    """
    Entrada: ruta del archivo JSON de nodos.
    Salida: dict con los nodos registrados incluyendo n_nodes.
    """
    with open(path) as f:
        return json.load(f)
    

def append_metrics(metrics_list: list[dict], round_n: int, path: str = METRICS_CSV):
    """
    Entrada: lista de dicts de métricas por nodo, número de ronda, ruta del CSV.
    Salida: ninguna (añade filas al CSV, crea cabecera si no existe).
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fieldnames = ["round", "node", "accuracy", "precision", "recall", "f1_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for entry in metrics_list:
            writer.writerow({"round": round_n, **entry})
    print(f"[CENTRAL] Métricas de ronda {round_n} guardadas en {path}")