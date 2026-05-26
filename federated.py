import asyncio
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import ROUNDS, EPOCHS, LEARNING_RATE, IN_FEATURES, LISTENER_DURATION, NODES_JSON, METRICS_CSV, MODEL_PATH, DATA_PATH, SLEEP_INTERVAL, POST_ROUND_DELAY, NODES_LISTENER_DELAY, LABEL_COLUMN
from connections.client import send_identified, send_file_identified, send_file_to_nodes, send_message_to_nodes, send
from connections.server import listener_nodes, listener_server
from model.create_model import MLP
from model.fed_model import ModelTrainer, federated_average
from network_metrics import collect_system_metrics
from utils import save_nodes, append_metrics, get_ipport
from logging_config import get_logger

logger = get_logger(__name__)

async def central_main(addr: str, ips: list[str], h_ronda: int | None = None):
    """
    Entrada: dirección propia del servidor central (host:port).
    """

    nodes = save_nodes(ips, NODES_JSON)
    n = nodes["n_nodes"]

    nodes_by_addr = {a: nid for nid, a in nodes.items() if nid != "n_nodes"}

    _, listen_port = get_ipport(addr)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")
        round_start_time = time.time()

        # Métricas de distribución del modelo global (TX desde central a nodos)
        # Clave: addr del nodo → dict de métricas de red del envío
        node_dist_metrics: dict[str, dict] = {}

        if round_n > 1:
            await asyncio.sleep(POST_ROUND_DELAY)
            logger.info(f"[CENTRAL] Distribuyendo modelo agregado a {n} nodo(s)...")
            node_dist_metrics = await send_file_to_nodes(ips, MODEL_PATH, delay=LISTENER_DURATION * 6)
            logger.info(f"[CENTRAL] Métricas de distribución recopiladas de {len(node_dist_metrics)} nodo(s)")

        logger.info(f"[CENTRAL] Esperando modelos y métricas de {n} nodo(s)...")
        round_dir = f"round_{round_n}"
        os.makedirs(round_dir, exist_ok=True)
        results = await listener_nodes(
            local_listen_addr,
            nodes=nodes_by_addr,
            delay=NODES_LISTENER_DELAY * 100,
            save_path=round_dir,
        )

        metrics_list = []
        model_paths = []
        # Métricas de red capturadas por el servidor al recibir modelos (RX)
        central_rx_metrics: dict[str, dict] = {}

        for node_id, received in results.items():
            for item in received:
                if isinstance(item, str) and os.path.isfile(item) and item.endswith(".pt"):
                    model_paths.append(item)
                elif isinstance(item, dict):
                    central_rx_metrics[node_id] = item
                    logger.info(f"[CENTRAL] Métricas de red RX capturadas para {node_id}: {item}")
                elif isinstance(item, str):
                    try:
                        m = json.loads(item)
                        m["node"] = node_id
                        metrics_list.append(m)
                    except json.JSONDecodeError:
                        pass

        # Construir un mapa de node_id → addr para poder cruzar con node_dist_metrics
        # nodes_by_addr es {addr -> node_id}, necesitamos la inversa
        node_id_to_addr: dict[str, str] = {nid: addr for addr, nid in nodes_by_addr.items()}

        for metrics in metrics_list:
            node_id = metrics.get("node")
            if h_ronda is not None:
                metrics["h_ronda"] = h_ronda

            # ── Métricas RX del central al recibir el modelo del nodo ──
            if node_id in central_rx_metrics:
                rx_m = central_rx_metrics[node_id]
                metrics["central_net_bytes_rx_model"] = rx_m.get("net_bytes_rx_model", 0)
                metrics["central_net_bytes_rx_system"] = rx_m.get("net_bytes_rx_system", 0)
                metrics["central_net_bandwidth_rx_kbps"] = rx_m.get("net_bandwidth_rx_kbps", 0)
                metrics["central_net_throughput_kbps"] = rx_m.get("net_throughput_kbps", 0)
                metrics["central_net_transmission_time_s"] = rx_m.get("net_transmission_time_s", 0)
                metrics["comm_overhead_bytes"] = int(
                    rx_m.get("net_bytes_rx_model", 0) * max(0, len(ips) - 1)
                )

            # ── Métricas TX del central al distribuir el modelo global al nodo ──
            if round_n > 1 and node_id in node_id_to_addr:
                node_addr = node_id_to_addr[node_id]
                dist_m = node_dist_metrics.get(node_addr, {})
                metrics["central_net_bytes_tx_model"] = dist_m.get("net_bytes_tx_model", 0)
                metrics["central_net_bytes_tx_system"] = dist_m.get("net_bytes_tx_system", 0)
                metrics["central_net_bandwidth_tx_kbps"] = dist_m.get("net_bandwidth_tx_kbps", 0)
                metrics["central_latency_model_dist_s"] = dist_m.get("net_transmission_time_s", 0)
            else:
                # Ronda 1: no hubo distribución previa
                metrics["central_net_bytes_tx_model"] = 0
                metrics["central_net_bytes_tx_system"] = 0
                metrics["central_net_bandwidth_tx_kbps"] = 0
                metrics["central_latency_model_dist_s"] = 0

        if not model_paths:
            logger.error("[CENTRAL] No se recibieron modelos. Abortando.")
            break

        aggregation_start = time.time()
        avg_state = federated_average(model_paths)
        aggregation_time_s = round(time.time() - aggregation_start, 3)
        torch.save(avg_state, MODEL_PATH)
        logger.info(f"[CENTRAL] Modelo global actualizado → {MODEL_PATH}")

        # Tiempo total desde inicio de ronda hasta fin de agregación
        convergence_time_s = round(time.time() - round_start_time, 3)

        for metrics in metrics_list:
            metrics["aggregation_time_s"] = aggregation_time_s

        losses = []
        for m in metrics_list:
            try:
                losses.append(float(m.get("loss", 0)))
            except (TypeError, ValueError):
                pass
        inter_silo_variance = float(np.var(losses)) if losses else 0.0
        for metrics in metrics_list:
            metrics["inter_silo_variance"] = round(inter_silo_variance, 6)

        converged = append_metrics(
            metrics_list,
            round_n,
            path=METRICS_CSV,
            K=3,
            convergence_time_s=convergence_time_s,
        )

        if round_n < ROUNDS:
            logger.info(f"[CENTRAL] Enviando mensaje de convergencia: CONVERGED={converged}")
            if converged:
                await send_message_to_nodes(ips, "CONVERGED", LISTENER_DURATION)
                break
            else:
                await send_message_to_nodes(ips, "NOT CONVERGED", LISTENER_DURATION)
            logger.info("[CENTRAL] Mensaje de convergencia enviado")

    logger.info("\n[CENTRAL] Entrenamiento federado completado.")


async def client_main(addr: str, server_addr: str, h_ronda: int | None = None):
    """
    Entrada: dirección propia del nodo (host:port), dirección del servidor central.
    """
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    _, listen_port = get_ipport(addr)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[NODE] ══ Ronda {round_n}/{ROUNDS} ══")

        # Métricas de descarga (RX) y subida (TX) se acumulan por separado
        # para evitar que update() sobreescriba claves con el mismo nombre.
        net_metrics_rx: dict = {}   # Métricas de la descarga del modelo global
        net_metrics_tx: dict = {}   # Métricas de la subida del modelo entrenado

        # ── Descarga del modelo global (rondas > 1) ──────────────────────────
        latency_model_download_s = 0.0
        if round_n > 1:
            logger.info(f"[NODE] Descargando modelo agregado desde el servidor central...")
            received, net_metrics_rx = await listener_server(
                local_listen_addr, delay=LISTENER_DURATION * 6, file_path=MODEL_PATH
            )
            if received is None:
                logger.error(f"[NODE] No se recibió modelo. Abortando.")
                break
            latency_model_download_s = net_metrics_rx.get("net_transmission_time_s", 0.0)
            logger.info(f"[NODE] Métricas de red RX: BW={net_metrics_rx.get('net_bandwidth_rx_kbps', 0):.2f}kbps")

        # ── Entrenamiento local ───────────────────────────────────────────────
        logger.info(f"[NODE] Entrenando con {DATA_PATH} por {EPOCHS} épocas...")
        architecture = MLP(in_features=IN_FEATURES)
        trainer = ModelTrainer(model_path=MODEL_PATH, model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(DATA_PATH, label_col=LABEL_COLUMN)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)
        trainer_time, trainer_loss = trainer.fit(train_loader, criterion, optimizer, epochs=EPOCHS)

        # evaluate() devuelve un dict nuevo — lo usamos como base y luego
        # añadimos encima todas las métricas complementarias sin sobreescribir.
        metrics = trainer.evaluate(test_loader)
        metrics["trainning_time"] = trainer_time
        metrics["loss"] = trainer_loss

        # Métricas de hardware del nodo
        metrics.update(collect_system_metrics())

        # Latencias de red (download ya conocido, upload se completa tras el envío)
        metrics["latency_model_download_s"] = latency_model_download_s
        metrics["latency_model_upload_s"] = 0.0  # Se actualiza tras el envío

        # Métricas de red RX (descarga del modelo global) con prefijo para no
        # colisionar con las métricas TX que se añadirán después.
        if net_metrics_rx:
            metrics["net_bytes_rx_model"] = net_metrics_rx.get("net_bytes_rx_model", 0)
            metrics["net_bytes_rx_system"] = net_metrics_rx.get("net_bytes_rx_system", 0)
            metrics["net_bandwidth_rx_kbps"] = net_metrics_rx.get("net_bandwidth_rx_kbps", 0)
            metrics["net_packets_recv"] = net_metrics_rx.get("net_packets_recv", 0)
            metrics["net_errors_in"] = net_metrics_rx.get("net_errors_in", 0)
            metrics["net_drops_in"] = net_metrics_rx.get("net_drops_in", 0)

        if h_ronda is not None:
            metrics["h_ronda"] = h_ronda

        trainer.save(MODEL_PATH)

        # ── Subida del modelo entrenado al servidor ───────────────────────────
        await asyncio.sleep(SLEEP_INTERVAL)
        logger.info(f"[NODE] Enviando modelo entrenado...")
        success, net_metrics_tx = await send_file_identified(addr, server_addr, MODEL_PATH)

        # Rellenar métricas TX siempre (éxito o fallo)
        metrics["latency_model_upload_s"] = net_metrics_tx.get("net_transmission_time_s", 0.0)
        metrics["net_bytes_tx_model"] = net_metrics_tx.get("net_bytes_tx_model", 0)
        metrics["net_bytes_tx_system"] = net_metrics_tx.get("net_bytes_tx_system", 0)
        metrics["net_bandwidth_tx_kbps"] = net_metrics_tx.get("net_bandwidth_tx_kbps", 0)
        metrics["net_packets_sent"] = net_metrics_tx.get("net_packets_sent", 0)
        metrics["net_errors_out"] = net_metrics_tx.get("net_errors_out", 0)
        metrics["net_drops_out"] = net_metrics_tx.get("net_drops_out", 0)
        metrics["net_throughput_kbps"] = net_metrics_tx.get("net_throughput_kbps", 0)
        metrics["net_transmission_time_s"] = net_metrics_tx.get("net_transmission_time_s", 0)

        if success:
            logger.info(f"[NODE] Métricas de red TX: BW={net_metrics_tx.get('net_bandwidth_tx_kbps', 0):.2f}kbps")
        else:
            logger.warning("[NODE] Envío del modelo falló. Métricas TX registradas como 0.")

        await send_identified(addr, server_addr, json.dumps(metrics))
        logger.info(f"[NODE] Métricas enviadas: {metrics}")

        if round_n < ROUNDS:
            logger.info("[CLIENT] Esperando posible convergencia")
            received, _ = await listener_server(local_listen_addr, LISTENER_DURATION * 100)
            converged = (received == "CONVERGED")
            logger.info(f"[CLIENT] Convergencia: {converged}")
            if converged:
                logger.info("[NODE] Entrenamiento centralizado ha convergido")
                break

    logger.info("\n[NODE] Entrenamiento federado completado.")


def main(central: bool, addr: list, server_addr: str, childs: list[str]=None, h_ronda: int | None = None):

    if central:
        asyncio.run(central_main(addr, childs, h_ronda))
    else:
        asyncio.run(client_main(addr, server_addr, h_ronda))


if __name__ == "__main__":
    main()