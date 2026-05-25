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

    # Extraemos el puerto en el que este nodo específico debe escuchar
    _, listen_port = get_ipport(addr)
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[CENTRAL] ══ Ronda {round_n}/{ROUNDS} ══")

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
        central_rx_metrics: dict[str, dict] = {}
        for node_id, received in results.items():
            for item in received:
                if isinstance(item, str) and os.path.isfile(item) and item.endswith(".pt"):
                    model_paths.append(item)
                elif isinstance(item, dict):
                    central_rx_metrics[node_id] = item
                    logger.info(f"[CENTRAL] Métricas de red capturadas para {node_id}: {item}")
                elif isinstance(item, str):
                    try:
                        m = json.loads(item)
                        m["node"] = node_id
                        metrics_list.append(m)
                    except json.JSONDecodeError:
                        pass

        for metrics in metrics_list:
            node_id = metrics.get("node")
            if h_ronda is not None:
                metrics["h_ronda"] = h_ronda
            if node_id in central_rx_metrics:
                metrics.update(central_rx_metrics[node_id])
                metrics["comm_overhead_bytes"] = int(
                    central_rx_metrics[node_id].get("net_bytes_rx_model", 0) * max(0, len(ips) - 1)
                )

        if not model_paths:
            logger.error("[CENTRAL] No se recibieron modelos. Abortando.")
            break

        aggregation_start = time.time()
        avg_state = federated_average(model_paths)
        aggregation_time_s = round(time.time() - aggregation_start, 3)
        torch.save(avg_state, MODEL_PATH)
        logger.info(f"[CENTRAL] Modelo global actualizado → {MODEL_PATH}")

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
            convergence_time_s=time.time() - aggregation_start,
        )
        
        if round_n < ROUNDS:
            logger.info(f"[CENTRAL] Enviando mensaje de convergencia: CONVERGED {converged}")
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
    # Usamos 0.0.0.0 para que escuche en todas las tarjetas de red (Docker o Raspberrys)
    local_listen_addr = f"0.0.0.0:{listen_port}"

    for round_n in range(1, ROUNDS + 1):
        logger.info(f"\n[NODE] ══ Ronda {round_n}/{ROUNDS} ══")

        net_metrics_round = {}  # Almacenar métricas de red de esta ronda
        metrics = {}

        if round_n > 1:
            logger.info(f"[NODE] Descargando modelo agregado desde el servidor central...")
            received, net_metrics_rx = await listener_server(local_listen_addr, delay=LISTENER_DURATION * 6, file_path=MODEL_PATH)
            if received is None:
                logger.error(f"[NODE] No se recibió modelo. Abortando.")
                break
            net_metrics_round.update(net_metrics_rx)
            logger.info(f"[NODE] Métricas de red RX: BW={net_metrics_rx.get('net_bandwidth_rx_kbps', 0):.2f}kbps")
            metrics["latency_model_download_s"] = net_metrics_rx.get("net_transmission_time_s", 0)

        logger.info(f"[NODE] Entrenando con {DATA_PATH} por {EPOCHS} épocas...")
        architecture = MLP(in_features=IN_FEATURES)
        trainer = ModelTrainer(model_path=MODEL_PATH, model_architecture=architecture)
        train_loader, test_loader = trainer.load_csv(DATA_PATH, label_col=LABEL_COLUMN)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(architecture.parameters(), lr=LEARNING_RATE)
        trainer_time, trainer_loss = trainer.fit(train_loader, criterion, optimizer, epochs=EPOCHS)
        metrics = trainer.evaluate(test_loader)
        metrics["trainning_time"] = trainer_time
        metrics["loss"] = trainer_loss
        metrics.update(collect_system_metrics())
        if h_ronda is not None:
            metrics["h_ronda"] = h_ronda
        trainer.save(MODEL_PATH)

        await asyncio.sleep(SLEEP_INTERVAL)
        logger.info(f"[NODE] Enviando modelo entrenado...")
        success, net_metrics_tx = await send_file_identified(addr, server_addr, MODEL_PATH)
        net_metrics_round.update(net_metrics_tx)
        if success:
            metrics["latency_model_upload_s"] = net_metrics_tx.get("net_transmission_time_s", 0)
            logger.info(f"[NODE] Métricas de red TX: BW={net_metrics_tx.get('net_bandwidth_tx_kbps', 0):.2f}kbps")

        if net_metrics_round:
            metrics.update(net_metrics_round)

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