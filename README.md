# Hierarchical Federated Learning System for NCD Mortality Classification

A privacy-preserving, tree-structured federated learning system for binary classification of premature mortality caused by Non-Communicable Diseases (NCDs). Each node trains a local MLP model on its own hospital dataset; only model weights are exchanged — raw patient data never leaves the node.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model](#model)
- [Metrics Collected](#metrics-collected)
- [Configuration](#configuration)
- [Quick Start — Docker Compose](#quick-start--docker-compose)
- [Quick Start — Bare Metal / Raspberry Pi](#quick-start--bare-metal--raspberry-pi)
- [Adding or Removing Nodes](#adding-or-removing-nodes)
- [Analyzing Results](#analyzing-results)
- [Requirements](#requirements)

---

## Architecture Overview

The system uses a **hierarchical tree topology**. Communication is WebSocket-based and fully asynchronous.

```
                    ┌─────────┐
                    │ CENTRAL │  Level 0 — root, aggregator
                    └────┬────┘
              ┌──────────┴──────────┐
          ┌───┴───┐             ┌───┴───┐
          │ node1 │             │ node2 │  Level 1 — intermediate
          └───┬───┘             └───┬───┘
        ┌─────┴─────┐         ┌────┴────┐
    ┌───┴┐ ┌───┴┐ ┌─┴──┐  ┌──┴─┐   ┌──┴─┐
    │ n3 │ │ n4 │ │ n5 │  │ n6 │   │ n7 │  Level 2 — leaf nodes
    └────┘ └────┘ └────┘  └────┘   └────┘
```

Each **leaf node** holds a private hospital dataset and trains locally. Each **intermediate node** acts as a sub-aggregator for its subtree before forwarding to the root. The **central node** runs FedAvg across all received models and distributes the updated global model.

The outer loop is the **hierarchical round** (`H_ROUNDS`); inside each hierarchical round, the federated loop runs for `ROUNDS` inner rounds before checking global convergence.

---

## Project Structure

```
.
├── main.py                    # Entry point; orchestrates hierarchy setup and federated rounds
├── federated.py               # central_main() and client_main() — core FL logic
├── config.py                  # All hyperparameters and file paths
├── utils.py                   # CSV metrics writer, node registry, convergence helpers
├── network_metrics.py         # NetworkMetricsCollector, NetworkMetricsContext, collect_system_metrics
├── logging_config.py          # Centralized logging (console + rotating file)
├── analyze_network_metrics.py # CLI tool for post-hoc analysis of metrics.csv
│
├── connections/
│   ├── client.py              # send(), send_identified(), send_file_identified(),
│   │                          # send_file_to_nodes(), send_message_to_nodes()
│   └── server.py              # listener_ips(), listener_nodes(), listener_server()
│
├── model/
│   ├── create_model.py        # MLP and MLPLarger architectures + create_model()
│   └── fed_model.py           # ModelTrainer (fit, evaluate, save) + federated_average()
│
├── data/
│   ├── Hospital.csv           # Full dataset (central reference)
│   ├── Hospital_1.csv         # Partition for node1
│   ├── Hospital_2.csv         # Partition for node2
│   └── ...                    # One CSV per leaf node
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## How It Works

### Startup — Hierarchy Discovery

1. Every node calls `main.py <own_addr> <parent_addr>`.
2. Each non-root node registers itself with its parent by sending its address over WebSocket.
3. The root creates `model.pt` with random weights and distributes it down the tree.
4. Once every node has the initial model, the federated training loop begins.

### Inner Federated Round (repeated `ROUNDS` times per hierarchical round)

| Step | Central node | Leaf / client node |
|------|--------------|--------------------|
| **1** | Distributes current `model.pt` to direct children (rounds > 1) | Receives `model.pt` from parent, records download latency |
| **2** | Waits for trained models + JSON metrics from all children | Trains locally for `EPOCHS` epochs, evaluates on local test split |
| **3** | Receives `model.pt` + metrics JSON from each node | Sends trained `model.pt` + JSON metrics to parent |
| **4** | Runs FedAvg → saves new `model.pt` | Waits for convergence signal |
| **5** | Evaluates convergence (loss window K=3, tol=1e-4), appends `metrics.csv` | Stops if `CONVERGED`, otherwise proceeds to next round |
| **6** | Broadcasts `CONVERGED` or `NOT CONVERGED` | — |

### Outer Hierarchical Round (repeated `H_ROUNDS` times)

After each inner FL session, intermediate nodes forward aggregated models up the tree, the root re-aggregates, and convergence is voted on from leaves to root. If all subtrees agree on `CONVERGED`, training stops early.

---

## Model

**MLP** — Multilayer Perceptron for binary classification (premature NCD mortality).

```
Input (79 features)
  → Linear(79 → 120) → ReLU → Dropout(0.3)
  → Linear(120 → 84) → ReLU → Dropout(0.3)
  → Linear(84 → 1)   → logit
```

Output is a raw logit; loss function is `BCEWithLogitsLoss`. Predictions use a threshold of 0.5 on `sigmoid(logit)`.

A larger variant **MLPLarger** (256 → 128 → 64 → 1) is available in `model/create_model.py` for datasets with more features.

**Input preprocessing** (inside `ModelTrainer.load_csv`):
- Rows with any `NaN` are dropped.
- All features are standardized with `StandardScaler` (zero mean, unit variance).
- An 80/20 train-test split is applied with a fixed random seed.

---

## Metrics Collected

All metrics are appended to `metrics.csv` after each inner round by the central node. Every row represents one node's contribution to one round.

### Training & Model Quality

| Column | Description |
|--------|-------------|
| `round` | Inner federated round number |
| `h_ronda` | Outer hierarchical round number |
| `node` | Node identifier (e.g. `Nodo_1`) |
| `accuracy` | Overall accuracy on local test set |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `f1_score` | Macro-averaged F1 |
| `specificity` | True Negative Rate |
| `sensitivity` | True Positive Rate |
| `trainning_time` | Local training wall-clock time (seconds) |
| `loss` | BCE loss on last training epoch |

### Communication Latency

| Column | Description |
|--------|-------------|
| `latency_model_download_s` | Time the node spent downloading the global model |
| `latency_model_upload_s` | Time the node spent uploading its trained model |

### Network — Node Side (measured at the node during TX/RX)

| Column | Description |
|--------|-------------|
| `net_bytes_tx_model` | Model bytes sent by the node |
| `net_bytes_tx_system` | Total system bytes TX during upload |
| `net_bandwidth_tx_kbps` | Upload bandwidth (kbps) |
| `net_packets_sent` | Packets sent |
| `net_errors_out` / `net_drops_out` | TX errors and drops |
| `net_bytes_rx_model` | Model bytes received by the node |
| `net_bytes_rx_system` | Total system bytes RX during download |
| `net_bandwidth_rx_kbps` | Download bandwidth (kbps) |
| `net_packets_recv` | Packets received |
| `net_errors_in` / `net_drops_in` | RX errors and drops |
| `net_throughput_kbps` | Combined TX+RX throughput (kbps) |
| `net_transmission_time_s` | Upload transmission duration |

### Network — Central Server Side

| Column | Description |
|--------|-------------|
| `central_net_bytes_rx_model` | Model bytes received by the central server |
| `central_net_bytes_rx_system` | Total system bytes RX at the central during reception |
| `central_net_bandwidth_rx_kbps` | Bandwidth at the central while receiving |
| `central_net_throughput_kbps` | Throughput at the central during reception |
| `central_net_transmission_time_s` | Time central spent receiving a node's model |
| `central_net_bytes_tx_model` | Model bytes distributed by the central |
| `central_net_bytes_tx_system` | Total system bytes TX at the central during distribution |
| `central_net_bandwidth_tx_kbps` | Bandwidth at the central while distributing |
| `central_latency_model_dist_s` | Time central spent sending the global model to a node |

### Hardware (sampled at the node after training)

| Column | Description |
|--------|-------------|
| `cpu_percent` | CPU utilization (%) |
| `ram_percent` | RAM utilization (%) |
| `cpu_freq_mhz` | Current CPU frequency (MHz) |
| `open_sockets` | Open network sockets |

### Aggregation & Convergence

| Column | Description |
|--------|-------------|
| `comm_overhead_bytes` | Extra communication cost: `bytes_rx_model × (n_nodes − 1)` |
| `aggregation_time_s` | FedAvg computation time at the central (seconds) |
| `inter_silo_variance` | Variance of loss values across all nodes in this round |
| `converged` | `True` if convergence was detected this round, `False` otherwise |
| `converged_round` | Round number at convergence, `None` if not yet converged |
| `convergence_time_s` | Wall-clock time from round start to convergence, `None` if not converged |

> Convergence is declared when the mean loss across nodes has changed by less than `tol=1e-4` over the last `K=3` consecutive rounds.

---

## Configuration

All tuneable parameters live in `config.py`. No environment variables are required.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROUNDS` | `5` | Inner federated rounds per hierarchical round |
| `H_ROUNDS` | `5` | Outer hierarchical rounds |
| `EPOCHS` | `5` | Local training epochs per round |
| `LEARNING_RATE` | `1e-3` | Adam optimizer learning rate |
| `IN_FEATURES` | `79` | Number of input features |
| `BATCH_SIZE` | `32` | Mini-batch size |
| `TEST_SIZE` | `0.2` | Local train/test split ratio |
| `DROPOUT_PROB` | `0.3` | MLP dropout probability |
| `LABEL_COLUMN` | `is_premature_ncd` | Target column in the CSV |
| `DATA_PATH` | `data/Hospital.csv` | Path to the local dataset |
| `MODEL_PATH` | `model.pt` | Shared model weights file |
| `METRICS_CSV` | `metrics.csv` | Output metrics file |
| `LISTENER_DURATION` | `20` | Seconds of inactivity before the IP-discovery server closes |
| `NODES_LISTENER_DELAY` | `300` | Max seconds to wait for all node models per round |
| `POST_ROUND_DELAY` | `2` | Pause between rounds to avoid port collisions |

---

## Quick Start — Docker Compose

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- One partitioned CSV per node placed in `./data/` (e.g. `Hospital_1.csv` … `Hospital_7.csv`)

### Build and run

```bash
# Build the shared image once
docker compose build

# Start all nodes (central + node1..node7)
docker compose up
```

Docker Compose manages startup order via `depends_on`. The central node listens first; child nodes register once the central is ready.

### Stopping

```bash
docker compose down
```

Metrics are written to `metrics.csv` in the working directory of the central container. Mount a host volume if you need to persist them:

```yaml
# Add to the 'central' service in docker-compose.yml
volumes:
  - ./output:/app
```

### Scaling to fewer nodes

Comment out unused services in `docker-compose.yml` and remove the corresponding data volumes. The system discovers connected nodes dynamically — no code changes required.

---

## Quick Start — Bare Metal / Raspberry Pi

This mode is useful for running each node on a separate physical machine (e.g. a Raspberry Pi cluster).

### 1. Install dependencies on every machine

```bash
pip install -r requirements.txt
```

### 2. Create the initial model (central machine only)

```bash
python create_model.py
```

This writes `model.pt` in the current directory.

### 3. Start the central node

```bash
python main.py <central_ip>:8765 <central_ip>:8765
```

Pass the same address twice to signal that this node is the root.

### 4. Start intermediate and leaf nodes

Each node takes its own address and its parent's address:

```bash
# Level-1 nodes (connect to central)
python main.py <node1_ip>:8766 <central_ip>:8765
python main.py <node2_ip>:8767 <central_ip>:8765

# Level-2 leaf nodes (connect to their Level-1 parent)
python main.py <node3_ip>:8768 <node1_ip>:8766
python main.py <node4_ip>:8769 <node1_ip>:8766
python main.py <node5_ip>:8770 <node2_ip>:8767
python main.py <node6_ip>:8771 <node2_ip>:8767
python main.py <node7_ip>:8772 <node3_ip>:8768
```

Each node expects its local dataset at `data/Hospital.csv`. Place the correct partition there before starting.

### 5. Collect results

`metrics.csv` and the `logs/` directory are written to the working directory of the central node.

---

## Adding or Removing Nodes

The topology is defined entirely by the command-line arguments passed to `main.py`. To add a node:

1. Add a new service in `docker-compose.yml` (Docker) or start a new process (bare metal).
2. Point its parent address at the desired parent node.
3. Mount or place the partitioned dataset at `data/Hospital.csv`.
4. No changes to source code are needed.

To remove a node, simply stop its process or remove its service from `docker-compose.yml`. The parent node will time out and proceed with the nodes that did respond.

---

## Analyzing Results

After training, `metrics.csv` contains one row per node per inner round. Use the bundled analysis script:

```bash
# Print a full summary (global stats, per-node, per-round)
python analyze_network_metrics.py metrics.csv

# Also compare nodes side-by-side
python analyze_network_metrics.py metrics.csv --compare-nodes

# Show TX/RX error and drop counts
python analyze_network_metrics.py metrics.csv --errors

# Export a CSV containing only the network columns
python analyze_network_metrics.py metrics.csv --export-network
```

For custom analysis, load `metrics.csv` directly with pandas:

```python
import pandas as pd

df = pd.read_csv("metrics.csv")

# Average F1 per round across all nodes
df.groupby("round")["f1_score"].mean().plot(title="F1 per round")

# Upload bandwidth per node
df.pivot(index="round", columns="node", values="net_bandwidth_tx_kbps").plot()

# Check which round convergence was reached
print(df[df["converged"] == True][["round", "node", "convergence_time_s"]].drop_duplicates())
```

Training logs are saved under `logs/federated_<timestamp>.log` on each node.

---

## Requirements

```
torch
websockets
psutil
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

Python 3.12 is recommended (used in the official Docker image). Python 3.10+ is required for the `match`/`|` type union syntax used in type hints.