import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from config import TEST_SIZE, BATCH_SIZE, LABEL_COLUMN, RANDOM_STATE, EPOCHS
from logging_config import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, model_path: str, model_architecture: nn.Module, device: str = None):
        """
        Entrada: ruta del archivo de pesos .pt, instancia de la arquitectura, dispositivo opcional.
        Salida: instancia lista para entrenar con los pesos cargados.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[INIT] Utilizando dispositivo: {self.device.upper()}")

        self.model = model_architecture.to(self.device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"[INIT] Pesos cargados desde: {model_path}")
        except Exception as e:
            logger.warning(f"[WARNING] No se pudieron cargar los pesos: {e}. Se entrena desde cero.")

    def load_csv(self, path, label_col=LABEL_COLUMN, test_size=TEST_SIZE, batch_size=BATCH_SIZE):
        df = pd.read_csv(path)
        
        # 1. Eliminar filas con NaN
        df = df.dropna()
        
        X = df.drop(columns=[label_col]).values.astype("float32")
        y = df[label_col].values.astype("float32")
        
        # 2. Normalizar features (media 0, std 1)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype("float32")
        
        # 3. Debug rápido
        logger.info(f"[DATA] X: shape={X.shape}, nan={np.isnan(X).sum()}, max={X.max():.2f}")
        logger.info(f"[DATA] y: unique={np.unique(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

        to_loader = lambda xd, yd: DataLoader(
            TensorDataset(torch.tensor(xd), torch.tensor(yd).unsqueeze(1)),
            batch_size=batch_size,
            shuffle=True,
        )
        return to_loader(X_train, y_train), to_loader(X_test, y_test)

    def fit(self, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epochs: int = EPOCHS) -> tuple[float, float]:
        """
        Entrada: DataLoader de entrenamiento, función de pérdida, optimizador, número de épocas.
        Salida: tiempo total de entrenamiento en segundos.
        """
        logger.info(f"\n[TRAINING] Iniciando entrenamiento por {epochs} épocas...")
        start = time.time()
        self.model.train()
        
        loss = 0
        for epoch in range(epochs):
            running_loss = 0.0
            t0 = time.time()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            loss = epoch_loss
            logger.info(f" Época [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | {time.time()-t0:.2f}s")

        total = time.time() - start
        logger.info(f"[TRAINING] Completado en {total:.2f}s")
        return total, loss

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Entrada: DataLoader de test.
        Salida: dict con accuracy, precision, recall y f1_score.
        """
        logger.info("\n[EVAL] Evaluando modelo...")
        self.model.eval()
        all_preds, all_labels = [], []

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            preds = (torch.sigmoid(self.model(inputs)) >= 0.5).long().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(1).long().numpy())

        metrics = {
            "accuracy":  sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1_score":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
        self._print_metrics(metrics)
        return metrics

    def _print_metrics(self, metrics: dict):
        """Entrada: dict de métricas. Salida: ninguna (imprime tabla)."""
        logger.info("-" * 35)
        for k, v in metrics.items():
            logger.info(f" {k.capitalize():<14} | {v:.4f}")
        logger.info("-" * 35)

    def save(self, output_path: str):
        """Entrada: ruta destino. Salida: ninguna (guarda state_dict del modelo)."""
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"[SAVED] Modelo guardado en: {output_path}")


def federated_average(model_paths: list[str]) -> dict:
    """
    Entrada: lista de rutas a archivos .pt con state_dicts de modelos entrenados.
    Salida: state_dict promediado (FedAvg).
    """
    state_dicts = [torch.load(p, map_location="cpu") for p in model_paths]
    avg = {}
    for key in state_dicts[0]:
        avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    return avg