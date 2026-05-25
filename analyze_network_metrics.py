#!/usr/bin/env python3
"""
Script de análisis de métricas de red para el sistema de federated learning.
Extrae, analiza y visualiza métricas de comunicación de red de los CSV de métricas.
"""

import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas no disponible. Instalalo con: pip install pandas")


class NetworkMetricsAnalyzer:
    """Analiza métricas de red del archivo CSV de métricas."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
        
        if PANDAS_AVAILABLE:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
            self.raw_data = self._read_csv_manually()

    def _read_csv_manually(self) -> List[Dict]:
        """Lee CSV manualmente sin pandas."""
        data = []
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def print_summary(self):
        """Imprime un resumen de las métricas de red."""
        if PANDAS_AVAILABLE:
            self._print_summary_pandas()
        else:
            self._print_summary_manual()

    def _print_summary_pandas(self):
        """Resumen usando pandas."""
        print("\n" + "="*80)
        print("RESUMEN DE MÉTRICAS DE RED - FEDERATED LEARNING")
        print("="*80)

        net_cols = [c for c in self.df.columns if c.startswith('net_')]
        if not net_cols:
            print("❌ No se encontraron métricas de red en el CSV")
            return

        print(f"\n📊 Total de registros: {len(self.df)}")
        print(f"🔢 Rondas: {self.df['round'].max():.0f}")
        print(f"🖥️  Nodos únicos: {self.df['node'].nunique()}")

        # Estadísticas globales de red
        print("\n" + "-"*80)
        print("ESTADÍSTICAS GLOBALES DE RED")
        print("-"*80)

        net_metrics_summary = [
            ("Bytes TX Modelo", "net_bytes_tx_model", "B"),
            ("Bytes RX Modelo", "net_bytes_rx_model", "B"),
            ("Bytes TX Sistema", "net_bytes_tx_system", "B"),
            ("Bytes RX Sistema", "net_bytes_rx_system", "B"),
            ("Paquetes Enviados", "net_packets_sent", "pkt"),
            ("Paquetes Recibidos", "net_packets_recv", "pkt"),
            ("Ancho TX", "net_bandwidth_tx_kbps", "kbps"),
            ("Ancho RX", "net_bandwidth_rx_kbps", "kbps"),
            ("Throughput", "net_throughput_kbps", "kbps"),
        ]

        for label, col, unit in net_metrics_summary:
            if col in self.df.columns:
                values = pd.to_numeric(self.df[col], errors='coerce')
                mean_val = values.mean()
                max_val = values.max()
                min_val = values.min()
                print(f"\n{label}:")
                print(f"  Media: {mean_val:.2f} {unit}")
                print(f"  Máx:   {max_val:.2f} {unit}")
                print(f"  Mín:   {min_val:.2f} {unit}")

        # Análisis por nodo
        print("\n" + "-"*80)
        print("ANÁLISIS POR NODO")
        print("-"*80)

        for node in sorted(self.df['node'].unique()):
            node_data = self.df[self.df['node'] == node]
            print(f"\n{node}:")
            
            for label, col, unit in net_metrics_summary[:3]:  # Mostrar solo bytes para no saturar
                if col in node_data.columns:
                    values = pd.to_numeric(node_data[col], errors='coerce')
                    avg = values.mean()
                    if pd.notna(avg) and avg > 0:
                        print(f"  {label}: {avg:.2f} {unit}")

        # Análisis por ronda
        print("\n" + "-"*80)
        print("ANÁLISIS POR RONDA")
        print("-"*80)

        for round_n in sorted(self.df['round'].unique()):
            round_data = self.df[self.df['round'] == round_n]
            print(f"\nRonda {round_n:.0f}:")
            
            bw_tx = pd.to_numeric(round_data['net_bandwidth_tx_kbps'], errors='coerce').mean()
            bw_rx = pd.to_numeric(round_data['net_bandwidth_rx_kbps'], errors='coerce').mean()
            throughput = pd.to_numeric(round_data['net_throughput_kbps'], errors='coerce').mean()
            
            if pd.notna(bw_tx):
                print(f"  Ancho TX: {bw_tx:.2f} kbps")
            if pd.notna(bw_rx):
                print(f"  Ancho RX: {bw_rx:.2f} kbps")
            if pd.notna(throughput):
                print(f"  Throughput: {throughput:.2f} kbps")

    def _print_summary_manual(self):
        """Resumen sin pandas."""
        print("\n" + "="*80)
        print("RESUMEN DE MÉTRICAS DE RED - FEDERATED LEARNING")
        print("="*80)

        if not self.raw_data:
            print("❌ No hay datos en el CSV")
            return

        print(f"\n📊 Total de registros: {len(self.raw_data)}")
        
        nodes = set(r.get('node', '') for r in self.raw_data if r.get('node'))
        rounds = set(r.get('round', '') for r in self.raw_data if r.get('round'))
        
        print(f"🔢 Rondas: {len(rounds)}")
        print(f"🖥️  Nodos únicos: {len(nodes)}")

        print("\n" + "-"*80)
        print("PRIMEROS 5 REGISTROS CON MÉTRICAS DE RED")
        print("-"*80)
        
        for i, row in enumerate(self.raw_data[:5]):
            print(f"\nRegistro {i+1}:")
            for key in ['node', 'round', 'net_bytes_tx_model', 'net_bytes_rx_model',
                       'net_bandwidth_tx_kbps', 'net_bandwidth_rx_kbps']:
                if key in row and row[key]:
                    print(f"  {key}: {row[key]}")

    def export_network_only(self, output_file: str = "network_metrics_only.csv"):
        """Exporta solo las columnas de métricas de red."""
        if PANDAS_AVAILABLE:
            net_cols = ['round', 'node'] + [c for c in self.df.columns if c.startswith('net_')]
            net_cols = [c for c in net_cols if c in self.df.columns]
            self.df[net_cols].to_csv(output_file, index=False)
            print(f"✅ Métricas de red exportadas a: {output_file}")
        else:
            print("❌ pandas requerido para esta función")

    def print_node_comparison(self):
        """Compara métricas entre nodos."""
        if not PANDAS_AVAILABLE:
            print("❌ pandas requerido para esta función")
            return

        print("\n" + "="*80)
        print("COMPARACIÓN DE NODOS")
        print("="*80)

        comparison_metrics = ['net_bandwidth_tx_kbps', 'net_bandwidth_rx_kbps', 'net_throughput_kbps']
        
        for metric in comparison_metrics:
            if metric in self.df.columns:
                print(f"\n{metric}:")
                node_stats = self.df.groupby('node')[metric].agg(['mean', 'max', 'min', 'std'])
                node_stats = node_stats.apply(pd.to_numeric, errors='coerce')
                print(node_stats.to_string())

    def print_errors_summary(self):
        """Resumen de errores y drops de red."""
        if not PANDAS_AVAILABLE:
            print("❌ pandas requerido para esta función")
            return

        print("\n" + "="*80)
        print("RESUMEN DE ERRORES Y DROPS")
        print("="*80)

        error_cols = ['net_errors_in', 'net_errors_out', 'net_drops_in', 'net_drops_out']
        error_cols = [c for c in error_cols if c in self.df.columns]

        if not error_cols:
            print("✅ No hay datos de errores o drops")
            return

        for col in error_cols:
            total_errors = pd.to_numeric(self.df[col], errors='coerce').sum()
            if total_errors > 0:
                print(f"\n⚠️  {col}: {int(total_errors)} (TOTAL)")
            else:
                print(f"\n✅ {col}: 0")


def main():
    parser = argparse.ArgumentParser(description="Analizar métricas de red del federated learning")
    parser.add_argument("csv_file", help="Ruta al archivo metrics.csv")
    parser.add_argument("--export-network", action="store_true", help="Exportar solo métricas de red")
    parser.add_argument("--compare-nodes", action="store_true", help="Comparar métricas entre nodos")
    parser.add_argument("--errors", action="store_true", help="Mostrar resumen de errores")
    
    args = parser.parse_args()

    try:
        analyzer = NetworkMetricsAnalyzer(args.csv_file)
        
        analyzer.print_summary()
        
        if args.compare_nodes:
            analyzer.print_node_comparison()
        
        if args.errors:
            analyzer.print_errors_summary()
        
        if args.export_network:
            analyzer.export_network_only()

        print("\n" + "="*80)
        print("✅ Análisis completado")
        print("="*80 + "\n")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        exit(1)


if __name__ == "__main__":
    main()
