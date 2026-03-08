"""Utilitários comuns do Kintuadi Energy.

Inclui:
- serialização segura para JSON (pandas, numpy, datetime)
- persistência de registros/datasets em CSV
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import date, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class JSONEncoder(json.JSONEncoder):
    """Encoder JSON robusto para pandas, numpy e datas."""

    def default(self, obj: Any):
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        return super().default(obj)


def make_serializable(obj: Any) -> Any:
    """Converte recursivamente qualquer objeto em JSON-serializável."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.tolist()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def save_json(data: Any, filename: str) -> bool:
    """Salva dados em JSON usando encoder customizado com fallback robusto."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=JSONEncoder, ensure_ascii=False, indent=2)
        return True
    except Exception:
        try:
            serializable_data = make_serializable(data)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


def load_json(filename: str) -> Any:
    """Carrega dados de um arquivo JSON."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_records_to_csv(
    records: List[Dict[str, Any]],
    dataset_name: str,
    base_dir: str = "data",
) -> str:
    """Salva lista de registros em CSV e remove versões antigas do dataset."""
    if not records:
        raise ValueError("Nenhum registro fornecido para salvar em CSV.")

    os.makedirs(base_dir, exist_ok=True)

    for filename in os.listdir(base_dir):
        if filename.startswith(dataset_name) and filename.endswith(".csv"):
            os.remove(os.path.join(base_dir, filename))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(base_dir, f"{dataset_name}_{timestamp}.csv")

    fieldnames = sorted({key for record in records for key in record.keys()})

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return filepath


def save_raw_csv_file(
    source_path: str,
    dataset_name: str,
    base_dir: str = "data",
) -> str:
    """Copia CSV bruto para diretório de destino e remove versões antigas."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {source_path}")

    os.makedirs(base_dir, exist_ok=True)

    for filename in os.listdir(base_dir):
        if filename.startswith(dataset_name) and filename.endswith(".csv"):
            os.remove(os.path.join(base_dir, filename))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(base_dir, f"{dataset_name}_{timestamp}.csv")

    shutil.copy2(source_path, target_path)

    return target_path
