"""Premium user-data module.

This module imports user data via Excel templates, normalizes it to the
CORE timeline and submarket scope, and computes energy/financial exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PREMIUM_SHEET = "dados_usuario"
SHEET_CONSUMO = "consumo"
SHEET_GERACAO = "geracao"
SHEET_CONTRATOS = "contratos"

TEMPLATE_COLUMNS = [
    "data",
    "hora",
    "submercado",
    "consumo_mwh",
    "geracao_mwh",
    "contratos_mwh",
    "preco_contrato",
]


@dataclass
class PremiumResult:
    data: pd.DataFrame
    resumo: Dict[str, Any]
    alertas: List[str]


def generate_premium_template(path: str) -> None:
    """Generate an Excel template for premium user inputs."""
    df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=PREMIUM_SHEET, index=False)


def load_premium_excel(file_path_or_buffer: Any) -> pd.DataFrame:
    """Load user data from Excel.

    Supports a single sheet named 'dados_usuario' or three sheets
    (consumo, geracao, contratos) that will be merged by data/hora/submercado.
    """
    xls = pd.ExcelFile(file_path_or_buffer)
    sheets = xls.sheet_names

    if PREMIUM_SHEET in sheets:
        df = pd.read_excel(xls, PREMIUM_SHEET)
        return _normalize_single_sheet(df)

    if all(sheet in sheets for sheet in [SHEET_CONSUMO, SHEET_GERACAO, SHEET_CONTRATOS]):
        consumo = pd.read_excel(xls, SHEET_CONSUMO)
        geracao = pd.read_excel(xls, SHEET_GERACAO)
        contratos = pd.read_excel(xls, SHEET_CONTRATOS)
        return _merge_multi_sheet(consumo, geracao, contratos)

    raise ValueError(
        "Template inválido. Use a aba 'dados_usuario' ou as abas 'consumo', "
        "'geracao' e 'contratos'."
    )


def _normalize_single_sheet(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in TEMPLATE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no template: {missing}")

    normalized = df.copy()
    normalized = _normalize_types(normalized)
    return normalized


def _merge_multi_sheet(
    consumo: pd.DataFrame,
    geracao: pd.DataFrame,
    contratos: pd.DataFrame,
) -> pd.DataFrame:
    consumo = _normalize_basic(consumo, "consumo_mwh")
    geracao = _normalize_basic(geracao, "geracao_mwh")
    contratos = _normalize_basic(contratos, "contratos_mwh")

    merged = consumo.merge(geracao, on=["data", "hora", "submercado"], how="outer")
    merged = merged.merge(contratos, on=["data", "hora", "submercado"], how="outer")
    merged["preco_contrato"] = merged.get("preco_contrato", 0)
    merged = _normalize_types(merged)
    return merged


def _normalize_basic(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    expected = {"data", "hora", "submercado", value_column}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    normalized = df.copy()
    if value_column not in normalized.columns:
        normalized[value_column] = 0
    return normalized


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["data"] = pd.to_datetime(normalized["data"]).dt.date
    normalized["hora"] = normalized["hora"].astype(int)
    normalized["submercado"] = normalized["submercado"].astype(str).str.upper()

    for col in ["consumo_mwh", "geracao_mwh", "contratos_mwh", "preco_contrato"]:
        if col not in normalized.columns:
            normalized[col] = 0
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0)

    normalized["timestamp"] = pd.to_datetime(normalized["data"].astype(str)) + pd.to_timedelta(
        normalized["hora"], unit="h"
    )
    return normalized


def build_pld_lookup(ccee_records: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], float]:
    lookup: Dict[Tuple[str, int, str], float] = {}
    for record in ccee_records:
        data = record.get("DIA") or record.get("data") or record.get("DATA")
        hora = record.get("HORA") or record.get("hora")
        submercado = record.get("SUBMERCADO") or record.get("submercado")
        pld = record.get("PLD_HORA") or record.get("pld_valor") or record.get("PLD")
        if data is None or hora is None or submercado is None or pld is None:
            continue
        key = (str(data), int(hora), str(submercado).upper())
        lookup[key] = float(pld)
    return lookup


def calculate_exposures(user_df: pd.DataFrame, pld_lookup: Dict[Tuple[str, int, str], float]) -> pd.DataFrame:
    df = user_df.copy()
    df["exposicao_energetica_mwh"] = df["consumo_mwh"] - df["geracao_mwh"] - df["contratos_mwh"]

    def _lookup_pld(row: pd.Series) -> float:
        key = (str(row["data"]), int(row["hora"]), str(row["submercado"]).upper())
        return pld_lookup.get(key, 0.0)

    df["pld"] = df.apply(_lookup_pld, axis=1)
    df["exposicao_financeira"] = df["exposicao_energetica_mwh"] * df["pld"]
    return df


def build_premium_summary(exposure_df: pd.DataFrame) -> PremiumResult:
    if exposure_df.empty:
        return PremiumResult(data=exposure_df, resumo={}, alertas=["Sem dados do usuário."])

    total_exposicao_mwh = exposure_df["exposicao_energetica_mwh"].sum()
    total_financeiro = exposure_df["exposicao_financeira"].sum()
    overhedge = exposure_df[exposure_df["exposicao_energetica_mwh"] < 0].shape[0]
    underhedge = exposure_df[exposure_df["exposicao_energetica_mwh"] > 0].shape[0]

    alertas = []
    if underhedge > overhedge:
        alertas.append("Predominância de underhedge (exposição comprada).")
    if overhedge > underhedge:
        alertas.append("Predominância de overhedge (exposição vendida).")

    resumo = {
        "total_exposicao_mwh": float(total_exposicao_mwh),
        "total_exposicao_financeira": float(total_financeiro),
        "linhas_underhedge": int(underhedge),
        "linhas_overhedge": int(overhedge),
    }

    return PremiumResult(data=exposure_df, resumo=resumo, alertas=alertas)
