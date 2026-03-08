# Novo arquivo: scripts/data_models.py
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import List, Dict, Optional, Any
import json

@dataclass
class DataMetadata:
    """Metadados padrão para todos os dados"""
    source: str
    collection_time: str
    status: str  # success, error, partial
    records_processed: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ReservoirData:
    """Dados padronizados de reservatório"""
    nome: str
    subsistema: str
    volume_percentual: float
    volume_util: float
    energia_armazenada: float
    data_atualizacao: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PLDData:
    """Dados padronizados de PLD"""
    data: str
    hora: str
    submercado: str
    pld_valor: float
    periodo_comercializacao: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class MarketAnalysis:
    """Análise de mercado padronizada"""
    timestamp: str
    tendencia_mercado: str
    indice_seguranca: float
    alerta: bool
    nivel_alerta: str  # baixo, medio, alto, critico
    recomendacoes: List[str]
    indicadores: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class KintuadiDataset:
    """Conjunto completo de dados padronizados"""
    metadata: DataMetadata
    reservatorios: List[ReservoirData]
    pld_data: List[PLDData]
    analysis: MarketAnalysis
    summary: Dict[str, Any]
    
    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)
    
    def to_dict(self):
        return asdict(self)