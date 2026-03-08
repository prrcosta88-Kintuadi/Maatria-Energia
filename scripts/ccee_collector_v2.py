# scripts/ccee_collector_v2.py - COM AUDITORIA
import requests
import pandas as pd
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, date
import logging
from typing import List, Dict, Optional, Any
from io import StringIO
import os
import glob
try:
    from .data_models import PLDData, DataMetadata
except Exception:
    from dataclasses import dataclass

    @dataclass
    class DataMetadata:
        source: str
        collection_time: str
        status: str
        records_processed: int = 0
        error_message: str = ""

        def to_dict(self):
            return {
                "source": self.source,
                "collection_time": self.collection_time,
                "status": self.status,
                "records_processed": self.records_processed,
                "error_message": self.error_message,
            }

    @dataclass
    class PLDData:
        data: str
        submercado: str
        pld_valor: float
        hora: int = 0
        mes_referencia: int = 0
        periodo_comercializacao: int = 0

        def to_dict(self):
            return {
                "data": self.data,
                "submercado": self.submercado,
                "pld": self.pld_valor,
                "pld_hora": self.pld_valor,
                "hora": self.hora,
                "mes_referencia": self.mes_referencia,
                "periodo_comercializacao": self.periodo_comercializacao,
            }

try:
    from .audit_logger import AuditLogger
except Exception:
    class AuditLogger:
        def save_raw_data(self, *args, **kwargs):
            return None
        def log_data_transformation(self, *args, **kwargs):
            return None
        def log_consolidation(self, *args, **kwargs):
            return None
        def log_api_call(self, *args, **kwargs):
            return None
        def log_anomaly(self, *args, **kwargs):
            return None

logger = logging.getLogger(__name__)

class CCEEPLDCollector:
    """Coletor otimizado de dados PLD da CCEE com auditoria"""

    CKAN_BASE_URL = "https://dadosabertos.ccee.org.br/api/3/action/datastore_search"
    PLD_HISTORICAL_RESOURCES = {
        "2021": "51922462-16b4-4c64-8327-4e14d6ee8c6c",
        "2022": "723cf7e6-6c29-4da6-aa39-e4c8804baf65",
        "2023": "5fc317af-7191-4f8a-94e7-f77c56c747b3",
        "2024": "1b5b6946-8036-4622-a7a3-b21f33fc52b7",
        "2025": "2a180a6b-f092-43eb-9f82-a48798b803dc",
        "2026": "3f279d6b-1069-42f7-9b0a-217b084729c4",
    }
    
    def __init__(self, cache_ttl_minutes: int = 60, enable_audit: bool = True):
        self.base_url = "https://dadosabertos.ccee.org.br/api/3/action"
        self.resource_id = "3f279d6b-1069-42f7-9b0a-217b084729c4"
        self.cache_ttl = cache_ttl_minutes
        self._cache = {}
        self._cache_time = {}
        
        # Sistema de auditoria
        self.enable_audit = enable_audit
        if enable_audit:
            self.audit_logger = AuditLogger()

        self._additional_datasets = {
            "contabilizacao_montante_perfil_agente": "76d1cf4c-da8c-47a5-9f0d-8b50079be960",
            "sumario_balanco_energetico_horario": "9418da65-0f9f-4f66-a43f-6517db9653f3",
            "sumario_distribuicao_mensal": "9e8e3f5f-58a8-4744-b6da-7309a4513fcb",
        }
        self._resource_show_url = f"{self.base_url}/resource_show"
        self._pld_2026_dump_csv = "https://dadosabertos.ccee.org.br/datastore/dump/3f279d6b-1069-42f7-9b0a-217b084729c4?bom=True"
    
    def collect_pld_data(self, days: int = 7) -> Dict:
        """Coleta dados PLD com auditoria completa"""
        
        metadata = DataMetadata(
            source="CCEE_PLD",
            collection_time=datetime.now().isoformat(),
            status="pending",
            records_processed=0
        )
        
        try:
            # Verifica cache
            cache_key = f"pld_{days}d"
            if self._is_cache_valid(cache_key):
                logger.info(f"Usando dados em cache: {cache_key}")
                return self._cache[cache_key]
            
            # 1. Busca dados brutos
            logger.info("Coletando dados brutos da CCEE...")
            raw_data = self._fetch_pld_data(days)
            
            if not raw_data:
                metadata.status = "error"
                metadata.error_message = "Nenhum dado coletado"
                return {"metadata": metadata.to_dict(), "data": []}
            
            # Auditoria: Salva dados brutos
            if self.enable_audit:
                self.audit_logger.save_raw_data(
                    source="CCEE_PLD",
                    raw_data=raw_data,
                    metadata={
                        'resource_id': self.resource_id,
                        'days_requested': days,
                        'collection_time': metadata.collection_time
                    }
                )
            
            # 2. Processa dados
            logger.info("Processando dados PLD...")
            pld_objects = self._create_pld_objects(raw_data)
            
            # Auditoria: Log da transformação
            if self.enable_audit:
                self.audit_logger.log_data_transformation(
                    source="CCEE",
                    raw_data=raw_data[:3],
                    processed_data=[p.to_dict() for p in pld_objects[:3]],
                    transformation="raw_json_to_pld_objects"
                )
            
            # 3. Calcula estatísticas
            stats = self._calculate_statistics(pld_objects)
            
            # 4. Valida anomalias
            self._validate_pld_anomalies(pld_objects, stats)
            
            # 5. Cria timeseries
            timeseries = self._create_timeseries(pld_objects, days)
            
            # Prepara resultado
            result = {
                "metadata": metadata,
                "data": [pld.to_dict() for pld in pld_objects],
                "statistics": stats,
                "timeseries": timeseries,
                "raw_data_sample": raw_data[:2]  # Para debug
            }
            
            # Atualiza metadados
            result["metadata"].status = "success"
            result["metadata"].records_processed = len(pld_objects)
            
            # Auditoria: Log da consolidação
            if self.enable_audit:
                self.audit_logger.log_consolidation(
                    sources=["CCEE_API"],
                    consolidated_data=result,
                    rules_applied=["date_parsing", "pld_calculation", "submarket_grouping", "timeseries_generation"]
                )
            
            # Atualiza cache
            self._cache[cache_key] = result
            self._cache_time[cache_key] = datetime.now()
            
            logger.info(f"CCEE: Processados {len(pld_objects)} registros PLD")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no coletor CCEE: {e}")
            metadata.status = "error"
            metadata.error_message = str(e)
            return {"metadata": metadata.to_dict(), "data": []}

    def collect_additional_datasets(self, limit: int = 5) -> Dict[str, Dict]:
        """Coleta datasets adicionais validados nos testes CCEE."""
        datasets = {}
        for name, resource_id in self._additional_datasets.items():
            datasets[name] = self._fetch_dataset(resource_id, limit=limit)
        return datasets


    def collect_pld_historical(self) -> Dict[str, Any]:
        """Coleta PLD histórico anual por resource_id CKAN dedicado."""
        datasets: List[Dict[str, Any]] = []
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            for year, resource_id in self.PLD_HISTORICAL_RESOURCES.items():
                logger.info(f"CCEE | Coletando PLD histórico {year}")
                records = self._fetch_open_dataset(
                    dataset_name=f"pld_{year}",
                    resource_id=resource_id,
                    limit=32000,
                )
                if not records:
                    continue

                df = pd.DataFrame(records)
                out_path = os.path.join("data", f"ccee_pld_{year}_{now_str}.csv")
                try:
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(out_path, index=False)
                    logger.info(f"CCEE: PLD anual salvo para {year}: {out_path} ({len(df)} linhas)")
                    datasets.append({"year": int(year), "file": out_path})
                except Exception as e:
                    logger.warning(f"CCEE: Falha ao salvar CSV anual {year}: {e}")
                    datasets.append({"year": int(year), "records": df.to_dict(orient="records")})

            if datasets:
                return {
                    "metadata": {
                        "source": "CCEE_PLD",
                        "status": "success",
                        "datasets_collected": len(datasets),
                        "collection_time": datetime.now().isoformat(),
                    },
                    "datasets": sorted(datasets, key=lambda d: int(d.get("year", 0))),
                }

            # fallback local
            files = sorted(glob.glob(os.path.join("data", "ccee_pld_*.csv")))
            for f in files:
                base = os.path.basename(f)
                parts = base.split("_")
                year = None
                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                        break
                if year is not None:
                    datasets.append({"year": year, "file": f})

            status = "success" if datasets else "error"
            return {
                "metadata": {
                    "source": "CCEE_PLD",
                    "status": status,
                    "datasets_collected": len(datasets),
                    "collection_time": datetime.now().isoformat(),
                },
                "datasets": sorted(datasets, key=lambda d: int(d.get("year", 0))),
            }
        except Exception as e:
            logger.error(f"Erro na coleta PLD histórico: {e}", exc_info=True)
            return {
                "metadata": {
                    "source": "CCEE_PLD",
                    "status": "error",
                    "error_message": str(e),
                    "collection_time": datetime.now().isoformat(),
                },
                "datasets": [],
            }

    def collect_open_data_csv(self, limit: int = 500) -> Dict[str, Dict[str, Any]]:
        """Coleta datasets adicionais via links CSV (open data)."""
        datasets = {}
        for name, resource_id in self._additional_datasets.items():
            datasets[name] = self._fetch_dataset_csv(resource_id, limit=limit)
        return datasets

    def _fetch_dataset(self, resource_id: str, limit: int = 5) -> Dict:
        """Consulta um dataset específico da CCEE."""
        params = {"resource_id": resource_id, "limit": limit}
        try:
            response = requests.get(f"{self.base_url}/datastore_search", params=params, timeout=30)
            response.raise_for_status()
        except Exception as exc:
            logger.error(f"CCEE: Erro ao buscar dataset {resource_id}: {exc}")
            return {"success": False, "error": str(exc), "records": []}

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error(f"CCEE: JSON inválido para {resource_id}: {exc}")
            return {"success": False, "error": str(exc), "records": []}

        result = payload.get("result", {})
        return {
            "success": payload.get("success", False),
            "records": result.get("records", []),
            "total": result.get("total"),
        }

    def _fetch_dataset_csv(self, resource_id: str, limit: int = 500) -> Dict[str, Any]:
        """Busca o link CSV do dataset via resource_show e retorna amostra."""
        try:
            response = requests.get(self._resource_show_url, params={"id": resource_id}, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error(f"CCEE: Erro ao buscar resource_show {resource_id}: {exc}")
            return {"success": False, "error": str(exc), "records": []}

        if not payload.get("success"):
            return {"success": False, "error": "resource_show failed", "records": []}

        resource = payload.get("result", {})
        csv_url = resource.get("url")
        if not csv_url:
            return {"success": False, "error": "CSV url ausente", "records": []}

        try:
            df = pd.read_csv(csv_url, nrows=limit)
        except Exception as exc:
            logger.error(f"CCEE: Erro ao ler CSV {resource_id}: {exc}")
            return {"success": False, "error": str(exc), "records": []}

        return {
            "success": True,
            "records": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "source_url": csv_url,
            "sample_size": len(df),
        }
    
    def _fetch_pld_data(self, days: int) -> List[Dict]:
        """Busca PLD do resource_id corrente usando requests + paginação CKAN."""
        try:
            target_records = max(40000, int(days) * 24 * 4)
        except Exception:
            target_records = 40000

        records = self._fetch_open_dataset(
            dataset_name="pld_current",
            resource_id=self.resource_id,
            limit=32000,
        )
        if len(records) > target_records:
            records = records[:target_records]

        logger.info(f"CCEE: Total bruto coletado via CKAN/requests: {len(records)} registros (target={target_records})")
        return records

    def _fetch_open_dataset(
        self,
        dataset_name: str,
        resource_id: str,
        limit: int = 32000,
    ) -> List[Dict[str, Any]]:
        """Consulta dataset CKAN com paginação por offset."""
        all_records: List[Dict[str, Any]] = []
        offset = 0

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        }

        while True:
            params = {
                "resource_id": resource_id,
                "limit": limit,
                "offset": offset,
            }
            try:
                response = requests.get(
                    self.CKAN_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as e:
                logger.error(f"CCEE: Erro na consulta datastore_search ({dataset_name}): {e}")
                break

            if self.enable_audit and offset == 0:
                self.audit_logger.log_api_call(
                    source="CCEE_PLD",
                    url=self.CKAN_BASE_URL,
                    params=params,
                    response_status=response.status_code,
                    data_sample=(payload.get("result", {}).get("records", [])[:2] if isinstance(payload, dict) else []),
                )

            if not payload.get("success"):
                logger.warning(f"CCEE: dataset {dataset_name} retornou success=False")
                break

            result = payload.get("result", {})
            records = result.get("records", [])
            if not records:
                break

            all_records.extend(records)
            logger.info(f"CCEE | {dataset_name} | +{len(records)} registros (total: {len(all_records)})")

            if len(records) < limit:
                break
            offset += limit

        return all_records

    def _create_pld_objects(self, raw_data: List[Dict]) -> List[PLDData]:
        """Converte dados brutos para objetos PLD com validação"""
        pld_objects = []
        validation_errors = []
        
        for i, record in enumerate(raw_data):
            try:
                # Obtém data do formato MES_REFERENCIA (AAAAMM) e DIA
                mes_ref = record.get("MES_REFERENCIA", "")
                dia = record.get("DIA", "")
                
                # Valida campos obrigatórios
                if not mes_ref or not dia:
                    validation_errors.append({
                        'index': i,
                        'erro': 'Campos MES_REFERENCIA ou DIA vazios',
                        'dados': {k: v for k, v in record.items() if k in ['MES_REFERENCIA', 'DIA', 'HORA', 'PLD_HORA']}
                    })
                    continue
                
                # Constrói data
                data_str = self._build_date_string(mes_ref, dia)
                
                # Obtém hora
                hora = record.get("HORA", 0)
                try:
                    hora_int = int(hora)
                    hora_str = f"{hora_int:04d}"
                except:
                    hora_str = "0000"
                
                # Converte hora (HHMM -> HH:MM)
                if len(hora_str) == 4:
                    hora_formatada = f"{hora_str[:2]}:{hora_str[2:]}"
                else:
                    hora_formatada = "00:00"
                
                # Converte PLD para float
                pld_raw = record.get("PLD_HORA", 0)
                try:
                    pld_valor = float(pld_raw)
                except:
                    pld_valor = 0.0
                
                # Valida PLD (normalmente entre 0-1000 R$/MWh)
                if pld_valor < 0 or pld_valor > 5000:
                    validation_errors.append({
                        'index': i,
                        'erro': f'PLD fora do range aceitável: {pld_valor}',
                        'dados': record
                    })
                    continue
                
                pld_obj = PLDData(
                    data=data_str,
                    hora=hora_formatada,
                    submercado=record.get("SUBMERCADO", "N/A"),
                    pld_valor=pld_valor,
                    periodo_comercializacao=str(record.get("PERIODO_COMERCIALIZACAO", "N/A"))
                )
                pld_objects.append(pld_obj)
                
            except (ValueError, TypeError, KeyError) as e:
                validation_errors.append({
                    'index': i,
                    'erro': str(e),
                    'dados': {k: v for k, v in record.items() if k in ['MES_REFERENCIA', 'DIA', 'HORA', 'PLD_HORA']}
                })
                continue
        
        # Log de validações
        if validation_errors:
            logger.warning(f"CCEE: {len(validation_errors)} erros de validação")
            for error in validation_errors[:3]:
                logger.debug(f"  Erro: {error}")
        
        logger.info(f"CCEE: {len(pld_objects)} registros processados com sucesso")
        return pld_objects
    
    def _build_date_string(self, mes_ref: str, dia: str) -> str:
        """Constrói string de data no formato YYYY-MM-DD com validação"""
        try:
            # Mes_ref deve ser AAAAMM (ex: 202602)
            if len(mes_ref) != 6 or not mes_ref.isdigit():
                logger.warning(f"CCEE: MES_REFERENCIA inválido: {mes_ref}")
                return datetime.now().strftime("%Y-%m-%d")
            
            ano = mes_ref[:4]
            mes = mes_ref[4:6]
            
            # Valida ano (deve ser entre 2020-2030)
            ano_int = int(ano)
            if ano_int < 2020 or ano_int > 2030:
                logger.warning(f"CCEE: Ano inválido: {ano}")
                return datetime.now().strftime("%Y-%m-%d")
            
            # Valida mês (1-12)
            mes_int = int(mes)
            if mes_int < 1 or mes_int > 12:
                logger.warning(f"CCEE: Mês inválido: {mes}")
                return datetime.now().strftime("%Y-%m-%d")
            
            # Dia pode vir como string (ex: "6")
            dia_str = str(dia).strip()
            if not dia_str.isdigit():
                logger.warning(f"CCEE: Dia não numérico: {dia}")
                dia_str = "1"
            
            dia_int = int(dia_str)
            if dia_int < 1 or dia_int > 31:
                logger.warning(f"CCEE: Dia inválido: {dia}")
                dia_str = "1"
            
            # Garante que dia tenha 2 dígitos
            dia_fmt = dia_str.zfill(2)
            
            return f"{ano}-{mes}-{dia_fmt}"
            
        except Exception as e:
            logger.warning(f"CCEE: Erro ao construir data: {e}")
            return datetime.now().strftime("%Y-%m-%d")
    
    def _calculate_statistics(self, pld_objects: List[PLDData]) -> Dict:
        """Calcula estatísticas dos dados PLD com detalhes"""
        if not pld_objects:
            return {}
        
        values = [pld.pld_valor for pld in pld_objects if pld.pld_valor > 0]
        
        if not values:
            return {}
        
        df = pd.Series(values)
        
        # Estatísticas detalhadas
        stats = {
            "geral": {
                "pld_medio": float(df.mean()),
                "pld_min": float(df.min()),
                "pld_max": float(df.max()),
                "pld_std": float(df.std()),
                "pld_mediana": float(df.median()),
                "pld_q1": float(df.quantile(0.25)),
                "pld_q3": float(df.quantile(0.75)),
                "quantidade": len(values),
                "distribuicao": {
                    "abaixo_100": len([v for v in values if v < 100]),
                    "entre_100_200": len([v for v in values if 100 <= v < 200]),
                    "entre_200_300": len([v for v in values if 200 <= v < 300]),
                    "acima_300": len([v for v in values if v >= 300])
                }
            }
        }
        
        # Por submercado
        submercados = {}
        for pld in pld_objects:
            sub = pld.submercado
            if sub not in submercados:
                submercados[sub] = []
            submercados[sub].append(pld.pld_valor)
        
        stats["por_submercado"] = {}
        for sub, valores in submercados.items():
            if valores:
                s = pd.Series(valores)
                stats["por_submercado"][sub] = {
                    "pld_medio": float(s.mean()),
                    "pld_min": float(s.min()),
                    "pld_max": float(s.max()),
                    "pld_std": float(s.std()),
                    "quantidade": len(valores),
                    "percentual_total": (len(valores) / len(values)) * 100
                }
        
        # Log das estatísticas
        logger.info(f"CCEE Estatísticas: PLD médio = R$ {stats['geral']['pld_medio']:.2f}/MWh")
        logger.info(f"  Range: R$ {stats['geral']['pld_min']:.2f} - R$ {stats['geral']['pld_max']:.2f}")
        
        return stats
    
    def _validate_pld_anomalies(self, pld_objects: List[PLDData], stats: Dict):
        """Valida anomalias nos dados PLD"""
        if not self.enable_audit:
            return
        
        pld_medio = stats.get('geral', {}).get('pld_medio', 0)
        
        # Alerta para PLD muito alto (> 400 R$/MWh)
        if pld_medio > 400:
            self.audit_logger.log_anomaly(
                source="CCEE",
                data_point="pld_medio_sistema",
                expected=(100, 300),  # Range esperado
                actual=pld_medio,
                severity="HIGH"
            )
        
        # Verifica valores extremos (> 1000 R$/MWh)
        extreme_values = [
            p for p in pld_objects 
            if p.pld_valor > 1000
        ]
        
        for pld in extreme_values[:3]:
            self.audit_logger.log_anomaly(
                source="CCEE",
                data_point=f"pld_extremo_{pld.submercado}",
                expected=(0, 500),
                actual=pld.pld_valor,
                severity="MEDIUM"
            )
    
    def _create_timeseries(self, pld_objects: List[PLDData], days: int) -> List[Dict]:
        """Cria série temporal para gráficos"""
        if not pld_objects:
            return []
        
        # Agrupa por data
        data_map = {}
        for pld in pld_objects:
            if pld.data not in data_map:
                data_map[pld.data] = []
            data_map[pld.data].append(pld.pld_valor)
        
        # Calcula estatísticas por dia
        timeseries = []
        for data_str, valores in sorted(data_map.items()):
            if valores:
                s = pd.Series(valores)
                timeseries.append({
                    "data": data_str,
                    "pld_medio": float(s.mean()),
                    "pld_min": float(s.min()),
                    "pld_max": float(s.max()),
                    "pld_std": float(s.std()),
                    "quantidade": len(valores)
                })
        
        # Filtra últimos N dias
        if days and timeseries:
            try:
                # Converte datas para ordenar
                timeseries_with_dates = []
                for item in timeseries:
                    try:
                        date_obj = datetime.strptime(item["data"], "%Y-%m-%d").date()
                        timeseries_with_dates.append((date_obj, item))
                    except:
                        continue
                
                # Ordena por data e pega os últimos N dias
                timeseries_with_dates.sort(key=lambda x: x[0])
                recent_items = timeseries_with_dates[-days:]
                return [item[1] for item in recent_items]
            except Exception as e:
                logger.warning(f"Erro ao filtrar timeseries: {e}")
        
        return timeseries[-days:] if days else timeseries
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica se cache ainda é válido"""
        if cache_key not in self._cache or cache_key not in self._cache_time:
            return False
        
        cache_age = datetime.now() - self._cache_time[cache_key]
        return cache_age.total_seconds() < (self.cache_ttl * 60)
    
    def get_detailed_report(self) -> Dict:
        """Gera relatório detalhado da coleta"""
        data = self.collect_pld_data()
        
        if data['metadata'].status != 'success':
            return {"error": "Coleta não foi bem-sucedida"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "resumo": {
                "total_registros": data['statistics']['geral']['quantidade'],
                "pld_medio": data['statistics']['geral']['pld_medio'],
                "pld_min": data['statistics']['geral']['pld_min'],
                "pld_max": data['statistics']['geral']['pld_max'],
                "volatilidade": data['statistics']['geral']['pld_std']
            },
            "submercados": {},
            "distribuicao_precos": data['statistics']['geral']['distribuicao']
        }
        
        # Detalhes por submercado
        for subm, stats in data['statistics']['por_submercado'].items():
            report["submercados"][subm] = {
                "pld_medio": stats.get("pld_medio", 0),
                "registros": stats.get("quantidade", 0),
                "percentual": stats.get("percentual_total", 0)
            }
        
        # Timeseries resumida
        if data['timeseries']:
            report["evolucao_recente"] = data['timeseries'][-3:]  # Últimos 3 dias
        
        return report