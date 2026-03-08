# scripts/audit_logger.py
import json
import logging
from datetime import datetime
import os
from typing import Dict, Any, Optional
import hashlib

class AuditLogger:
    """Sistema completo de auditoria de dados"""
    
    def __init__(self, audit_dir: str = "audit_logs"):
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)
        
        # Configura logging detalhado
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.DEBUG)
        
        # Handler para arquivo
        log_file = os.path.join(self.audit_dir, f"audit_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Formato detalhado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_api_call(self, source: str, url: str, params: Dict, 
                    response_status: int, data_sample: Any):
        """Registra chamada de API"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'url': url,
            'params': params,
            'response_status': response_status,
            'data_sample': data_sample[:3] if isinstance(data_sample, list) else data_sample,
            'data_hash': self._generate_hash(data_sample)
        }
        
        self.logger.info(f"API CALL: {source} - Status: {response_status}")
        self._save_audit_entry(f"api_{source}", log_entry)
    
    def log_data_transformation(self, source: str, raw_data: Any, 
                               processed_data: Any, transformation: str):
        """Registra transformação de dados"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'transformation': transformation,
            'raw_data_sample': self._get_sample(raw_data),
            'processed_data_sample': self._get_sample(processed_data),
            'raw_hash': self._generate_hash(raw_data),
            'processed_hash': self._generate_hash(processed_data)
        }
        
        self.logger.info(f"TRANSFORMATION: {source} - {transformation}")
        self._save_audit_entry(f"transform_{source}", log_entry)
    
    def log_anomaly(self, source: str, data_point: Any, expected: Any, 
                   actual: Any, severity: str = "WARNING"):
        """Registra anomalia detectada"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'anomaly_type': 'data_validation',
            'severity': severity,
            'data_point': data_point,
            'expected_range': expected,
            'actual_value': actual,
            'deviation_percentage': self._calculate_deviation(expected, actual)
        }
        
        self.logger.warning(f"ANOMALY: {source} - {data_point} = {actual} (expected: {expected})")
        self._save_audit_entry(f"anomaly_{source}", log_entry)
    
    def log_consolidation(self, sources: list, consolidated_data: Any, 
                         rules_applied: list):
        """Registra consolidação de dados"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'consolidation',
            'sources': sources,
            'rules_applied': rules_applied,
            'consolidated_sample': self._get_sample(consolidated_data),
            'consolidated_hash': self._generate_hash(consolidated_data)
        }
        
        self.logger.info(f"CONSOLIDATION: {len(sources)} sources")
        self._save_audit_entry("consolidation", log_entry)
    
    def save_raw_data(self, source: str, raw_data: Any, metadata: Dict = None):
        """Salva dados brutos completos para auditoria"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_{source}_{timestamp}.json"
        filepath = os.path.join(self.audit_dir, filename)
        
        save_data = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'data': raw_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"RAW DATA SAVED: {filename}")
        return filepath
    
    def _save_audit_entry(self, category: str, entry: Dict):
        """Salva entrada de auditoria"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{category}_{timestamp}.json"
        filepath = os.path.join(self.audit_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_hash(self, data: Any) -> str:
        """Gera hash para verificação de integridade"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _get_sample(self, data: Any, max_items: int = 3) -> Any:
        """Obtém amostra dos dados"""
        if isinstance(data, list):
            return data[:max_items]
        elif isinstance(data, dict):
            return {k: self._get_sample(v, max_items) for k, v in list(data.items())[:max_items]}
        else:
            return data
    
    def _calculate_deviation(self, expected: Any, actual: Any) -> Optional[float]:
        """Calcula porcentagem de desvio"""
        try:
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                # Esperado é um range (min, max)
                expected_min, expected_max = expected
                expected_mid = (expected_min + expected_max) / 2
                return abs((actual - expected_mid) / expected_mid * 100)
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs((actual - expected) / expected * 100)
        except:
            return None