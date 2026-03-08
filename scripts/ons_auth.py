# scripts/ons_auth.py - ATUALIZADO
import requests
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ONSAuthenticator:
    """Gerencia autenticação com a API do ONS - VERSÃO CORRIGIDA"""
    
    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password
        self.token = None
        self.token_expiry = None
        self.refresh_token = None
        self.token_type = None
    
    def authenticate(self):
        """Autentica no ONS usando as credenciais fornecidas"""
        if not self.username or not self.password:
            logger.warning("⚠️ Credenciais ONS não configuradas")
            return False
        
        url = "https://integra.ons.org.br/api/autenticar"
        
        payload = {
            "usuario": self.username,
            "senha": self.password
        }
        
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        try:
            logger.info("🔐 Autenticando no ONS...")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # CORREÇÃO: A resposta tem campos diferentes
            self.token = data.get("access_token")
            self.token_type = data.get("token_type", "bearer")
            self.refresh_token = data.get("refresh_token")
            
            if self.token:
                expires_in = data.get("expires_in", 1199)  # 1199 segundos = ~20 minutos
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                
                logger.info(f"✅ Autenticação bem-sucedida")
                logger.info(f"   Token válido até: {self.token_expiry.strftime('%H:%M:%S')}")
                logger.info(f"   Token type: {self.token_type}")
                logger.debug(f"   Token (primeiros 20 chars): {self.token[:20]}...")
                
                return True
            else:
                logger.error("❌ Token não encontrado na resposta")
                logger.debug(f"Resposta completa: {data}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de conexão na autenticação: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro na autenticação: {str(e)}")
            return False
    
    def get_auth_headers(self):
        """Retorna headers de autenticação com token atualizado"""
        # Verifica se o token está válido
        if not self.token or not self.token_expiry or datetime.now() >= self.token_expiry:
            logger.info("🔄 Token expirado ou inválido, reautenticando...")
            if not self.authenticate():
                logger.error("❌ Falha na reautenticação")
                return {}
        
        headers = {
            "Authorization": f"{self.token_type.capitalize()} {self.token}",
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        return headers
    
    def get_auth_headers_with_pagination(self, pagina=1, quantidade=240):
        """Headers com paginação para endpoints específicos"""
        headers = self.get_auth_headers()
        
        if not headers:
            return {}
        
        # CORREÇÃO: Adiciona headers de paginação corretamente
        headers["Pagina"] = str(pagina)
        headers["Quantidade"] = str(quantidade)
        
        return headers
    
    def is_authenticated(self):
        """Verifica se está autenticado"""
        if not self.token or not self.token_expiry:
            return False
        
        # Adiciona margem de segurança de 60 segundos
        return datetime.now() < (self.token_expiry - timedelta(seconds=60))
    
    def get_token_info(self):
        """Retorna informações sobre o token atual"""
        return {
            "tem_token": bool(self.token),
            "token_valido": self.is_authenticated(),
            "expira_em": self.token_expiry.isoformat() if self.token_expiry else None,
            "tempo_restante": (self.token_expiry - datetime.now()).total_seconds() if self.token_expiry and self.token_expiry > datetime.now() else 0
        }