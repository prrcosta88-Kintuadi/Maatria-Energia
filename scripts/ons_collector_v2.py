from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging
import csv
import io
import os
import requests
from dateutil.relativedelta import relativedelta


# Import defensivo
try:
    from .utils import save_records_to_csv, save_raw_csv_file
except ImportError:
    from scripts.utils import save_records_to_csv, save_raw_csv_file


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3000
OPEN_DATA_TIMEOUT = 6000

ENERGIA_AGORA_BASE_URL = "https://integra.ons.org.br/api/energiaagora/Get"
EXPECTED_API_DISABLE_MESSAGE = "API desabilitada"


# ======================================================================
# API Energia Agora – configuração
# ======================================================================

ENERGIA_AGORA_ENDPOINTS = [
    # SIN
    "Geracao_SIN_Eolica_json",
    "Geracao_SIN_Hidraulica_json",
    "Geracao_SIN_Nuclear_json",
    "Geracao_SIN_Solar_json",
    "Geracao_SIN_Termica_json",
    # Submercados
    "Geracao_Norte_Eolica_json",
    "Geracao_Norte_Hidraulica_json",
    "Geracao_Norte_Nuclear_json",
    "Geracao_Norte_Solar_json",
    "Geracao_Norte_Termica_json",
    "Geracao_Nordeste_Eolica_json",
    "Geracao_Nordeste_Hidraulica_json",
    "Geracao_Nordeste_Nuclear_json",
    "Geracao_Nordeste_Solar_json",
    "Geracao_Nordeste_Termica_json",
    "Geracao_Sudeste_Eolica_json",
    "Geracao_Sudeste_Hidraulica_json",
    "Geracao_Sudeste_Nuclear_json",
    "Geracao_Sudeste_Solar_json",
    "Geracao_Sudeste_Termica_json",
    "Geracao_Sul_Eolica_json",
    "Geracao_Sul_Hidraulica_json",
    "Geracao_Sul_Nuclear_json",
    "Geracao_Sul_Solar_json",
    "Geracao_Sul_Termica_json",
    "Geracao_SudesteECentroOeste_Eolica_json",
    "Geracao_SudesteECentroOeste_Hidraulica_json",
    "Geracao_SudesteECentroOeste_Nuclear_json",
    "Geracao_SudesteECentroOeste_Solar_json",
    "Geracao_SudesteECentroOeste_Termica_json",
]

CARGA_AGORA_ENDPOINTS = [
    "Carga_SIN_json",
    "Carga_Norte_json",
    "Carga_Nordeste_json",
    "Carga_SudesteECentroOeste_json",
    "Carga_Sul_json",
]


# ======================================================================
# Collector
# ======================================================================



class ONSCollectorV2:

    # OpenData ONS prioriza XLSX para evitar inconsistências de separador decimal em CSV.
    OPEN_DATASETS: List[Tuple[str, str]] = [
        ("Reservatorios", "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/reservatorio/RESERVATORIOS.xlsx"),
        ("Capacidade_Instalada", "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/capacidade-geracao/CAPACIDADE_GERACAO.xlsx"),
    ]

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        enable_audit: bool = True,
    ):
        self.username = username
        self.password = password
        self.enable_audit = enable_audit
        self.api_headers: Optional[Dict[str, str]] = None

        # adiciona datasets dinâmicos
        self.OPEN_DATASETS = (
            self.OPEN_DATASETS +
            self._build_yearly_datasets() +
            self._build_dynamic_datasets()
        )


    # ==========================================================
    # DATASETS DINÂMICOS
    # ==========================================================

    def _generate_year_range(self, start_year: int) -> List[int]:
        current_year = datetime.today().year
        return list(range(start_year, current_year + 1))


    def _generate_month_range(self, start: str) -> List[str]:
        start_dt = datetime.strptime(start, "%Y-%m")
        end_dt = datetime.today()

        months = []
        current = start_dt

        while current <= end_dt:
            months.append(current.strftime("%Y-%m"))
            current += relativedelta(months=1)

        return months

    def _build_yearly_datasets(self) -> List[Tuple[str, str]]:

        dynamic = []

        # ============================
        # HIDROLOGIA (2010 → atual)
        # ============================

        hydro_years = self._generate_year_range(2010)

        for year in hydro_years:

            dynamic.append((
                f"EAR_Diario_Reservatorios_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/ear_reservatorio_di/"
                f"EAR_DIARIO_RESERVATORIOS_{year}.xlsx"
            ))

            dynamic.append((
                f"ENA_Diario_Reservatorios_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/ena_reservatorio_di/"
                f"ENA_DIARIO_RESERVATORIOS_{year}.xlsx"
            ))

            dynamic.append((
                f"EAR_Diario_Subsistema_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/ear_subsistema_di/"
                f"EAR_DIARIO_SUBSISTEMA_{year}.xlsx"
            ))

            dynamic.append((
                f"ENA_Diario_Subsistema_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/ena_subsistema_di/"
                f"ENA_DIARIO_SUBSISTEMA_{year}.xlsx"
            ))

        # ============================
        # CVU TÉRMICA (2005 → atual)
        # ============================

        cvu_years = self._generate_year_range(2005)

        for year in cvu_years:

            dynamic.append((
                f"CVU_Usina_Termica_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/cvu_usitermica_se/"
                f"CVU_USINA_TERMICA_{year}.xlsx"
            ))

        # ============================
        # CMO SEMI-HORÁRIO (2020 → atual)
        # ============================
        cmo_years = self._generate_year_range(2020)
        for year in cmo_years:
            dynamic.append((
                f"CMO_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/cmo_tm/"
                f"CMO_SEMIHORARIO_{year}.xlsx"
            ))

        # ============================
        # INTERCÂMBIO NACIONAL HORÁRIO (2000 → atual)
        # ============================
        intercambio_years = self._generate_year_range(2000)
        for year in intercambio_years:
            dynamic.append((
                f"Intercambio_Nacional_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/intercambio_nacional_ho/"
                f"INTERCAMBIO_NACIONAL_{year}.xlsx"
            ))

        # ============================
        # CURVA DE CARGA (2018 → atual) - XLSX
        # ============================
        curva_carga_years = self._generate_year_range(2018)
        for year in curva_carga_years:
            dynamic.append((
                f"Curva_Carga_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/curva-carga-ho/"
                f"CURVA_CARGA_{year}.xlsx"
            ))

        # ============================
        # GERAÇÃO POR USINA HORÁRIA (2018-2021)
        # ============================
        for year in range(2018, 2022):
            dynamic.append((
                f"Geracao_Usina_Horaria_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/geracao_usina_2_ho/"
                f"GERACAO_USINA-2_{year}.xlsx"
            ))

        # ============================
        # DESPACHO GFOM ANUAL (2013-2021)
        # ============================
        for year in range(2013, 2022):
            dynamic.append((
                f"Despacho_GFOM_{year}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/geracao_termica_despacho_2_ho/"
                f"GERACAO_TERMICA_DESPACHO_{year}.xlsx"
            ))

        return dynamic

    
    def _build_dynamic_datasets(self) -> List[Tuple[str, str]]:

        dynamic = []

        months = self._generate_month_range("2021-10")
        months_disponibilidade = self._generate_month_range("2015-01")

        for m in months:
            year, month = m.split("-")

            # GFOM mensal
            dynamic.append((
                f"Despacho_GFOM_{m}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/geracao_termica_despacho_2_ho/"
                f"GERACAO_TERMICA_DESPACHO-2_{year}_{month}.xlsx"
            ))

            # Restrição FV (disponível desde 2024-04)
            if m >= "2024-04":
                dynamic.append((
                    f"Restricao_fotovoltaica_{m}",
                    f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                    f"dataset/restricao_coff_fotovoltaica_tm/"
                    f"RESTRICAO_COFF_FOTOVOLTAICA_{year}_{month}.xlsx"
                ))

            # Restrição Eólica
            dynamic.append((
                f"Restricao_eolica_{m}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/restricao_coff_eolica_tm/"
                f"RESTRICAO_COFF_EOLICA_{year}_{month}.xlsx"
            ))

        # Disponibilidade (histórico desde 2015-01)
        for m in months_disponibilidade:
            year, month = m.split("-")
            dynamic.append((
                f"Disponibilidade_Usina_{m}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/disponibilidade_usina_ho/"
                f"DISPONIBILIDADE_USINA_{year}_{month}.xlsx"
            ))

        # Geração por usina (mensal: 2022-01 → atual)
        months_geracao_usina = self._generate_month_range("2022-01")
        for m in months_geracao_usina:
            year, month = m.split("-")
            dynamic.append((
                f"Geracao_Usina_Horaria_{m}",
                f"https://ons-aws-prod-opendata.s3.amazonaws.com/"
                f"dataset/geracao_usina_2_ho/"
                f"GERACAO_USINA-2_{year}_{month}.xlsx"
            ))

        return dynamic


    # ==========================================================
    # COLETA OPEN DATA
    # ==========================================================

    def _should_download(self, dataset_name: str) -> bool:
        """
        Verifica se o arquivo já existe localmente.
        Retorna True se PRECISA baixar.
        """

        try:
            year = dataset_name.split("_")[-1].split("-")[0]
        except Exception:
            return True

        path = os.path.join("data", "ons", year)
        os.makedirs(path, exist_ok=True)

        csv_path = os.path.join(path, f"{dataset_name}.csv")
        xlsx_path = os.path.join(path, f"{dataset_name}.xlsx")
        return not (os.path.exists(csv_path) or os.path.exists(xlsx_path))

    
    def collect_open_data(self) -> Dict[str, Any]:

        datasets = []

        for name, url in self.OPEN_DATASETS:
            try:
                logger.info(f"ONS | OpenData | {name}")
                path, rows = self._fetch_and_save_open_data(url, name)
                if not path:
                    continue

                datasets.append({
                    "dataset": name,
                    "type": "xlsx" if str(path).lower().endswith(".xlsx") else "csv",
                    "records": rows,
                    "file": path,
                    "origin": "open_data",
                })
            except Exception as e:
                logger.warning(f"Falha {name}: {e}")

        datasets.extend(self._collect_api_series())

        return {
            "metadata": {
                "source": "ONS",
                "datasets_collected": len(datasets),
                "collection_time": datetime.now().isoformat(),
            },
            "datasets": datasets,
        }


    # ==================================================================
    # Open Data helpers
    # ==================================================================


    
    def _fetch_and_save_open_data(self, url: str, dataset_name: str) -> Tuple[Optional[str], int]:

        try:
            year = dataset_name.split("_")[-1].split("-")[0]
        except Exception:
            year = "misc"

        path = os.path.join("data", "ons", year)
        os.makedirs(path, exist_ok=True)

        ext = ".xlsx" if url.lower().endswith(".xlsx") else ".csv"
        file_path = os.path.join(path, f"{dataset_name}{ext}")

        # -----------------------------
        # Se já existe localmente
        # -----------------------------
        if os.path.exists(file_path):
            logger.info(f"{dataset_name} já existe. Pulando download.")
            return file_path, 0

        # -----------------------------
        # Download
        # -----------------------------
        try:
            response = requests.get(url, timeout=OPEN_DATA_TIMEOUT)

            if response.status_code == 404:
                alt_url = None
                if url.lower().endswith('.xlsx'):
                    alt_url = url[:-5] + '.csv'
                elif url.lower().endswith('.csv'):
                    alt_url = url[:-4] + '.xlsx'
                if alt_url:
                    try:
                        alt_resp = requests.get(alt_url, timeout=OPEN_DATA_TIMEOUT)
                        if alt_resp.status_code == 200:
                            response = alt_resp
                            url = alt_url
                            logger.info(f"{dataset_name} fallback para {alt_url}")
                        else:
                            logger.info(f"{dataset_name} não disponível (404).")
                            return None, 0
                    except requests.RequestException:
                        logger.info(f"{dataset_name} não disponível (404).")
                        return None, 0
                else:
                    logger.info(f"{dataset_name} não disponível (404).")
                    return None, 0

            if response.status_code == 403:
                logger.warning(f"{dataset_name} acesso negado (403).")
                return None, 0

            response.raise_for_status()

        except requests.RequestException as e:
            logger.warning(f"Erro ao baixar {dataset_name}: {e}")
            return None, 0

        # -----------------------------
        # Salvamento
        # -----------------------------
        with open(file_path, "wb") as f:
            f.write(response.content)

        final_ext = ".xlsx" if (url.lower().endswith(".xlsx")) else ".csv"
        if final_ext != ext:
            # ajusta nome salvo caso fallback troque extensão
            file_path = os.path.join(path, f"{dataset_name}{final_ext}")
            with open(file_path, "wb") as f:
                f.write(response.content)
        rows = self._count_csv_rows(response.content) if final_ext == ".csv" else 0

        return file_path, rows

    def _count_csv_rows(self, content: bytes) -> int:
        text_stream = io.StringIO(content.decode("utf-8", errors="ignore"))
        reader = csv.reader(text_stream)
        return max(sum(1 for _ in reader) - 1, 0)

    # ==================================================================
    # API Energia Agora helpers
    # ==================================================================

    def _authenticate(self) -> bool:
        if not self.username or not self.password:
            logger.info("ONS | API Energia Agora não configurada (sem credenciais)")
            return False

        try:
            resp = requests.post(
                f"{ENERGIA_AGORA_BASE_URL}/autenticar",
                json={"usuario": self.username, "senha": self.password},
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            token = resp.json().get("access_token")
            token_type = resp.json().get("token_type", "bearer")

            self.api_headers = {
                "Authorization": f"{token_type.capitalize()} {token}",
                "accept": "application/json",
            }
            logger.info("ONS | Autenticação API bem-sucedida")
            return True

        except Exception as e:
            logger.warning(f"ONS | Falha na autenticação API: {e}")
            return False

    def _is_api_disabled(self, resp: requests.Response) -> bool:
        """
        Detecta respostas padrão do ONS quando a API Energia Agora está desabilitada.
        """
        try:
            if resp.status_code in (401, 403, 404):
                return True

            text = resp.text.lower()
            return EXPECTED_API_DISABLE_MESSAGE.lower() in text

        except Exception:
            return False

    def _collect_api_series(self) -> List[Dict[str, Any]]:
        collected = []

        headers = {"accept": "application/json"}

        for endpoint in ENERGIA_AGORA_ENDPOINTS + CARGA_AGORA_ENDPOINTS:
            url = f"{ENERGIA_AGORA_BASE_URL}/{endpoint}"

            try:
                logger.info(f"ONS | Energia Agora | {endpoint}")
                resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)

                if self._is_api_disabled(resp):
                    logger.warning(f"ONS | API Energia Agora desabilitada ({endpoint})")
                    continue

                resp.raise_for_status()
                records = resp.json()

                if not isinstance(records, list) or not records:
                    logger.warning(f"ONS | Energia Agora vazio ({endpoint})")
                    continue

                path = save_records_to_csv(
                    records,
                    dataset_name=f"ons_{endpoint.lower().replace('_json','')}",
                )

                collected.append({
                    "dataset": endpoint.replace("_json", ""),
                    "type": "csv",
                    "records": len(records),
                    "file": path,
                    "origin": "energia_agora",
                })

            except Exception as e:
                logger.warning(f"ONS | Falha Energia Agora {endpoint}: {e}")

        return collected

    