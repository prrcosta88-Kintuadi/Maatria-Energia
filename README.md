# Kintuadi Energy v2

Plataforma de coleta, integração e análise de dados do setor elétrico brasileiro
(ONS, CCEE e fontes correlatas).

## Visão geral
O projeto integra dados públicos do ONS e da CCEE para gerar análises técnicas
sobre fundamentos físicos do SIN, formação de preços (PLD) e resultados do MCP.
A solução está estruturada em dois ambientes complementares:

### CORE (Genérico/Público)
- Baseado exclusivamente em dados ONS + CCEE.
- Produz análises sistêmicas do SIN (sem dados de usuários).
- Fornece KPIs, séries temporais e indicadores explicáveis.

### PREMIUM (Personalizado)
- Importa dados do usuário via template Excel padronizado.
- Normaliza dados na mesma base temporal/submercado do CORE.
- Calcula exposição energética e financeira, cenários e indicadores de hedge.

## Estrutura do projeto
```
Kintuadi-Energy-v2/
├── data/                     # Dados coletados e outputs
├── logs/                     # Logs de execução
├── scripts/
│   ├── data_models.py         # Modelos de dados
│   ├── ons_collector_v2.py    # Coletor ONS
│   ├── ccee_collector_v2.py   # Coletor CCEE (PLD)
│   ├── analyzer_v2.py         # Análise integrada básica
│   ├── core_analysis.py       # Análises CORE explicáveis
│   ├── premium_module.py      # Importação/normalização Premium
│   ├── integrated_collector_v2.py
│   └── utils.py
├── dashboard_integrado.py     # Dashboard integrado CORE + PREMIUM
├── run_collector.py           # Script de execução
├── requirements.txt           # Dependências
├── test_ons_api.py            # Testes ONS
├── test_ccee_api.py           # Testes CCEE
└── README.md
```

## Fluxo de dados (CORE vs PREMIUM)
1. **Coleta**
   - ONS: reservatórios e operações (coletores ONS).
   - CCEE: PLD e datasets abertos selecionados (coletores CCEE).
2. **Normalização**
   - Padronização de datas, horas e submercados.
   - Conversão de tipos e validações básicas.
3. **Análise**
   - CORE: indicadores hidrológicos, operação, preços, MCP, consumo e contratos.
   - PREMIUM: exposições energéticas/financeiras e sensibilidade ao PLD.
4. **Visualização**
   - Dashboard integrado com visão sistêmica (CORE) e personalizada (PREMIUM).

## Escopo CORE implementado
- **Estado hidrológico do SIN**: EAR/ENA (quando disponível), tendência e conforto hídrico.
- **Operação do sistema**: geração por subsistema/modalidade, carga, térmicas e CVU (quando disponíveis).
- **Formação de preços**: PLD horário por submercado e indicadores de coerência preço-fundamento.
- **Liquidação do MCP**: sumários mensais por perfil de agente (quando disponíveis).
- **Consumo mensal ACL vs ACR**: previsto para integração em datasets CCEE.
- **Perdas da rede básica**: previsto para integração em datasets ONS.
- **Contratos agregados por duração**: previsto para integração em datasets CCEE.

> Observação: o CORE é resiliente a dados parciais. Indicadores ainda não coletados
> são sinalizados como indisponíveis, mantendo a estrutura para evolução.

## Escopo PREMIUM (modular)
- Importa dados do usuário (Excel).
- Normaliza dados para a base CORE (hora, submercado).
- Calcula:
  - Exposição energética horária e mensal.
  - Exposição financeira ao PLD.
  - Sensibilidade a cenários de preço.
  - Indicadores de overhedge/underhedge.

## Template Excel (PREMIUM)
O template possui uma aba chamada **dados_usuario** com as colunas:

| coluna           | descrição |
|------------------|-----------|
| data             | Data no formato YYYY-MM-DD |
| hora             | Hora (0-23) |
| submercado       | N, NE, SE, S (ou Norte, Nordeste, Sudeste, Sul) |
| consumo_mwh      | Consumo do usuário em MWh |
| geracao_mwh      | Geração própria em MWh |
| contratos_mwh    | Contratos alocados em MWh |
| preco_contrato   | Preço do contrato (R$/MWh) |

Para gerar um template automaticamente:
```bash
python -c "from scripts.premium_module import generate_premium_template; generate_premium_template('premium_template.xlsx')"
```

## Execução
### Coleta de dados
```bash
python run_collector.py
```

### Dashboard
```bash
streamlit run dashboard_integrado.py
```

### Testes das APIs
```bash
python test_ons_api.py
python test_ccee_api.py
```

## Requisitos
- Python 3.10+
- Dependências listadas em `requirements.txt`

## Diretrizes
- Não remover funcionalidades existentes.
- Priorizar refatoração incremental.
- Manter separação entre coleta, normalização, análise e visualização.
