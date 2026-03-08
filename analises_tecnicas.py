"""
Módulo de análises técnicas detalhadas para o dashboard Kintuadi.
ATUALIZADO para compatibilidade com análise térmica revisada (v5 - perspectiva dupla)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def _format_value(value, format_str="", default="—"):
    """Formata valores seguramente, lidando com None."""
    if value is None:
        return default
    try:
        if format_str:
            return format_str.format(value)
        return str(value)
    except (ValueError, TypeError):
        return default

def mostrar_analises_tecnicas(core_data: dict):
    """
    Mostra explicações técnicas detalhadas de todas as análises.
    
    Args:
        core_data: Dicionário com os dados do CORE analysis
    """
    
    # Configuração CSS
    st.markdown("""
    <style>
    .analise-section {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.1), rgba(30, 58, 138, 0.05));
        border-radius: 12px;
        padding: 1.8rem;
        margin: 2rem 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .formula-box {
        background-color: rgba(59, 130, 246, 0.15);
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        font-family: 'Consolas', 'Monaco', monospace;
        color: #93c5fd;
        font-size: 0.95rem;
    }
    
    .indicator-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.12);
        transition: all 0.3s ease;
    }
    
    .indicator-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    .metric-highlight {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60a5fa;
        padding: 0.5rem 0;
    }
    
    .interpretation-box {
        background: rgba(234, 179, 8, 0.1);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Badges para perspectiva dupla */
    .perspectiva-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-sistema {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .badge-gerador {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .card-perspectiva {
        background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.12);
    }
    
    .card-sistema {
        border-left: 4px solid #3b82f6;
    }
    
    .card-gerador {
        border-left: 4px solid #10b981;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Cabeçalho
    st.markdown("# 🔬 Análises Técnicas - Kintuadi Energy Intelligence")
    st.markdown("### Explicação detalhada das métricas, metodologias e algoritmos")
    st.markdown("---")
    
    # Verificar se está usando análise térmica revisada v5
    metadata = core_data.get("metadata", {})
    correcao_conceitual = metadata.get("correcao_conceitual", False)
    
    if correcao_conceitual:
        st.success("✅ **Versão da Análise:**")
    
    # Barra lateral com navegação
    with st.sidebar:
        st.markdown("## 📚 Navegação")
        secoes = {
            "Hidrologia": "💧",
            "Pulso do Sistema": "📈", 
            "Ciclo do SIN": "🌎",
            "PLD e Mercado": "💰",
            "Análise Térmica Revisada": "🔥",
            "MCP Econômico": "📊",
            "Limites Regulatórios": "📏",
            "Formação do Preço": "🔍",
            "Fontes de Dados": "🔗",
            "Glossário": "📖"
        }
        
        secao_selecionada = st.radio(
            "Selecione a seção:",
            list(secoes.keys()),
            format_func=lambda x: f"{secoes[x]} {x}"
        )
    
    # Seção selecionada
    if secao_selecionada == "Hidrologia":
        mostrar_hidrologia(core_data)
    elif secao_selecionada == "Pulso do Sistema":
        mostrar_pulso_sistema(core_data)
    elif secao_selecionada == "Ciclo do SIN":
        mostrar_ciclo_sin(core_data)
    elif secao_selecionada == "PLD e Mercado":
        mostrar_pld_mercado(core_data)
    elif secao_selecionada == "Análise Térmica Revisada":
        mostrar_analise_termica_revisada(core_data)
    elif secao_selecionada == "MCP Econômico":
        mostrar_mcp_economico(core_data)
    elif secao_selecionada == "Limites Regulatórios":
        mostrar_limites_regulatorios(core_data)
    elif secao_selecionada == "Formação do Preço":
        mostrar_formacao_preco(core_data)
    elif secao_selecionada == "Fontes de Dados":
        mostrar_fontes_dados()
    else:
        mostrar_glossario()

def mostrar_hidrologia(core):
    """Seção de análise técnica de hidrologia."""
    
    st.markdown("## 💧 1. Hidrologia - Análise Técnica")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 1.1 Definições Fundamentais
        
        **EAR (Energia Armazenada)**  
        Percentual da capacidade total de armazenamento dos reservatórios do SIN. 
        Representa a **segurança operacional de médio/longo prazo**.
        
        **ENA (Energia Natural Afluente)**  
        Média da energia disponível nas bacias hidrográficas. 
        Indicador de **disponibilidade hídrica atual**.
        """)
        
        # Fórmulas
        st.markdown("### 1.2 Fórmulas de Cálculo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **EAR por Subsistema:**
            """)
            st.markdown("""
            <div class="formula-box">
            EAR_subsistema = (Volume_armazenado / Volume_útil) × 100%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **EAR Médio SIN (Ponderado):**
            """)
            st.markdown("""
            <div class="formula-box">
            EAR_SIN = Σ(EAR_i × Capacidade_i) / Σ(Capacidade_i)
            </div>
            """, unsafe_allow_html=True)
        
        # Classificação
        st.markdown("### 1.3 Matriz de Classificação")
        
        classificacao_df = pd.DataFrame({
            "EAR (%)": ["< 40", "40 - 55", "55 - 70", "70 - 85", "≥ 85"],
            "Classe": ["Crítico", "Alerta", "Atenção", "Confortável", "Abundante"],
            "Risco": ["Alto", "Moderado-Alto", "Moderado", "Baixo", "Muito Baixo"],
            "Ação Recomendada": [
                "Redução de carga crítica",
                "Ativação térmica planejada", 
                "Monitoramento intensivo",
                "Operação normal",
                "Folga operacional"
            ]
        })
        
        st.dataframe(classificacao_df, use_container_width=True, hide_index=True)
        
        # Tendência
        st.markdown("### 1.4 Indicador de Tendência")
        
        st.markdown("""
        <div class="formula-box">
        Tendência = EAR(média 7 dias) - EAR(média 30 dias)
        <br><br>
        Interpretação:
        <br>• Tendência > 0 → Recuperação hídrica
        <br>• Tendência < 0 → Degradação hídrica
        <br>• |Tendência| > 2% → Movimento significativo
        </div>
        """, unsafe_allow_html=True)
        
        # Dados atuais
        st.markdown("### 1.5 Dados Atuais")
        
        hyd = core.get("hydrology", {})
        
        # Obter valores com tratamento de None
        ear_medio = hyd.get('ear_medio')
        ena_media = hyd.get('ena_media')
        tendencia = hyd.get('tendencia')
        classificacao = hyd.get('classificacao', {}).get('classe', 'N/A')
        
        # Formatar valores
        ear_str = _format_value(ear_medio, "{:.1f}%")
        ena_str = _format_value(ena_media, "{:,.0f} MWmed")
        tend_str = _format_value(tendencia, "{:+.1f}%")
        
        # Determinar cores
        ear_color = "green" if ear_medio and ear_medio > 70 else "orange" if ear_medio and ear_medio > 55 else "red"
        tend_color = "green" if tendencia and tendencia > 0 else "red" if tendencia and tendencia < 0 else "gray"
        class_color = (
            "red" if classificacao == "crítico" else
            "orange" if classificacao in ["alerta", "atenção"] else
            "green"
        )
        
        cols = st.columns(4)
        metrics = [
            ("EAR Médio", ear_str, ear_color),
            ("ENA Média", ena_str, "blue"),
            ("Tendência", tend_str, tend_color),
            ("Classe", classificacao.title(), class_color)
        ]
        
        for col, (label, value, color) in zip(cols, metrics):
            with col:
                st.markdown(f'<div class="metric-highlight" style="color:{color}">{label}<br>{value}</div>', 
                           unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_pulso_sistema(core):
    """Seção de análise do Pulso do Sistema - ATUALIZADA para nova análise térmica v5."""
    
    st.markdown("## 📈 2. Pulso do Sistema - Análise Técnica")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 2.1 Conceito
        
        O **Pulso do Sistema** é um indicador composto que integra três dimensões 
        fundamentais do SIN em tempo real:
        
        1. **Dimensão Física (Hidrologia)** - Condição estrutural
        2. **Dimensão Econômica (Preços)** - Sinal de mercado
        3. **Dimensão Operacional (Relação CVU/PLD)** - Folga operacional do sistema
        """)
        
        # Nova lógica de interpretação
        st.markdown("### 2.2 CVU/PLD como Indicador de Folga")
        
        st.markdown("""
        - CVU alto vs PLD baixo = Folga Estrutural (perspectiva operacional)
        
        **Explicação:**
        Quando o CVU está muito acima do PLD (>150%), significa que as térmicas estão 
        **fora do despacho econômico**, indicando FOLGA operacional do sistema, 
        não risco. O sistema pode atender a demanda sem despachar térmicas caras.
        
        **Interpretação do Percentual CVU/PLD:**
        - **< 80%:** Folga operacional (CVU significativamente menor que PLD)
        - **80-95%:** Atenção (CVU próximo do PLD)
        - **95-100%:** Pressão moderada (CVU muito próximo do PLD)
        - **100-150%:** Risco de custos (CVU ≥ PLD, possível despacho com prejuízo)
        - **> 150%:** Folga estrutural (CVU muito maior que PLD, térmicas fora do despacho econômico)
        """)
        
        # Algoritmo de integração atualizado
        st.markdown("### 2.3 Algoritmo de Integração")
        
        st.markdown("""
        <div class="formula-box">
        Pulso = f(H, P, R)
        <br><br>
        Onde:
        <br>H = Score_Hidrologia (0-100)
        <br>P = Score_Preços (0-100)  
        <br>R = Score_Relação_CVU_PLD (0-100)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Conceito do Score CVU/PLD:**
        
        Se o percentual for menor ou igual a 80% = Folga operacional e Score máximo (100)
                    
        Se o percentual for entre 80% e 95% = Atenção e Score médio alto (80)
        
        Se o percentual for entre 95% e 100% = Pressão = e Score médio (50)
        
        Se o percentual for entre 100% e 150% = Risco de custos e Score baixo (20)
        
        Se o percentual for maior que 150% = Folga estrutural e Score médio alto (80)
        """)
            # =============================================
            # RESUMO VISUAL
            # =============================================
        
        # Obter percentual CVU/PLD
        thermal = core.get("thermal_analysis", {})
        indicadores = thermal.get("indicadores_quantitativos", {})
        percentual_cvu_pld = indicadores.get("percentual_cvu_pld")

        st.markdown("### 📊 **Resumo Visual**")
            
            # Criar visualização da relação CVU/PLD
        if percentual_cvu_pld is not None:
                # Calcular posição na escala 0-200%
            posicao = min(200, max(0, percentual_cvu_pld))
                
                # Definir zonas de cores
            zonas = [
                {"min": 0, "max": 80, "cor": "#22c55e", "label": "Folga Operacional"},
                {"min": 80, "max": 95, "cor": "#f59e0b", "label": "Atenção"},
                {"min": 95, "max": 100, "cor": "#ef4444", "label": "Pressão"},
                {"min": 100, "max": 150, "cor": "#dc2626", "label": "Risco"},
                {"min": 150, "max": 200, "cor": "#22c55e", "label": "Folga Estrutural"}
            ]
                
            fig = go.Figure()
                
                # Adicionar zonas
            for zona in zonas:
                fig.add_trace(go.Indicator(
                    mode="gauge",
                    value=posicao,
                    title={'text': f"CVU/PLD: {percentual_cvu_pld:.0f}%"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#3b82f6", 'thickness': 0.75},
                        'steps': [
                            {'range': [zona["min"], zona["max"]], 'color': zona["cor"]}
                            for zona in zonas
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': posicao
                        }
                    }
                ))
                
            fig.update_layout(
                height=300,
                template="plotly_dark",
                margin=dict(t=50, b=10, l=10, r=10)
            )
                
            st.plotly_chart(fig, use_container_width=True)
                
                # Legenda
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown('<div style="background-color:#22c55e; padding:5px; border-radius:3px; text-align:center; color:white;">Folga</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="background-color:#f59e0b; padding:5px; border-radius:3px; text-align:center; color:white;">Atenção</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div style="background-color:#ef4444; padding:5px; border-radius:3px; text-align:center; color:white;">Pressão</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div style="background-color:#dc2626; padding:5px; border-radius:3px; text-align:center; color:white;">Risco</div>', unsafe_allow_html=True)
            with col5:
                st.markdown('<div style="background-color:#22c55e; padding:5px; border-radius:3px; text-align:center; color:white;">Folga Estrutural</div>', unsafe_allow_html=True)

        # Matriz de decisão dos cards - ATUALIZADA
        st.markdown("### 2.4 Matriz de Decisão - Cores dos Cards")
        
        matriz_cards = pd.DataFrame({
            "Cor": ["🟢 Verde", "🟡 Amarelo", "🔴 Vermelho", "🟢 Verde"],
            "Razão CVU/PLD": [
                "< 80%",
                "80-95%", 
                "95-150%",
                "> 150%"
            ],
            "Interpretação": [
                "Folga Operacional",
                "Atenção - Monitorar",
                "Pressão/Risco de custos",
                "Folga Estrutural"
            ],
            "Perspectiva Sistema": [
                "CVU < PLD - Despacho econômico",
                "CVU próximo do PLD",
                "CVU ≥ PLD - Risco de prejuízo",
                "CVU >> PLD - Térmicas fora do despacho"
            ],
            "Significado Operacional": [
                "Risco baixo, operação normal",
                "Requer atenção e monitoramento",
                "Intervenção recomendada",
                "Folga estrutural, baixa pressão"
            ]
        })
        
        st.dataframe(matriz_cards, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Explicação da Nova Lógica:**
        
        **Folga Estrutural (>150%):**
        - CVU muito maior que PLD
        - Térmicas estão fora do despacho econômico
        - Sistema pode atender demanda sem térmicas caras
        - **É uma condição de FOLGA, não de risco**
        
        **Risco de Custos (100-150%):**
        - CVU igual ou maior que PLD
        - Despacho térmico pode gerar prejuízo
        - Pressão sobre modicidade tarifária
        - Risco econômico para o sistema
        """)
        
        # Dados atuais
        st.markdown("### 2.5 Componentes Atuais")
        
        cols = st.columns(3)
        
        # Hidrologia
        with cols[0]:
            hyd = core.get("hydrology", {})
            ear_medio = hyd.get('ear_medio')
            score_h = min(100, max(0, ear_medio)) if ear_medio is not None else 0
            st.markdown(f"""
            <div class="indicator-card">
            <h4>💧 Hidrologia</h4>
            <div class="metric-highlight">Score: {score_h:.0f}/100</div>
            <small>EAR: {_format_value(ear_medio, "{:.1f}%")}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Preços
        with cols[1]:
            prices = core.get("prices", {})
            # Usar volatilidade normalizada se disponível
            volatilidade_norm = prices.get('pld_volatilidade_norm')
            
            if volatilidade_norm is not None:
                # Score ajustado para volatilidade normalizada
                score_p = 100 - min(100, volatilidade_norm * 2)
                vol_text = f"{volatilidade_norm:.1f}% (banda)"
            else:
                score_p = 0
                vol_text = "—"
                
            st.markdown(f"""
            <div class="indicator-card">
            <h4>💰 Preços</h4>
            <div class="metric-highlight">Score: {score_p:.0f}/100</div>
            <small>Volatilidade: {vol_text}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Relação CVU/PLD (NOVA MÉTRICA)
        with cols[2]:
            thermal = core.get("thermal_analysis", {})
            indicadores = thermal.get('indicadores_quantitativos', {})
            percentual_cvu_pld = indicadores.get('percentual_cvu_pld')
            
            if percentual_cvu_pld is not None:
                # Calcular score baseado na nova lógica
                if percentual_cvu_pld <= 80:
                    score_r = 100  # Folga operacional
                elif percentual_cvu_pld <= 95:
                    score_r = 80   # Atenção
                elif percentual_cvu_pld <= 100:
                    score_r = 50   # Pressão
                elif percentual_cvu_pld <= 150:
                    score_r = 20   # Risco de custos
                else:
                    score_r = 80   # Folga estrutural
                
                percentual_text = f"{percentual_cvu_pld:.0f}%"
            else:
                score_r = 0
                percentual_text = "—"
                
            st.markdown(f"""
            <div class="indicator-card">
            <h4>🔥 Relação CVU/PLD</h4>
            <div class="metric-highlight">Score: {score_r:.0f}/100</div>
            <small>Percentual: {percentual_text}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretação integrada
        st.markdown("### 2.6 Interpretação Integrada")
        
        scores = [score_h, score_p, score_r]
        valid_scores = [s for s in scores if s is not None]
        score_medio = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        if score_medio > 70:
            interpretacao = "🟢 **Sistema Saudável**: Todas as dimensões em níveis adequados"
        elif score_medio > 50:
            interpretacao = "🟡 **Sistema em Observação**: Alguma dimensão requer atenção"
        else:
            interpretacao = "🔴 **Sistema em Risco**: Intervenção recomendada em múltiplas frentes"
        
        st.markdown(f"""
        <div class="interpretation-box">
        <h4>📊 Score Integrado: {score_medio:.0f}/100</h4>
        <p><strong>Método de cálculo:</strong> Média aritmética entre Hidrologia ({score_h:.0f}), Preços ({score_p:.0f}) e Relação CVU/PLD ({score_r:.0f})</p>
        {interpretacao}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_ciclo_sin(core):
    """Seção de análise do Ciclo do SIN."""
    
    st.markdown("## 🌎 3. Ciclo do SIN - Análise Técnica")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 3.1 Conceito
        
        O **Ciclo do SIN** classifica o regime operacional do sistema baseado em:
        - **EAR Médio**: Condição hídrica estrutural
        - **Stress Index**: Relação demanda/oferta hidráulica
        """)
        
        # Algoritmo de classificação
        st.markdown("### 3.2 Algoritmo de Classificação")
        
        st.markdown("""
        <div class="formula-box">
        Se (EAR > 75% e Stress < 0.9) → Ciclo Úmido
        <br>Se (EAR < 45% e Stress > 1.1) → Ciclo Crítico
        <br>Se (Stress > 1.0) → Ciclo Seco
        <br>Para nenhuma das alternativas → Ciclo Transição
        </div>
        """, unsafe_allow_html=True)
        
        # Matriz de decisão
        st.markdown("### 3.3 Matriz de Decisão")
        
        matriz_ciclo = pd.DataFrame({
            "Ciclo": ["Úmido", "Crítico", "Seco", "Transição"],
            "EAR": ["> 75%", "< 45%", "Qualquer", "Qualquer"],
            "Stress Index": ["< 0.9", "> 1.1", "> 1.0", "Outros"],
            "Características": [
                "Abundância hídrica, folga operacional",
                "Escassez hídrica, estresse estrutural",
                "Oferta pressionada, térmica elevada",
                "Equilíbrio instável, monitoramento"
            ]
        })
        
        st.dataframe(matriz_ciclo, use_container_width=True, hide_index=True)
        
        # Dados atuais
        st.markdown("### 3.4 Estado Atual")
        
        ciclo = core.get("sin_cycle", {})
        hyd = core.get("hydrology", {})
        mcp = core.get("mcp_economico", {})
        
        cols = st.columns(3)
        
        with cols[0]:
            ciclo_atual = ciclo.get('cycle', 'N/A')
            st.metric("Ciclo Atual", ciclo_atual.upper() if ciclo_atual != 'N/A' else 'N/A')
        
        with cols[1]:
            ear = hyd.get('ear_medio')
            ear_str = _format_value(ear, "{:.1f}%")
            st.metric("EAR Médio", ear_str)
        
        with cols[2]:
            stress = mcp.get('stress_index')
            stress_str = _format_value(stress, "{:.2f}")
            st.metric("Stress Index", stress_str)
        
        # Descrição
        descricao = ciclo.get('description', 'Descrição não disponível.')
        st.info(f"**Descrição:** {descricao}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_pld_mercado(core):
    """Seção de análise de PLD e Mercado."""
    
    st.markdown("## 💰 4. PLD e Mercado - Análise Técnica")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 4.1 Definição e Conceitos
        
        **PLD (Preço de Liquidação das Diferenças)**
        - Preço horário de equilibro entre oferta e demanda
        - Calculado pela CCEE para cada submercado
        - Principal sinal de preço de curto prazo
        
        **Volatilidade Normalizada do PLD**
        - Métrica que considera os limites regulatórios
        - Mede o desvio padrão como percentual da banda total (limites regulatórios)
        
        **Correlação PLD vs Carga (r)**
        - Mede quanto o preço responde à demanda
        - Correlação positiva forte: preços sensíveis à carga
        - Indica formação de preço por pressão da demanda
        
        **Correlação PLD vs Hidrologia (r)**
        - Mede quanto o preço responde à disponibilidade hídrica
        - Correlação negativa forte: preços sensíveis à hidrologia
        - Correlação positiva: comportamento anômalo (investigar)
        """)
        
        # Fórmulas
        st.markdown("### 4.2 Fórmulas de Cálculo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Volatilidade Normalizada:**
            """)
            st.markdown("""
            <div class="formula-box">
            Volatilidade_norm = (σ(PLD) / (Teto_estrutural - Piso)) × 100%
            <br>
            <br>Onde:
            <br>σ = desvio padrão dos preços horários
            <br>Teto_estrutural = R$ 785,27 (2026)
            <br>Piso = R$ 57,31 (2026)
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Classificação da volatilidade:**
            - < 10%: Baixa volatilidade
            - 10-25%: Volatilidade moderada  
            - 25-40%: Alta volatilidade
            - > 40%: Volatilidade extrema
            """)
        
        with col2:
            st.markdown("""
            **Posição Relativa na Banda:**
            """)
            st.markdown("""
            <div class="formula-box">
            Posição = ((PLD_médio - Piso) / (Teto_estrutural - Piso)) × 100%
            <br>
            <br>Onde:
            <br>PLD_médio = Preço médio observado
            <br>Teto_estrutural = R$ 751,73 (média semanal)
            <br>Piso = R$ 58,60
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Interpretação da posição:**
            - < 33%: PLD baixo (abaixo de R$ 287,72)
            - 33-66%: PLD moderado (💰287,72 - 💰516,84)
            - > 66%: PLD elevado (acima de R$ 516,84)
            """)
        
        # Dados atuais
        st.markdown("### 4.3 Dados Atuais")
        
        prices = core.get("prices", {})
        mcp = core.get("mcp_economico", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pld_medio = prices.get('pld_medio')
            pld_str = _format_value(pld_medio, "R$ {:.2f}")
            st.metric("PLD Médio", pld_str)
            st.caption("Preço médio de referência")
        
        with col2:
            volatilidade_norm = prices.get('pld_volatilidade_norm')
            vol_str = _format_value(volatilidade_norm, "{:.1f}%")
            classificacao_vol = prices.get('pld_classificacao_vol', 'N/A')
            st.metric("Volatilidade", vol_str)
            st.caption(f"Classificação: {classificacao_vol}")
        
        with col3:
            correlacoes = mcp.get('correlacoes', {})
            corr_carga = correlacoes.get('pld_vs_carga')
            corr_str = _format_value(corr_carga, "{:.2f}")
            st.metric("PLD vs Carga", corr_str)
            st.caption("Sensibilidade à demanda")
        
        with col4:
            corr_hidro = correlacoes.get('pld_vs_hidraulica')
            corr_h_str = _format_value(corr_hidro, "{:.2f}")
            st.metric("PLD vs Hidro", corr_h_str)
            st.caption("Dependência hídrica")
        
        # Análise de formação do preço
        st.markdown("### 4.4 Análise da Formação do Preço")
        
        # Calcular análise
        corr_carga = mcp.get("correlacoes", {}).get("pld_vs_carga")
        corr_hidro = mcp.get("correlacoes", {}).get("pld_vs_hidraulica")
        
        abs_corr_carga = abs(corr_carga) if corr_carga is not None else 0
        abs_corr_hidro = abs(corr_hidro) if corr_hidro is not None else 0
        
        if corr_carga is not None and corr_hidro is not None:
            if abs_corr_carga < 0.3 and abs_corr_hidro < 0.3:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### ⚠️ Comportamento do PLD NÃO EXPLICADO pelos Fatores Analisados
                
                **Análise:**
                - Correlação com carga: {:.2f} (fraca)
                - Correlação com hidrologia: {:.2f} (fraca)
                
                **Interpretação:**
                Os fatores tradicionais (demanda e hidrologia) não explicam a formação do preço do PLD.
                
                **Possíveis causas a investigar:**
                1. **Restrições operacionais** (transmissão, geração)
                2. **Térmicas marginais** com CVU elevado
                3. **Fatores administrativos** do mercado
                4. **Eventos pontuais** não capturados
                5. **Sazonalidade específica**
                
                **Recomendação:** Análise mais detalhada necessária.
                """.format(corr_carga, corr_hidro))
                st.markdown('</div>', unsafe_allow_html=True)
            
            elif corr_hidro > 0.3:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 🚨 Comportamento ANÔMALO do PLD
                
                **Análise:**
                - Correlação com hidrologia: {:.2f} (POSITIVA)
                - Comportamento esperado: Correlação NEGATIVA
                
                **Interpretação:**
                PLD sobe quando há MAIS hidrologia disponível (contraintuitivo).
                
                **Possíveis explicações:**
                1. **Período úmido com térmicas caras despachadas**
                2. **Restrições operacionais** limitam uso da hidrologia
                3. **Efeito sazonal** confundindo correlação
                4. **Problemas nos dados** ou alinhamento temporal
                
                **Recomendação:** Investigar causas imediatamente.
                """.format(corr_hidro))
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretação das correlações
        st.markdown("### 4.5 Guia de Interpretação")
        
        interpretacao_df = pd.DataFrame({
            "Força |r|": ["Forte (> 0,6)", "Moderada (0,3-0,6)", "Fraca (< 0,3)"],
            "PLD vs Carga": [
                "Demanda explica maioria da variação do PLD",
                "Demanda tem influência significativa",
                "Demanda tem pouca influência"
            ],
            "PLD vs Hidro (negativo)": [
                "PLD determinado pela hidrologia (esperado)",
                "Hidrologia influencia moderadamente",
                "Hidrologia tem pouca influência"
            ],
            "PLD vs Hidro (positivo)": [
                "🚨 ANÔMALO: Investigar causas",
                "⚠️ Comportamento atípico",
                "Pouco significativo"
            ]
        })
        
        st.dataframe(interpretacao_df, use_container_width=True, hide_index=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_analise_termica_revisada(core):
    """SEÇÃO ATUALIZADA: Análise térmica - perspectiva dupla)."""
    
    st.markdown("## 🔥 5. Análise Térmica - Perspectiva Dupla)")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 5.1 Conceito da análise
        
        - **Perspectiva dupla** - Sistema vs Gerador (dono da Usina térmica)
        
        **Dupla Perspectiva:**
        
        1. **Perspectiva do Sistema (Modicidade Tarifária):**
           - Foco: **Segurança operacional** e **folga do sistema**
           - Métrica: Percentual **CVU/PLD**
           - Pergunta: O sistema precisa despachar térmicas caras?
        
        2. **Perspectiva do Gerador (Viabilidade Econômica):**
           - Foco: **Rentabilidade** das usinas térmicas
           - Métrica: **Spread absoluto** (PLD - CVU)
           - Pergunta: É economicamente viável despachar térmicas?
        """)
        
        # Nova lógica conceitual
        st.markdown("### 5.2 Lógica Conceitual")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        
        **CVU ALTO vs PLD BAIXO = FOLGA**
        
        **Explicação:**
        - CVU muito maior que PLD (>150%) significa que as térmicas estão **fora do despacho econômico**
        - O sistema pode atender a demanda **sem precisar despachar térmicas (energia mais cara)**
        - Isto representa **FOLGA ESTRUTURAL** do sistema
        
        **Por que isto é importante?**
        - Quando o CVU está alto e o PLD baixo, o sistema tem **mais opções**
        - Esta correção alinha a análise com a realidade operacional do SIN
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fórmulas principais - PERSPECTIVA DO SISTEMA
        st.markdown("### 5.3 Fórmulas - Perspectiva do Sistema")
        
        st.markdown('<div class="card-perspectiva card-sistema">', unsafe_allow_html=True)
        st.markdown('<span class="perspectiva-badge badge-sistema">Sistema - Modicidade Tarifária</span>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Percentual CVU/PLD:**
            """)
            st.markdown("""
            <div class="formula-box">
            %CVU/PLD = (CVU / PLD) × 100%
            <br><br>
            Onde:
            <br>CVU = Custo Variável Unitário médio (R$/MWh)
            <br>PLD = Preço de Liquidação das Diferenças (R$/MWh)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Classificação:**
            - **< 80%:** Folga operacional
            - **80-95%:** Atenção
            - **95-100%:** Pressão moderada
            - **100-150%:** Risco de custos
            - **> 150%:** Folga estrutural
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fórmulas principais - PERSPECTIVA DO GERADOR
        st.markdown('<div class="card-perspectiva card-gerador">', unsafe_allow_html=True)
        st.markdown('<span class="perspectiva-badge badge-gerador">Gerador - Viabilidade Econômica</span>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Spread Absoluto:**
            """)
            st.markdown("""
            <div class="formula-box">
            Spread = PLD - CVU
            <br><br>
            Interpretação:
            <br>• Spread > 0: Viável economicamente
            <br>• Spread ≤ 0: Despacho estrutural (não econômico)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **Viabilidade Econômica:**
            - **Spread > 0:** ✅ Econômica (lucrativa)
            - **Spread ≤ 0:** 🔄 Estrutural (necessidade operacional)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fórmulas adicionais
        st.markdown("### 5.4 Fórmulas Adicionais - Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Margem de Segurança do Sistema:**
            """)
            st.markdown("""
            <div class="formula-box">
            Margem_Sistema = ((PLD - CVU) / PLD) × 100%
            <br><br>
            Se PLD ≤ CVU → Margem = 0%
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Interpretação:**
            - > 20%: Margem adequada
            - 10-20%: Margem reduzida
            - 5-10%: Margem crítica
            - < 5%: Margem insuficiente
            """)
        
        with col2:
            st.markdown("""
            **Dependência Térmica Efetiva:**
            """)
            st.markdown("""
            <div class="formula-box">
            Dependência = (%CVU/PLD > 80%) × (1 - EAR_normalizado)
            <br><br>
            Onde:
            <br>EAR_normalizado = EAR / 100 (0-1)
            <br>%CVU/PLD > 80% = 1 se verdadeiro, 0 se falso
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Interpretação:**
            - > 0.5: Alta dependência térmica
            - 0.2-0.5: Dependência moderada
            - < 0.2: Baixa dependência térmica
            """)
        
        # Matriz de classificação completa
        st.markdown("### 5.5 Matriz de Classificação Completa v5")
        
        classificacao_df = pd.DataFrame({
            "%CVU/PLD": ["< 80%", "80-95%", "95-100%", "100-150%", "> 150%"],
            "Classificação Sistema": ["Folga Operacional", "Atenção", "Pressão Moderada", "Risco de Custos", "Folga Estrutural"],
            "Spread Gerador": ["Positivo", "Positivo (reduzido)", "Zero/Negativo", "Negativo", "Negativo (grande)"],
            "Perspectiva Sistema": [
                "CVU < 80% PLD - Folga",
                "CVU próximo PLD - Monitorar",
                "CVU ≥ 95% PLD - Pressão",
                "CVU ≥ PLD - Risco econômico",
                "CVU >> PLD - Térmicas fora"
            ],
            "Perspectiva Gerador": [
                "✅ Viável econômica",
                "⚠️ Margem reduzida",
                "🔄 Despacho estrutural",
                "🔴 Prejuízo operacional",
                "🔄 Fora do despacho econômico"
            ]
        })
        
        st.dataframe(classificacao_df, use_container_width=True, hide_index=True)
        
        # Dados atuais - PERSPECTIVA DUPLA
        st.markdown("### 5.6 Indicadores Atuais - Perspectiva Dupla")
        
        thermal = core.get("thermal_analysis", {})
        
        if thermal:
            # PERSPECTIVA DO SISTEMA
            st.markdown("#### 🏭 **Perspectiva do Sistema**")
            st.markdown('<span class="perspectiva-badge badge-sistema">Modicidade Tarifária</span>', unsafe_allow_html=True)
            
            analise_sistema = thermal.get("analise_sistema", {})
            indicadores = thermal.get("indicadores_quantitativos", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                percentual_cvu_pld = indicadores.get('percentual_cvu_pld')
                percentual_str = _format_value(percentual_cvu_pld, "{:.0f}%")
                classificacao = analise_sistema.get('classificacao', 'N/A')
                st.metric("%CVU/PLD", percentual_str, delta=classificacao.replace('_', ' ').title())
                st.caption("Indicador de folga do sistema")
            
            with col2:
                margem_seguranca = indicadores.get('margem_seguranca_sistema')
                margem_str = _format_value(margem_seguranca, "{:.1f}%")
                st.metric("Margem Sistema", margem_str)
                st.caption("(PLD-CVU)/PLD")
            
            with col3:
                margem_teto = indicadores.get('margem_vs_teto')
                margem_teto_str = _format_value(margem_teto, "{:.1f}%")
                st.metric("Margem vs Teto", margem_teto_str)
                st.caption("Segurança regulatória")
            
            with col4:
                dependencia = indicadores.get('dependencia_termica_efetiva')
                dependencia_str = _format_value(dependencia, "{:.2f}")
                st.metric("Dependência Térmica", dependencia_str)
                st.caption("CVU>80% × (1-EAR)")
            
            # Descrição do sistema
            if analise_sistema.get("descricao"):
                st.markdown('<div class="card-perspectiva card-sistema">', unsafe_allow_html=True)
                st.markdown(f"**📋 Análise do Sistema:** {analise_sistema['descricao']}")
                if analise_sistema.get("recomendacao"):
                    st.markdown(f"**🎯 Recomendação:** {analise_sistema['recomendacao']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # PERSPECTIVA DO GERADOR
            st.markdown("#### ⚡ **Perspectiva do Gerador**")
            st.markdown('<span class="perspectiva-badge badge-gerador">Viabilidade Econômica</span>', unsafe_allow_html=True)
            
            analise_gerador = thermal.get("analise_gerador", {})
            dados_referencia = thermal.get("dados_referencia", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                spread = analise_gerador.get('spread_absoluto')
                spread_str = _format_value(spread, "R$ {:.1f}")
                perspectiva = analise_gerador.get('perspectiva_gerador', 'N/A')
                st.metric("Spread Absoluto", spread_str, delta=perspectiva.title())
                st.caption("PLD - CVU")
            
            with col2:
                viabilidade = analise_gerador.get('viabilidade_economica')
                if viabilidade is True:
                    viabilidade_str = "✅ Econômica"
                elif viabilidade is False:
                    viabilidade_str = "🔄 Estrutural"
                else:
                    viabilidade_str = "—"
                st.metric("Viabilidade", viabilidade_str)
                st.caption("Perspectiva do gerador")
            
            with col3:
                pld_medio = dados_referencia.get('pld_medio')
                cvu_medio = dados_referencia.get('cvu_medio')
                pld_str = _format_value(pld_medio, "R$ {:.1f}")
                cvu_str = _format_value(cvu_medio, "R$ {:.1f}")
                st.metric("PLD vs CVU", pld_str, delta=f"CVU: {cvu_str}")
                st.caption("Valores absolutos")
            
            # Descrição do gerador
            if analise_gerador.get("descricao"):
                st.markdown('<div class="card-perspectiva card-gerador">', unsafe_allow_html=True)
                st.markdown(f"**📋 Perspectiva do Gerador:** {analise_gerador['descricao']}")
                
                # Contexto hídrico
                contexto_hidrologico = thermal.get("contexto_hidrologico", {})
                if contexto_hidrologico.get("interpretacao"):
                    st.markdown(f"**💧 Contexto Hídrico:** {contexto_hidrologico['interpretacao']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualização da relação CVU/PLD
            st.markdown("### 5.7 Visualização da Relação CVU/PLD")
            
            if percentual_cvu_pld is not None:
                # Criar visualização
                fig = go.Figure()
                
                # Definir zonas
                zonas = [
                    {"min": 0, "max": 80, "cor": "#22c55e", "label": "Folga Operacional"},
                    {"min": 80, "max": 95, "cor": "#f59e0b", "label": "Atenção"},
                    {"min": 95, "max": 100, "cor": "#ef4444", "label": "Pressão"},
                    {"min": 100, "max": 150, "cor": "#dc2626", "label": "Risco"},
                    {"min": 150, "max": 200, "cor": "#22c55e", "label": "Folga Estrutural"}
                ]
                
                # Adicionar zonas ao gráfico
                fig.add_trace(go.Indicator(
                    mode="gauge",
                    value=min(200, max(0, percentual_cvu_pld)),
                    title={'text': f"CVU/PLD: {percentual_cvu_pld:.0f}%"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#3b82f6", 'thickness': 0.75},
                        'steps': [
                            {'range': [zona["min"], zona["max"]], 'color': zona["cor"]}
                            for zona in zonas
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': min(200, max(0, percentual_cvu_pld))
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    template="plotly_dark",
                    margin=dict(t=50, b=10, l=10, r=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Dados de análise térmica revisada não disponíveis.")
        
def mostrar_mcp_economico(core):
    """Seção de análise do MCP Econômico."""
    
    st.markdown("## 📊 6. MCP Econômico - Análise Técnica")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 6.1 Conceito
        
        **No contexto deste dashboard**, usamos **MCP Econômico** para análise do:
        - Mercado de curto prazo (spot)
        - Formação de preços estruturais
        - Condições de oferta/demanda
        """)
        
        # Fórmulas
        st.markdown("### 6.2 Fórmulas de Cálculo")
        
        st.markdown("""
        <div class="formula-box">
        Stress Index = Carga_média / Geração_hidráulica_média
        <br>
        <br>Onde:
        <br>Carga_média = Demanda média do SIN (MW)
        <br>Geração_hidráulica_média = Geração hídrica média (MW)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Interpretação do Stress Index:**
        
        - **SI < 0.95**: Excedente hídrico → Oferta > Demanda
        - **SI = 1.00**: Equilíbrio perfeito → Oferta = Demanda  
        - **SI > 1.05**: Déficit hídrico → Demanda > Oferta
        - **SI > 1.10**: Estresse estrutural → Dependência térmica crítica
        
        **Relevância:**
        - Mede a **dependência estrutural** do sistema na hidrologia
        - Indica necessidade de **despacho térmico complementar**
        - Sinal de **pressão de longo prazo** sobre preços
        """)
        
        # Regimes
        st.markdown("### 6.3 Regimes do MCP")
        
        regimes_df = pd.DataFrame({
            "Regime": ["Excedente Estrutural", "Equilíbrio", "Escassez Estrutural"],
            "Stress Index": ["< 0.95", "0.95 - 1.10", "> 1.10"],
            "Características": [
                "Oferta excede demanda, preços deprimidos",
                "Sistema balanceado, preços estáveis",
                "Demanda excede oferta, pressão nos preços"
            ],
            "Implicações": [
                "Redução de térmicas, preços baixos",
                "Operação normal, despacho otimizado",
                "Ativação térmica, preços elevados"
            ]
        })
        
        st.dataframe(regimes_df, use_container_width=True, hide_index=True)
        
        # Interpretação
        mcp = core.get("mcp_economico", {})
        interpretacao = mcp.get('interpretação', {})
        st.markdown("### 6.4 Interpretação Atual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            preco = interpretacao.get('preço', 'N/A')
            st.info(f"**Formação de Preço:** {preco}")
            st.caption("""
            **Estrutural**: Preços determinados por condições hídricas de longo prazo
            **Conjuntural**: Preços determinados por fatores de curto prazo (demanda, térmicas)
            **Mista**: Combinação de fatores estruturais e conjunturais
            """)
        
        with col2:
            termica = interpretacao.get('térmica', 'N/A')
            st.info(f"**Posição Térmica:** {termica}")
            st.caption("""
            **Pressionada**: Térmicas despachadas com prejuízo (CVU > PLD)
            **Competitiva**: Térmicas despachadas com lucro (CVU < PLD)
            **Equilibrada**: CVU próximo do PLD
            """)
        
        # Análise especial para casos problemáticos
        stress = mcp.get('stress_index')
        regime = mcp.get('regime_mcp', 'N/A')
        regime_display = regime.title() if regime != 'N/A' else 'N/A'
        corr = mcp.get("correlacoes", {})
        corr_carga = corr.get("pld_vs_carga")
        corr_hidro = corr.get("pld_vs_hidraulica")
        
        if corr_carga is not None and corr_hidro is not None:
            if abs(corr_carga) < 0.3 and abs(corr_hidro) < 0.3:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### ⚠️ ALERTA: Formação de Preço Não Explicada
                
                **Análise:**
                - Stress Index: {:.2f} ({})
                - Correlação PLD-Carga: {:.2f} (fraca)
                - Correlação PLD-Hidro: {:.2f} (fraca)
                
                **Interpretação:**
                O comportamento do PLD não está sendo explicado pelos fatores tradicionais 
                (demanda e hidrologia). Isto pode indicar:
                
                1. **Fatores não capturados** dominando a formação de preço
                2. **Restrições operacionais** significativas
                3. **Comportamento administrativo** do mercado
                4. **Problemas nos dados** ou metodologia
                
                **Recomendação:** Investigação detalhada necessária.
                """.format(
                    stress if stress else 0,
                    regime_display,
                    corr_carga,
                    corr_hidro
                ))
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Síntese do MCP Econômico:**
        
        Esta análise combina:
        1. **Dados físicos** (hidrologia, geração, carga)
        2. **Dados econômicos** (preços, custos)
        3. **Análise de correlações** (relações estruturais)
        
        Para fornecer uma visão integrada do **mercado spot** brasileiro.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_limites_regulatorios(core):
    """Seção sobre limites regulatórios do PLD."""
    
    st.markdown("## 📏 7. Limites Regulatórios do PLD - ANEEL/CCEE 2025")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 7.1 Definição dos Limites
        
        Para 2026, a ANEEL e a CCEE estabeleceram os seguintes limites para o PLD:
        
        **Piso (Mínimo):** R$ 57,31/MWh
        - Corresponde ao maior valor entre:
          - TEO (Tarifa de Energia de Otimização)
          - TEO Itaipu
        
        **Teto Estrutural:** R$ 785,27/MWh
        - Utilizado como limite para a **média semanal**
        - Referência para análise de nível de preço
        
        **Teto Horário:** R$ 1.611,04/MWh  
        - Limite absoluto para preços horários
        - Base para cálculo da volatilidade normalizada
        
        **Aplicação:**
        - Valores válidos para **todos os submercados**
        - Calculados diariamente pela CCEE
        - Consideram o Custo Marginal de Operação (CMO)
        """)
        
        # Dados atuais
        st.markdown("### 7.2 Posição Atual do PLD na Banda Regulatória")
        
        prices = core.get("prices", {})
        limites = prices.get("limites_regulatorios", {})
        pld_medio = prices.get("pld_medio")
        pos_rel = prices.get("pld_posicao_relativa")
        classificacao_nivel = prices.get("pld_classificacao_nivel", "N/A")
        
        if limites and pld_medio is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Piso Regulatório", f"R$ {limites.get('piso', 58.60):.2f}")
                st.caption("Mínimo absoluto")
            
            with col2:
                st.metric("PLD Médio Atual", f"R$ {pld_medio:.2f}")
                if pos_rel:
                    st.caption(f"{pos_rel:.0f}% da banda")
            
            with col3:
                st.metric("Teto Estrutural", f"R$ {limites.get('teto_estrutural', 751.73):.2f}")
                st.caption("Limite para média semanal")
            
            with col4:
                st.metric("Teto Horário", f"R$ {limites.get('teto_horario', 1542.23):.2f}")
                st.caption("Máximo horário absoluto")
            
            # Visualização gráfica
            if pos_rel:
                st.markdown("### 7.3 Visualização da Posição na Banda")
                
                # Cálculo dos limites de faixa
                faixa_baixa_max = 33  # Até 33% = baixo
                faixa_moderada_max = 66  # 33-66% = moderado
                
                # Criar visualização
                fig = go.Figure()
                
                # Adicionar faixas
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=pos_rel,
                    title={'text': f"Posição do PLD na Banda"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, faixa_baixa_max], 'color': "#22c55e"},
                            {'range': [faixa_baixa_max, faixa_moderada_max], 'color': "#f59e0b"},
                            {'range': [faixa_moderada_max, 100], 'color': "#ef4444"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': pos_rel
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    template="plotly_dark",
                    margin=dict(t=50, b=10, l=10, r=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretação
                st.markdown(f"**Classificação Atual:** {classificacao_nivel.upper()}")
                
                if pos_rel < 33:
                    st.success("""
                    **✅ PLD na faixa BAIXA da banda regulatória**
                    - Pressão de preços reduzida
                    - Condições favoráveis ao consumidor
                    - Risco de alta limitado
                    """)
                elif pos_rel < 66:
                    st.info("""
                    **ℹ️ PLD na faixa MODERADA da banda regulatória**
                    - Pressão de preços moderada
                    - Condições equilibradas
                    - Monitoramento recomendado
                    """)
                else:
                    st.warning("""
                    **⚠️ PLD na faixa ELEVADA da banda regulatória**
                    - Pressão de preços significativa
                    - Condições desfavoráveis ao consumidor
                    - Risco de atingir teto estrutural
                    """)
        
        # Implicações da alta volatilidade
        st.markdown("### 7.4 Implicações da Alta Volatilidade Intrínseca")
        
        st.markdown("""
        **O PLD por definição já é altamente volátil devido a:**
        
        1. **Amplitude da banda regulatória:**
           - Diferença piso-teto estrutural
           - Naturalmente gera alta volatilidade percentual
        
        2. **Natureza do mecanismo:**
           - Preço de equilíbrio horário
           - Sensível a variações instantâneas
           - Sujeito a eventos pontuais
        
        3. **Consequências para análise:**
           - Métricas tradicionais (σ/μ) podem ser enganosas
           - Necessidade de normalização pela banda
           - Contextualização obrigatória dos resultados
        
        **Por isso desenvolvemos a métrica de volatilidade normalizada**, 
        que expressa o desvio padrão como percentual da banda total, 
        proporcionando uma visão mais realista da instabilidade dos preços.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_formacao_preco(core):
    """Seção específica sobre formação do preço do PLD."""
    
    st.markdown("## 🔍 8. Formação do Preço do PLD - Análise Detalhada")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 8.1 Fatores que Influenciam o PLD
        
        **Fatores Estruturais (Longo Prazo):**
        1. **Hidrologia** - Disponibilidade hídrica
        2. **Capacidade instalada** - Oferta total
        3. **Mix de geração** - Composição das fontes
        4. **Limites regulatórios** - Bandas de preço
        
        **Fatores Conjunturais (Curto Prazo):**
        1. **Demanda horária** - Consumo instantâneo
        2. **Despacho térmico** - CVU das usinas marginais
        3. **Restrições operacionais** - Transmissão, geração
        4. **Eventos pontuais** - Manutenções, indisponibilidades
        
        **Fatores Administrativos:**
        1. **Regras do mercado** - Mecanismos da CCEE
        2. **Intervenções regulatórias** - Ações da ANEEL
        3. **Condições contratuais** - Contratos existentes
        """)
        
        # Análise das correlações atuais
        st.markdown("### 8.2 Diagnóstico da Formação Atual do Preço")
        
        mcp = core.get("mcp_economico", {})
        corr = mcp.get("correlacoes", {})
        
        corr_carga = corr.get("pld_vs_carga")
        corr_hidro = corr.get("pld_vs_hidraulica")
        
        if corr_carga is not None and corr_hidro is not None:
            # Determinar cenário
            abs_corr_carga = abs(corr_carga)
            abs_corr_hidro = abs(corr_hidro)
            
            # Cenário 1: Nenhum fator explica
            if abs_corr_carga < 0.3 and abs_corr_hidro < 0.3:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 🎯 CENÁRIO 1: Formação de Preço NÃO EXPLICADA
                
                **Diagnóstico:**
                - Correlação com carga: {:.2f} (|r| < 0,3)
                - Correlação com hidrologia: {:.2f} (|r| < 0,3)
                - **Nenhum fator tradicional explica o comportamento do PLD**
                
                **O que pode estar acontecendo:**
                
                1. **Fatores não capturados** dominando:
                   - Térmicas marginais com CVU específico
                   - Restrições de transmissão críticas
                   - Eventos operacionais pontuais
                
                2. **Comportamento administrativo:**
                   - Preços administrados ou intervencionados
                   - Regras específicas do período
                   - Efeito de contratos existentes
                
                3. **Problemas metodológicos:**
                   - Dados incompletos ou inconsistentes
                   - Defasagem temporal não considerada
                   - Sazonalidade não ajustada
                
                **Ações Recomendadas:**
                1. Investigar restrições operacionais ativas
                2. Analisar despacho térmico específico
                3. Verificar eventos extraordinários
                4. Revisar qualidade e alinhamento dos dados
                """.format(corr_carga, corr_hidro))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Cenário 2: Hidrologia positiva (anômalo)
            elif corr_hidro > 0.3:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 🎯 CENÁRIO 2: Comportamento ANÔMALO
                
                **Diagnóstico:**
                - Correlação com hidrologia: {:.2f} (POSITIVA e > 0,3)
                - **PLD sobe quando há MAIS água disponível**
                - Comportamento CONTRAINTUITIVO
                
                **Possíveis explicações:**
                
                1. **Período úmido com restrições:**
                   - Reservatórios cheios mas não podem gerar
                   - Restrições operacionais ou ambientais
                   - Transmissão limitada das áreas úmidas
                
                2. **Térmicas caras despachadas:**
                   - Mesmo com água, térmicas necessárias
                   - CVU elevado determinando preço marginal
                   - Contratos ou obrigações específicas
                
                3. **Efeito sazonal confundindo:**
                   - Período úmido coincide com alta demanda
                   - Ar condicionado no verão com chuvas
                   - Sazonalidade não removida da análise
                
                4. **Problemas nos dados:**
                   - Séries temporais não alinhadas
                   - Defasagem não considerada
                   - Dados inconsistentes
                
                **Ações Recomendadas:**
                1. Verificar restrições operacionais ativas
                2. Analisar despacho térmico no período
                3. Investigar eventos específicos
                4. Revisar alinhamento temporal dos dados
                """.format(corr_hidro))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Cenário 3: Fatores explicam normalmente
            else:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                ### 🎯 CENÁRIO 3: Formação de Preço EXPLICADA
                
                **Diagnóstico:**
                - Pelo menos um fator tradicional explica o PLD
                - Comportamento consistente com expectativas
                
                **Análise detalhada:**
                - Correlação com carga: {:.2f} {}
                - Correlação com hidrologia: {:.2f} {}
                
                **Interpretação:**
                {}
                """.format(
                    corr_carga,
                    "(|r| > 0,3)" if abs_corr_carga > 0.3 else "(|r| < 0,3)",
                    corr_hidro,
                    "(|r| > 0,3)" if abs_corr_hidro > 0.3 else "(|r| < 0,3)",
                    "Demanda explica o PLD" if abs_corr_carga > 0.3 and corr_carga > 0 else
                    "Hidrologia explica o PLD" if abs_corr_hidro > 0.3 and corr_hidro < 0 else
                    "Ambos fatores têm influência" if abs_corr_carga > 0.3 and abs_corr_hidro > 0.3 else
                    "Análise inconclusiva"
                ))
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Metodologia de investigação
        st.markdown("### 8.3 Metodologia para Investigação Adicional")
        
        st.markdown("""
        **Quando os fatores não explicam o PLD, investigue:**
        
        1. **Análise de Restrições:**
           - Verificar restrições operacionais ativas
           - Analisar capacidade de transmissão
           - Identificar usinas indisponíveis
        
        2. **Despacho Térmico Detalhado:**
           - CVU das térmicas marginais
           - Ordem de mérito do despacho
           - Custos específicos por usina
        
        3. **Fatores Externos:**
           - Condições climáticas extremas
           - Eventos de força maior
           - Intervenções regulatórias
        
        4. **Qualidade dos Dados:**
           - Completude das séries temporais
           - Alinhamento temporal correto
           - Consistência entre fontes
        
        **Ferramentas Recomendadas:**
        - Análise de correlação com defasagens
        - Modelos de regressão múltipla
        - Análise de componentes principais
        - Machine learning para padrões complexos
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_fontes_dados():
    """Seção sobre fontes de dados."""
    
    st.markdown("## 🔗 9. Fontes de Dados - Informações Técnicas")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        st.markdown("""
        ### 9.1 ONS - Operador Nacional do Sistema
        
        **Datasets utilizados:**
        
        | Dataset | Descrição | Frequência | Atualização |
        |---------|-----------|------------|-------------|
        | EAR_Diario_Subsistema | Energia Armazenada por subsistema | Diária | 00:00 |
        | ENA_Diario_Subsistema | Energia Natural Afluente | Diária | 00:00 |
        | CVU_Usina_Termica | Custo Variável Unitário | Mensal | 1º dia útil |
        | Energia Agora - Geração | Séries horárias de geração | Horária | A cada hora |
        | Energia Agora - Carga | Séries horárias de carga | Horária | A cada hora |
        """)
        
        st.markdown("""
        ### 9.2 CCEE - Câmara de Comercialização de Energia Elétrica
        
        **Dados utilizados:**
        
        | Dataset | Descrição | Frequência | Atualização |
        |---------|-----------|------------|-------------|
        | PLD Horário | Preços por submercado | Horária | A cada hora |
        | Histórico PLD | Série temporal 7 dias | Diária | 00:00 |
        """)
        
        st.markdown("""
        ### 9.3 ANEEL - Agência Nacional de Energia Elétrica
        
        **Informações utilizadas:**
        - Limites regulatórios do PLD para 2025
        - Regras de formação de preço
        - Parâmetros do mercado
        
        ### 9.4 Metadados Técnicos
        
        - **Versão da análise**: core-5.0 (com análise térmica v5 - perspectiva dupla)
        - **Formato de dados**: JSON estruturado
        - **Frequência de atualização**: Horária/diária conforme fonte
        - **Latência máxima**: 60 minutos
        - **Disponibilidade histórica**: 7 dias (PLD), 30 dias (hidrologia)
        - **Cobertura geográfica**: SIN completo + 4 submercados
        """)
        
        st.markdown("""
        ### 9.5 Validação e Qualidade
        
        **Processos de validação:**
        1. Verificação de completude dos dados
        2. Validação de ranges (EAR: 0-100%, PLD: dentro de limites)
        3. Consistência temporal (timestamps sequenciais)
        4. Cross-check entre fontes (ONS vs CCEE)
        5. Verificação de limites regulatórios
        
        **Indicadores de qualidade:**
        - Completude: >95% (requerido)
        - Consistência: >98% (alvo)
        - Atualidade: <60 minutos (meta)
        - Conformidade regulatória: 100% (obrigatório)
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_glossario():
    """Seção do glossário técnico."""
    
    st.markdown("## 📖 10. Glossário Técnico")
    
    with st.container():
        st.markdown('<div class="analise-section">', unsafe_allow_html=True)
        
        glossario = {
            "EAR": "Energia Armazenada - Percentual da capacidade útil dos reservatórios",
            "ENA": "Energia Natural Afluente - Energia média disponível nas bacias",
            "PLD": "Preço de Liquidação das Diferenças - Preço horário de equilíbrio de mercado",
            "CVU": "Custo Variável Unitário - Custo operacional das usinas térmicas",
            "SIN": "Sistema Interligado Nacional - Rede elétrica brasileira",
            "ONS": "Operador Nacional do Sistema - Órgão responsável pela operação do SIN",
            "CCEE": "Câmara de Comercialização de Energia Elétrica - Responsável pelo mercado de energia",
            "ANEEL": "Agência Nacional de Energia Elétrica - Órgão regulador",
            "MCP": "Mercado de Curto Prazo",
            "Stress Index": "Índice que relaciona demanda com oferta hídrica",
            "Spread": "Diferença entre PLD e CVU - Indicador de rentabilidade térmica",
            "%CVU/PLD": "Nova métrica v5 - Percentual do CVU em relação ao PLD - Mede folga operacional",
            "Perspectiva Sistema": "Análise do sistema como um todo - Foco em modicidade tarifária e segurança",
            "Perspectiva Gerador": "Análise do ponto de vista das usinas térmicas - Foco em viabilidade econômica",
            "Folga Estrutural": "Nova classificação v5 - CVU muito maior que PLD (>150%) - Térmicas fora do despacho econômico",
            "Risco de Custos": "Classificação v5 - CVU igual ou maior que PLD (100-150%) - Risco de despacho com prejuízo",
            "Volatilidade Normalizada": "Desvio padrão do PLD como % da banda regulatória",
            "Posição Relativa": "Posição do PLD médio na banda piso-teto estrutural",
            "Correlação (r)": "Medida de associação entre variáveis (-1 a +1)",
            "Rampa": "Variação horária da geração ou carga (MW/h)",
            "Submercado": "Divisão geográfica do SIN (Sul, Sudeste/Centro-Oeste, Nordeste, Norte)",
            "Margem de Segurança": "Diferença percentual entre PLD e CVU, relativa ao PLD",
            "Margem vs Teto": "Diferença percentual entre teto estrutural e CVU",
            "Dependência Térmica Efetiva": "Índice v5 que combina pressão térmica com condição hídrica"
        }
        
        for termo, definicao in glossario.items():
            st.markdown(f"**{termo}** - {definicao}")
        
        st.markdown("""
        ### Siglas Comuns
        
        - **MW**: Megawatt - Unidade de potência (1.000.000 watts)
        - **MWmed**: Megawatt médio - Potência média em um período
        - **MWh**: Megawatt-hora - Unidade de energia (1 MW por 1 hora)
        - **R$/MWh**: Reais por megawatt-hora - Unidade de preço de energia
        - **%**: Percentual - Proporção em relação ao total
        - **σ**: Sigma - Desvio padrão (medida de dispersão)
        - **μ**: Mu - Média (valor central)
        - **r**: Coeficiente de correlação
        """)
        
        st.markdown("""
        ### Métricas de Performance v5
        
        - **%CVU/PLD**: Percentual CVU/PLD (mede folga operacional do sistema)
        - **Spread Absoluto**: PLD - CVU (mede viabilidade econômica do gerador)
        - **Margem de Segurança**: Percentual (PLD-CVU)/PLD (mede risco econômico)
        - **Volatilidade Normalizada**: Variação dos preços como % da banda
        - **Correlação**: Grau de associação entre variáveis (-1 a +1)
        - **Tendência**: Mudança direcional em um período
        - **Score**: Pontuação padronizada para comparação (0-100)
        - **Stress Index**: Razão entre demanda e oferta hídrica
        - **Dependência Térmica Efetiva**: Índice v5 - Combina %CVU/PLD >80% com (1-EAR)
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)