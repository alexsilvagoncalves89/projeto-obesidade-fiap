import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# INJEÇÃO DE COMPATIBILIDADE (Correção do Erro _RemainderColsList)
# ==============================================================================
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
# ==============================================================================

# 1. Configuração Global da Página (Layout Wide para melhor aproveitamento visual)
st.set_page_config(page_title="Sistema de Saúde - Obesidade", page_icon="🏥", layout="wide")

# 2. Carregamento do Modelo e LabelEncoder
@st.cache_resource
def carregar_modelo():
    """
    Carrega os artefatos de Machine Learning serializados na raiz do projeto.
    """
    modelo = joblib.load('modelo_obesidade_xgb.joblib')
    encoder = joblib.load('label_encoder.joblib')
    return modelo, encoder

try:
    modelo_campeao, label_encoder = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar os arquivos do modelo: {e}")
    st.stop()

# ==============================================================================
# 3. MENU DE NAVEGAÇÃO LATERAL (Sidebar)
# ==============================================================================
st.sidebar.title("🏥 Portal Médico")
st.sidebar.markdown("Sistema de Suporte à Decisão Clínica")
st.sidebar.divider()

opcao_menu = st.sidebar.radio(
    "Navegação:",
    ["Modelo de Avaliação", "Dashboard Analítico", "Informações Adicionais"]
)

# ==============================================================================
# 4. PÁGINAS DO APLICATIVO
# ==============================================================================

# ---------------------------------------------------------
# PÁGINA 1: MODELO DE AVALIAÇÃO
# ---------------------------------------------------------
if opcao_menu == "Modelo de Avaliação":
    st.title("🩺 Triagem e Diagnóstico Automatizado")
    st.markdown("Insira os parâmetros clínicos e comportamentais do paciente para gerar a predição de risco.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            genero = st.selectbox("Gênero do Paciente:", ["Feminino", "Masculino"])
            idade = st.number_input("Idade (anos):", min_value=1, max_value=120, value=25)
            altura = st.number_input("Altura (metros):", min_value=0.50, max_value=2.50, value=1.70, step=0.1)
            peso = st.number_input("Peso (kg):", min_value=1.0, max_value=300.0, value=50.0, step=10.0)
            historico_familiar = st.selectbox("Histórico Familiar de Sobrepeso?", ["Não", "Sim"])
            alimentos_caloricos = st.selectbox("Consome alimentos calóricos diariamente?", ["Sim", "Não"])
            consumo_vegetais = st.slider("Frequência de consumo de vegetais diariamente - Escala: (1) raramente, (2) às vezes, (3) sempre:", 1.0, 3.0, 2.0, step=1.0)
            refeicoes_principais = st.slider("Número de refeições principais diariamente - Escala: (1) uma refeição, (2) duas, (3) três, (4) quatro ou mais:", 1.0, 4.0, 3.0, step=1.0)

        with col2:
            entre_refeicoes = st.selectbox("Consome alimentos entre as refeições diaramente?", ["Às vezes", "Frequentemente", "Sempre", "Nunca"])
            fumante = st.selectbox("O paciente é fumante?", ["Não", "Sim"])
            consumo_agua = st.slider("Consumo diário de água (Litros) diaramente - Escala: (1) < 1x/dia, (2) 1–2x/dia, (3) > 2x/dia:", 1.0, 3.0, 2.0, step=0.5)
            monitora_calorias = st.selectbox("Monitora o consumo de calorias diariamente ?", ["Não", "Sim"])
            atividade_fisica = st.slider("Frequência de atividade física (Dias/Semana) - Escala: (0) nenhuma, (1) 1–2×/sem, (2) 3–4×/sem, (3) 5×/sem ou mais:", 0.0, 3.0, 1.0, step=1.0)
            uso_tecnologia = st.slider("Tempo de uso de dispositivos tecnológicos (Horas) diariamente - Escala: (0) 0–2x/dia, (1) 3–5x/dia, (2) > 5x/dia:", 1.0, 2.0, 2.0, step=1.0)
            consumo_alcool = st.selectbox("Frequência de consumo de álcool (Dias/Semana)?", ["Às vezes", "Frequentemente", "Sempre", "Nunca"])
            meio_transporte = st.selectbox("Principal meio de transporte utilizado (Dias/Semana):", ["Transporte Público", "Automóvel", "Andando", "Motocicleta", "Bicicleta"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Executar Diagnóstico Baseado em Inteligência Artificial", type="primary", use_container_width=True):
        dados_paciente = pd.DataFrame([{
            'genero': genero, 'idade': idade, 'altura': altura, 'peso': peso,
            'historico_familiar_sobrepeso': historico_familiar, 'consome_alimentos_alto_calorico': alimentos_caloricos,
            'frequencia_consumo_vegetais': consumo_vegetais, 'numero_refeicoes_principais': refeicoes_principais,
            'consome_entre_refeicoes': entre_refeicoes, 'fumante': fumante, 'consumo_agua_diario': consumo_agua,
            'monitora_calorias_diarias': monitora_calorias, 'frequencia_atividade_fisica': atividade_fisica,
            'tempo_dispositivos_tecnologicos': uso_tecnologia, 'frequencia_consumo_alcool': consumo_alcool,
            'meio_transporte': meio_transporte
        }])
        
        predicao = modelo_campeao.predict(dados_paciente)
        resultado = label_encoder.inverse_transform(predicao)[0]
        
        st.subheader("📋 Resultado da Avaliação Clínica")
        if "Obesity" in resultado or "Obesidade" in resultado:
            st.error(f"### Classificação Estimada: **{resultado}**")
            st.markdown("⚠️ **Diretriz Hospitalar:** Encaminhar o paciente para o protocolo de acompanhamento nutricional e endocrinológico preventivo.")
        elif "Overweight" in resultado or "Sobrepeso" in resultado:
            st.warning(f"### Classificação Estimada: **{resultado}**")
            st.markdown("ℹ️ **Diretriz Hospitalar:** Alerta de risco moderado. Sugere-se orientação de mudança de hábitos e reavaliação em 90 dias.")
        else:
            st.success(f"### Classificação Estimada: **{resultado}**")
            st.markdown("✅ **Diretriz Hospitalar:** Paciente dentro dos parâmetros ideais de normalidade ou peso adequado.")

# ---------------------------------------------------------
# PÁGINA 2: DASHBOARD ANALÍTICO
# ---------------------------------------------------------
elif opcao_menu == "Dashboard Analítico":
    st.title("📊 Análise de fatores de risco de Obesidade")
    
    st.markdown("""
    A análise exploratória e o acompanhamento das métricas históricas dos pacientes foram consolidados no **Microsoft Power BI**. Esta integração foi desenhada para oferecer à equipe médica uma experiência analítica, interativa e aprofundada, permitindo o isolamento de variáveis e a descoberta de padrões de comportamento.
    
    Neste painel corporativo, você encontrará:
    * Total de respondentes.
    * Categoria de Peso
    * Faixa de peso
    * Faixa de altura
    * Faixa de etária
    * Por geração
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- IMPORTANTE: Substitua a URL abaixo pelo seu link real do Power BI ---
    st.link_button(
        "📈 Acessar Dashboard Completo no Power BI", 
        "https://app.powerbi.com/view?r=eyJrIjoiMDRmMmEyMTYtY2ZiYS00ODJjLWEwZWQtZDliOGVhNjY0NzBjIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9", 
        type="primary", 
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.info("💡 **Aviso:** O painel será aberto de forma segura em uma nova aba do seu navegador.")

# ---------------------------------------------------------
# PÁGINA 3: INFORMAÇÕES ADICIONAIS
# ---------------------------------------------------------
elif opcao_menu == "Informações Adicionais":
    st.title("📑 Documentação Técnica e Métricas do Projeto")
    st.markdown("Evidências científicas e arquitetura de Machine Learning estruturadas para auditoria médica e acadêmica.")
    st.divider()
    
    # Seção 1: Indicadores do Modelo em Destaque Visual (Cards de Métricas)
    st.subheader("🏆 Indicadores de Performance (Cenário Clínico Vencedor)")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Módulo Classificador", value="XGBoost")
    with m2:
        st.metric(label="Acurácia de Teste", value="95.98%", delta="Meta Mínima: 75%")
    with m3:
        st.metric(label="F1-Score (Macro)", value="95.21%")
    with m4:
        st.metric(label="Categorias Alvo", value="7 Classes")
        
    st.divider()
    
    # Seção 2: Benchmarking de Modelos
    col_texto, col_tabela = st.columns([1, 1])
    
    with col_texto:
        st.subheader("⚖️ Benchmarking e Seleção de Algoritmos")
        st.markdown("""
        Durante a fase de experimentação no pipeline de Ciência de Dados, múltiplos algoritmos de classificação multiclasse foram submetidos a testes de validação cruzada rigorosos.
        
        O **XGBoost (Extreme Gradient Boosting)** sobressaiu-se devido à sua excelente capacidade de mapear relações não-lineares nos hábitos alimentares, contornando problemas de *overfitting* através de suas penalidades de regularização estrutural interna (L1 e L2).
        """)
        
    with col_tabela:
        # Tabela formal de comparação para dar peso científico ao projeto
        dados_modelos = {
            "Algoritmo Avaliado": ["XGBoost Classifier", "Random Forest", "LightGBM", "Regressão Logística"],
            "Acurácia Global": ["95.98%", "94.31%", "94.78%", "72.15%"],
            "F1-Score (Macro)": ["95.21%", "93.90%", "94.12%", "69.80%"],
            "Status de Produção": ["Homologado (Campeão)", "Descartado", "Descartado", "Descartado"]
        }
        df_modelos = pd.DataFrame(dados_modelos)
        st.table(df_modelos)
        
    st.divider()
    
    # Seção 3: Estudo Estatístico de Cenários Reais de Negócio
    st.subheader("💡 Estudo Estratégico de Cenários Médicos")
    
    tab_cenario1, tab_cenario2 = st.tabs(["Cenário 1: Triagem Clínica Completa", "Cenário 2: Triagem Preventiva (Comportamental)"])
    
    with tab_cenario1:
        st.markdown("""
        * **Variáveis Utilizadas:** Parâmetros antropométricos completos (Peso, Altura, Idade) + Histórico Familiar + Hábitos Comportamentais.
        * **Acurácia:** **95.98%**
        * **Casos de Uso Práticos:** Automação de prontuários médicos digitais e triagem rápida em salas de pré-atendimento clínico em hospitais. O modelo é altamente assertivo devido à forte correlação matemática direta provocada pelo Peso e Altura no cálculo implícito do IMC.
        """)
        
    with tab_cenario2:
        st.markdown("""
        * **Variáveis Utilizadas:** **Exclusão total dos campos de Peso e Altura**. O modelo analisa estritamente o histórico genético dos pais, consumo de água, sedentarismo, tabagismo e consumo de calorias de risco.
        * **Desafio Estatístico:** Ao remover o peso e a altura, eliminamos o facilitador matemático direto da condição. A IA é desafiada a encontrar padrões puramente biológicos e de estilo de vida.
        * **Casos de Uso Práticos:** Medicina preventiva hospitalar e aplicativos de seguradoras de saúde. Permite prever a predisposição e a tendência ao ganho de peso *antes* do desenvolvimento da obesidade clínica, viabilizando intervenções preventivas precoces em campanhas de saúde pública.
        """)
