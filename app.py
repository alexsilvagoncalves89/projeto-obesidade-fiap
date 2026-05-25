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

# 1. Configuração Global da Página
st.set_page_config(page_title="Sistema de Saúde - Obesidade", page_icon="🏥", layout="wide")

# 2. Carregamento do Modelo (Oculto nos bastidores)
@st.cache_resource
def carregar_modelo():
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
st.sidebar.markdown("Sistema Preditivo de Obesidade")
st.sidebar.divider()

# Criação das opções do menu
opcao_menu = st.sidebar.radio(
    "Navegação:",
    ["🩺 Modelo de Avaliação", "📊 Dashboard Analítico", "📑 Informações Adicionais"]
)

# ==============================================================================
# 4. PÁGINAS DO APLICATIVO (Roteamento)
# ==============================================================================

# ---------------------------------------------------------
# PÁGINA 1: MODELO DE AVALIAÇÃO (Formulário Clínico)
# ---------------------------------------------------------
if opcao_menu == "🩺 Modelo de Avaliação":
    st.title("🩺 Triagem de Pacientes")
    st.markdown("Preencha os dados clínicos e comportamentais do paciente para estimar o risco de obesidade.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            genero = st.selectbox("Gênero do Paciente:", ["Feminino", "Masculino"])
            idade = st.number_input("Idade (anos):", min_value=1, max_value=120, value=25)
            altura = st.number_input("Altura (metros):", min_value=0.50, max_value=2.50, value=1.70, step=0.01)
            peso = st.number_input("Peso (kg):", min_value=1.0, max_value=300.0, value=70.0, step=0.1)
            historico_familiar = st.selectbox("Histórico Familiar de Sobrepeso?", ["Sim", "Não"])
            alimentos_caloricos = st.selectbox("Consome alimentos calóricos?", ["Sim", "Não"])
            consumo_vegetais = st.slider("Consumo de vegetais (1 a 3):", 1.0, 3.0, 2.0, step=0.5)
            refeicoes_principais = st.slider("Refeições principais por dia:", 1.0, 4.0, 3.0, step=1.0)

        with col2:
            entre_refeicoes = st.selectbox("Consome entre as refeições?", ["Nunca", "Às vezes", "Frequentemente", "Sempre"])
            fumante = st.selectbox("É fumante?", ["Sim", "Não"])
            consumo_agua = st.slider("Consumo de água (Litros):", 1.0, 3.0, 2.0, step=0.5)
            monitora_calorias = st.selectbox("Monitora calorias?", ["Sim", "Não"])
            atividade_fisica = st.slider("Atividade física (0 a 3):", 0.0, 3.0, 1.0, step=0.5)
            uso_tecnologia = st.slider("Uso de tecnologia (0 a 2):", 0.0, 2.0, 1.0, step=0.5)
            consumo_alcool = st.selectbox("Consumo de álcool?", ["Nunca", "Às vezes", "Frequentemente", "Sempre"])
            meio_transporte = st.selectbox("Meio de transporte:", ["Transporte Público", "Automóvel", "Andando", "Motocicleta", "Bicicleta"])

    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento

    if st.button("🔮 Executar Diagnóstico", type="primary", use_container_width=True):
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
        
        # Alertas baseados no resultado (Visual de Saúde)
        if "Obesity" in resultado or "Obesidade" in resultado:
            st.error(f"### 🚨 Diagnóstico Estimado: **{resultado}**")
            st.markdown("Recomenda-se encaminhamento imediato para equipe multidisciplinar (Nutricionista/Endocrinologista).")
        elif "Overweight" in resultado or "Sobrepeso" in resultado:
            st.warning(f"### ⚠️ Diagnóstico Estimado: **{resultado}**")
            st.markdown("Recomenda-se acompanhamento de rotina e ajuste de hábitos.")
        else:
            st.success(f"### ✅ Diagnóstico Estimado: **{resultado}**")
            st.markdown("Paciente encontra-se dentro dos parâmetros de normalidade.")

# ---------------------------------------------------------
# PÁGINA 2: DASHBOARD
# ---------------------------------------------------------
elif opcao_menu == "📊 Dashboard Analítico":
    st.title("📊 Painel de Insights Hospitalares")
    st.markdown("Visualize o comportamento demográfico e os hábitos de vida dos pacientes analisados.")
    
    col_grafico1, col_grafico2 = st.columns(2)
    with col_grafico1:
        with st.container(border=True):
            st.subheader("Distribuição de Peso e Altura")
            st.markdown("*(Espaço reservado para o gráfico)*")
    with col_grafico2:
        with st.container(border=True):
            st.subheader("Matriz de Hábitos Alimentares")
            st.markdown("*(Espaço reservado para o gráfico)*")

# ---------------------------------------------------------
# PÁGINA 3: INFORMAÇÕES ADICIONAIS
# ---------------------------------------------------------
elif opcao_menu == "📑 Informações Adicionais":
    st.title("📑 Documentação Técnica e Desempenho")
    
    st.markdown("""
    ### 🏆 O Modelo Campeão: XGBoost
    Para este desafio, desenvolvemos um pipeline completo de Machine Learning. O algoritmo que apresentou a melhor aderência aos dados hospitalares foi o **XGBoost Classifier**.
    
    ### 🎯 Métricas de Validação (Cenário 1)
    * **Acurácia Global:** 95.98%
    * **F1-Score (Macro):** ~95%
    * *Nota:* Estas métricas provam a altíssima assertividade do modelo em equilibrar os acertos entre todas as 7 classes médicas (desde 'Abaixo do Peso' até 'Obesidade Grau III').
    
    ---
    
    ### ⚖️ Estudo de Cenários: Com e Sem Peso/Altura
    Durante a análise exploratória (Matriz de Correlação), percebeu-se que as variáveis **Peso** e **Altura** possuíam uma relação matemática quase direta (cálculo natural do IMC) com a variável alvo.
    
    Para testar a robustez dos nossos dados comportamentais, dividimos a modelagem em dois cenários:
    1. **Cenário 1 (Clínico - Atual na Plataforma):** Mantendo Peso e Altura. O modelo atua como uma ferramenta infalível de triagem rápida automatizada. *(Acurácia de 95.98%)*
    2. **Cenário 2 (Comportamental):** Removendo o Peso e a Altura do treinamento. Desafiamos a IA a prever o nível de obesidade olhando **exclusivamente** para o histórico familiar, consumo de água, atividades físicas e hábitos tecnológicos. 
    """)
