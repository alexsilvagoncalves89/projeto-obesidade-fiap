import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Configuração da página
st.set_page_config(page_title="Predição de Obesidade", page_icon="🩺", layout="centered")

# 2. Função de carregamento direto (arquivos na mesma pasta raiz)
@st.cache_resource
def carregar_modelo():
    with open('modelo_obesidade_xgb.pkl', 'rb') as f_modelo:
        modelo = pickle.load(f_modelo)
    with open('label_encoder.pkl', 'rb') as f_encoder:
        encoder = pickle.load(f_encoder)
    return modelo, encoder

# 3. Tratamento de erro simplificado
try:
    modelo_campeao, label_encoder = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo. Detalhes: {e}")
    st.stop()

# 4. Interface Visual
st.title("🩺 Sistema de Diagnóstico de Obesidade")
st.markdown("Plataforma de triagem utilizando inteligência artificial (XGBoost).")
st.divider()

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

st.divider()

if st.button("🔮 Executar Diagnóstico", type="primary"):
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
    
    st.success(f"### 📋 Diagnóstico Estimado: **{resultado}**")
