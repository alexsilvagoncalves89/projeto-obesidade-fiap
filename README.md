# 🏥 Sistema Inteligente de Suporte ao Diagnóstico de Obesidade
> **FIAP - Pós-Graduação em Data Analytics** > *Tech Challenge - Fase 4: Modelagem Preditiva e Engenharia de Machine Learning*

Análise de dados avançada e deploy de um modelo preditivo baseado em algoritmos de Gradient Boosting para triagem e classificação automatizada dos níveis de obesidade em ambientes hospitalares e clínicos.

---

## 🚀 Link da Aplicação em Produção
O sistema web foi implantado com sucesso no Streamlit Community Cloud e pode ser acessado em tempo real por equipes médicas através do link abaixo:

🔗 **[Acessar o Portal de Triagem Médica](https://projeto-obesidade-fiap-alt.streamlit.app/)**

---

## 📌 O Problema de Negócio
A obesidade é uma condição crônica multifatorial associada a graves riscos cardiometabólicos. Em ambientes de pronto-atendimento ou consultas de rotina, a identificação rápida e precisa do perfil antropométrico e de hábitos do paciente ajuda na intervenção precoce. 

Esta aplicação atua como uma ferramenta de **Suporte à Decisão Clínica (CDSS)**, permitindo que médicos ou enfermeiros insiram dados demográficos, físicos e comportamentais para obter instantaneamente a classificação do paciente em uma das 7 categorias de peso estabelecidas pela Organização Mundial da Saúde (OMS).

---

## 📊 Performance e Resultados do Modelo

O projeto foi estruturado através de um pipeline robusto de Ciência de Dados no ambiente de desenvolvimento, avaliando múltiplos algoritmos de classificação. O modelo campeão foi o **XGBoost Classifier**, otimizado via validação cruzada.

### Estudo de Cenários Clínicos
Para avaliar o impacto prático da solução, realizamos um estudo comparativo dividindo os dados em dois cenários distintos:

1. **Cenário 1 (Modelo Clínico Completo):** Inclui variáveis tradicionais como Peso e Altura (que correlacionam diretamente com o IMC). 
   * **Acurácia alcançada:** `95.98%`
   * **F1-Score (Macro):** `~95.00%`
   * *Aplicação:* Triagem rápida e automatização de prontuários eletrônicos.

2. **Cenário 2 (Modelo Comportamental / Preventivo):** Remove os dados de Peso e Altura do treinamento. Desafia a Inteligência Artificial a estimar a tendência de peso avaliando **exclusivamente** o histórico familiar, consumo de água, hábitos alimentares de risco, tabagismo e atividade física.
   * *Aplicação:* Medicina preventiva e questionários de telemedicina preventiva.

---

## 📂 Estrutura do Repositório (Arquitetura Flat)

Para garantir máxima eficiência no deploy contínuo da aplicação e evitar quebras de caminhos relativos em ambientes conteinerizados, o repositório adota a estrutura unificada na raiz:

```text
├── app.py                      # Código-fonte da aplicação Streamlit (Multi-páginas)
├── requirements.txt            # Especificação de dependências e versões do ecossistema Python
├── modelo_obesidade_xgb.joblib # Pipeline do modelo preditivo campeão serializado
├── label_encoder.joblib       # Tradutor binário das classes de diagnóstico médico
├── Obesity.csv                 # Base de dados histórica utilizada no estudo
└── README.md                   # Documentação estratégica do projeto (esta vitrine)
