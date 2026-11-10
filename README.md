# 💳 Detecção de Fraude em Cartão de Crédito

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

Projeto final da disciplina **Sistemas Inteligentes** (UFABC).

## 🎯 Objetivo

Identificar transações fraudulentas de cartão de crédito, **minimizando fraudes não detectadas (Recall)** sem gerar alarmes falsos demais (Precision).

O problema é difícil por causa do **desbalanceamento extremo**: só ~0,17% das transações são fraude. Um modelo que responde "tudo é normal" teria ~99,8% de acurácia e seria inútil. Por isso avaliamos com Precision, Recall, F1 e curva Precision-Recall, e não só Accuracy.

## 👥 Equipe

| **Integrante** | **GitHub** | **Responsabilidades** |
|---|---|---|
| **Integrante 1** | @usuario1 | Análise exploratória dos dados (EDA), investigação da distribuição das classes, análise do desbalanceamento e produção de gráficos e conclusões. |
| **Integrante 2** | @usuario2 | Pré-processamento dos dados, implementação de técnicas de balanceamento (SMOTE e/ou undersampling), integração ao pipeline e prevenção de vazamento de dados (*data leakage*). |
| **Integrante 3** | @usuario3 | Desenvolvimento e otimização dos modelos de classificação, ajuste de hiperparâmetros, comparação de configurações e implementação de um quarto modelo, se necessário. |
| Erik | @erikhoshima | Execução do pipeline, organização da validação cruzada, definição do threshold com dados de validação e geração das métricas finais de avaliação. |
| **Integrante 5** | @usuario5 | Elaboração de gráficos comparativos e tabelas de métricas, desenvolvimento do dashboard em Streamlit, integração dos resultados e testes de classificação de novas transações. |
| **Integrante 6** | @usuario6 | Elaboração e atualização do README, documentação da metodologia, consolidação dos resultados e conclusões, preparação do relatório e dos slides da apresentação. |

## 📦 Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284.807 transações, 492 fraudes. Colunas `V1..V28` são componentes PCA anonimizados. Instruções de download em [`data/README.md`](data/README.md).

## 🔬 Metodologia

Segue a estrutura sugerida nas aulas:

1. **Objetivo:** detectar fraude priorizando Recall e F1.
2. **Pré-processamento:** `RobustScaler` em `Time` e `Amount`.
3. **Desbalanceamento:** pesos de classe (`class_weight`). _Extra: testar SMOTE._
4. **Modelos:** Regressão Logística, Random Forest e Isolation Forest (detecção de anomalia).
5. **Validação e seleção de threshold:**
   - Experimento OOF: holdout estratificado 80/20, com
     Stratified 5-Fold no conjunto de treino.
   - Experimento independente: divisão em treino (60%),
     calibração (10%), validação (10%) e teste (20%).
   - O threshold é escolhido para maximizar o F1-score
     nos dados de calibração ou nas predições OOF.

6. **Métricas:** Precision, Recall, F1, PR-AUC, ROC-AUC
   e matriz de confusão.

7. **Seleção do modelo:** modelo com maior PR-AUC na
   validação. O conjunto de teste é reservado para a
   avaliação final.

## 📊 Resultados

Os experimentos foram realizados com o dataset completo: 284.807 transações, das quais 492 são fraudulentas (0,173%).

### Experimento 1 — Validação OOF

| Modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8298 | 0.7959 | 0.8125 | 0.7175 | 0.9720 |
| Random Forest | 0.8163 | 0.8163 | 0.8163 | **0.8169** | 0.9678 |
| Isolation Forest | 0.2976 | 0.2551 | 0.2747 | 0.1558 | 0.9560 |

### Experimento 2 — Calibração independente

| Modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8081 | 0.8163 | 0.8122 | 0.7191 | **0.9725** |
| Random Forest | 0.8370 | 0.7857 | 0.8105 | **0.8170** | 0.9478 |
| Isolation Forest | 0.2600 | 0.2653 | 0.2626 | 0.1486 | 0.9512 |

### Análise

O Random Forest apresentou o maior PR-AUC de teste nos dois experimentos, mantendo aproximadamente 0.817.

A Regressão Logística apresentou ROC-AUC ligeiramente superior no experimento independente, enquanto o Isolation Forest teve desempenho inferior nas métricas avaliadas.

Os resultados indicam que o Random Forest foi o modelo com melhor capacidade de distinguir transações fraudulentas neste estudo, considerando principalmente a curva Precision-Recall.

**Observação:** o experimento independente utiliza menos dados para treinamento (60%, contra 80% no OOF). Portanto, diferenças entre os resultados não devem ser atribuídas exclusivamente à metodologia de validação.

_Adicionar aqui `results/pr_curves.png` e um GIF do dashboard em `images/`._

## 🚀 Como rodar

```bash
# 1. Clonar e entrar
git clone https://github.com/SEU_USUARIO/fraud-detection.git
cd fraud-detection

# 2. Ambiente virtual e dependências
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Baixar o dataset para data/creditcard.csv (veja data/README.md)

# 4. Treinar e comparar os modelos
python -m src.train              # use --fast para um teste rápido com subamostra
                                 # use --independent para executar a avaliação com calibração independente

# 6. Abrir o dashboard
streamlit run app.py
```

## 🗂️ Estrutura

```
fraud-detection/
├── app.py                 # Dashboard Streamlit
├── src/
│   ├── data.py            # carga e divisão dos dados
│   ├── models.py          # modelos (adicione novos em get_models)
│   ├── evaluate.py        # métricas e gráficos
│   └── train.py           # pipeline de treino/validação
├── notebooks/             # análise exploratória
├── data/                  # dataset (não versionado)
├── models/                # modelo treinado (não versionado)
├── results/               # métricas e gráficos gerados
└── images/                # imagens para o README
```

## ✅ Roadmap

- [ ] Baixar o dataset e rodar `python -m src.train`
- [ ] Análise exploratória no notebook (distribuições, desbalanceamento)
- [ ] Testar SMOTE / undersampling (`imbalanced-learn`)
- [ ] Ajuste de hiperparâmetros (`GridSearchCV` ou `RandomizedSearchCV`)
- [ ] Adicionar um 4º modelo (ex.: Gradient Boosting)
- [ ] Preencher a tabela de resultados e gráficos neste README
- [ ] Gravar GIF do dashboard
- [ ] Relatório e apresentação final

## 🤝 Contribuindo

Veja [`CONTRIBUTING.md`](CONTRIBUTING.md).

## 📄 Licença

[MIT](LICENSE)
