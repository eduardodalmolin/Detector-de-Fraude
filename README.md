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

| Nome | GitHub | Responsabilidade |
|------|--------|------------------|
| _Nome 1_ | @usuario1 | _ex.: EDA e pré-processamento_ |
| _Nome 2_ | @usuario2 | _ex.: modelos e validação_ |
| _Nome 3_ | @usuario3 | _ex.: dashboard e README_ |

## 📦 Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284.807 transações, 492 fraudes. Colunas `V1..V28` são componentes PCA anonimizados. Instruções de download em [`data/README.md`](data/README.md).

## 🔬 Metodologia

Segue a estrutura sugerida nas aulas:

1. **Objetivo:** detectar fraude priorizando Recall e F1.
2. **Pré-processamento:** `RobustScaler` em `Time` e `Amount`.
3. **Desbalanceamento:** pesos de classe (`class_weight`). _Extra: testar SMOTE._
4. **Modelos:** Regressão Logística, Random Forest e Isolation Forest (detecção de anomalia).
5. **Validação:** holdout estratificado 80/20 + Stratified 5-Fold no treino.
6. **Métricas:** Precision, Recall, F1, PR-AUC, ROC-AUC e matriz de confusão.
7. **Resultados:** tabela comparativa, curvas e dashboard.

## 📊 Resultados

> ⚠️ Preencha depois de rodar o treino com o dataset completo.

| Modelo | Precision | Recall | F1 | PR-AUC |
|--------|-----------|--------|----|--------|
| Logistic Regression | – | – | – | – |
| Random Forest | – | – | – | – |
| Isolation Forest | – | – | – | – |

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

# 5. Abrir o dashboard
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
