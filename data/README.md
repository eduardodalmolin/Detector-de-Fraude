# Dados

O dataset **não** é versionado (é grande). Baixe manualmente:

1. Acesse https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Baixe `creditcard.csv`
3. Coloque o arquivo nesta pasta: `data/creditcard.csv`

Ou, com a [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configurada:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Colunas: `Time`, `V1`..`V28` (componentes PCA anonimizados), `Amount` e `Class` (1 = fraude, 0 = normal).
