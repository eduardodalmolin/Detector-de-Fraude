"""Carregamento e divisão dos dados."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PATH = Path("data/creditcard.csv")
TARGET = "Class"


def load_data(path=DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Veja data/README.md para baixar o dataset."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]


def subsample(df: pd.DataFrame, ratio: int = 20, seed: int = 42) -> pd.DataFrame:
    """Mantém todas as fraudes e `ratio` vezes esse número de transações normais.

    Útil para rodar rápido (--fast). Não use para reportar resultados finais.
    """
    fraud = df[df[TARGET] == 1]
    n_normal = (df[TARGET] == 0).sum()
    normal = df[df[TARGET] == 0].sample(n=min(len(fraud) * ratio, n_normal), random_state=seed)
    return pd.concat([fraud, normal]).sample(frac=1, random_state=seed).reset_index(drop=True)


def holdout_split(X, y, test_size: float = 0.2, seed: int = 42):
    """Holdout estratificado (mantém a proporção de fraudes em treino e teste)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
