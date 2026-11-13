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


def independent_validation_split(X, y, seed=42):

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=seed,
    )

    X_cal, X_val, y_cal, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )

    return (
        X_train, X_cal, X_val, X_test,
        y_train, y_cal, y_val, y_test,
    )
