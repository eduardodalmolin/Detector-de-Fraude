"""Definição dos modelos comparados no projeto.

Balanceamento de classes
------------------------
O balanceamento (SMOTE / undersampling) é um *step do pipeline* (imblearn),
posicionado entre o pré-processamento e o classificador:

    prep (RobustScaler) -> [sampler(s)] -> clf

Por ser parte do pipeline, o sampler só atua no ``fit`` e apenas sobre os dados
de treino que o pipeline recebe (em ``cross_val_predict``: o treino de cada
fold). Em ``predict`` / ``predict_proba`` / ``decision_function`` o sampler é
ignorado, então validação, calibração e teste nunca são reamostrados e o
threshold é escolhido sempre na distribuição real das classes.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

SCALE_COLS = ["Time", "Amount"]

# Estratégias de balanceamento disponíveis (--balance em src.train).
#   none        -> sem reamostragem; usa class_weight (baseline)
#   smote       -> SMOTE (oversampling sintético da classe minoritária)
#   under       -> RandomUnderSampler (descarta exemplos da classe majoritária)
#   smote_under -> SMOTE seguido de undersampling (combinação clássica)
BALANCE_CHOICES = ("none", "smote", "under", "smote_under")
DEFAULT_SAMPLING_RATIO = 0.1  # razão minoritária/majoritária após reamostrar


class IsolationForestClassifier(ClassifierMixin, BaseEstimator):
    """Adapta o Isolation Forest à interface de classificador.

    Utiliza scores brutos de anomalia, em que valores maiores indicam
    maior probabilidade de uma observação ser anômala.
    """

    def __init__(self, n_estimators=200, contamination=0.002, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        cont = float(np.mean(y)) if y is not None else self.contamination
        cont = min(max(cont, 1e-4), 0.5)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=cont,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X)

        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(X)

    def predict(self, X):
        threshold = -self.model_.offset_
        return (self.decision_function(X) > threshold).astype(int)


def get_samplers(balance: str = "none", seed: int = 42,
                 ratio: float = DEFAULT_SAMPLING_RATIO) -> list:
    """Retorna a lista de steps (nome, sampler) do balanceamento escolhido.

    `ratio` é a razão minoritária/majoritária desejada após a reamostragem
    (0.1 = uma fraude para cada 10 transações normais).
    """
    if balance not in BALANCE_CHOICES:
        raise ValueError(f"balance deve ser um de {BALANCE_CHOICES}, recebido: {balance!r}")
    if balance == "none":
        return []
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio deve estar em (0, 1], recebido: {ratio}")

    if balance == "smote":
        return [("sampler", SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed))]
    if balance == "under":
        return [("sampler", RandomUnderSampler(sampling_strategy=ratio, random_state=seed))]

    # smote_under: sobe a minoritária até `ratio` e depois reduz a majoritária
    # até 0.5. O undersampling só pode *aumentar* a razão, logo ratio <= 0.5.
    if ratio > 0.5:
        raise ValueError("em smote_under, ratio deve ser <= 0.5")
    return [
        ("over", SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed)),
        ("under", RandomUnderSampler(sampling_strategy=0.5, random_state=seed)),
    ]


def _pipeline(estimator, samplers=None) -> Pipeline:
    """prep -> [samplers] -> clf."""
    prep = ColumnTransformer([("scale", RobustScaler(), SCALE_COLS)], remainder="passthrough")
    steps = [("prep", prep)]
    steps += list(samplers or [])
    steps.append(("clf", estimator))
    return Pipeline(steps)


def get_models(seed: int = 42, balance: str = "none",
               ratio: float = DEFAULT_SAMPLING_RATIO) -> dict:
    """Retorna os modelos a comparar.

    balance: estratégia de balanceamento (ver BALANCE_CHOICES). Com qualquer
        valor diferente de "none", o `class_weight` é desligado nos modelos que
        recebem o sampler, para não compensar o desbalanceamento duas vezes.
    ratio:   razão minoritária/majoritária após a reamostragem.

    O Isolation Forest (não supervisionado) nunca recebe sampler: dar fraudes
    sintéticas a ele distorceria a `contamination`, calculada a partir de y.
    """
    resampling = balance != "none"
    weight = None if resampling else "balanced"

    def samp():  # instâncias novas por modelo
        return get_samplers(balance, seed, ratio)

    models = {
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=1000, class_weight=weight, random_state=seed),
            samp(),
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=12,
                class_weight=None if resampling else "balanced_subsample",
                n_jobs=-1, random_state=seed,
            ),
            samp(),
        ),
        "Isolation Forest": _pipeline(IsolationForestClassifier(random_state=seed)),
    }
    return models
