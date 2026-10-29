"""Definição dos modelos comparados no projeto."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

SCALE_COLS = ["Time", "Amount"]


class IsolationForestClassifier(ClassifierMixin, BaseEstimator):
    """Adapta o Isolation Forest (não supervisionado) à interface de classificador.

    A taxa de fraude do treino define a `contamination`. O score de anomalia é
    normalizado para [0, 1] e exposto em `predict_proba`.
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
        scores = -self.model_.score_samples(X)
        self.lo_, self.hi_ = scores.min(), scores.max()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = -self.model_.score_samples(X)
        s = np.clip((s - self.lo_) / (self.hi_ - self.lo_ + 1e-12), 0, 1)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.model_.predict(X) == -1).astype(int)


def _pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer([("scale", RobustScaler(), SCALE_COLS)], remainder="passthrough")
    return Pipeline([("prep", prep), ("clf", estimator)])


def get_models(seed: int = 42) -> dict:
    """Retorna os modelos a comparar. Adicione novos aqui!"""
    return {
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed,
            )
        ),
        "Isolation Forest": _pipeline(IsolationForestClassifier(random_state=seed)),
    }
