"""Testes do balanceamento: estrutura do pipeline e ausência de vazamento.

Rodar (qualquer uma das formas):
    pytest tests/test_balance.py -v
    python tests/test_balance.py

Usam dados sintéticos pequenos; não precisam do creditcard.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from imblearn.under_sampling import RandomUnderSampler  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

from src.models import BALANCE_CHOICES, get_models, get_samplers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLED_MODELS = ["Logistic Regression", "Random Forest"]
RESAMPLING = [b for b in BALANCE_CHOICES if b != "none"]


def make_xy(n=3000, frac=0.04, seed=0):
    """Dados no mesmo formato do creditcard.csv: Time, V1..V28, Amount."""
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=10,
        weights=[1 - frac, frac], class_sep=1.0, random_state=seed,
    )
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.uniform(0, 172_800, n))
    df["Amount"] = rng.exponential(80, n)
    return df, pd.Series(y, name="Class")


# --------------------------------------------------------------------------- #
# Estrutura do pipeline
# --------------------------------------------------------------------------- #
def test_sampler_fica_entre_prep_e_clf():
    for balance in RESAMPLING:
        models = get_models(0, balance=balance)
        for name in SAMPLED_MODELS:
            steps = [n for n, _ in models[name].steps]
            assert steps[0] == "prep" and steps[-1] == "clf", (balance, name, steps)
            assert len(steps) > 2, f"{name} deveria ter sampler com balance={balance}"
        # Isolation Forest nunca recebe sampler
        assert [n for n, _ in models["Isolation Forest"].steps] == ["prep", "clf"]


def test_sem_balanceamento_nao_ha_sampler():
    for name, pipe in get_models(0, balance="none").items():
        assert [n for n, _ in pipe.steps] == ["prep", "clf"], name


def test_class_weight_desligado_ao_reamostrar():
    for balance in RESAMPLING:
        models = get_models(0, balance=balance)
        for name in SAMPLED_MODELS:
            assert models[name].get_params()["clf__class_weight"] is None, (balance, name)
    base = get_models(0, balance="none")
    assert base["Logistic Regression"].get_params()["clf__class_weight"] == "balanced"
    assert base["Random Forest"].get_params()["clf__class_weight"] == "balanced_subsample"


def test_argumentos_invalidos():
    for bad in (
        lambda: get_samplers("xyz"),
        lambda: get_samplers("smote", ratio=0),
        lambda: get_samplers("smote_under", ratio=0.7),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("era esperado ValueError")


# --------------------------------------------------------------------------- #
# Vazamento: o sampler só pode ver dados de treino
# --------------------------------------------------------------------------- #
CALLS: list = []


class CountingUnderSampler(RandomUnderSampler):
    """RandomUnderSampler que registra o tamanho dos dados em cada fit_resample."""

    def fit_resample(self, X, y, **kwargs):
        CALLS.append(len(y))
        return super().fit_resample(X, y, **kwargs)


def _counting_pipeline():
    from sklearn.compose import ColumnTransformer

    prep = ColumnTransformer([("scale", RobustScaler(), ["Time", "Amount"])], remainder="passthrough")
    return ImbPipeline([
        ("prep", prep),
        ("sampler", CountingUnderSampler(sampling_strategy=0.5, random_state=0)),
        ("clf", LogisticRegression(max_iter=500)),
    ])


def test_sampler_so_ve_o_treino_de_cada_fold():
    X, y = make_xy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    expected = sorted(len(tr) for tr, _ in cv.split(X, y))

    CALLS.clear()
    proba = cross_val_predict(_counting_pipeline(), X, y, cv=cv, method="predict_proba")

    assert sorted(CALLS) == expected, "o sampler deve ver só o treino de cada fold"
    assert all(c < len(X) for c in CALLS), "o sampler não pode ver o dataset inteiro"
    assert proba.shape == (len(X), 2), "as predições OOF não podem ser reamostradas"


def test_sampler_nao_atua_na_predicao():
    X, y = make_xy()
    X_tr, X_te, y_tr = X.iloc[:2000], X.iloc[2000:], y.iloc[:2000]
    pipe = _counting_pipeline()

    CALLS.clear()
    pipe.fit(X_tr, y_tr)
    assert CALLS == [len(X_tr)]

    pipe.predict(X_te)
    pipe.predict_proba(X_te)
    pipe.decision_function(X_te)
    assert CALLS == [len(X_tr)], "predict/predict_proba/decision_function não devem reamostrar"
    assert pipe.predict_proba(X_te).shape[0] == len(X_te)


def test_nenhum_script_reamostra_fora_do_pipeline():
    """Reamostrar dados completos antes do split seria vazamento."""
    for rel in ("src/train.py", "src/data.py", "src/evaluate.py", "app.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "fit_resample" not in text, f"{rel}: use o sampler só dentro do pipeline"
        assert "SMOTE(" not in text, f"{rel}: use get_samplers() em src/models.py"


# --------------------------------------------------------------------------- #
# Fumaça: todos os modelos treinam e pontuam com cada estratégia
# --------------------------------------------------------------------------- #
def test_todos_os_modelos_rodam_em_cv_com_cada_estrategia():
    X, y = make_xy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for balance in RESAMPLING:
        for name, model in get_models(0, balance=balance).items():
            method = "decision_function" if name == "Isolation Forest" else "predict_proba"
            out = cross_val_predict(model, X, y, cv=cv, method=method)
            score = out if out.ndim == 1 else out[:, 1]
            assert score.shape == (len(X),), (balance, name)
            assert np.isfinite(score).all(), (balance, name)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, fn in tests:
        try:
            fn()
            print(f"OK    {nome}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"FALHA {nome}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - falhas}/{len(tests)} testes passaram")
    sys.exit(1 if falhas else 0)

