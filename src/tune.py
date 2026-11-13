"""Ajuste de hiperparâmetros e comparação de configurações (só no treino).

Uso:
    python -m src.tune --fast                    # teste rápido
    python -m src.tune --n-iter 15 --folds 3     # execução real
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.data import DEFAULT_PATH, holdout_split, load_data, split_xy, subsample
from src.models import get_models

SCORING = "average_precision"  # PR-AUC

SPACES = {
    "Logistic Regression": {
        "clf__C": loguniform(1e-3, 1e2),
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [8, 12, 16, None],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", 0.3],
    },
    "Isolation Forest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_samples": ["auto", 0.25, 0.5, 1.0],
    },
    "Hist Gradient Boosting": {
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": [3, 5, 8, None],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__min_samples_leaf": [20, 50, 100],
        "clf__l2_regularization": loguniform(1e-3, 10),
    },
}


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="usa subamostra (só para testes)")
    ap.add_argument("--results-dir", default="results/tuning")
    args = ap.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    if args.fast:
        df = subsample(df, seed=args.seed)
        args.n_iter = min(args.n_iter, 4)
        print(f"[fast] subamostra com {len(df)} linhas")
    X, y = split_xy(df)

    # O teste NÃO é usado aqui: ele fica reservado para a avaliação final.
    X_train, _, y_train, _ = holdout_split(X, y, seed=args.seed)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    summary, best_params = [], {}
    for name, model in get_models(args.seed).items():
        print(f"\n== {name} ==")
        base = cross_val_score(model, X_train, y_train, cv=cv, scoring=SCORING).mean()

        search = RandomizedSearchCV(
            model, SPACES[name], n_iter=args.n_iter, scoring=SCORING,
            cv=cv, random_state=args.seed, refit=False,
        ).fit(X_train, y_train)

        cfg = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cols = ["mean_test_score", "std_test_score", "mean_fit_time"] + \
               [c for c in cfg.columns if c.startswith("param_")]
        cfg[cols].round(4).to_csv(out / f"{name.replace(' ', '_').lower()}_configs.csv", index=False)

        best_params[name] = search.best_params_
        summary.append({
            "model": name,
            "pr_auc_padrao": round(base, 4),
            "pr_auc_otimizado": round(search.best_score_, 4),
            "ganho": round(search.best_score_ - base, 4),
        })
        print(summary[-1])

    pd.DataFrame(summary).set_index("model").to_csv(out / "summary.csv")
    (out / "best_params.json").write_text(json.dumps(best_params, indent=2, default=_plain))
    print("\n", pd.DataFrame(summary).to_string(index=False))
    print(f"\nCopie {out}/best_params.json para BEST_PARAMS em src/models.py")


if __name__ == "__main__":
    main()