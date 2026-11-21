"""Treina e compara os modelos.

Uso:
    python -m src.train                 # dataset completo
    python -m src.train --fast          # subamostra para testes rápidos
    python -m src.train --data caminho/creditcard.csv --folds 5
    python -m src.train --independent  # avaliação independente

Balanceamento (imblearn, dentro do pipeline; só atua no treino):
    python -m src.train --balance smote --results-dir results/smote --models-dir models/smote
    python -m src.train --balance under --results-dir results/under --models-dir models/under
    python -m src.train --balance smote_under --sampling-ratio 0.1 \\
        --results-dir results/smote_under --models-dir models/smote_under
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import (
    DEFAULT_PATH,
    holdout_split,
    independent_validation_split,
    load_data,
    split_xy,
    subsample,
)
from src.evaluate import (
    compute_metrics,
    find_best_threshold,
    plot_confusion,
    plot_pr_curves,
)
from src.models import BALANCE_CHOICES, DEFAULT_SAMPLING_RATIO, get_models


def build_models(args):
    """Monta os modelos conforme as flags de balanceamento."""
    return get_models(
        args.seed,
        balance=args.balance,
        ratio=args.sampling_ratio,
    )


def run_independent_experiment(X, y, args):

    (
        X_train, X_cal, X_val, X_test,
        y_train, y_cal, y_val, y_test,
    ) = independent_validation_split(X, y, seed=args.seed)

    results_dir = Path(args.results_dir) / "independent"
    models_dir = Path(args.models_dir) / "independent"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows, test_scores, fitted = [], {}, {}

    for name, model in build_models(args).items():
        print(f"\n== {name} ==")

        model.fit(X_train, y_train)

        if name == "Isolation Forest":
            cal_score = model.decision_function(X_cal)
        else:
            cal_score = model.predict_proba(X_cal)[:, 1]

        threshold, _ = find_best_threshold(y_cal, cal_score)

        if name == "Isolation Forest":
            val_score = model.decision_function(X_val)
        else:
            val_score = model.predict_proba(X_val)[:, 1]

        validation = compute_metrics(
            y_val,
            val_score,
            threshold=threshold,
        )

        if name == "Isolation Forest":
            test_score = model.decision_function(X_test)
        else:
            test_score = model.predict_proba(X_test)[:, 1]

        test = compute_metrics(
            y_test,
            test_score,
            threshold=threshold,
        )

        row = {"model": name, "threshold": threshold}
        row.update({f"validation_{k}": v for k, v in validation.items()})
        row.update({f"test_{k}": v for k, v in test.items()})
        rows.append(row)

        test_scores[name] = test_score
        fitted[name] = model

        print(f"Threshold: {threshold:.10f}")
        print("Validação:", {k: round(v, 4) for k, v in validation.items()})
        print("Teste:", {k: round(v, 4) for k, v in test.items()})

        plot_confusion(
            y_test,
            test_score,
            name,
            results_dir / f"confusion_{name.replace(' ', '_').lower()}.png",
            threshold=threshold,
        )

    table = pd.DataFrame(rows).set_index("model")
    best = table["validation_pr_auc"].idxmax()

    table.round(4).to_csv(results_dir / "metrics.csv")
    plot_pr_curves(y_test, test_scores, results_dir / "pr_curves.png")

    joblib.dump(fitted[best], models_dir / "model.joblib")
    pd.DataFrame({
        "y_true": y_test.values,
        "y_score": test_scores[best],
    }).to_csv(results_dir / "test_scores.csv", index=False)

    (results_dir / "best_model.txt").write_text(best)

    print("\nResultados da avaliação independente:")
    print(table.round(4).to_string())
    print(f"\nMelhor modelo (PR-AUC na validação): {best}")
    print(f"Artefatos salvos em {results_dir} e {models_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_PATH))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="usa subamostra (só para testes)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument(
        "--independent",
        action="store_true",
        help="executa avaliação com calibração e validação independentes",
    )
    ap.add_argument(
        "--balance",
        choices=BALANCE_CHOICES,
        default="none",
        help="estratégia de balanceamento aplicada só ao treino (default: none = class_weight)",
    )
    ap.add_argument(
        "--sampling-ratio",
        type=float,
        default=DEFAULT_SAMPLING_RATIO,
        help="razão minoritária/majoritária após reamostrar (default: %(default)s)",
    )
    args = ap.parse_args()

    if args.balance != "none" and (args.results_dir == "results" or args.models_dir == "models"):
        ap.error(
            "com --balance, informe --results-dir e --models-dir próprios "
            "(ex.: results/smote e models/smote) para não sobrescrever o baseline"
        )

    results_dir, models_dir = Path(args.results_dir), Path(args.models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    if args.fast:
        df = subsample(df, seed=args.seed)
        print(f"[fast] subamostra com {len(df)} linhas")
    X, y = split_xy(df)
    print(f"{len(df)} transações | fraudes: {int(y.sum())} ({y.mean():.3%})")
    print(
        f"Balanceamento: {args.balance}"
        + (f" (razão {args.sampling_ratio})" if args.balance != "none" else " (class_weight)")
    )

    if args.independent:
        run_independent_experiment(X, y, args)
        return

    X_train, X_test, y_train, y_test = holdout_split(X, y, seed=args.seed)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows, test_scores, fitted = [], {}, {}

    for name, model in build_models(args).items():
        print(f"\n== {name} ==")

        if name == "Isolation Forest":
            oof_scores = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=cv,
                method="decision_function",
            )
        else:
            oof_scores = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
            )[:, 1]

        best_threshold, _ = find_best_threshold(y_train, oof_scores)

        validation = compute_metrics(
            y_train,
            oof_scores,
            threshold=best_threshold,
        )

        model.fit(X_train, y_train)

        if name == "Isolation Forest":
            score = model.decision_function(X_test)
        else:
            score = model.predict_proba(X_test)[:, 1]

        test = compute_metrics(
            y_test,
            score,
            threshold=best_threshold,
        )

        row = {
            "model": name,
            "threshold": best_threshold,
        }

        row.update({
            f"validation_{k}": v
            for k, v in validation.items()
        })

        row.update({
            f"test_{k}": v
            for k, v in test.items()
        })

        rows.append(row)

        test_scores[name] = score
        fitted[name] = model

        print(f"Threshold: {best_threshold:.10f}")
        print("Validação:", {k: round(v, 4) for k, v in validation.items()})
        print("Teste:", {k: round(v, 4) for k, v in test.items()})

        plot_confusion(
            y_test,
            score,
            name,
            results_dir / f"confusion_{name.replace(' ', '_').lower()}.png",
            threshold=best_threshold,
        )

    table = pd.DataFrame(rows).set_index("model")

    best = table["validation_pr_auc"].idxmax()

    table.round(4).to_csv(results_dir / "metrics.csv")
    print("\n", table.round(4).to_string())

    plot_pr_curves(y_test, test_scores, results_dir / "pr_curves.png")

    print(f"\nMelhor modelo (PR-AUC na validação): {best}")
    joblib.dump(fitted[best], models_dir / "model.joblib")
    pd.DataFrame({"y_true": y_test.values, "y_score": test_scores[best]}).to_csv(
        results_dir / "test_scores.csv", index=False
    )
    (results_dir / "best_model.txt").write_text(best)
    print(f"Artefatos salvos em {results_dir} e {models_dir}")


if __name__ == "__main__":
    main()
