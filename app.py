"""Dashboard Streamlit. Rode `python -m src.train` antes.

    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.evaluate import compute_metrics, plot_confusion

st.set_page_config(page_title="Detecção de Fraude", page_icon="💳", layout="wide")
st.title("💳 Detecção de fraude em cartão de crédito")

scores_path = Path("results/test_scores.csv")
metrics_path = Path("results/metrics.csv")
model_path = Path("models/model.joblib")

if not scores_path.exists():
    st.warning("Resultados não encontrados. Rode `python -m src.train` primeiro.")
    st.stop()

best = Path("results/best_model.txt").read_text() if Path("results/best_model.txt").exists() else "modelo"
scores = pd.read_csv(scores_path)

tab1, tab2, tab3 = st.tabs(["🎚️ Limiar de decisão", "📊 Comparação de modelos", "📁 Classificar CSV"])

with tab1:
    st.subheader(f"Modelo: {best}")
    st.caption("Ajuste o limiar: quanto menor, mais fraudes detectadas (Recall) e mais alarmes falsos (menor Precision).")
    thr = st.slider("Limiar de decisão", 0.01, 0.99, 0.50, 0.01)
    m = compute_metrics(scores["y_true"], scores["y_score"], thr)
    c = st.columns(4)
    c[0].metric("Precision", f"{m['precision']:.3f}")
    c[1].metric("Recall", f"{m['recall']:.3f}")
    c[2].metric("F1", f"{m['f1']:.3f}")
    c[3].metric("PR-AUC", f"{m['pr_auc']:.3f}")
    plot_confusion(scores["y_true"], scores["y_score"], f"Limiar = {thr:.2f}", "results/_tmp_confusion.png", thr)
    st.image("results/_tmp_confusion.png", width=420)

with tab2:
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path).set_index("model"), use_container_width=True)
    if Path("results/pr_curves.png").exists():
        st.image("results/pr_curves.png", width=600)

with tab3:
    st.caption("Envie um CSV com as colunas Time, V1..V28 e Amount (e opcionalmente Class).")
    up = st.file_uploader("CSV de transações", type="csv")
    if up and model_path.exists():
        data = pd.read_csv(up).drop(columns=["Class"], errors="ignore")
        model = joblib.load(model_path)
        data["fraud_score"] = model.predict_proba(data)[:, 1]
        data["suspeita"] = data["fraud_score"] >= st.slider("Limiar", 0.01, 0.99, 0.5, 0.01, key="thr2")
        st.write(f"{int(data['suspeita'].sum())} transações suspeitas de {len(data)}")
        st.dataframe(data.sort_values("fraud_score", ascending=False).head(100))
