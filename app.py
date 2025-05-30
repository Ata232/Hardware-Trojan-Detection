# app.py  v3 ─────────────────────────────────────────────────────────────
import os, io, glob, base64, time, datetime as dt
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import streamlit as st
from joblib import load, dump
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score, average_precision_score)
from preprocessing import (clean, load_concat, build_pipelines, cross_val)
import shap, xlsxwriter, warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hardware-Trojan Detector",
                   page_icon="🛡️", layout="wide")
sns.set_style("whitegrid")

# ── utils ----------------------------------------------------------------
def list_models():
    return sorted(glob.glob("models/*.joblib"))

@st.cache_resource(show_spinner=False)
def load_model(path):
    return load(path)

@st.cache_resource(show_spinner=False)
def cv_scores():
    if os.path.exists("scores.csv"):
        return pd.read_csv("scores.csv")
    df = clean(load_concat())
    X, y = df[[c for c in df.select_dtypes("number").columns if c != "Label"]], df["Label"]
    pipes = build_pipelines(X, y)
    with st.spinner("Calculating CV scores…"):
        scores = cross_val(pipes, X, y)
    pd.DataFrame(scores).to_csv("scores.csv", index=False)
    return pd.DataFrame(scores)

def to_xlsx(df):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as wr:
        df.to_excel(wr, index=False, sheet_name="results")
    return bio.getvalue()

# ── sidebar ---------------------------------------------------------------
st.sidebar.header("⚙️  Options")

# model selector
models_available = list_models()
sel_model = st.sidebar.selectbox("Select model", models_available, index=0)
MODEL = load_model(sel_model)
EXP_COLS = MODEL.named_steps["prep"].transformers_[0][2]

st.sidebar.markdown("---")
sample_btn = st.sidebar.button("Load example HEROdata2.xlsx")
uploaded   = st.sidebar.file_uploader("📤  Upload CSV / XLSX", type=["csv", "xlsx"])
threshold  = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
show_shap  = st.sidebar.checkbox("Show SHAP (slow)")

st.sidebar.markdown("---")
# retrain section
with st.sidebar.expander("🔄  Retrain model"):
    retrain_file = st.file_uploader("Labeled CSV/XLSX", type=["csv", "xlsx"],
                                    key="retrain")
    if st.button("🚀 Retrain", key="train_btn") and retrain_file:
        ext = os.path.splitext(retrain_file.name)[1].lower()
        df_r = pd.read_csv(retrain_file) if ext==".csv" \
               else pd.read_excel(retrain_file, engine="openpyxl")
        df_r = clean(df_r)
        Xr, yr = df_r[[c for c in df_r.select_dtypes("number").columns
                       if c != "Label"]], df_r["Label"]
        st.info(f"Training BalancedRandomForest on {len(df_r)} rows…")
        from sklearn.ensemble import BalancedRandomForestClassifier
        from imblearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        prep = ColumnTransformer(
            [("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler())
            ]), Xr.columns)], remainder="drop")
        brf = BalancedRandomForestClassifier(n_estimators=800, random_state=42)
        pipe = Pipeline([("prep", prep), ("clf", brf)])
        with st.spinner("Fitting…"):
            pipe.fit(Xr, yr)
        fname = f"models/custom_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        dump(pipe, fname)
        st.success(f"Saved new model ➜ {fname}")
        st.experimental_rerun()

# ── main tabs -------------------------------------------------------------
tabs = st.tabs(["🏠 Predict", "📊 EDA", "📈 CV Scores"])
tab_pred, tab_eda, tab_cv = tabs

# --- CV tab
with tab_cv:
    SCORES = cv_scores()
    st.dataframe(SCORES.style.format(precision=4), use_container_width=True)
    fig = plt.figure(figsize=(6,3))
    sns.barplot(data=SCORES, x="model", y="f1_macro", palette="mako")
    plt.title("Macro-F1"); plt.tight_layout(); st.pyplot(fig)

# --- EDA tab
with tab_eda:
    st.header("Exploratory Data Analysis")
    eda_file = uploaded or (sample_btn and "Data/HEROdata2.xlsx")
    if not eda_file:
        st.info("Upload a file first.")
    else:
        ext = ".xlsx" if isinstance(eda_file, str) else os.path.splitext(eda_file.name)[1]
        df_e = pd.read_excel(eda_file, engine="openpyxl") if ext==".xlsx" else pd.read_csv(eda_file)
        df_e = clean(df_e)
        num = df_e.select_dtypes("number")
        st.subheader("Summary stats")
        st.dataframe(num.describe().T.round(3), height=300)
        st.subheader("Correlation heatmap")
        plt.figure(figsize=(8,6))
        sns.heatmap(num.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.clf()

# --- Predict tab
with tab_pred:
    if sample_btn:
        uploaded = "Data/HEROdata2.xlsx"

    if not uploaded:
        st.info("⬅️  Upload a dataset to get predictions.")
        st.stop()

    # read file
    ext = ".xlsx" if isinstance(uploaded, str) else os.path.splitext(uploaded.name)[1].lower()
    df_raw = pd.read_excel(uploaded, engine="openpyxl") if ext==".xlsx" else pd.read_csv(uploaded)

    df = clean(df_raw.copy())

    # align columns
    missing = sorted(set(EXP_COLS) - set(df.columns))
    extra   = sorted(set(df.columns) - set(EXP_COLS) - {"Label"})
    for col in missing: df[col] = np.nan
    df = df.drop(columns=extra)
    if missing: st.warning(f"🛈 Added NaN for {len(missing)} missing cols")
    if extra:   st.info(f"⚠️ Dropped {len(extra)} unused cols")

    X = df[[c for c in EXP_COLS if c != "Label"]]

    st.subheader("Data preview"); st.dataframe(df.head(), use_container_width=True)

    # progress & prediction
    prog = st.progress(0, text="Predicting…"); start = time.time()
    proba = load_model(sel_model).predict_proba(X)[:,1]; prog.progress(50)
    pred  = (proba >= threshold).astype(int); prog.progress(100); prog.empty()
    st.success(f"Finished in {time.time()-start:.2f}s | threshold={threshold:.2f}")

    df_out = df.copy()
    df_out["Trojan_prob"] = np.round(proba,3)
    df_out["Pred"]        = pred
    st.dataframe(df_out[["Trojan_prob","Pred"]], use_container_width=True)

    # download buttons
    col1,col2 = st.columns(2)
    with col1:
        csv = df_out.to_csv(index=False).encode()
        st.download_button("⬇️ CSV", csv, "predictions.csv", mime="text/csv")
    with col2:
        st.download_button("⬇️ XLSX", to_xlsx(df_out),
                           "predictions.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # confusion & metrics
    if "Label" in df:
        cm = confusion_matrix(df["Label"], pred, labels=[0,1])
        ConfusionMatrixDisplay(cm, display_labels=["Clean","Trojan"]).plot(cmap="Blues")
        plt.tight_layout(); st.pyplot(plt.gcf()); plt.clf()
        f1  = f1_score(df["Label"], pred, average="macro")
        roc = roc_auc_score(df["Label"], proba)
        pr  = average_precision_score(df["Label"], proba)
        st.write(f"**Macro-F1:** {f1:.3f} | **ROC-AUC:** {roc:.3f} | **PR-AUC:** {pr:.3f}")

    # SHAP
    if show_shap:
        with st.spinner("Calculating SHAP…"):
            try:
                expl = shap.TreeExplainer(load_model(sel_model).named_steps["clf"])
                Xtr  = load_model(sel_model).named_steps["prep"].transform(X)
                shap_vals = expl.shap_values(Xtr)
                shap.summary_plot(shap_vals, Xtr, max_display=20, show=False)
                st.pyplot(plt.gcf()); plt.clf()
            except Exception as e:
                st.error(f"SHAP not supported: {e}")

# footer
st.caption("🛡️  Hardware-Trojan Detector · Bitirme Projesi 2025")
