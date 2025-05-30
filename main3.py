import os, seaborn as sns, matplotlib.pyplot as plt, shap, pandas as pd
from joblib import dump
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, f1_score,
                             roc_auc_score, average_precision_score)
from preprocessing import load_concat, clean, build_pipelines, cross_val, tune

# ─── 1. VERİ ────────────────────────────────────────────────────────────
df = clean(load_concat())
num_cols = [c for c in df.select_dtypes(include="number").columns
            if c != "Label"]          #  << düzeltme
X = df[num_cols]
y = df["Label"]
print(f"Rows:{len(df)} Clean:{(y==0).sum()} Trojan:{(y==1).sum()}")
# ─── 2. MODELLER & CV ──────────────────────────────────────────────────
pipes  = build_pipelines(X, y)
scores = cross_val(pipes, X, y)
print(pd.DataFrame(scores).round(4).to_string(index=False))

best_name = scores[0]["model"]
best_pipe = pipes[best_name]
best_model, best_pars = tune(best_pipe, X, y)
if best_pars:
    print("Best params:", best_pars)

# ─── 3. TAM VERİDE EĞİT & KAYDET ───────────────────────────────────────
best_model.fit(X, y)
os.makedirs("models", exist_ok=True)
dump(best_model, "models/best_model.joblib")

proba = best_model.predict_proba(X)[:, 1]
pred  = best_model.predict(X)
print("\nFULL  Macro-F1:",
      f"{f1_score(y, pred, average='macro'):.3f}",
      "| ROC-AUC:", f"{roc_auc_score(y, proba):.3f}",
      "| PR-AUC:",  f"{average_precision_score(y, proba):.3f}")

# ─── 4. GRAFİKLER ──────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# barplot
plt.figure(figsize=(6,3))
sns.barplot(data=pd.DataFrame(scores), x="model", y="f1_macro", palette="mako")
plt.title("Macro-F1 (5-fold Stratified)"); plt.tight_layout()
plt.savefig("plots/f1_bar.png")

# confusion
cm = confusion_matrix(y, pred, labels=[0,1])
ConfusionMatrixDisplay(cm, display_labels=["Clean","Trojan"])\
    .plot(cmap="Blues"); plt.tight_layout()
plt.savefig("plots/confusion.png")

# PR curve
pr, re, _ = precision_recall_curve(y, proba)
plt.figure(figsize=(4,4))
plt.step(re, pr, where="post")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (train)")
plt.tight_layout(); plt.savefig("plots/pr_curve.png")

# SHAP (sadece XGB ise)
if best_name == "XGB":
    expl = shap.TreeExplainer(best_model.named_steps["clf"])
    Xtr  = best_model.named_steps["prep"].transform(X)
    shap_vals = expl.shap_values(Xtr)
    shap.summary_plot(shap_vals, Xtr, max_display=20, show=False)
    plt.tight_layout(); plt.savefig("plots/shap_top20.png")

print("🖼️  plots/*  |  ✅  models/best_model.joblib")
