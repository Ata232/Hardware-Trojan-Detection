import glob, os, warnings, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from sklearn.ensemble import VotingClassifier
from imblearn.pipeline import Pipeline as ImbPipe
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

RS = 42
DATA_DIR = "Data"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# ─── 1. VERİ ──────────────────────────────────────────────────────────────
def load_concat(pattern="Data/*"):
    """
    Data klasöründeki CSV, XLSX ve eski XLS dosyalarını birleştir.
    Okunamayan veya desteklenmeyen dosyaları atlar.
    """
    frames = []
    for path in glob.glob(pattern):
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == ".csv":
                frames.append(pd.read_csv(path))
            elif ext == ".xlsx":
                # XLSX → openpyxl
                frames.append(pd.read_excel(path, engine="openpyxl"))
            elif ext == ".xls":
                # Eski XLS → xlrd (pandas otomatik seçer)
                frames.append(pd.read_excel(path, engine=None))
            else:
                print(f"🚫 Skipped {path} (unsupported ext)")
        except Exception as e:
            print(f"⚠️ {path} okunamadı → {e}")

    if not frames:
        raise RuntimeError("Hiçbir veri dosyası okunamadı!")

    return pd.concat(frames, ignore_index=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["Label"] = (df["Label"].astype(str)
                   .str.strip("'").str.strip()
                   .map({"Trojan Free": 0, "Trojan Infected": 1}))
    df = df.drop_duplicates()
    df = df.loc[:, df.nunique() > 1]      # constant drop
    df = df.drop(columns=[c for c in ["Circuit", "IP"] if c in df.columns],
                 errors="ignore")
    return df

# ─── 2. ÖN-İŞLEME ────────────────────────────────────────────────────────
def make_preprocessor(cols):
    return ColumnTransformer(
        [("num",
          Pipeline([
              ("imp", SimpleImputer(strategy="median")),
              ("sel", SelectKBest(f_classif, k=min(30, len(cols)))),
              ("sc",  StandardScaler())
          ]),
          cols)],
        remainder="drop"
    )

# ─── 3. MODELLER ─────────────────────────────────────────────────────────
def build_pipelines(X, y):
    ratio = (y == 0).sum() / (y == 1).sum()
    eas = EasyEnsembleClassifier(n_estimators=60, random_state=RS)
    brf = BalancedRandomForestClassifier(n_estimators=800, random_state=RS)
    xgb = XGBClassifier(n_estimators=600, learning_rate=0.05,
                        max_depth=6, subsample=0.8,
                        scale_pos_weight=ratio,
                        eval_metric="logloss", random_state=RS)
    vote = VotingClassifier([("eas", eas), ("brf", brf)],
                            voting="soft", n_jobs=-1)
    models = {"EasyEns": eas, "BalRF": brf, "XGB": xgb, "Vote": vote}
    pipes  = {}
    for n, m in models.items():
        pipes[n] = ImbPipe([("prep", make_preprocessor(X.columns)),
                            ("clf",  m)])
    return pipes

# ─── 4. DEĞERLENDİRME ────────────────────────────────────────────────────
def _scorers():
    return {"f1_macro": make_scorer(f1_score, average="macro"),
            "roc_auc":  "roc_auc",
            "pr_auc":   make_scorer(average_precision_score)}

def cross_val(pipes, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    rows = []
    for n, p in tqdm(pipes.items(), desc="CV",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        s = cross_validate(p, X, y, cv=cv, scoring=_scorers(), n_jobs=-1)
        rows.append({"model": n,
                     "f1_macro": s["test_f1_macro"].mean(),
                     "roc_auc":  s["test_roc_auc"].mean(),
                     "pr_auc":   s["test_pr_auc"].mean()})
    return sorted(rows, key=lambda d: d["f1_macro"], reverse=True)

def tune(pipe, X, y):
    grid = {"clf__n_estimators": [40, 60, 80]} if "EasyEns" in pipe.steps[-1][0] else {}
    if not grid:
        return pipe, {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
    rs = RandomizedSearchCV(pipe, grid, n_iter=len(grid), cv=cv,
                            scoring="f1_macro", random_state=RS, n_jobs=-1)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_
