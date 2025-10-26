import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# -------- 1) Charger tes données --------
# df doit contenir : bank_id, date, risk_rate (cible), et les features réglementaires
# Exemple : df = pd.read_csv("banks_panel.csv", parse_dates=["date"])
# Ici on suppose df déjà chargé :
# df = ...
df = pd.read_csv("bank_KM1.csv", sep=",")

# -------- 2) Préparer et featurer --------
def make_lags(group, cols, lags=(1,4)):
    g = group.sort_values("date").copy()
    for c in cols:
        for L in lags:
            g[f"{c}_l{L}"] = g[c].shift(L)
    return g

def make_rollings(group, cols, windows=(4,)):
    g = group.sort_values("date").copy()
    for c in cols:
        for w in windows:
            g[f"{c}_ma{w}"] = g[c].rolling(w, min_periods=1).mean()
    return g

def feature_engineering(df):
    df = df.copy()
    df = df.sort_values(["bank_id","date"])

    # Exemples de features dérivées (ajuste selon tes colonnes réelles)
    if set(["CET1_ratio","EU_7d","CCB","CCyB"]).issubset(df.columns):
        df["distance_CET1"] = df["CET1_ratio"] - (df["EU_7d"] + df["CCB"] + df["CCyB"])
    elif set(["CET1_ratio","EU_7d"]).issubset(df.columns):
        df["distance_CET1"] = df["CET1_ratio"] - df["EU_7d"]

    if set(["RWA","TotalAssets"]).issubset(df.columns):
        df["RWA_density"] = df["RWA"] / df["TotalAssets"]

    # Liste de colonnes sur lesquelles créer lags & moyennes
    base_cols = [c for c in [
        "CET1_ratio","Tier1_ratio","Total_ratio","Leverage_ratio","LCR","NSFR",
        "EU_7a","EU_7b","EU_7c","EU_7d","distance_CET1","RWA_density",
        "CET1","Tier1","TotalCapital","RWA"
    ] if c in df.columns]

    # Lags par banque
    df = df.groupby("bank_id", group_keys=False).apply(make_lags, cols=base_cols, lags=(1,4))
    # Moyennes mobiles courtes
    df = df.groupby("bank_id", group_keys=False).apply(make_rollings, cols=[c for c in base_cols if "ratio" in c or c in ["LCR","NSFR"]], windows=(4,))

    # Cible décalée (prédire T+1 à partir de T)
    df = df.sort_values(["bank_id","date"])
    df["risk_rate_t1"] = df.groupby("bank_id")["risk_rate"].shift(-1)

    # Supprimer lignes sans cible ou sans lags
    df = df.dropna(subset=["risk_rate_t1"])
    return df

# df = feature_engineering(df)

# -------- 3) Split temporel + pipeline --------
def train_backtest(df, feature_cols, n_splits=5, model_choice="ridge"):
    df = df.sort_values(["date","bank_id"]).copy()

    X = df[feature_cols]
    y = df["risk_rate_t1"]

    # Prétraitement numérique standard
    preproc = ColumnTransformer([
        ("num", StandardScaler(), feature_cols)
    ], remainder="drop")

    if model_choice == "ridge":
        model = Ridge(alpha=1.0, random_state=42)
    elif model_choice == "gbrt":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("model_choice must be 'ridge' or 'gbrt'")

    pipe = Pipeline([
        ("prep", preproc),
        ("model", model)
    ])

    # TimeSeriesSplit global (toutes banques mélangées temporellement) — robuste si les dates sont alignées
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_pred = np.zeros(len(df))
    metrics = []

    for fold, (tr, te) in enumerate(tscv.split(X, y), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred = pipe.predict(X.iloc[te])
        oof_pred[te] = y_pred

        mae = mean_absolute_error(y.iloc[te], y_pred)
        rmse = mean_squared_error(y.iloc[te], y_pred, squared=False)
        metrics.append({"fold": fold, "MAE": mae, "RMSE": rmse})

    # Fit final sur tout l'historique
    pipe.fit(X, y)

    metrics_df = pd.DataFrame(metrics)
    return pipe, metrics_df, oof_pred

# Exemple :
# feature_cols = [c for c in df.columns if c not in ["bank_id","date","risk_rate","risk_rate_t1"]]
# model, metrics_df, oof = train_backtest(df, feature_cols, n_splits=5, model_choice="gbrt")
# print(metrics_df.describe())

# -------- 4) Prévision dernière période par banque --------
def forecast_last_period(df, model, feature_cols):
    # On prédit le T+1 à partir de la DERNIÈRE ligne dispo par banque (features connues à T)
    last_rows = df.sort_values("date").groupby("bank_id").tail(1).copy()
    preds = model.predict(last_rows[feature_cols])
    out = last_rows[["bank_id","date"]].copy()
    out["pred_risk_rate_Tplus1"] = preds
    return out

# -------- 5) Importance des variables (permutation) --------
def perm_importance(df, model, feature_cols, n_repeats=10):
    X = df[feature_cols]
    y = df["risk_rate_t1"]
    # Récupérer l'étape modèle du pipeline
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    return pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)