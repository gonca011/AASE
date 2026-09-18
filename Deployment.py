import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
main_table = "ScreenTimevsMentalWellness"
deploy_table = "Deployment_Output"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
deploy_csv = f"deployment_output_{timestamp}.csv"

# Pasta para artefactos
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================
# 1) Ler dados
# =========================
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
conn.close()

df = df.copy()
df["row_id"] = np.arange(1, len(df) + 1)

print("Linhas lidas:", len(df))
print("Colunas:", len(df.columns))

# =========================
# 2) Inputs (features) base
# =========================
needed_base = [
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
    "sleep_hours",
    "sleep_quality_1_5",
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "stress_level_0_10",
    "productivity_0_10",
]

missing = [c for c in needed_base if c not in df.columns]
if missing:
    raise ValueError(f"Faltam colunas necessárias na BD: {missing}")

df[needed_base] = df[needed_base].apply(pd.to_numeric, errors="coerce")
df[needed_base] = df[needed_base].fillna(df[needed_base].mean(numeric_only=True))

target = "mental_wellness_index_0_10"
has_target = target in df.columns
if has_target:
    df[target] = pd.to_numeric(df[target], errors="coerce")

# =========================
# 3) Regressão (LinearRegression)
# =========================
X_reg = df[needed_base].copy()

reg_model = None
df["mw_pred_reg_lin_raw"] = np.nan  # diagnóstico opcional

if has_target and df[target].notna().any():
    df_train_reg = df[df[target].notna()].copy()
    X_train_reg = df_train_reg[needed_base]
    y_train_reg = df_train_reg[target]

    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)

    df["mw_pred_reg_lin_raw"] = reg_model.predict(X_reg)

    # CORREÇÃO: clipping 0–10
    df["mw_pred_reg_lin"] = df["mw_pred_reg_lin_raw"].clip(0.0, 10.0)
else:
    df["mw_pred_reg_lin"] = np.nan
    print("Aviso: não foi possível treinar regressão (target ausente ou sem valores).")

# =========================
# 4) Classificação (RandomForest)
# =========================
labels = ["Baixo", "Medio", "Alto"]
df["mw_class_true"] = np.nan

clf_model = None

# garantir que as colunas de probabilidade existem sempre
df["mw_class_pred_rf"] = np.nan
df["mw_prob_Baixo"] = np.nan
df["mw_prob_Medio"] = np.nan
df["mw_prob_Alto"] = np.nan

if has_target and df[target].notna().any():
    bins = [-0.1, 3.0, 7.0, 10.0]
    df["mw_class_true"] = pd.cut(df[target], bins=bins, labels=labels, include_lowest=True)

    df_train_clf = df[df["mw_class_true"].notna()].copy()
    X_train_clf = df_train_clf[needed_base]
    y_train_clf = df_train_clf["mw_class_true"]

    clf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf_model.fit(X_train_clf, y_train_clf)

    df["mw_class_pred_rf"] = clf_model.predict(df[needed_base])

    # Probabilidades com mapeamento seguro por classe
    proba = clf_model.predict_proba(df[needed_base])
    class_order = list(clf_model.classes_)  # ordem real do modelo

    proba_map = {cls: proba[:, i] for i, cls in enumerate(class_order)}

    # CORREÇÃO: garantir que as 3 colunas existem sempre (0 se a classe não existir)
    df["mw_prob_Baixo"] = proba_map.get("Baixo", np.zeros(len(df)))
    df["mw_prob_Medio"] = proba_map.get("Medio", np.zeros(len(df)))
    df["mw_prob_Alto"]  = proba_map.get("Alto",  np.zeros(len(df)))
else:
    print("Aviso: não foi possível treinar classificação (target ausente ou sem valores).")

# =========================
# 5) Clustering Objetivo 1 (k=4)
# =========================
df["sleep_index"] = pd.to_numeric(df["sleep_hours"], errors="coerce") + pd.to_numeric(df["sleep_quality_1_5"], errors="coerce")
df["sleep_index"] = df["sleep_index"].fillna(df["sleep_index"].mean())

cluster_cols_obj1 = [
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "sleep_index",
    "stress_level_0_10",
    "productivity_0_10",
    "screen_time_hours",
]

X_clust1 = df[cluster_cols_obj1].copy()
X_clust1 = X_clust1.apply(pd.to_numeric, errors="coerce")
X_clust1 = X_clust1.fillna(X_clust1.mean(numeric_only=True))

scaler1 = StandardScaler()
X_clust1_scaled = scaler1.fit_transform(X_clust1)

kmeans1 = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_obj1_k4"] = kmeans1.fit_predict(X_clust1_scaled)

# =========================
# 6) Clustering Objetivo 2 (k=4)
# =========================
cluster_cols_obj2 = [
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
    "sleep_hours",
    "sleep_quality_1_5",
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "stress_level_0_10",
    "productivity_0_10",
    "occupation_Desempregado",
    "occupation_Empregado",
    "occupation_Estudante",
    "occupation_Reformado",
    "work_mode_Hybrid",
    "work_mode_In-person",
    "work_mode_Remote",
    "age_Jovem_16-25",
    "age_Adulto_25-50",
    "age_Senior_50-60",
]

cluster_cols_obj2 = [c for c in cluster_cols_obj2 if c in df.columns]
if len(cluster_cols_obj2) < 5:
    raise ValueError(
        f"Poucas colunas disponíveis para o clustering do objetivo 2: {cluster_cols_obj2}"
    )

X_clust2 = df[cluster_cols_obj2].copy()
X_clust2 = X_clust2.apply(pd.to_numeric, errors="coerce")
X_clust2 = X_clust2.fillna(X_clust2.mean(numeric_only=True))

scaler2 = StandardScaler()
# Treinar scaler com DataFrame (mantém feature_names_in_)
X_clust2_scaled = scaler2.fit_transform(X_clust2)

kmeans2 = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_obj2_k4"] = kmeans2.fit_predict(X_clust2_scaled)

# =========================
# 7) Dataset final de Deployment
# =========================
final_cols = [
    "row_id",
    *needed_base,

    # outputs principais
    "mw_pred_reg_lin",
    "mw_class_pred_rf",
    "mw_prob_Baixo",
    "mw_prob_Medio",
    "mw_prob_Alto",
    "cluster_obj1_k4",
    "cluster_obj2_k4",
]

# opcional: manter target real + classe real + reg raw 
if has_target:
    final_cols.insert(final_cols.index("mw_pred_reg_lin"), target)
    final_cols.insert(final_cols.index("mw_pred_reg_lin"), "mw_class_true")
    final_cols.insert(final_cols.index("mw_pred_reg_lin"), "mw_pred_reg_lin_raw")

final_cols = [c for c in final_cols if c in df.columns]
df_deploy = df[final_cols].copy()

# =========================
# 8) Exportar CSV e gravar tabela na BD
# =========================
df_deploy.to_csv(deploy_csv, index=False)

conn = sqlite3.connect(db_path)
df_deploy.to_sql(deploy_table, conn, if_exists="replace", index=False)
conn.close()

print("\nDeployment gerado com sucesso.")
print("CSV:", deploy_csv)
print("Tabela BD:", deploy_table)

# =========================
# 9) Guardar modelos (joblib)
# =========================
# Guardar com nomes "estáveis" (última versão) + opcional timestamp
REG_PATH = os.path.join(ARTIFACT_DIR, "model_regressao.pkl")
CLF_PATH = os.path.join(ARTIFACT_DIR, "model_classificacao.pkl")

K1_PATH = os.path.join(ARTIFACT_DIR, "kmeans_obj1_k4.pkl")
S1_PATH = os.path.join(ARTIFACT_DIR, "scaler_obj1.pkl") 

K2_PATH = os.path.join(ARTIFACT_DIR, "kmeans_obj2_k4.pkl")
S2_PATH = os.path.join(ARTIFACT_DIR, "scaler_obj2.pkl")

if reg_model is not None:
    joblib.dump(reg_model, REG_PATH)

if clf_model is not None:
    joblib.dump(clf_model, CLF_PATH)

joblib.dump(kmeans1, K1_PATH)
joblib.dump(scaler1, S1_PATH)

joblib.dump(kmeans2, K2_PATH)
joblib.dump(scaler2, S2_PATH)

print("\nModelos e scalers guardados com sucesso em:", ARTIFACT_DIR)
print(" -", REG_PATH)
print(" -", CLF_PATH)
print(" -", K1_PATH, "|", S1_PATH)
print(" -", K2_PATH, "|", S2_PATH)
