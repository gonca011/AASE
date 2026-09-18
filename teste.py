import sqlite3
import pandas as pd
import numpy as np

# =========================
# Config
# =========================
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
deploy_table = "Deployment_Output"

# =========================
# 1) Ler tabela de Deployment
# =========================
conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {deploy_table}", conn)
conn.close()

print("\n==============================")
print("CHECKLIST AUTOMÁTICO DEPLOYMENT")
print("==============================\n")

print("Linhas:", len(df))
print("Colunas:", len(df.columns))
print("")

# =========================
# 2) Verificação estrutural
# =========================
required_cols = [
    "mw_pred_reg_lin",
    "mw_class_pred_rf",
    "mw_prob_Baixo",
    "mw_prob_Medio",
    "mw_prob_Alto",
    "cluster_obj1_k4",
    "cluster_obj2_k4",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("❌ ERRO: Colunas obrigatórias em falta:", missing)
else:
    print("✅ Estrutura: todas as colunas obrigatórias existem.")

# =========================
# 3) Regressão – sanity check
# =========================
if "mw_pred_reg_lin" in df.columns:
    print("\n--- Regressão ---")
    print("Min:", df["mw_pred_reg_lin"].min())
    print("Max:", df["mw_pred_reg_lin"].max())

    invalid_reg = df[
        (df["mw_pred_reg_lin"] < -1) | (df["mw_pred_reg_lin"] > 11)
    ]
    if len(invalid_reg) == 0:
        print("✅ Valores de regressão dentro do intervalo esperado.")
    else:
        print("⚠️ Atenção:", len(invalid_reg), "valores fora do intervalo esperado.")

# =========================
# 4) Classificação – classes previstas
# =========================
valid_classes = {"Baixo", "Medio", "Alto"}
if "mw_class_pred_rf" in df.columns:
    invalid_classes = df[
        ~df["mw_class_pred_rf"].isin(valid_classes)
        & df["mw_class_pred_rf"].notna()
    ]
    if len(invalid_classes) == 0:
        print("\n✅ Classificação: todas as classes previstas são válidas.")
    else:
        print("\n❌ ERRO: Classes inválidas encontradas.")
        print(invalid_classes["mw_class_pred_rf"].value_counts())

# =========================
# 5) Probabilidades – soma ≈ 1
# =========================
print("\n--- Probabilidades ---")
if all(c in df.columns for c in ["mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto"]):
    prob_sum = (
        df["mw_prob_Baixo"] +
        df["mw_prob_Medio"] +
        df["mw_prob_Alto"]
    )

    bad_probs = df[(prob_sum < 0.99) | (prob_sum > 1.01)]
    if len(bad_probs) == 0:
        print("✅ Probabilidades somam aproximadamente 1.")
    else:
        print("⚠️ Atenção:", len(bad_probs), "linhas com soma de probabilidades anómala.")

# =========================
# 6) Clustering – valores válidos
# =========================
print("\n--- Clustering ---")
for c in ["cluster_obj1_k4", "cluster_obj2_k4"]:
    if c in df.columns:
        unique_vals = sorted(df[c].dropna().unique())
        print(f"{c} -> clusters:", unique_vals)

        if set(unique_vals).issubset({0,1,2,3}):
            print(f"✅ {c}: valores válidos.")
        else:
            print(f"⚠️ {c}: valores fora do esperado.")

# =========================
# 7) Distribuição dos clusters
# =========================
print("\nDistribuição cluster_obj1_k4:")
print(df["cluster_obj1_k4"].value_counts().sort_index())

print("\nDistribuição cluster_obj2_k4:")
print(df["cluster_obj2_k4"].value_counts().sort_index())

# =========================
# 8) Coerência entre regressão e classificação
# =========================
print("\n--- Coerência Regressão vs Classificação ---")

if all(c in df.columns for c in ["mw_pred_reg_lin", "mw_class_pred_rf"]):
    mismatch = df[
        ((df["mw_pred_reg_lin"] < 4) & (df["mw_class_pred_rf"] == "Alto")) |
        ((df["mw_pred_reg_lin"] > 7) & (df["mw_class_pred_rf"] == "Baixo"))
    ]

    print("Casos potencialmente incoerentes:", len(mismatch))
    if len(mismatch) > 0:
        print("Exemplo:")
        print(mismatch[[
            "mw_pred_reg_lin",
            "mw_class_pred_rf"
        ]].head())

# =========================
# 9) Resumo final rápido
# =========================
print("\n==============================")
print("RESUMO FINAL")
print("==============================")
print("Classes previstas:")
print(df["mw_class_pred_rf"].value_counts())

print("\nMédia do bem-estar previsto (regressão):",
      round(df["mw_pred_reg_lin"].mean(), 2))

print("\nChecklist concluído.")
