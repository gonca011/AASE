import sqlite3
import pandas as pd
from datetime import datetime

db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
conn = sqlite3.connect(db_path)

# === 0) Ler tabela ===
df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

# === 0.1) Guardar backup imediato da tabela original ===
backup_name = f"Backup_STvMW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
df.to_sql(backup_name, conn, if_exists="replace", index=False)
print(f" Backup criado: {backup_name}")

# === 1) Identificar colunas dummies de work_mode ===
work_dummy_cols = [c for c in df.columns if c.lower().startswith("work_mode_")]

def rebuild_categorical_from_dummies(prefix_cols, labels_map):
    if not prefix_cols:
        return None, pd.DataFrame()
    X = df[prefix_cols].fillna(0).astype(int)
    row_sum = X.sum(axis=1)
    rebuilt = pd.Series(pd.NA, index=df.index, dtype="object")
    for col, label in labels_map.items():
        if col in X.columns:
            mask = (X[col] == 1) & (row_sum == 1)
            rebuilt.loc[mask] = label
    conflicts = df.loc[row_sum > 1, :].copy()
    return rebuilt, conflicts

# === 1.1) Mapas de nomes das dummies ===
labels_map_work = {}
for col in work_dummy_cols:
    base = col.split("work_mode_", 1)[-1]
    key = base.lower().replace("-", "").replace(" ", "")
    if key in ["remote"]:
        labels_map_work[col] = "Remote"
    elif key in ["hybrid"]:
        labels_map_work[col] = "Hybrid"
    elif key in ["inperson", "in_person", "in-person", "presencial"]:
        labels_map_work[col] = "In-person"
    else:
        labels_map_work[col] = base

# === 2) Reconstruir work_mode textual, se necessário ===
need_rebuild_work = ("work_mode" not in df.columns) or df["work_mode"].isna().all()
if need_rebuild_work and work_dummy_cols:
    work_mode_rebuilt, conflicts_work = rebuild_categorical_from_dummies(work_dummy_cols, labels_map_work)
    df["work_mode"] = work_mode_rebuilt
    if len(conflicts_work) > 0:
        conflicts_work.to_sql("Audit_WorkMode_Conflicts", conn, if_exists="replace", index=False)
        print(f" Conflitos work_mode detectados: {len(conflicts_work)} (tabela Audit_WorkMode_Conflicts)")
else:
    if "work_mode" in df.columns:
        df["work_mode"] = df["work_mode"].replace(
            ["", "None", "none", "N/A", "n/a", "NA", "na"], pd.NA
        )

# === 3) Reconstruir ou identificar occupation_grouped ===
def rebuild_occupation_from_dummies():
    if "occupation_grouped" in df.columns:
        return "occupation_grouped"
    if "occupation" in df.columns:
        return "occupation"

    occ_dummy_cols = [c for c in df.columns if c.lower().startswith("occupation_")]
    if not occ_dummy_cols:
        return None

    map_occ = {}
    for col in occ_dummy_cols:
        base = col.split("occupation_", 1)[-1]
        key = base.lower().replace("-", "").replace(" ", "")
        if key in ["empregado", "employed", "selfemployed", "self-employed"]:
            map_occ[col] = "Empregado"
        elif key in ["estudante", "student"]:
            map_occ[col] = "Estudante"
        elif key in ["desempregado", "unemployed"]:
            map_occ[col] = "Desempregado"
        elif key in ["reformado", "retired"]:
            map_occ[col] = "Reformado"
        else:
            map_occ[col] = base
    occ_series, conflicts = rebuild_categorical_from_dummies(occ_dummy_cols, map_occ)
    df["occupation_grouped"] = occ_series
    if len(conflicts) > 0:
        conflicts.to_sql("Audit_Occupation_Conflicts", conn, if_exists="replace", index=False)
        print(f"Conflitos occupation detectados: {len(conflicts)} (tabela Audit_Occupation_Conflicts)")
    return "occupation_grouped"

occ_col = rebuild_occupation_from_dummies()
if occ_col is None:
    raise ValueError("Não encontrei 'occupation' ou 'occupation_grouped' nem consegui reconstruir de dummies.")

# === 4) Corrigir regra: Desempregado/Reformado ⇒ work_mode = NULL ===
unemp_vals = {"Desempregado", "Unemployed"}
ret_vals   = {"Reformado", "Retired"}

mask_fix = df[occ_col].isin(list(unemp_vals | ret_vals))
audit = df.loc[mask_fix & df["work_mode"].notna(), :].copy()
if len(audit) > 0:
    audit.to_sql("Audit_OccWorkMode_BeforeFix", conn, if_exists="replace", index=False)
    print(f" Auditoria guardada: {len(audit)} casos com work_mode definido em {list(unemp_vals|ret_vals)}")

df.loc[mask_fix, "work_mode"] = pd.NA  # aplicar regra

# === 5) Remover dummies antigas e refazer ===
if work_dummy_cols:
    df = df.drop(columns=work_dummy_cols, errors="ignore")

df_clean = pd.get_dummies(df, columns=["work_mode"], prefix="work_mode")

# ===  6) Remover 'occupation_grouped' do resultado final ===
if "occupation_grouped" in df_clean.columns:
    df_clean = df_clean.drop(columns=["occupation_grouped"])
    print(" Coluna 'occupation_grouped' removida do resultado final (mantida nos backups/auditorias).")

# === 7) Gravar resultado final na BD ===
df_clean.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)

# === 8) Resumo rápido ===
summary = {
    "NULL_work_mode": int(df["work_mode"].isna().sum()),
    "Remote": int((df["work_mode"] == "Remote").sum()),
    "Hybrid": int((df["work_mode"] == "Hybrid").sum()),
    "In-person": int((df["work_mode"] == "In-person").sum())
}
print("Correção concluída e dummies refeitas.")
print("Resumo work_mode:", summary)

conn.close()
