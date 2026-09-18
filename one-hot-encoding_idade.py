import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np

DB_PATH = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
MAIN_TABLE = "ScreenTimevsMentalWellness"

def list_tables(conn):
    return pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn
    )["name"].tolist()

def parse_ts(name):
    try:
        ts = name.split("Backup_STvMW_")[1]
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except Exception:
        return datetime.min

conn = sqlite3.connect(DB_PATH)
try:
    # Procurar backups
    tables = list_tables(conn)
    backups = sorted(
        [t for t in tables if t.startswith("Backup_STvMW_")],
        key=parse_ts,
        reverse=True,
    )
    if not backups:
        raise RuntimeError("Não existem backups na BD com o padrão 'Backup_STvMW_'.")

    # Escolher o mais recente que tenha a coluna age
    chosen_backup = None
    df_src = None
    for t in backups:
        tmp = pd.read_sql_query(f"SELECT * FROM {t}", conn)
        if "age" in tmp.columns:
            chosen_backup = t
            df_src = tmp
            break
    if chosen_backup is None:
        raise RuntimeError("Nenhum backup contém a coluna 'age'.")

    print(f"Backup escolhido: {chosen_backup} | Linhas: {len(df_src)}")

    # Remover dummies antigas de idade, se existirem
    old_age_dummies = [c for c in df_src.columns if c.startswith("age_")]
    if old_age_dummies:
        df_src = df_src.drop(columns=old_age_dummies)
        print(f"Colunas dummies antigas removidas do backup: {old_age_dummies}")

    # Criar faixas: [16–25), [25–50), [50–60)
    bins = [16, 25, 50, 61]
    labels = ["Jovem_16-25", "Adulto_25-50", "Senior_50-60"]

    df_src["age_group"] = pd.cut(
        df_src["age"], bins=bins, labels=labels, right=False, include_lowest=True
    )

    # One-hot encoding
    df_final = pd.get_dummies(df_src, columns=["age_group"], prefix="age")

    # Remover a coluna original age
    if "age" in df_final.columns:
        df_final = df_final.drop(columns=["age"])
        print("Coluna 'age' removida do resultado final.")

    # Gravar a tabela final
    df_final.to_sql(MAIN_TABLE, conn, if_exists="replace", index=False)
    print(f"Tabela '{MAIN_TABLE}' atualizada a partir do backup '{chosen_backup}'.")

    # Mostrar distribuição das faixas
    dist = df_src["age_group"].value_counts().sort_index()
    print("Distribuição por faixas etárias:")
    print(dist.to_string())

    print("NaNs em age_group:", df_src["age_group"].isna().sum())
    print(df_src.loc[df_src["age_group"].isna(), ["age"]].head())


finally:
    conn.close()
