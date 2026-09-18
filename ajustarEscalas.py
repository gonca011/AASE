import sqlite3
import pandas as pd
from datetime import datetime

# Caminho para a base de dados
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
main_table = "ScreenTimevsMentalWellness"

conn = sqlite3.connect(db_path)

try:
    # 1) Ler a tabela principal
    df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
    print(f"Tabela '{main_table}' carregada com {len(df)} linhas e {len(df.columns)} colunas.")

    # 2) Criar backup antes da modificação
    backup_name = f"Backup_STvMW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    df.to_sql(backup_name, conn, if_exists="replace", index=False)
    print(f"Backup criado: {backup_name}")

    # 3) Converter as colunas desejadas (se existirem)
    cols_to_scale = ["mental_wellness_index_0_100", "productivity_0_100"]
    scaled = []
    for col in cols_to_scale:
        if col in df.columns:
            df[col] = (df[col] / 10).round(1)  
            scaled.append(col)
        else:
            print(f"A coluna '{col}' não existe na tabela e foi ignorada.")

    if scaled:
        print(f"As seguintes colunas foram convertidas para escala 0–10 e arredondadas: {scaled}")
    else:
        print("Nenhuma coluna encontrada para converter.")

    # 4) Renomear colunas para refletir a nova escala (opcional)
    rename_map = {
        "mental_wellness_index_0_100": "mental_wellness_index_0_10",
        "productivity_0_100": "productivity_0_10"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    print("Colunas renomeadas para refletir a nova escala (0–10).")

    # 5) Gravar a tabela atualizada
    df.to_sql(main_table, conn, if_exists="replace", index=False)
    print(f"Tabela '{main_table}' atualizada com sucesso.")

finally:
    conn.close()
