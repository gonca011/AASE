import sqlite3
import pandas as pd
from datetime import datetime

# Caminho para a base de dados
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Nome da tabela principal
main_table = "ScreenTimevsMentalWellness"

conn = sqlite3.connect(db_path)

try:
    # 1) Ler a tabela principal
    df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
    print(f"Tabela '{main_table}' carregada com {len(df)} linhas e {len(df.columns)} colunas.")

    # 2) Criar backup completo (mantém as colunas de género)
    backup_name = f"Backup_STvMW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    df.to_sql(backup_name, conn, if_exists="replace", index=False)
    print(f"Backup criado: {backup_name}")

    # 3) Identificar colunas relacionadas com género
    gender_cols = [c for c in df.columns if c.lower().startswith("gender")]
    if gender_cols:
        df = df.drop(columns=gender_cols)
        print(f"Colunas de género removidas: {gender_cols}")
    else:
        print("Não foram encontradas colunas relacionadas com género.")

    # 4) Gravar a tabela sem as colunas de género
    df.to_sql(main_table, conn, if_exists="replace", index=False)
    print(f"Tabela '{main_table}' atualizada sem colunas de género.")

finally:
    conn.close()
