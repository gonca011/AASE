import sqlite3
import pandas as pd
from datetime import datetime

# Caminho da base de dados
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Nome da tabela principal
main_table = "ScreenTimevsMentalWellness"

conn = sqlite3.connect(db_path)

try:
    # 1) Ler a tabela principal
    df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
    print(f"Tabela '{main_table}' carregada com {len(df)} linhas e {len(df.columns)} colunas.")

    # 2) Guardar backup antes de alterar
    backup_name = f"Backup_STvMW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    df.to_sql(backup_name, conn, if_exists="replace", index=False)
    print(f"Backup criado: {backup_name}")

    # 3) Remover a coluna 'occupation_grouped', se existir
    if "occupation_grouped" in df.columns:
        df = df.drop(columns=["occupation_grouped"])
        print("Coluna 'occupation_grouped' removida do resultado final.")
    else:
        print("A coluna 'occupation_grouped' não existe nesta tabela.")

    # 4) Gravar a versão limpa na tabela principal
    df.to_sql(main_table, conn, if_exists="replace", index=False)
    print(f"Tabela '{main_table}' atualizada sem a coluna 'occupation_grouped'.")

finally:
    conn.close()
