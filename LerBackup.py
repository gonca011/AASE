import sqlite3
import pandas as pd
from datetime import datetime
import os

# 0) Caminho do .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
print("DB path:", os.path.abspath(db_path), "| Exists:", os.path.exists(db_path))

conn = sqlite3.connect(db_path)

try:
    # 1) Listar tabelas existentes
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)["name"].tolist()
    print("Tabelas na BD:", tables)

    # 2) Se a principal não existe, procurar backups que criámos antes
    main_table = "ScreenTimevsMentalWellness"
    if main_table not in tables:
        # Filtrar tabelas de backup no formato Backup_STvMW_YYYYMMDD_HHMMSS
        backup_tables = [t for t in tables if t.startswith("Backup_STvMW_")]
        if not backup_tables:
            raise RuntimeError("Não encontrei a tabela principal nem backups do padrão 'Backup_STvMW_'. Estás no .db certo?")

        # Ordenar por timestamp no nome e escolher o mais recente
        def ts_key(name):
            # name ex.: Backup_STvMW_20251020_232537
            try:
                ts = name.split("Backup_STvMW_")[1]
                return datetime.strptime(ts, "%Y%m%d_%H%M%S")
            except Exception:
                return datetime.min

        backup_tables.sort(key=ts_key, reverse=True)
        latest_backup = backup_tables[0]
        print("Backup mais recente encontrado:", latest_backup)

        # 3) Restaurar a partir do backup
        df_backup = pd.read_sql_query(f"SELECT * FROM {latest_backup}", conn)
        df_backup.to_sql(main_table, conn, if_exists="replace", index=False)
        print(f"Tabela '{main_table}' restaurada a partir de '{latest_backup}'.")

    # 4) Verificar leitura da tabela restaurada
    df = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {main_table}", conn)
    print(f"Tabela '{main_table}' OK. Linhas:", int(df['n'][0]))

finally:
    conn.close()
