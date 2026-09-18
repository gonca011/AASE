import sqlite3
import pandas as pd

# Caminho para o ficheiro .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Criar ligação
conn = sqlite3.connect(db_path)

# Ler a tabela
df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

# Contar linhas antes
rows_before = len(df)

# Identificar linhas irregulares
irregular = df[(df["screen_time_hours"] + df["sleep_hours"]) > 24]
rows_irregular = len(irregular)

# Mostrar as linhas irregulares (opcional)
if rows_irregular > 0:
    print(" Linhas irregulares encontradas:")
    print(irregular[["screen_time_hours", "sleep_hours"]])
else:
    print(" Nenhuma linha irregular encontrada.")

# Remover as linhas irregulares
df_cleaned = df[(df["screen_time_hours"] + df["sleep_hours"]) <= 24]

# Contar linhas depois
rows_after = len(df_cleaned)

# Substituir a tabela no ficheiro .db
df_cleaned.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)

# Fechar ligação
conn.close()

print(f"\n Limpeza concluída com sucesso!")
print(f"Linhas antes: {rows_before}")
print(f"Linhas removidas: {rows_irregular}")
print(f"Linhas finais: {rows_after}")
