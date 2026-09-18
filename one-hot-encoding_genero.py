import sqlite3
import pandas as pd

# Caminho para o  ficheiro .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Criar ligação
conn = sqlite3.connect(db_path)

# Ver tabelas existentes
print(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn))

df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

df_encoded = pd.get_dummies(df, columns=["gender"], prefix="gender")
 # One-hot encoding

# Substituir a tabela existente
df_encoded.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)
print("Tabela atualizada com sucesso.")

# Fechar a ligação
conn.close()





