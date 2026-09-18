import sqlite3
import pandas as pd

# Caminho para o ficheiro .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Criar ligação
conn = sqlite3.connect(db_path)

# Ler tabela
df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

# Apagar a coluna 'user_id' se existir
if "user_id" in df.columns:
    df = df.drop(columns=["user_id"])
    print("Coluna 'user_id' removida com sucesso")
else:
    print("Coluna 'user_id' não encontrada.")

# Regravar a tabela sem a coluna
df.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)

# Fechar ligação
conn.close()
print("Tabela atualizada permanentemente no ficheiro .db")
