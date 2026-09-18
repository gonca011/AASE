import sqlite3
import pandas as pd

# Caminho para o  ficheiro .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

conn = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

df = df.drop(columns=["Unnamed: 15"])

df.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)
print("Coluna 'Unnamed: 15' removida com sucesso.")

conn.close()