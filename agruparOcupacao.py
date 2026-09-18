import sqlite3
import pandas as pd

# Caminho para o ficheiro .db
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Criar ligação
conn = sqlite3.connect(db_path)

# Ler a tabela
df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

# 🔧 Agrupar categorias da coluna 'occupation'
df["occupation_grouped"] = df["occupation"].replace({
    "Employed": "Empregado",
    "Self-employed": "Empregado",
    "Student": "Estudante",
    "Unemployed": "Desempregado",
    "Retired": "Reformado"
})


df = df.drop(columns=["occupation"])


df_encoded = pd.get_dummies(df, columns=["occupation_grouped"], prefix="occupation")

# Substituir a tabela na base de dados
df_encoded.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)

print("Coluna 'occupation' agrupada e tabela atualizada com sucesso")

# Fechar ligação
conn.close()
