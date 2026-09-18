import sqlite3
import pandas as pd

# Path to the .db file
db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"

# Connect to database
conn = sqlite3.connect(db_path)

# Read the table
df = pd.read_sql_query("SELECT * FROM ScreenTimevsMentalWellness", conn)

# ---- ONE-HOT ENCODING for 'work_mode' ----
df_encoded = pd.get_dummies(df, columns=["work_mode"], prefix="work_mode")


# df_encoded = df_encoded.drop(columns=["work_mode_In-person"])

# Replace the table in the database
df_encoded.to_sql("ScreenTimevsMentalWellness", conn, if_exists="replace", index=False)

print("One-hot encoding applied to 'work_mode' and table updated successfully.")

# Close connection
conn.close()
