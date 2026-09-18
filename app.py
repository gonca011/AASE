import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Hotel Detox - Dashboard Deployment",
    layout="wide"
)

st.title("Hotel Detox — Dashboard de Deployment (Regressão + Classificação + Clustering)")
st.caption("Fonte: SQLite (Deployment_Output).")

# -----------------------------
# Sidebar: caminho da BD e opções
# -----------------------------
st.sidebar.header("Configuração")

db_path = st.sidebar.text_input(
    "Caminho da BD (SQLite)",
    r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
)

table_name = st.sidebar.text_input(
    "Tabela de Deployment",
    "Deployment_Output"
)

# -----------------------------
# Carregamento de dados (com cache)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

try:
    df = load_data(db_path, table_name)
except Exception as e:
    st.error(f"Erro ao ler a tabela '{table_name}' na BD: {e}")
    st.stop()

# -----------------------------
# Validações mínimas e limpeza
# -----------------------------
# Colunas esperadas (ajusta se o teu output tiver nomes diferentes)
required = [
    "mw_pred_reg_lin",
    "mw_class_pred_rf",
    "mw_prob_Baixo",
    "mw_prob_Medio",
    "mw_prob_Alto",
    "cluster_obj1_k4",
    "cluster_obj2_k4"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Colunas esperadas em falta: {missing}")

# Converter colunas numéricas comuns (se existirem)
num_cols_candidates = [
    "screen_time_hours", "work_screen_hours", "leisure_screen_hours",
    "sleep_hours", "sleep_quality_1_5", "exercise_minutes_per_week",
    "social_hours_per_week", "stress_level_0_10", "productivity_0_10",
    "mw_pred_reg_lin", "mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto",
]
for c in num_cols_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Sidebar: filtros
# -----------------------------
st.sidebar.header("Filtros")

# Filtro por classe prevista
if "mw_class_pred_rf" in df.columns:
    classes = sorted([x for x in df["mw_class_pred_rf"].dropna().unique()])
    selected_classes = st.sidebar.multiselect(
        "Classe prevista (mw_class_pred_rf)",
        options=classes,
        default=classes
    )
else:
    selected_classes = None

# Filtros por clusters
def cluster_filter(col_name: str):
    if col_name in df.columns:
        vals = sorted(df[col_name].dropna().unique())
        return st.sidebar.multiselect(col_name, options=vals, default=vals)
    return None

sel_c1 = cluster_filter("cluster_obj1_k4")
sel_c2 = cluster_filter("cluster_obj2_k4")

# Filtro por probabilidade de "Baixo" (risco)
risk_threshold = st.sidebar.slider(
    "Threshold de risco (mw_prob_Baixo >=)",
    min_value=0.0, max_value=1.0, value=0.60, step=0.05
)

# Filtro por intervalo de screen_time_hours (se existir)
if "screen_time_hours" in df.columns:
    min_st, max_st = float(df["screen_time_hours"].min()), float(df["screen_time_hours"].max())
    screen_range = st.sidebar.slider(
        "Intervalo de screen_time_hours",
        min_value=min_st, max_value=max_st,
        value=(min_st, max_st)
    )
else:
    screen_range = None

# Aplicar filtros
df_f = df.copy()

if selected_classes is not None:
    df_f = df_f[df_f["mw_class_pred_rf"].isin(selected_classes)]

if sel_c1 is not None:
    df_f = df_f[df_f["cluster_obj1_k4"].isin(sel_c1)]

if sel_c2 is not None:
    df_f = df_f[df_f["cluster_obj2_k4"].isin(sel_c2)]

if "mw_prob_Baixo" in df_f.columns:
    df_f["is_risk"] = df_f["mw_prob_Baixo"] >= risk_threshold
else:
    df_f["is_risk"] = False

if screen_range is not None and "screen_time_hours" in df_f.columns:
    df_f = df_f[(df_f["screen_time_hours"] >= screen_range[0]) & (df_f["screen_time_hours"] <= screen_range[1])]

# -----------------------------
# KPIs (topo)
# -----------------------------
st.subheader("KPIs (após filtros)")

colA, colB, colC, colD = st.columns(4)

colA.metric("Nº de registos", f"{len(df_f)}")

if "mw_pred_reg_lin" in df_f.columns:
    colB.metric("Bem-estar previsto (média)", f"{df_f['mw_pred_reg_lin'].mean():.2f}")
else:
    colB.metric("Bem-estar previsto (média)", "N/A")

if "mw_prob_Baixo" in df_f.columns:
    pct_risk = 100 * df_f["is_risk"].mean()
    colC.metric(f"% em risco (Prob_Baixo ≥ {risk_threshold:.2f})", f"{pct_risk:.1f}%")
else:
    colC.metric("% em risco", "N/A")

if "mw_class_pred_rf" in df_f.columns:
    top_class = df_f["mw_class_pred_rf"].value_counts().idxmax() if len(df_f) else "N/A"
    colD.metric("Classe mais frequente", str(top_class))
else:
    colD.metric("Classe mais frequente", "N/A")

st.divider()

# -----------------------------
# Layout: 2 colunas
# -----------------------------
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("Distribuição de classes previstas")
    if "mw_class_pred_rf" in df_f.columns and len(df_f):
        fig = px.histogram(df_f, x="mw_class_pred_rf", category_orders={"mw_class_pred_rf": ["Baixo", "Medio", "Alto"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados ou coluna mw_class_pred_rf indisponível.")

    st.subheader("Bem-estar previsto por cluster (Objetivo 2)")
    if "mw_pred_reg_lin" in df_f.columns and "cluster_obj2_k4" in df_f.columns and len(df_f):
        fig = px.box(df_f, x="cluster_obj2_k4", y="mw_pred_reg_lin")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados suficientes para boxplot.")

with right:
    st.subheader("Risco (Prob_Baixo) vs variáveis-chave")
    ycol = "mw_prob_Baixo" if "mw_prob_Baixo" in df_f.columns else None

    x_options = [c for c in ["screen_time_hours", "sleep_hours", "stress_level_0_10", "productivity_0_10"] if c in df_f.columns]
    x_choice = st.selectbox("Variável (eixo X)", options=x_options) if x_options else None

    if ycol and x_choice and len(df_f):
        fig = px.scatter(
        df_f,
        x=x_choice,
        y=ycol,
        color="mw_class_pred_rf" if "mw_class_pred_rf" in df_f.columns else None,
        hover_data=[c for c in ["row_id", "cluster_obj2_k4"] if c in df_f.columns]
    )


        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados suficientes para scatter.")

    st.subheader("Matriz de correlação (inputs + outputs)")
    corr_cols = [c for c in [
        "screen_time_hours", "sleep_hours", "stress_level_0_10",
        "productivity_0_10", "mw_pred_reg_lin", "mw_prob_Baixo"
    ] if c in df_f.columns]

    if len(corr_cols) >= 2 and len(df_f):
        corr = df_f[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem colunas numéricas suficientes para correlação.")

st.divider()

# -----------------------------
# Tabela de dados (para auditoria)
# -----------------------------
st.subheader("Tabela (amostra)")
st.dataframe(df_f.head(200), use_container_width=True)

# Download CSV filtrado
st.download_button(
    "Descarregar CSV (dados filtrados)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="deployment_filtrado.csv",
    mime="text/csv"
)
