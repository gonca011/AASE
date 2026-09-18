import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Hotel Detox - Painel de Bem-Estar", layout="wide")

# -------------------------
# Defaults (SQLite)
# -------------------------
DB_DEFAULT = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
TABLE_DEFAULT = "Deployment_Output"

NUM_COLS = [
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
    "sleep_hours",
    "sleep_quality_1_5",
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "stress_level_0_10",
    "productivity_0_10",
    "mental_wellness_index_0_10",  # pode não existir
    "mw_pred_reg_lin",
    "mw_prob_Baixo",
    "mw_prob_Medio",
    "mw_prob_Alto",
]

CLUST_COLS = ["cluster_obj1_k4", "cluster_obj2_k4"]
CLASS_COLS = ["mw_class_true", "mw_class_pred_rf"]

# Sugestões por segmento — manter alinhado com HotelDetoxSegmentos.py
CLUSTER_LABELS_OBJ2 = {
    0: "Stress elevado (moderado)",
    1: "Intermédio (estável mas vulnerável)",
    2: "Risco elevado (hiperconectado + stress alto)",
    3: "Equilibrado / Referência saudável",
}

MEASURES_BY_CLUSTER_OBJ2 = {
    0: [
        "Gestão de stress: mindfulness + respiração guiada (10–15 min/dia)",
        "Rotina de sono consistente + higiene do sono (sem ecrãs 60 min antes)",
        "Redução do ecrã noturno + regras de uso (limitar lazer digital)",
        "Atividade física leve/moderada regular (caminhada + sessão guiada 2×/semana)",
        "Check-in semanal para monitorizar stress e hábitos",
    ],
    1: [
        "Plano equilibrado: sono + exercício moderado + atividades sociais presenciais",
        "Intervenções leves: mindfulness curta + educação digital (reduzir scrolling)",
        "Rotina de foco (Pomodoro) para reduzir carga cognitiva",
        "Metas pequenas e consistentes (gamificação) com acompanhamento semanal",
    ],
    2: [
        "Detox digital estruturado: janelas sem ecrãs + bloqueio de notificações",
        "Intervenção anti-stress intensiva: sauna/massagem + mindfulness diário",
        "Plano físico progressivo (caminhada diária + 2 sessões guiadas/semana)",
        "Rotina de sono completa: higiene do sono + otimização do quarto (blackout/temperatura)",
        "Atividades sociais presenciais para reduzir dependência digital",
        "Check-ins frequentes (ex.: diário curto) para reforço comportamental",
    ],
    3: [
        "Manutenção: rotina equilibrada (sono consistente + atividade física moderada)",
        "Oferta premium: experiências avançadas (natureza + mindfulness + workshops)",
        "Programa de fidelização: benefícios para repetição/upgrade de estadia",
        "Partilha de boas práticas e metas de manutenção (check-in mensal)",
    ],
}

CLUSTER_LABELS_OBJ1 = {
    0: "Crítico: stress extremo + sono fraco + ecrãs altos + baixo exercício/social",
    1: "Físico forte, social baixo: exercício alto mitiga parcialmente stress",
    2: "Perfil ideal: sono muito bom + stress baixo + ecrãs baixos + produtividade alta",
    3: "Social alto mas stress alto e sono fraco: bem-estar baixo",
}

MEASURES_BY_CLUSTER_OBJ1 = {
    0: [
        "Intervenção prioritária: programa anti-stress intensivo (mindfulness diário + respiração + relaxamento guiado)",
        "Plano de sono estruturado (higiene do sono + redução total de ecrãs 60–90 min antes de dormir + rotina consistente)",
        "Detox digital forte (janelas sem ecrãs; limitar lazer digital; bloqueio de notificações)",
        "Plano de atividade física progressivo (iniciar com caminhada diária + 2 sessões leves/semana)",
        "Plano de socialização assistida (atividades em grupo pequenas + refeições partilhadas sem ecrãs)",
        "Acompanhamento frequente (check-ins curtos 2–3×/semana) e monitorização de stress",
    ],
    3: [
        "Foco principal: reduzir stress (protocolos de relaxamento; sauna/massagem; mindfulness diário)",
        "Reforçar sono (higiene do sono; otimização do quarto; consistência horários)",
        "Reequilibrar socialização: manter social, mas com atividades de baixa estimulação (natureza, caminhada, yoga em grupo)",
        "Limitar ecrãs à noite (regras de uso e substituição por rotinas relaxantes)",
        "Micro-pausas estruturadas e redução de multitasking",
    ],
    1: [
        "Manter exercício (é um ativo do segmento): plano de manutenção e prevenção de lesão",
        "Aumentar socialização de forma incremental (atividades em grupo 2×/semana; social hour; experiências partilhadas)",
        "Intervenções leves de stress (mindfulness curta; educação sobre gestão de agenda e carga cognitiva)",
        "Ajustes no sono (ritual noturno simples; redução de ecrãs noturnos; exposição à luz natural pela manhã)",
        "Coaching de foco e produtividade (blocos de trabalho profundo + pausas)",
    ],
    2: [
        "Manutenção do estilo de vida: reforçar hábitos atuais (sono consistente + exercício + socialização saudável)",
        "Oferta premium: experiências avançadas (natureza, mindfulness, workshops de performance sustentável)",
        "Plano de prevenção: check-in semanal de stress/hábitos digitais; manutenção de limites de ecrã",
        "Fidelização: benefícios e recomendações personalizadas para manter o perfil ideal",
    ],
}

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def wellness_rate(series, threshold):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return 100.0 * (s >= threshold).mean()


def segment_suggestion(seg_col: str, seg_value):
    """Devolve etiqueta e recomendações do segmento (alinhado com HotelDetoxSegmentos.py)."""
    try:
        seg_int = int(seg_value)
    except Exception:
        seg_int = seg_value

    if seg_col == "cluster_obj2_k4":
        label = CLUSTER_LABELS_OBJ2.get(seg_int, f"Cluster {seg_int}")
        measures = MEASURES_BY_CLUSTER_OBJ2.get(seg_int, [])
        return label, measures

    if seg_col == "cluster_obj1_k4":
        label = CLUSTER_LABELS_OBJ1.get(seg_int, f"Cluster {seg_int}")
        measures = MEASURES_BY_CLUSTER_OBJ1.get(seg_int, [])
        return label, measures

    return f"Segmento {seg_value}", []


@st.cache_data(show_spinner=True)
def load_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df_ = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df_


@st.cache_resource(show_spinner=True)
def load_regression_model(model_path: str):
    """Carregar modelo de regressão para previsão de bem-estar."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_wellness(model, values_dict):
    """Prever bem-estar com base nos valores ajustados."""
    if model is None:
        return None
    
    try:
        # Features esperadas pela regressão
        needed_base = [
            "screen_time_hours",
            "work_screen_hours",
            "leisure_screen_hours",
            "sleep_hours",
            "sleep_quality_1_5",
            "exercise_minutes_per_week",
            "social_hours_per_week",
            "stress_level_0_10",
            "productivity_0_10",
        ]
        
        df_input = pd.DataFrame([values_dict])
        X = df_input[[c for c in needed_base if c in df_input.columns]].fillna(0)
        
        # Garantir que todas as colunas existem
        for c in needed_base:
            if c not in X.columns:
                X[c] = 0.0
        
        X = X[needed_base]
        pred = float(model.predict(X)[0])
        return float(np.clip(pred, 0.0, 10.0))
    except Exception:
        return None


def predict_class(model, values_dict):
    """Prever classe de bem-estar com base nos valores ajustados."""
    if model is None:
        return None
    
    try:
        # Features esperadas pela classificação
        needed_base = [
            "screen_time_hours",
            "work_screen_hours",
            "leisure_screen_hours",
            "sleep_hours",
            "sleep_quality_1_5",
            "exercise_minutes_per_week",
            "social_hours_per_week",
            "stress_level_0_10",
            "productivity_0_10",
        ]
        
        df_input = pd.DataFrame([values_dict])
        X = df_input[[c for c in needed_base if c in df_input.columns]].fillna(0)
        
        # Garantir que todas as colunas existem
        for c in needed_base:
            if c not in X.columns:
                X[c] = 0.0
        
        X = X[needed_base]
        pred_class = str(model.predict(X)[0])
        return pred_class
    except Exception:
        return None


def predict_class_with_proba(model, values_dict):
    """Prever classe e probabilidades."""
    default_probs = {"mw_prob_Baixo": np.nan, "mw_prob_Medio": np.nan, "mw_prob_Alto": np.nan}
    if model is None:
        return None, default_probs

    try:
        needed_base = [
            "screen_time_hours",
            "work_screen_hours",
            "leisure_screen_hours",
            "sleep_hours",
            "sleep_quality_1_5",
            "exercise_minutes_per_week",
            "social_hours_per_week",
            "stress_level_0_10",
            "productivity_0_10",
        ]

        df_input = pd.DataFrame([values_dict])
        X = df_input[[c for c in needed_base if c in df_input.columns]].fillna(0)
        for c in needed_base:
            if c not in X.columns:
                X[c] = 0.0
        X = X[needed_base]

        pred_class = str(model.predict(X)[0])

        prob_cols = default_probs.copy()
        try:
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            for cls_label, p in zip(classes, proba):
                if cls_label == "Baixo":
                    prob_cols["mw_prob_Baixo"] = float(p)
                elif cls_label == "Medio":
                    prob_cols["mw_prob_Medio"] = float(p)
                elif cls_label == "Alto":
                    prob_cols["mw_prob_Alto"] = float(p)
        except Exception:
            pass

        return pred_class, prob_cols
    except Exception:
        return None, default_probs


def merge_adjusted_rows(base_df: pd.DataFrame, updates_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Merge updated rows into base_df using id_col; keep other rows intact."""
    if id_col not in base_df.columns or id_col not in updates_df.columns:
        return base_df

    base_df = base_df.copy()
    updates_df = updates_df.copy()

    if "mw_pred_reg_lin_novo" in updates_df.columns:
        updates_df["mw_pred_reg_lin"] = updates_df["mw_pred_reg_lin_novo"]
    if "mw_class_pred_rf_novo" in updates_df.columns:
        updates_df["mw_class_pred_rf"] = updates_df["mw_class_pred_rf_novo"]

    updates_df = updates_df[[c for c in updates_df.columns if not c.endswith("_novo")]]

    base_df = base_df.set_index(id_col)
    updates_df = updates_df.set_index(id_col)
    base_df.update(updates_df)
    return base_df.reset_index()

# -------------------------
# Sidebar: Load + filters
# -------------------------
st.sidebar.title("Fonte de Dados")
db_path = st.sidebar.text_input("Caminho da Base de Dados", value=DB_DEFAULT)
table_name = st.sidebar.text_input("Tabela", value=TABLE_DEFAULT)

try:
    df = load_sqlite(db_path, table_name).copy()
except Exception as e:
    st.error(f"Erro a ler SQLite '{db_path}' / tabela '{table_name}': {e}")
    st.stop()

df = to_numeric(df, NUM_COLS)

# Derived
if "sleep_hours" in df.columns and "sleep_quality_1_5" in df.columns:
    df["sleep_index"] = df["sleep_hours"] + df["sleep_quality_1_5"]
else:
    df["sleep_index"] = np.nan

st.sidebar.markdown("---")
st.sidebar.subheader("Índice de Bem-Estar")

threshold = st.sidebar.slider(
    "Limiar para bem-estar elevado (0–10)",
    min_value=0.0,
    max_value=10.0,
    value=8.5,
    step=0.1,
    help="Valores acima deste limiar são considerados bem-estar elevado"
)

# Business Objective
st.sidebar.markdown("---")
st.sidebar.subheader("Metas de Sucesso")
st.sidebar.markdown("**Objetivo:** Atingir taxa mínima de bem-estar elevado")

target_wellness_pct = st.sidebar.number_input(
    "Meta: % mínima de clientes com bem-estar elevado",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=1.0,
    help="Percentual de clientes que devem atingir bem-estar acima do limiar"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

# Modelo de regressão (opcional para previsão na aba "Ajustar Cliente")
st.sidebar.markdown("---")
st.sidebar.subheader("Modelo de Bem-estar (Opcional)")
model_folder = st.sidebar.text_input(
    "Pasta do modelo de regressão (.pkl)",
    value=".",
    help="Deixe vazio ou '.' para a pasta atual. Usado para prever bem-estar ao ajustar clientes."
)
reg_model_filename = st.sidebar.text_input(
    "Nome do ficheiro (.pkl)",
    value="model_regressao.pkl"
)
cl_model_filename = st.sidebar.text_input(
    "Classificador (.pkl)",
    value="model_classificacao.pkl",
    help="Modelo de classificação para prever classe de bem-estar"
)

# Tentar carregar os modelos
regression_model = None
classifier_model = None
if model_folder and reg_model_filename:
    model_path = os.path.abspath(os.path.join(model_folder, reg_model_filename))
    regression_model = load_regression_model(model_path)
    if regression_model is not None:
        st.sidebar.success("✓ Regressão carregada")
    else:
        st.sidebar.info(f"ℹ Regressão não encontrada")

if model_folder and cl_model_filename:
    cl_path = os.path.abspath(os.path.join(model_folder, cl_model_filename))
    classifier_model = load_regression_model(cl_path)  # pode usar a mesma função
    if classifier_model is not None:
        st.sidebar.success("✓ Classificador carregado")
    else:
        st.sidebar.info(f"ℹ Classificador não encontrado")

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

df_f = df.copy()

# Cluster filters
for c in CLUST_COLS:
    if c in df_f.columns:
        vals = sorted(df_f[c].dropna().unique().tolist())
        sel = st.sidebar.multiselect(f"{c}", vals, default=vals)
        df_f = df_f[df_f[c].isin(sel)]

# Class filters
for c in CLASS_COLS:
    if c in df_f.columns:
        vals = sorted(df_f[c].dropna().unique().tolist())
        sel = st.sidebar.multiselect(f"{c}", vals, default=vals)
        df_f = df_f[df_f[c].isin(sel)]

# Numeric range filters
def add_range_filter(df_in: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if col not in df_in.columns:
        return df_in
    s = pd.to_numeric(df_in[col], errors="coerce")
    if s.dropna().empty:
        return df_in
    mn = float(s.min())
    mx = float(s.max())
    a, b = st.sidebar.slider(label, mn, mx, (mn, mx))
    return df_in[(df_in[col] >= a) & (df_in[col] <= b)]

df_f = add_range_filter(df_f, "screen_time_hours", "Tempo de ecrã (horas/dia)")
df_f = add_range_filter(df_f, "stress_level_0_10", "Nível de stress (0–10)")
df_f = add_range_filter(df_f, "sleep_hours", "Horas de sono")

# Available segment columns for suggestions and group planning
available_seg_cols = [c for c in CLUST_COLS if c in df_f.columns and df_f[c].notna().any()]

# -------------------------
# Main: KPIs
# -------------------------
st.title("Hotel Detox — Painel de Bem-Estar")
st.caption("📊 Monitoramento do índice de bem-estar dos clientes")

# O real pode não existir no deployment (depende se guardaste target)
has_real = "mental_wellness_index_0_10" in df_f.columns and df_f["mental_wellness_index_0_10"].notna().any()
has_pred = "mw_pred_reg_lin" in df_f.columns and df_f["mw_pred_reg_lin"].notna().any()

# -----------------------------
# KPIs Principais - Destaque
# -----------------------------
st.header("🎯 KPIs Principais do Projeto")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# KPI 0: Crescimento de clientes
with kpi_col1:
    st.metric(
        "KPI 0: Crescimento de Clientes",
        "+15% YoY",
        delta="⚠️ Objetivo estratégico",
        delta_color="off"
    )
    st.caption("Sem dados disponíveis para cálculo atual.")

# KPI 1.1: Clientes em risco
risk_count = len(df_f[
    (df_f['mw_prob_Baixo'] >= 0.5) |  # Usando threshold padrão
    (df_f['mw_class_pred_rf'] == 'Baixo')
])
risk_pct = (risk_count / len(df_f) * 100) if len(df_f) > 0 else 0

with kpi_col2:
    st.metric(
        "KPI 1.1: Clientes em Risco",
        f"{risk_pct:.1f}%",
        delta="❌ Elevado" if risk_pct > 20 else "✅ Aceitável",
        delta_color="inverse" if risk_pct > 20 else "normal"
    )
    st.caption(f"{risk_count} clientes identificados como em risco.")

# KPI 1.2: Bem-estar elevado
wellness_pct = wellness_rate(df_f["mw_pred_reg_lin"], threshold) if has_pred else 0

with kpi_col3:
    st.metric(
        "KPI 1.2: Bem-estar Elevado",
        f"{wellness_pct:.1f}%" if not np.isnan(wellness_pct) else "N/A",
        delta="❌ Muito baixo" if wellness_pct < 15 else "✅ Adequado",
        delta_color="inverse" if wellness_pct < 15 else "normal"
    )
    st.caption(f"Meta: ≥ {target_wellness_pct:.0f}% com bem-estar ≥ {threshold:.1f}.")

# KPI 2: Medidas por segmento
with kpi_col4:
    st.metric(
        "KPI 2: Medidas por Segmento",
        "Clusters OBJ1/OBJ2",
        delta="✅ Cumprido",
        delta_color="normal"
    )
    st.caption("Recomendações personalizadas implementadas.")

st.markdown("---")

k1, k2, k3, k4 = st.columns(4)

rate_real = wellness_rate(df_f["mental_wellness_index_0_10"], threshold) if has_real else np.nan
rate_pred = wellness_rate(df_f["mw_pred_reg_lin"], threshold) if has_pred else np.nan

# Initialize target achievement variables
real_meets_target = False
pred_meets_target = False

# KPI 1: Taxa Real com target
if not np.isnan(rate_real):
    real_meets_target = rate_real >= target_wellness_pct
    real_delta = rate_real - target_wellness_pct
    k1.metric(
        f"Bem-estar Elevado (Medido)",
        f"{rate_real:.1f}%",
        delta=f"{real_delta:+.1f}pp vs meta ({target_wellness_pct:.0f}%)",
        delta_color="normal",
        help=f"Clientes com bem-estar ≥ {threshold:.1f} | {'✓ Meta atingida' if real_meets_target else '✗ Abaixo da meta'}"
    )
else:
    k1.metric(f"Bem-estar Elevado (Medido)", "N/A")

# KPI 2: Taxa Prevista com target
if not np.isnan(rate_pred):
    pred_meets_target = rate_pred >= target_wellness_pct
    pred_delta = rate_pred - target_wellness_pct
    k2.metric(
        f"Bem-estar Elevado (Previsto)",
        f"{rate_pred:.1f}%",
        delta=f"{pred_delta:+.1f}pp vs meta ({target_wellness_pct:.0f}%)",
        delta_color="normal",
        help=f"Clientes com bem-estar ≥ {threshold:.1f} | {'✓ Meta atingida' if pred_meets_target else '✗ Abaixo da meta'}"
    )
else:
    k2.metric(f"Bem-estar Elevado (Previsto)", "N/A")

real_mean = df_f["mental_wellness_index_0_10"].mean() if has_real else np.nan
k3.metric("Índice Médio (Medido)", f"{real_mean:.2f}/10" if not np.isnan(real_mean) else "N/A")

k4.metric("Total de Clientes", f"{len(df_f)}")

# Status do objetivo de negócio
st.markdown("---")

# Determinar qual KPI usar para o status (priorizar real, depois previsto)
if not np.isnan(rate_real):
    current_rate = rate_real
    rate_type = "medido"
    meets_target = real_meets_target
elif not np.isnan(rate_pred):
    current_rate = rate_pred
    rate_type = "previsto"
    meets_target = pred_meets_target
else:
    current_rate = None
    rate_type = None
    meets_target = None

if current_rate is not None:
    if meets_target:
        st.success(f"✓ 🎯 Meta atingida! {current_rate:.1f}% dos clientes apresentam bem-estar elevado (meta: {target_wellness_pct:.0f}%)")
    else:
        gap = target_wellness_pct - current_rate
        st.error(f"✗ Meta não atingida: {current_rate:.1f}% com bem-estar elevado (meta: {target_wellness_pct:.0f}%, faltam {gap:.1f}pp). Recomenda-se intervenções adicionais.")
else:
    st.warning("⚠ Não há dados suficientes para avaliar o progresso em relação à meta")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral",
    "🔍 Fatores de Influência",
    "🎯 Classificação & Probabilidades",
    "👥 Segmentos & Perfis",
    "🧪 Ajustar Cliente",
    "🎯 Planeamento em Grupo"
])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        if has_real:
            fig = px.histogram(df_f, x="mental_wellness_index_0_10", nbins=20, title="Distribuição do Índice de Bem-Estar (Medido)")
            fig.update_layout(height=420, xaxis_title="Índice de Bem-estar (0–10)")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("📄 Dados de bem-estar medido não disponíveis nesta visualização.")

    with c2:
        if has_pred:
            fig2 = px.histogram(df_f, x="mw_pred_reg_lin", nbins=20, title="Distribuição do Índice de Bem-Estar (Previsto)")
            fig2.update_layout(height=420, xaxis_title="Índice de bem-estar previsto (0–10)")
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("📄 Predições de bem-estar não disponíveis nesta visualização.")

    st.subheader("🔄 Comparação: Medido vs Previsto")

    if has_real and has_pred:
        df_sc = df_f[["mental_wellness_index_0_10", "mw_pred_reg_lin"]].copy()
        df_sc["erro"] = df_sc["mw_pred_reg_lin"] - df_sc["mental_wellness_index_0_10"]

        # limites para linha y=x
        vmin = float(np.nanmin([df_sc["mental_wellness_index_0_10"].min(), df_sc["mw_pred_reg_lin"].min()]))
        vmax = float(np.nanmax([df_sc["mental_wellness_index_0_10"].max(), df_sc["mw_pred_reg_lin"].max()]))

        fig3 = px.scatter(
            df_sc,
            x="mental_wellness_index_0_10",
            y="mw_pred_reg_lin",
            color="erro",
            color_continuous_scale="RdBu",
            title="Índice Medido vs Índice Previsto",
            opacity=0.75
        )

        # linha perfeita y=x
        fig3.add_shape(
            type="line",
            x0=vmin, y0=vmin,
            x1=vmax, y1=vmax,
            line=dict(dash="dash")
        )

        fig3.update_layout(height=520, xaxis_title="Índice Medido", yaxis_title="Índice Previsto")
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("📄 Para esta comparação são necessários dados medidos e previstos.")


    st.subheader("📋 Amostra de Dados")
    st.dataframe(df_f.head(30), width='stretch')

# -------------------------
# Tab 2: Drivers
# -------------------------
with tab2:
    if not has_real:
        st.info("📄 Esta análise requer dados medidos de bem-estar.")
    else:
        st.subheader("🔍 Fatores que Influenciam o Bem-estar")

        driver_cols = [
            "screen_time_hours",
            "work_screen_hours",
            "leisure_screen_hours",
            "sleep_hours",
            "sleep_quality_1_5",
            "exercise_minutes_per_week",
            "social_hours_per_week",
            "stress_level_0_10",
            "productivity_0_10",
            "sleep_index",
        ]
        driver_cols = [c for c in driver_cols if c in df_f.columns]

        xcol = st.selectbox("Selecione a variável para análise", driver_cols, index=0 if driver_cols else None)

        if xcol:
            fig = px.scatter(
                df_f, x=xcol, y="mental_wellness_index_0_10",
                title=f"Índice de Bem-estar vs {xcol}",
                opacity=0.7
            )
            fig.update_layout(height=520, yaxis_title="Índice de Bem-estar (0–10)")
            st.plotly_chart(fig, width='stretch')

        st.markdown("**Força da correlação com bem-estar**")
        corr = []
        for c in driver_cols:
            s = pd.to_numeric(df_f[c], errors="coerce")
            t = pd.to_numeric(df_f["mental_wellness_index_0_10"], errors="coerce")
            corr.append((c, s.corr(t)))
        corr_df = pd.DataFrame(corr, columns=["variavel", "corr_pearson"]).sort_values("corr_pearson", ascending=False)
        st.dataframe(corr_df, width='stretch')

# -------------------------
# Tab 3: Classification
# -------------------------
with tab3:
    st.subheader("🎯 Classificação dos Níveis de Bem-estar")

    # Evitar crash se não existirem (depende do deployment)
    has_true_class = "mw_class_true" in df_f.columns and df_f["mw_class_true"].notna().any()
    has_pred_class = "mw_class_pred_rf" in df_f.columns and df_f["mw_class_pred_rf"].notna().any()

    c1, c2 = st.columns([1, 1.1])

    with c1:
        if has_true_class:
            ct = df_f["mw_class_true"].value_counts().reset_index()
            ct.columns = ["Nível", "Quantidade"]
            fig = px.pie(ct, names="Nível", values="Quantidade", title="Níveis de bem-estar (Medidos)")
            fig.update_layout(height=420)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("📄 Classificação medida não disponível.")

    with c2:
        if has_pred_class:
            cp = df_f["mw_class_pred_rf"].value_counts().reset_index()
            cp.columns = ["Nível", "Quantidade"]
            fig2 = px.pie(cp, names="Nível", values="Quantidade", title="Níveis de bem-estar (Previstos)")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("📄 Classificação prevista não disponível.")

    if all(c in df_f.columns for c in ["mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto"]):
        st.subheader("📊 Probabilidades Médias por Nível")
        probs = df_f[["mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto"]].mean(numeric_only=True).reset_index()
        probs.columns = ["Nível", "Probabilidade Média"]
        probs["Nível"] = probs["Nível"].str.replace("mw_prob_", "", regex=False)
        fig3 = px.bar(probs, x="Nível", y="Probabilidade Média", title="Probabilidade média por nível de bem-estar")
        fig3.update_layout(height=420, yaxis_title="Probabilidade média")
        st.plotly_chart(fig3, width='stretch')
    else:
        st.info("📄 Dados de probabilidade não disponíveis.")

    if has_pred_class and all(c in df_f.columns for c in ["stress_level_0_10", "productivity_0_10"]):
        st.subheader("🔍 Stress vs Produtividade por Nível de Bem-estar")
        fig4 = px.scatter(
            df_f, x="stress_level_0_10", y="productivity_0_10",
            color="mw_class_pred_rf",
            title="Relação entre Stress e Produtividade",
            opacity=0.7
        )
        fig4.update_layout(height=520)
        st.plotly_chart(fig4, width='stretch')

# -------------------------
# Tab 4: Clusters
# -------------------------
with tab4:
    st.subheader("👥 Perfis por Segmento de Clientes")

    available_clusters = [c for c in CLUST_COLS if c in df_f.columns]
    if not available_clusters:
        st.info("📄 Não existem segmentações disponíveis.")
    else:
        chosen_cluster = st.selectbox("Escolha o tipo de segmentação", available_clusters, index=0)

        profile_cols = [
            "screen_time_hours",
            "work_screen_hours",
            "leisure_screen_hours",
            "sleep_hours",
            "sleep_quality_1_5",
            "exercise_minutes_per_week",
            "social_hours_per_week",
            "stress_level_0_10",
            "productivity_0_10",
            "sleep_index",
        ]
        profile_cols = [c for c in profile_cols if c in df_f.columns]

        g = df_f.groupby(chosen_cluster)[profile_cols].mean(numeric_only=True).reset_index()

        st.markdown("**📊 Mapa de calor: Características médias por segmento**")
        heat = g.set_index(chosen_cluster)
        fig = px.imshow(heat, aspect="auto", title=f"Médias por {chosen_cluster}")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width='stretch')

        st.markdown("**🎯 Taxa de bem-estar elevado por segmento**")

        if has_real:
            rate_df = (df_f.groupby(chosen_cluster)["mental_wellness_index_0_10"]
                       .apply(lambda s: wellness_rate(s, threshold))
                       .reset_index(name="wellness_rate_pct"))
            title = f"% com bem-estar elevado (≥ {threshold:.1f}) por {chosen_cluster}"
        elif has_pred:
            rate_df = (df_f.groupby(chosen_cluster)["mw_pred_reg_lin"]
                       .apply(lambda s: wellness_rate(s, threshold))
                       .reset_index(name="wellness_rate_pct"))
            title = f"% com bem-estar elevado previsto (≥ {threshold:.1f}) por {chosen_cluster}"
        else:
            rate_df = None

        if rate_df is not None:
            fig2 = px.line(rate_df, x=chosen_cluster, y="wellness_rate_pct", markers=True, title=title)
            fig2.update_layout(height=420, yaxis_title="Taxa (%)")
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("📄 Dados insuficientes para calcular taxa por segmento.")

        st.subheader("📋 Tabela de perfis por segmento")
        st.dataframe(g, width='stretch')

# -----------------------------
# Tab 5: Ajustar Cliente
# -----------------------------
with tab5:
    st.subheader("🧪 Ajustar Cliente")
    
    # Sub-seções: Clientes em Risco e Ajuste por Segmento
    sub_tab1, sub_tab2 = st.tabs(["🚨 Clientes em Risco", "👥 Por Segmento"])
    
    with sub_tab1:
        st.markdown("**Identificar e ajustar clientes em risco**")
        
        # Filtros para clientes em risco
        risk_threshold_tab5 = st.slider("Limiar de Risco (mw_prob_Baixo ≥)", 0.0, 1.0, 0.5, key="risk_thresh_tab5")
        wellness_threshold_tab5 = st.slider("Limiar Bem-Estar Baixo (mw_pred_reg_lin <)", 0.0, 10.0, 6.0, key="wellness_thresh_tab5")
        
        # Identificar clientes em risco
        risk_clients = df_f[
            (df_f['mw_prob_Baixo'] >= risk_threshold_tab5) |
            ((df_f['mw_pred_reg_lin'] < wellness_threshold_tab5) & df_f['mw_pred_reg_lin'].notna()) |
            (df_f['mw_class_pred_rf'] == 'Baixo')
        ].copy()
        
        st.metric("Clientes em Risco Identificados", len(risk_clients))
        
        if len(risk_clients) > 0:
            # Ordenar por risco
            if 'mw_prob_Baixo' in risk_clients.columns:
                risk_clients = risk_clients.sort_values('mw_prob_Baixo', ascending=False)
            
            # Mostrar lista
            display_cols = ['row_id', 'mw_pred_reg_lin', 'mw_class_pred_rf', 'mw_prob_Baixo', 
                           'screen_time_hours', 'stress_level_0_10', 'sleep_hours']
            display_cols = [c for c in display_cols if c in risk_clients.columns]
            
            st.dataframe(risk_clients[display_cols].head(10), width='stretch')
            
            # Selecionar cliente
            if 'row_id' in risk_clients.columns:
                selected_risk_client = st.selectbox(
                    "Selecionar Cliente em Risco", 
                    risk_clients['row_id'].tolist(),
                    key="select_risk_client_tab5"
                )
                
                client_risk_data = risk_clients[risk_clients['row_id'] == selected_risk_client].iloc[0]
                
                # Sugestões baseadas no segmento
                seg_col_risk = None
                for c in CLUST_COLS:
                    if c in client_risk_data.index and pd.notna(client_risk_data[c]):
                        seg_col_risk = c
                        break
                
                if seg_col_risk:
                    label, measures = segment_suggestion(seg_col_risk, client_risk_data[seg_col_risk])
                    st.info(f"**Segmento:** {label}")
                    if measures:
                        st.markdown("**Recomendações:**")
                        for m in measures[:3]:
                            st.write(f"- {m}")
                
                # Ajustes detalhados (igual à aba original)
                st.markdown("---")
                st.subheader("Ajustar Valores Detalhadamente")
                
                # Store original values
                st.session_state["original_client_risk"] = {
                    "work_screen_hours": float(client_risk_data.get("work_screen_hours", 0.0) or 0.0),
                    "leisure_screen_hours": float(client_risk_data.get("leisure_screen_hours", 0.0) or 0.0),
                    "sleep_hours": float(client_risk_data.get("sleep_hours", 7.0) or 7.0),
                    "sleep_quality_1_5": int(client_risk_data.get("sleep_quality_1_5", 3) or 3),
                    "exercise_minutes_per_week": int(client_risk_data.get("exercise_minutes_per_week", 150) or 0),
                    "social_hours_per_week": int(client_risk_data.get("social_hours_per_week", 8) or 0),
                    "stress_level_0_10": int(client_risk_data.get("stress_level_0_10", 5) or 0),
                    "productivity_0_10": int(client_risk_data.get("productivity_0_10", 6) or 0),
                    "mw_pred_reg_lin": float(client_risk_data.get("mw_pred_reg_lin", np.nan) or np.nan),
                    "mw_class_pred_rf": str(client_risk_data.get("mw_class_pred_rf", "N/A") or "N/A"),
                }
                
                with st.form("ajustar_cliente_risk_form"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        work_screen = st.slider(
                            "Tempo de ecrã no trabalho (h/dia)",
                            0.0, 14.0, float(client_risk_data.get("work_screen_hours", 0.0) or 0.0), 0.1
                        )
                        leisure_screen = st.slider(
                            "Tempo de ecrã no lazer (h/dia)",
                            0.0, 14.0, float(client_risk_data.get("leisure_screen_hours", 0.0) or 0.0), 0.1
                        )
                        screen_total = work_screen + leisure_screen
                        st.caption(f"Tempo de ecrã total calculado: {screen_total:.1f} h/dia")

                    with col2:
                        sleep_hours = st.slider(
                            "Horas de sono", 3.0, 10.0, float(client_risk_data.get("sleep_hours", 7.0) or 7.0), 0.1
                        )
                        sleep_quality = st.slider(
                            "Qualidade do sono (1–5)", 1, 5, int(client_risk_data.get("sleep_quality_1_5", 3) or 3), 1
                        )
                        exercise = st.slider(
                            "Exercício (min/semana)", 0, 600, int(client_risk_data.get("exercise_minutes_per_week", 150) or 0), 10
                        )

                    with col3:
                        social = st.slider(
                            "Horas sociais/semana", 0, 40, int(client_risk_data.get("social_hours_per_week", 8) or 0), 1
                        )
                        stress = st.slider(
                            "Stress (0–10)", 0, 10, int(client_risk_data.get("stress_level_0_10", 5) or 0), 1
                        )
                        productivity = st.slider(
                            "Produtividade (0–10)", 0, 10, int(client_risk_data.get("productivity_0_10", 6) or 0), 1
                        )

                    submitted_risk = st.form_submit_button("Aplicar ajustes")

                if submitted_risk:
                    adjusted = {
                        "row_id": client_risk_data.get("row_id", selected_risk_client),
                        seg_col_risk: client_risk_data.get(seg_col_risk),
                        "work_screen_hours": work_screen,
                        "leisure_screen_hours": leisure_screen,
                        "screen_time_hours": screen_total,
                        "sleep_hours": sleep_hours,
                        "sleep_quality_1_5": sleep_quality,
                        "exercise_minutes_per_week": exercise,
                        "social_hours_per_week": social,
                        "stress_level_0_10": stress,
                        "productivity_0_10": productivity,
                        "sleep_index": sleep_hours + sleep_quality,
                    }

                    # Calcular previsão
                    adjusted_wellness = predict_wellness(regression_model, adjusted)

                    # Build comparison table
                    original = st.session_state.get("original_client_risk", {})
                    comparison_data = []
                    for key in [
                        "work_screen_hours", "leisure_screen_hours", "screen_time_hours",
                        "sleep_hours", "sleep_quality_1_5", "exercise_minutes_per_week",
                        "social_hours_per_week", "stress_level_0_10", "productivity_0_10",
                        "sleep_index"
                    ]:
                        orig_val = original.get(key, 0)
                        adj_val = adjusted.get(key, 0)
                        delta = adj_val - orig_val
                        delta_pct = (delta / orig_val * 100) if orig_val != 0 else 0.0
                        
                        if delta > 0.01:
                            indicator = "📈"
                        elif delta < -0.01:
                            indicator = "📉"
                        else:
                            indicator = "➡️"
                        
                        comparison_data.append({
                            "Métrica": key,
                            "Anterior": f"{orig_val:.1f}" if isinstance(orig_val, float) else str(orig_val),
                            "Novo": f"{adj_val:.1f}" if isinstance(adj_val, float) else str(adj_val),
                            "Mudança": f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}",
                            "Indicador": indicator,
                        })

                    # Adicionar bem-estar
                    orig_wellness = original.get("mw_pred_reg_lin", np.nan)
                    if adjusted_wellness is not None:
                        wellness_delta = adjusted_wellness - orig_wellness
                        if wellness_delta > 0.01:
                            wellness_indicator = "📈"
                        elif wellness_delta < -0.01:
                            wellness_indicator = "📉"
                        else:
                            wellness_indicator = "➡️"
                        
                        comparison_data.append({
                            "Métrica": "mw_pred_reg_lin (Bem-estar previsto)",
                            "Anterior": f"{orig_wellness:.2f}" if not np.isnan(orig_wellness) else "N/A",
                            "Novo": f"{adjusted_wellness:.2f}",
                            "Mudança": f"{wellness_delta:+.2f}",
                            "Indicador": wellness_indicator,
                        })

                    # Adicionar classe
                    orig_class = original.get("mw_class_pred_rf", "N/A")
                    if classifier_model is not None:
                        new_class = predict_class(classifier_model, adjusted)
                        class_indicator = "✓" if orig_class == new_class else "➡️ Mudança"
                        comparison_data.append({
                            "Métrica": "mw_class_pred_rf (Classe prevista)",
                            "Anterior": str(orig_class),
                            "Novo": str(new_class),
                            "Mudança": class_indicator if orig_class != new_class else "Mantém",
                            "Indicador": "→" if orig_class != new_class else "✓",
                        })

                    st.markdown("---")
                    st.subheader("📊 Comparação: Antes vs Depois")
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')

                    st.markdown("---")
                    st.markdown("**Pré-visualização do cliente ajustado**")
                    df_out = pd.DataFrame([adjusted])
                    st.dataframe(df_out, width='stretch')

                    # Store adjusted client
                    st.session_state["adjusted_client_risk"] = df_out
                    st.session_state["adjusted_client_id_risk"] = selected_risk_client

        # Persistent save section
        if "adjusted_client_risk" in st.session_state:
            st.markdown("---")
            col_csv, col_db = st.columns(2)
            
            with col_csv:
                df_out_persist = st.session_state["adjusted_client_risk"]
                csv_data = df_out_persist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descarregar cliente ajustado (CSV)",
                    data=csv_data,
                    file_name="cliente_risco_ajustado.csv",
                    mime="text/csv",
                    key="download_client_risk"
                )
            
            with col_db:
                save_status_client_risk = st.empty()
                
                if "save_status_client_risk" in st.session_state:
                    msg = st.session_state["save_status_client_risk"]
                    if msg.get("type") == "success":
                        save_status_client_risk.success(msg.get("message", ""))
                    elif msg.get("type") == "error":
                        save_status_client_risk.error(msg.get("message", ""))
                        if msg.get("traceback"):
                            st.code(msg["traceback"])
                
                if st.button("💿 Guardar cliente na Base de Dados", key="save_client_risk_persistent"):
                    df_out = st.session_state["adjusted_client_risk"]
                    id_col_client = "row_id" if "row_id" in df_out.columns else None

                    st.session_state["save_status_client_risk"] = {"type": "info", "message": "🔄 A processar..."}
                    save_status_client_risk.info("🔄 A processar...")

                    try:
                        if not os.path.exists(db_path):
                            st.session_state["save_status_client_risk"] = {
                                "type": "error",
                                "message": f"❌ Base de dados não encontrada: {db_path}"
                            }
                            save_status_client_risk.error(st.session_state["save_status_client_risk"]["message"])
                        elif not id_col_client:
                            st.session_state["save_status_client_risk"] = {
                                "type": "error",
                                "message": "❌ Não foi possível guardar: coluna row_id em falta."
                            }
                            save_status_client_risk.error(st.session_state["save_status_client_risk"]["message"])
                        else:
                            conn = sqlite3.connect(db_path)
                            try:
                                try:
                                    base_df = pd.read_sql_query("SELECT * FROM Deployment_Output", conn)
                                except Exception:
                                    try:
                                        base_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                                    except Exception:
                                        base_df = df_out.copy()

                                try:
                                    adjusted_df = pd.read_sql_query("SELECT * FROM Deployment_Output_Adjusted", conn)
                                except Exception:
                                    adjusted_df = base_df.copy()

                                current_df = merge_adjusted_rows(base_df, adjusted_df, id_col_client)
                                merged = merge_adjusted_rows(current_df, df_out, id_col_client)
                                merged.to_sql("Deployment_Output_Adjusted", conn, if_exists="replace", index=False)
                                conn.commit()
                                st.session_state["save_status_client_risk"] = {
                                    "type": "success",
                                    "message": f"✅ Cliente guardado na tabela 'Deployment_Output_Adjusted'. Tabela completa com {len(merged)} linha(s)."
                                }
                                save_status_client_risk.success(st.session_state["save_status_client_risk"]["message"])
                            finally:
                                conn.close()
                    except Exception as e:
                        import traceback
                        st.session_state["save_status_client_risk"] = {
                            "type": "error",
                            "message": f"❌ Erro ao guardar cliente: {e}",
                            "traceback": traceback.format_exc()
                        }
                        save_status_client_risk.error(st.session_state["save_status_client_risk"]["message"])
                        st.code(st.session_state["save_status_client_risk"]["traceback"])
    
    with sub_tab2:
        seg_col = st.selectbox("Segmentação para sugestões", available_seg_cols, index=0)
        seg_values = sorted(df_f[seg_col].dropna().unique().tolist())
        seg_value = st.selectbox("Segmento", seg_values, index=0)

        df_seg = df_f[df_f[seg_col] == seg_value]
        if df_seg.empty:
            st.info("📄 Nenhum cliente neste segmento com os filtros atuais.")
        else:
            id_col = "row_id" if "row_id" in df_seg.columns else None
            client_options = df_seg[id_col].tolist() if id_col else df_seg.index.tolist()
            client_id = st.selectbox(
                "Cliente",
                client_options,
                format_func=lambda x: f"Cliente {x}"
            )

            if id_col:
                client_row = df_seg[df_seg[id_col] == client_id].iloc[0]
            else:
                client_row = df_seg.loc[client_id]

            label, measures = segment_suggestion(seg_col, seg_value)
            st.success(f"Sugestão do segmento: {label}")
            if measures:
                st.markdown("**Recomendações principais do segmento**")
                for m in measures:
                    st.markdown(f"- {m}")

            # Store original values for comparison
            st.session_state["original_client"] = {
                "work_screen_hours": float(client_row.get("work_screen_hours", 0.0) or 0.0),
                "leisure_screen_hours": float(client_row.get("leisure_screen_hours", 0.0) or 0.0),
                "sleep_hours": float(client_row.get("sleep_hours", 7.0) or 7.0),
                "sleep_quality_1_5": int(client_row.get("sleep_quality_1_5", 3) or 3),
                "exercise_minutes_per_week": int(client_row.get("exercise_minutes_per_week", 150) or 0),
                "social_hours_per_week": int(client_row.get("social_hours_per_week", 8) or 0),
                "stress_level_0_10": int(client_row.get("stress_level_0_10", 5) or 0),
                "productivity_0_10": int(client_row.get("productivity_0_10", 6) or 0),
                "mw_pred_reg_lin": float(client_row.get("mw_pred_reg_lin", np.nan) or np.nan),
                "mw_class_pred_rf": str(client_row.get("mw_class_pred_rf", "N/A") or "N/A"),
            }

            st.markdown("Ajuste os valores do cliente segundo as recomendações acima.")

            with st.form("ajustar_cliente_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    work_screen = st.slider(
                        "Tempo de ecrã no trabalho (h/dia)",
                        0.0, 14.0, float(client_row.get("work_screen_hours", 0.0) or 0.0), 0.1
                    )
                    leisure_screen = st.slider(
                        "Tempo de ecrã no lazer (h/dia)",
                        0.0, 14.0, float(client_row.get("leisure_screen_hours", 0.0) or 0.0), 0.1
                    )
                    screen_total = work_screen + leisure_screen
                    st.caption(f"Tempo de ecrã total calculado: {screen_total:.1f} h/dia")

                with col2:
                    sleep_hours = st.slider(
                        "Horas de sono", 3.0, 10.0, float(client_row.get("sleep_hours", 7.0) or 7.0), 0.1
                    )
                    sleep_quality = st.slider(
                        "Qualidade do sono (1–5)", 1, 5, int(client_row.get("sleep_quality_1_5", 3) or 3), 1
                    )
                    exercise = st.slider(
                        "Exercício (min/semana)", 0, 600, int(client_row.get("exercise_minutes_per_week", 150) or 0), 10
                    )

                with col3:
                    social = st.slider(
                        "Horas sociais/semana", 0, 40, int(client_row.get("social_hours_per_week", 8) or 0), 1
                    )
                    stress = st.slider(
                        "Stress (0–10)", 0, 10, int(client_row.get("stress_level_0_10", 5) or 0), 1
                    )
                    productivity = st.slider(
                        "Produtividade (0–10)", 0, 10, int(client_row.get("productivity_0_10", 6) or 0), 1
                    )

                submitted = st.form_submit_button("Aplicar ajustes")

            if submitted:
                adjusted = {
                    "row_id": client_row.get("row_id", client_id),
                    seg_col: seg_value,
                    "work_screen_hours": work_screen,
                    "leisure_screen_hours": leisure_screen,
                    "screen_time_hours": screen_total,
                    "sleep_hours": sleep_hours,
                    "sleep_quality_1_5": sleep_quality,
                    "exercise_minutes_per_week": exercise,
                    "social_hours_per_week": social,
                    "stress_level_0_10": stress,
                    "productivity_0_10": productivity,
                    "sleep_index": sleep_hours + sleep_quality,
                }

                # Calcular previsão de bem-estar para os valores ajustados
                adjusted_wellness = predict_wellness(regression_model, adjusted)

                # Build comparison table
                original = st.session_state.get("original_client", {})
                comparison_data = []
                for key in [
                    "work_screen_hours", "leisure_screen_hours", "screen_time_hours",
                    "sleep_hours", "sleep_quality_1_5", "exercise_minutes_per_week",
                    "social_hours_per_week", "stress_level_0_10", "productivity_0_10",
                    "sleep_index"
                ]:
                    orig_val = original.get(key, 0)
                    adj_val = adjusted.get(key, 0)
                    delta = adj_val - orig_val
                    delta_pct = (delta / orig_val * 100) if orig_val != 0 else 0.0
                    
                    # Format indicator
                    if delta > 0.01:
                        indicator = "📈"
                    elif delta < -0.01:
                        indicator = "📉"
                    else:
                        indicator = "➡️"
                    
                    comparison_data.append({
                        "Métrica": key,
                        "Anterior": f"{orig_val:.1f}" if isinstance(orig_val, float) else str(orig_val),
                        "Novo": f"{adj_val:.1f}" if isinstance(adj_val, float) else str(adj_val),
                        "Mudança": f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}",
                        "Indicador": indicator,
                    })

                # Adicionar bem-estar previsto à comparação
                orig_wellness = original.get("mw_pred_reg_lin", np.nan)
                if adjusted_wellness is not None:
                    wellness_delta = adjusted_wellness - orig_wellness
                    if wellness_delta > 0.01:
                        wellness_indicator = "📈"
                    elif wellness_delta < -0.01:
                        wellness_indicator = "📉"
                    else:
                        wellness_indicator = "➡️"
                    
                    comparison_data.append({
                        "Métrica": "mw_pred_reg_lin (Bem-estar previsto)",
                        "Anterior": f"{orig_wellness:.2f}" if not np.isnan(orig_wellness) else "N/A",
                        "Novo": f"{adjusted_wellness:.2f}",
                        "Mudança": f"{wellness_delta:+.2f}",
                        "Indicador": wellness_indicator,
                    })

                # Adicionar classe prevista à comparação
                orig_class = original.get("mw_class_pred_rf", "N/A")
                if classifier_model is not None:
                    adjusted_class = predict_class(classifier_model, adjusted)
                    class_indicator = "✓" if orig_class == adjusted_class else "➡️ Mudança"
                    comparison_data.append({
                        "Métrica": "mw_class_pred_rf (Classe prevista)",
                        "Anterior": str(orig_class),
                        "Novo": str(adjusted_class),
                        "Mudança": class_indicator if orig_class != adjusted_class else "Mantém",
                        "Indicador": "→" if orig_class != adjusted_class else "✓",
                    })

                st.markdown("---")
                st.subheader("📊 Comparação: Antes vs Depois")
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, width='stretch')

                st.markdown("---")
                st.markdown("**Pré-visualização do cliente ajustado**")
                df_out = pd.DataFrame([adjusted])
                st.dataframe(df_out, width='stretch')

                # Store adjusted client in session state for persistent save button
                st.session_state["adjusted_client_painel"] = df_out
                st.session_state["adjusted_client_id_painel"] = client_id

        # Persistent save section (outside form submit block)
        if "adjusted_client_painel" in st.session_state:
            st.markdown("---")
            col_csv, col_db = st.columns(2)
            
            with col_csv:
                df_out_persist = st.session_state["adjusted_client_painel"]
                csv_data = df_out_persist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descarregar cliente ajustado (CSV)",
                    data=csv_data,
                    file_name="cliente_ajustado.csv",
                    mime="text/csv",
                    key="download_client_painel"
                )
            
            with col_db:
                save_status_client = st.empty()
                
                if "save_status_client_painel" in st.session_state:
                    msg = st.session_state["save_status_client_painel"]
                    if msg.get("type") == "success":
                        save_status_client.success(msg.get("message", ""))
                    elif msg.get("type") == "error":
                        save_status_client.error(msg.get("message", ""))
                        if msg.get("traceback"):
                            st.code(msg["traceback"])
                
                if st.button("💿 Guardar cliente na Base de Dados", key=f"save_client_painel_persistent"):
                    df_out = st.session_state["adjusted_client_painel"]
                    db_path_client = db_path
                    id_col_client = "row_id" if "row_id" in df_out.columns else None

                    st.session_state["save_status_client_painel"] = {"type": "info", "message": "🔄 A processar..."}
                    save_status_client.info("🔄 A processar...")

                    try:
                        import os
                        if not os.path.exists(db_path_client):
                            st.session_state["save_status_client_painel"] = {
                                "type": "error",
                                "message": f"❌ Base de dados não encontrada: {db_path_client}"
                            }
                            save_status_client.error(st.session_state["save_status_client_painel"]["message"])
                        elif not id_col_client:
                            st.session_state["save_status_client_painel"] = {
                                "type": "error",
                                "message": "❌ Não foi possível guardar: coluna row_id em falta."
                            }
                            save_status_client.error(st.session_state["save_status_client_painel"]["message"])
                        else:
                            conn = sqlite3.connect(db_path_client)
                            try:
                                try:
                                    base_df = pd.read_sql_query("SELECT * FROM Deployment_Output", conn)
                                except Exception:
                                    try:
                                        base_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                                    except Exception:
                                        base_df = df_out.copy()

                                try:
                                    adjusted_df = pd.read_sql_query("SELECT * FROM Deployment_Output_Adjusted", conn)
                                except Exception:
                                    adjusted_df = base_df.copy()

                                current_df = merge_adjusted_rows(base_df, adjusted_df, id_col_client)
                                merged = merge_adjusted_rows(current_df, df_out, id_col_client)
                                merged.to_sql("Deployment_Output_Adjusted", conn, if_exists="replace", index=False)
                                conn.commit()
                                st.session_state["save_status_client_painel"] = {
                                    "type": "success",
                                    "message": f"✅ Cliente guardado na tabela 'Deployment_Output_Adjusted'. Tabela completa com {len(merged)} linha(s)."
                                }
                                save_status_client.success(st.session_state["save_status_client_painel"]["message"])
                            finally:
                                conn.close()
                    except Exception as e:
                        import traceback
                        st.session_state["save_status_client_painel"] = {
                            "type": "error",
                            "message": f"❌ Erro ao guardar cliente: {e}",
                            "traceback": traceback.format_exc()
                        }
                        save_status_client.error(st.session_state["save_status_client_painel"]["message"])
                        st.code(st.session_state["save_status_client_painel"]["traceback"])

# -------------------------
# Tab 6: Planeamento em Grupo
# -------------------------
with tab6:
    st.subheader("🎯 Planeamento em Grupo para um Segmento")
    st.caption("Aplique ajustes a todos os clientes de um segmento e visualize o impacto agregado")

    if not available_seg_cols:
        st.info("📄 Não existem colunas de segmento disponíveis.")
    else:
        seg_col = st.selectbox("Segmentação", available_seg_cols, index=0, key="group_seg_col")
        seg_values = sorted(df_f[seg_col].dropna().unique().tolist())
        seg_value = st.selectbox("Selecione o Segmento", seg_values, index=0, key="group_seg_value")

        df_seg_group = df_f[df_f[seg_col] == seg_value]
        
        if df_seg_group.empty:
            st.warning("📄 Nenhum cliente neste segmento.")
        else:
            # Mostrar perfil atual do segmento
            st.markdown("---")
            st.subheader("📊 Perfil Atual do Segmento")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total de Clientes", len(df_seg_group))
            if 'mw_pred_reg_lin' in df_seg_group.columns:
                col2.metric("Bem-estar Médio", f"{df_seg_group['mw_pred_reg_lin'].mean():.2f}")
            if 'screen_time_hours' in df_seg_group.columns:
                col3.metric("Ecrã Médio (h)", f"{df_seg_group['screen_time_hours'].mean():.1f}")
            if 'sleep_hours' in df_seg_group.columns:
                col4.metric("Sono Médio (h)", f"{df_seg_group['sleep_hours'].mean():.1f}")
            if 'stress_level_0_10' in df_seg_group.columns:
                col5.metric("Stress Médio", f"{df_seg_group['stress_level_0_10'].mean():.1f}")

            # Classes atuais no segmento
            if "mw_class_pred_rf" in df_seg_group.columns:
                class_dist = df_seg_group["mw_class_pred_rf"].value_counts()
                st.markdown("**Distribuição de Classes (Atual):**")
                for cls, count in class_dist.items():
                    pct = (count / len(df_seg_group)) * 100
                    st.write(f"- {cls}: {count} clientes ({pct:.1f}%)")

            # Recomendações do segmento
            label, measures = segment_suggestion(seg_col, seg_value)
            st.markdown("---")
            st.info(f"🎯 **Perfil do Segmento:** {label}")
            if measures:
                st.markdown("**📝 Recomendações de Intervenção para este Segmento:**")
                for i, m in enumerate(measures, start=1):
                    st.markdown(f"{i}. {m}")
                st.caption("💡 Use os ajustes abaixo para simular a aplicação destas recomendações a todo o grupo")

            st.markdown("---")
            st.subheader("⚙️ Ajustes do Grupo")
            st.caption("Defina os ajustes que devem ser aplicados a TODOS os clientes do segmento")

            with st.form("group_adjustment_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    work_screen_delta = st.slider(
                        "Ajuste de Ecrã de Trabalho (h/dia)",
                        -5.0, 5.0, 0.0, 0.1,
                        help="Positivo = aumentar, Negativo = diminuir"
                    )
                    leisure_screen_delta = st.slider(
                        "Ajuste de Ecrã de Lazer (h/dia)",
                        -5.0, 5.0, 0.0, 0.1
                    )
                    sleep_delta = st.slider(
                        "Ajuste de Sono (h/dia)",
                        -3.0, 3.0, 0.0, 0.1
                    )

                with col2:
                    sleep_quality_delta = st.slider(
                        "Ajuste de Qualidade do Sono (1-5)",
                        -2, 2, 0, 1
                    )
                    exercise_delta = st.slider(
                        "Ajuste de Exercício (min/semana)",
                        -150, 150, 0, 10
                    )
                    social_delta = st.slider(
                        "Ajuste de Horas Sociais (h/semana)",
                        -10, 10, 0, 1
                    )

                with col3:
                    stress_delta = st.slider(
                        "Ajuste de Stress (0-10)",
                        -5, 5, 0, 1
                    )
                    productivity_delta = st.slider(
                        "Ajuste de Produtividade (0-10)",
                        -5, 5, 0, 1
                    )

                group_submitted = st.form_submit_button("Calcular Impacto do Grupo")

            if group_submitted:
                # Aplicar ajustes a todos os clientes do segmento
                adjusted_group = df_seg_group.copy()
                
                # Calcular novos valores
                adjusted_group["work_screen_hours"] = (adjusted_group["work_screen_hours"] + work_screen_delta).clip(lower=0)
                adjusted_group["leisure_screen_hours"] = (adjusted_group["leisure_screen_hours"] + leisure_screen_delta).clip(lower=0)
                adjusted_group["screen_time_hours"] = adjusted_group["work_screen_hours"] + adjusted_group["leisure_screen_hours"]
                adjusted_group["sleep_hours"] = (adjusted_group["sleep_hours"] + sleep_delta).clip(3, 10)
                adjusted_group["sleep_quality_1_5"] = (adjusted_group["sleep_quality_1_5"] + sleep_quality_delta).clip(1, 5)
                adjusted_group["exercise_minutes_per_week"] = (adjusted_group["exercise_minutes_per_week"] + exercise_delta).clip(lower=0)
                adjusted_group["social_hours_per_week"] = (adjusted_group["social_hours_per_week"] + social_delta).clip(lower=0)
                adjusted_group["stress_level_0_10"] = (adjusted_group["stress_level_0_10"] + stress_delta).clip(0, 10)
                adjusted_group["productivity_0_10"] = (adjusted_group["productivity_0_10"] + productivity_delta).clip(0, 10)
                adjusted_group["sleep_index"] = adjusted_group["sleep_hours"] + adjusted_group["sleep_quality_1_5"]

                # Recalcular bem-estar previsto para cada cliente
                new_wellness_list = []
                new_class_list = []
                new_prob_baixo = []
                new_prob_medio = []
                new_prob_alto = []
                for idx, row in adjusted_group.iterrows():
                    row_dict = row.to_dict()
                    new_wellness = predict_wellness(regression_model, row_dict)
                    new_wellness_list.append(new_wellness if new_wellness is not None else np.nan)
                    
                    if classifier_model is not None:
                        new_class, prob_cols = predict_class_with_proba(classifier_model, row_dict)
                        new_class_list.append(new_class)
                        new_prob_baixo.append(prob_cols.get("mw_prob_Baixo", np.nan))
                        new_prob_medio.append(prob_cols.get("mw_prob_Medio", np.nan))
                        new_prob_alto.append(prob_cols.get("mw_prob_Alto", np.nan))
                    else:
                        new_class_list.append("N/A")
                        new_prob_baixo.append(np.nan)
                        new_prob_medio.append(np.nan)
                        new_prob_alto.append(np.nan)

                adjusted_group["mw_pred_reg_lin_novo"] = new_wellness_list
                adjusted_group["mw_class_pred_rf_novo"] = new_class_list
                adjusted_group["mw_prob_Baixo"] = new_prob_baixo
                adjusted_group["mw_prob_Medio"] = new_prob_medio
                adjusted_group["mw_prob_Alto"] = new_prob_alto

                # Calcular impacto
                st.markdown("---")
                st.subheader("📈 Impacto Agregado do Grupo")

                # Bem-estar médio
                if 'mw_pred_reg_lin' in df_seg_group.columns:
                    original_avg_wellness = df_seg_group["mw_pred_reg_lin"].mean()
                    new_avg_wellness = adjusted_group["mw_pred_reg_lin_novo"].mean()
                    wellness_delta_group = new_avg_wellness - original_avg_wellness
                else:
                    original_avg_wellness = 0
                    new_avg_wellness = 0
                    wellness_delta_group = 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Bem-estar Médio (Antes)", f"{original_avg_wellness:.2f}")
                col2.metric("Bem-estar Médio (Depois)", f"{new_avg_wellness:.2f}")
                col3.metric("Mudança Média", f"{wellness_delta_group:+.2f}", 
                           delta_color="normal" if wellness_delta_group >= 0 else "inverse")
                if 'mw_pred_reg_lin' in df_seg_group.columns:
                    improving_pct = (adjusted_group['mw_pred_reg_lin_novo'] > df_seg_group['mw_pred_reg_lin']).sum() / len(adjusted_group) * 100
                    col4.metric("% Clientes a Melhorar", f"{improving_pct:.1f}%")
                else:
                    col4.metric("% Clientes a Melhorar", "N/A")

                # Distribuição de classes antes/depois
                if classifier_model is not None and 'mw_class_pred_rf' in df_seg_group.columns:
                    st.markdown("**Mudança na Distribuição de Classes de Bem-estar:**")
                    
                    original_classes = df_seg_group["mw_class_pred_rf"].value_counts()
                    new_classes = pd.Series(new_class_list).value_counts()
                    
                    # Always show all three classes (match data format: Medio without accent)
                    all_classes = ["Baixo", "Medio", "Alto"]
                    display_names = ["Baixo", "Médio", "Alto"]
                    
                    class_comparison = pd.DataFrame({
                        "Classe": display_names,
                        "Antes": [int(original_classes.get(c, 0)) for c in all_classes],
                        "Depois": [int(new_classes.get(c, 0)) for c in all_classes]
                    })
                    class_comparison["Mudança"] = class_comparison["Depois"] - class_comparison["Antes"]

                    st.dataframe(class_comparison, width='stretch')

                    # Visualizar mudança
                    fig_class = px.bar(
                        class_comparison,
                        x="Classe",
                        y=["Antes", "Depois"],
                        barmode="group",
                        title="Distribuição de Classes: Antes vs Depois",
                        labels={"value": "Número de Clientes", "variable": "Período"}
                    )
                    fig_class.update_layout(height=400)
                    st.plotly_chart(fig_class, width='stretch')

                # Comparação de métricas
                st.markdown("**Mudança nas Métricas do Grupo:**")
                
                metric_cols = [
                    ("work_screen_hours", "Ecrã de Trabalho (h/dia)"),
                    ("leisure_screen_hours", "Ecrã de Lazer (h/dia)"),
                    ("sleep_hours", "Sono (h/dia)"),
                    ("sleep_quality_1_5", "Qualidade do Sono"),
                    ("exercise_minutes_per_week", "Exercício (min/semana)"),
                    ("social_hours_per_week", "Horas Sociais"),
                    ("stress_level_0_10", "Stress"),
                    ("productivity_0_10", "Produtividade")
                ]
                
                metrics_data = []
                for col_name, display_name in metric_cols:
                    if col_name in df_seg_group.columns and col_name in adjusted_group.columns:
                        metrics_data.append({
                            "Métrica": display_name,
                            "Antes": df_seg_group[col_name].mean(),
                            "Depois": adjusted_group[col_name].mean()
                        })
                
                metrics_comparison = pd.DataFrame(metrics_data)
                metrics_comparison["Mudança"] = metrics_comparison["Depois"] - metrics_comparison["Antes"]

                st.dataframe(metrics_comparison, width='stretch')

                # Dados detalhados dos clientes ajustados
                st.markdown("---")
                st.subheader("📋 Detalhes dos Clientes Ajustados")
                
                export_cols = [
                    "row_id", "mw_pred_reg_lin", "mw_pred_reg_lin_novo", "mw_class_pred_rf", "mw_class_pred_rf_novo",
                    "mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto",
                    "work_screen_hours", "leisure_screen_hours", "sleep_hours", "sleep_quality_1_5",
                    "exercise_minutes_per_week", "social_hours_per_week", "stress_level_0_10", "productivity_0_10",
                ]
                export_cols = [c for c in export_cols if c in adjusted_group.columns]
                
                st.dataframe(adjusted_group[export_cols], width='stretch')

                # Store in session state for persistent save button
                st.session_state["adjusted_group_wellness"] = adjusted_group[export_cols]

        # Persistent save section (outside form submit block)
        if "adjusted_group_wellness" in st.session_state:
            st.markdown("---")
            col_csv, col_db = st.columns(2)
            
            with col_csv:
                df_group_persist = st.session_state["adjusted_group_wellness"]
                csv_data = df_group_persist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descarregar Plano de Intervenção em Grupo (CSV)",
                    data=csv_data,
                    file_name=f"grupo_ajustado_wellness.csv",
                    mime="text/csv",
                    key="download_group_wellness"
                )
            
            with col_db:
                status_placeholder_group = st.empty()
                
                if "save_status_wellness_group" in st.session_state:
                    msg = st.session_state["save_status_wellness_group"]
                    if msg.get("type") == "success":
                        status_placeholder_group.success(msg.get("message", ""))
                    elif msg.get("type") == "error":
                        status_placeholder_group.error(msg.get("message", ""))
                        if msg.get("traceback"):
                            st.code(msg["traceback"])
                
                if st.button("💿 Guardar Grupo na Base de Dados", key=f"save_group_wellness_persistent"):
                    df_to_save_group = st.session_state["adjusted_group_wellness"]
                    id_col_group = "row_id" if "row_id" in df_to_save_group.columns else None

                    st.session_state["save_status_wellness_group"] = {"type": "info", "message": "🔄 A processar..."}
                    status_placeholder_group.info("🔄 A processar...")

                    try:
                        import os
                        if not os.path.exists(db_path):
                            st.session_state["save_status_wellness_group"] = {
                                "type": "error",
                                "message": f"❌ Base de dados não encontrada: {db_path}"
                            }
                            status_placeholder_group.error(st.session_state["save_status_wellness_group"]["message"])
                        elif not id_col_group:
                            st.session_state["save_status_wellness_group"] = {
                                "type": "error",
                                "message": "❌ Não foi possível guardar: coluna row_id em falta."
                            }
                            status_placeholder_group.error(st.session_state["save_status_wellness_group"]["message"])
                        else:
                            conn = sqlite3.connect(db_path)
                            try:
                                try:
                                    base_df = pd.read_sql_query("SELECT * FROM Deployment_Output", conn)
                                except Exception:
                                    try:
                                        base_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                                    except Exception:
                                        base_df = df_to_save_group.copy()

                                try:
                                    adjusted_df = pd.read_sql_query("SELECT * FROM Deployment_Output_Adjusted", conn)
                                except Exception:
                                    adjusted_df = base_df.copy()

                                current_df = merge_adjusted_rows(base_df, adjusted_df, id_col_group)
                                merged = merge_adjusted_rows(current_df, df_to_save_group, id_col_group)
                                merged.to_sql("Deployment_Output_Adjusted", conn, if_exists="replace", index=False)
                                conn.commit()

                                st.session_state["save_status_wellness_group"] = {
                                    "type": "success",
                                    "message": f"✅ Grupo guardado na tabela 'Deployment_Output_Adjusted'. Tabela completa com {len(merged)} linha(s)."
                                }
                                status_placeholder_group.success(st.session_state["save_status_wellness_group"]["message"])
                            finally:
                                conn.close()
                    except Exception as e:
                        import traceback
                        st.session_state["save_status_wellness_group"] = {
                            "type": "error",
                            "message": f"❌ Erro ao guardar: {e}",
                            "traceback": traceback.format_exc()
                        }
                        status_placeholder_group.error(st.session_state["save_status_wellness_group"]["message"])
                        st.code(st.session_state["save_status_wellness_group"]["traceback"])

# -----------------------------
