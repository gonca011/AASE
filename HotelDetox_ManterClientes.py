# kpi_dashboard.py
# Dashboard Streamlit para KPIs (Deployment_Output):
# KPI 1.1: Taxa de clientes em risco (mw_prob_Baixo >= threshold OU classe = "Baixo")
# KPI 1.2: Taxa de clientes com bem-estar elevado (mw_pred_reg_lin >= cutoff)
#
# Requisitos:
# pip install streamlit pandas numpy plotly

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import joblib
import os

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Hotel Detox - Painel de Risco e Bem-estar", layout="wide")
st.title("Hotel Detox — Painel de Clientes em Risco e Bem-estar")
st.caption("Visão clara de risco e bem-estar dos clientes (dados do Deployment)")

# -----------------------------
# Segment definitions and recommendations
# -----------------------------
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

def segment_suggestion(seg_col: str, seg_value):
    """Devolve etiqueta e recomendações do segmento."""
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

# -----------------------------
# Sidebar: configuração
# -----------------------------
st.sidebar.header("Fonte de Dados")

db_path = st.sidebar.text_input(
    "Caminho da Base de Dados",
    value=r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
)
table_name = st.sidebar.text_input(
    "Tabela",
    value="Deployment_Output"
)

# KPI settings
st.sidebar.header("Parâmetros")

risk_threshold = st.sidebar.slider(
    "Limite para considerar risco (prob_Baixo >=)",
    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
    help="Acima deste valor o cliente é marcado como risco"
)

use_class_as_risk = st.sidebar.checkbox(
    'Incluir também clientes com classe prevista = "Baixo" como risco',
    value=True
)

high_wellness_cutoff = st.sidebar.slider(
    "Limiar para bem-estar elevado (previsto >=)",
    min_value=0.0, max_value=10.0, value=8.5, step=0.5,
    help="Acima deste valor o cliente é considerado bem-estar elevado"
)

# Business Objective
st.sidebar.header("Metas")
st.sidebar.markdown("**Objetivo:** Maximizar bem-estar e reduzir risco")

target_high_wellness_pct = st.sidebar.number_input(
    "% mínima de clientes com bem-estar elevado",
    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
    help="Percentual mínimo desejado de clientes com bem-estar elevado"
)

target_max_risk_pct = st.sidebar.number_input(
    "% máxima de clientes em risco",
    min_value=0.0, max_value=100.0, value=15.0, step=1.0,
    help="Percentual máximo tolerado de clientes em risco"
)

# -----------------------------
# Carregar dados (cache)
# -----------------------------
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
    if model is None:
        return None, {"mw_prob_Baixo": np.nan, "mw_prob_Medio": np.nan, "mw_prob_Alto": np.nan}

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

        prob_cols = {"mw_prob_Baixo": np.nan, "mw_prob_Medio": np.nan, "mw_prob_Alto": np.nan}
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
        return None, {"mw_prob_Baixo": np.nan, "mw_prob_Medio": np.nan, "mw_prob_Alto": np.nan}


@st.cache_data(show_spinner=True)
def load_data(path: str, tbl: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    df_ = pd.read_sql_query(f"SELECT * FROM {tbl}", conn)
    conn.close()
    return df_


def merge_adjusted_rows(base_df: pd.DataFrame, updates_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Merge updated rows into base_df using id_col; keep other rows intact."""
    if id_col not in base_df.columns or id_col not in updates_df.columns:
        return base_df

    base_df = base_df.copy()
    updates_df = updates_df.copy()

    # If new prediction columns exist, map them to the canonical names
    if "mw_pred_reg_lin_novo" in updates_df.columns:
        updates_df["mw_pred_reg_lin"] = updates_df["mw_pred_reg_lin_novo"]
    if "mw_class_pred_rf_novo" in updates_df.columns:
        updates_df["mw_class_pred_rf"] = updates_df["mw_class_pred_rf_novo"]

    # Drop helper columns ending with _novo before merging
    updates_df = updates_df[[c for c in updates_df.columns if not c.endswith("_novo")]]

    base_df = base_df.set_index(id_col)
    updates_df = updates_df.set_index(id_col)
    base_df.update(updates_df)
    return base_df.reset_index()

try:
    df = load_data(db_path, table_name)
except Exception as e:
    st.error(f"Erro ao ler a tabela '{table_name}' na BD: {e}")
    st.stop()

# -----------------------------
# Validações mínimas
# -----------------------------
required_cols = [
    "mw_pred_reg_lin",
    "mw_class_pred_rf",
    "mw_prob_Baixo",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltam colunas necessárias para estes KPIs: {missing}")
    st.stop()

# Tipos numéricos
for c in ["mw_pred_reg_lin", "mw_prob_Baixo"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Sidebar: filtros (opcionais)
# -----------------------------
st.sidebar.header("Filtros")

# Filtrar por cluster (se existir)
def multiselect_filter(colname: str):
    if colname in df.columns:
        vals = sorted(df[colname].dropna().unique())
        return st.sidebar.multiselect(f"Filtro: {colname}", options=vals, default=vals)
    return None

sel_obj1 = multiselect_filter("cluster_obj1_k4")
sel_obj2 = multiselect_filter("cluster_obj2_k4")

# Filtrar por intervalo de screen_time_hours (se existir)
if "screen_time_hours" in df.columns:
    df["screen_time_hours"] = pd.to_numeric(df["screen_time_hours"], errors="coerce")
    st_min, st_max = float(df["screen_time_hours"].min()), float(df["screen_time_hours"].max())
    screen_range = st.sidebar.slider(
        "Intervalo de screen_time_hours",
        min_value=st_min, max_value=st_max,
        value=(st_min, st_max)
    )
else:
    screen_range = None

df_f = df.copy()

if sel_obj1 is not None:
    df_f = df_f[df_f["cluster_obj1_k4"].isin(sel_obj1)]

if sel_obj2 is not None:
    df_f = df_f[df_f["cluster_obj2_k4"].isin(sel_obj2)]

if screen_range is not None and "screen_time_hours" in df_f.columns:
    df_f = df_f[
        (df_f["screen_time_hours"] >= screen_range[0]) &
        (df_f["screen_time_hours"] <= screen_range[1])
    ]

# -----------------------------
# KPI 1.1 (Risco)
# -----------------------------
risk_by_prob = df_f["mw_prob_Baixo"].fillna(0) >= risk_threshold

if use_class_as_risk and "mw_class_pred_rf" in df_f.columns:
    risk_by_class = df_f["mw_class_pred_rf"].astype(str).str.strip().eq("Baixo")
    df_f["is_risk"] = risk_by_prob | risk_by_class
else:
    df_f["is_risk"] = risk_by_prob

kpi_11_risk_pct = float(df_f["is_risk"].mean() * 100) if len(df_f) else 0.0

# -----------------------------
# KPI 1.2 (Bem-estar elevado)
# -----------------------------
df_f["is_high_wellness"] = df_f["mw_pred_reg_lin"].fillna(-np.inf) >= high_wellness_cutoff
kpi_12_high_pct = float(df_f["is_high_wellness"].mean() * 100) if len(df_f) else 0.0

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral & KPIs",
    "🧪 Ajustar Cliente",
    "📋 Dados & Export",
    "🎯 Planeamento em Grupo"
])

# -------------------------
# Tab 1: Visão Geral & KPIs
# -------------------------
with tab1:
    st.subheader("Visão Geral dos KPIs (após filtros)")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total de Clientes", f"{len(df_f)}")

    # KPI 1.1 com indicação de objetivo
    risk_meets_target = kpi_11_risk_pct <= target_max_risk_pct
    risk_delta = kpi_11_risk_pct - target_max_risk_pct
    c2.metric(
        "Clientes em Risco",
        f"{kpi_11_risk_pct:.1f}%",
        delta=f"{risk_delta:+.1f}pp vs meta ({target_max_risk_pct:.0f}%)",
        delta_color="inverse",  # lower is better
        help=f"{'✓ Dentro da meta' if risk_meets_target else '✗ Acima da meta (precisa reduzir)'}"
    )

    # KPI 1.2 com indicação de objetivo
    wellness_meets_target = kpi_12_high_pct >= target_high_wellness_pct
    wellness_delta = kpi_12_high_pct - target_high_wellness_pct
    c3.metric(
        "Clientes com Bem-estar Elevado",
        f"{kpi_12_high_pct:.1f}%",
        delta=f"{wellness_delta:+.1f}pp vs meta ({target_high_wellness_pct:.0f}%)",
        delta_color="normal",
        help=f"{'✓ Dentro da meta' if wellness_meets_target else '✗ Abaixo da meta (precisa melhorar)'}"
    )

    c4.metric("Bem-estar Médio Previsto", f"{df_f['mw_pred_reg_lin'].mean():.2f}" if len(df_f) else "N/A")

    # Status do objetivo de negócio
    st.markdown("---")
    both_targets_met = risk_meets_target and wellness_meets_target
    if both_targets_met:
        st.success(
            f"✓ Excelente: {kpi_12_high_pct:.1f}% dos clientes estão com bem-estar elevado (meta: {target_high_wellness_pct:.0f}%) e "
            f"{kpi_11_risk_pct:.1f}% em risco (meta: ≤{target_max_risk_pct:.0f}%)."
        )
    elif wellness_meets_target and not risk_meets_target:
        st.warning(
            f"⚠ Parcial: bem-estar está bom ({kpi_12_high_pct:.1f}%), mas o risco ainda está alto ({kpi_11_risk_pct:.1f}% > {target_max_risk_pct:.0f}%)."
        )
    elif risk_meets_target and not wellness_meets_target:
        st.warning(
            f"⚠ Parcial: risco controlado ({kpi_11_risk_pct:.1f}%), porém bem-estar insuficiente ({kpi_12_high_pct:.1f}% < {target_high_wellness_pct:.0f}%)."
        )
    else:
        st.error(
            f"✗ Atenção: bem-estar baixo ({kpi_12_high_pct:.1f}% < {target_high_wellness_pct:.0f}%) e risco alto ({kpi_11_risk_pct:.1f}% > {target_max_risk_pct:.0f}%)."
        )

    st.divider()

    # -----------------------------
    # Visualizações
    # -----------------------------
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.subheader("Risco — Distribuição da Probabilidade")
        fig = px.histogram(
            df_f,
            x="mw_prob_Baixo",
            nbins=20,
            title="Distribuição de probabilidade de risco (mw_prob_Baixo)"
        )
        fig.add_vline(x=risk_threshold, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Bem-estar — Distribuição das Previsões")
        fig2 = px.histogram(
            df_f,
            x="mw_pred_reg_lin",
            nbins=20,
            title="Distribuição das previsões de bem-estar (mw_pred_reg_lin)"
        )
        fig2.add_vline(x=high_wellness_cutoff, line_dash="dash")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("Risco vs Bem-estar (dispersão)")
        fig3 = px.scatter(
            df_f,
            x="mw_pred_reg_lin",
            y="mw_prob_Baixo",
            color="mw_class_pred_rf" if "mw_class_pred_rf" in df_f.columns else None,
            hover_data=[c for c in ["row_id", "cluster_obj1_k4", "cluster_obj2_k4"] if c in df_f.columns],
            title="mw_prob_Baixo vs mw_pred_reg_lin"
        )
        fig3.add_hline(y=risk_threshold, line_dash="dash")
        fig3.add_vline(x=high_wellness_cutoff, line_dash="dash")
        st.plotly_chart(fig3, use_container_width=True)

        # KPI por cluster (se existir)
        if "cluster_obj2_k4" in df_f.columns:
            st.subheader("KPIs por Segmento (Obj2)")
            grp = df_f.groupby("cluster_obj2_k4", dropna=False).agg(
                n=("mw_pred_reg_lin", "size"),
                risk_pct=("is_risk", lambda s: float(s.mean() * 100)),
                high_pct=("is_high_wellness", lambda s: float(s.mean() * 100)),
                mw_mean=("mw_pred_reg_lin", "mean")
            ).reset_index()

            fig4 = px.bar(
                grp,
                x="cluster_obj2_k4",
                y="risk_pct",
                title="Clientes em risco por segmento"
            )
            st.plotly_chart(fig4, use_container_width=True)

            fig5 = px.bar(
                grp,
                x="cluster_obj2_k4",
                y="high_pct",
                title="Clientes com bem-estar elevado por segmento"
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("A coluna cluster_obj2_k4 não existe. Secção de KPIs por cluster não é apresentada.")

# Sidebar: Modelo (opcional)
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

# -------------------------
# Tab 2: Ajustar Cliente
# -------------------------
with tab2:
    st.subheader("🧪 Ajustar um cliente")

    id_col = "row_id" if "row_id" in df_f.columns else None
    client_options = df_f[id_col].tolist() if id_col else df_f.index.tolist()

    if client_options:
        client_id = st.selectbox(
            "Selecione um cliente",
            client_options,
            format_func=lambda x: f"Cliente {x}"
        )

        if id_col:
            client_row = df_f[df_f[id_col] == client_id].iloc[0]
        else:
            client_row = df_f.loc[client_id]

        st.markdown("Ajuste os valores do cliente para otimizar bem-estar e reduzir risco.")

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

        adjusted_wellness = predict_wellness(regression_model, adjusted)
        if adjusted_wellness is not None:
            adjusted["mw_pred_reg_lin"] = adjusted_wellness

        adjusted_class = None
        prob_cols = {"mw_prob_Baixo": np.nan, "mw_prob_Medio": np.nan, "mw_prob_Alto": np.nan}
        if classifier_model is not None:
            adjusted_class, prob_cols = predict_class_with_proba(classifier_model, adjusted)
            adjusted["mw_class_pred_rf"] = adjusted_class
            adjusted.update(prob_cols)

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
        if classifier_model is not None and adjusted_class is not None:
            class_indicator = "✓" if orig_class == adjusted_class else "➡️ Mudança"
            comparison_data.append({
                "Métrica": "mw_class_pred_rf (Classe prevista)",
                "Anterior": str(orig_class),
                "Novo": str(adjusted_class),
                "Mudança": class_indicator if orig_class != adjusted_class else "Mantém",
                "Indicador": "→" if orig_class != adjusted_class else "✓",
            })

        # Probabilidades previstas
        if classifier_model is not None and prob_cols:
            for k, label in [("mw_prob_Baixo", "Prob. Baixo"), ("mw_prob_Medio", "Prob. Médio"), ("mw_prob_Alto", "Prob. Alto")]:
                new_val = prob_cols.get(k, np.nan)
                old_val = original.get(k, np.nan)
                delta = new_val - old_val if not np.isnan(old_val) and not np.isnan(new_val) else np.nan
                comparison_data.append({
                    "Métrica": label,
                    "Anterior": f"{old_val:.3f}" if not np.isnan(old_val) else "N/A",
                    "Novo": f"{new_val:.3f}" if not np.isnan(new_val) else "N/A",
                    "Mudança": f"{delta:+.3f}" if not np.isnan(delta) else "N/A",
                    "Indicador": "📈" if not np.isnan(delta) and delta > 0.0001 else ("📉" if not np.isnan(delta) and delta < -0.0001 else "➡️"),
                })

        st.markdown("---")
        st.subheader("📊 Comparação: Antes vs Depois")
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("---")
        st.markdown("**Pré-visualização do cliente ajustado**")
        df_out = pd.DataFrame([adjusted])
        st.dataframe(df_out, use_container_width=True)
        
        # Store in session state for the button to access
        st.session_state["adjusted_client_data"] = df_out

        col_csv, col_db = st.columns(2)
        
        with col_csv:
            st.download_button(
                "💾 Descarregar cliente ajustado (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="cliente_ajustado.csv",
                mime="text/csv"
            )
        
        with col_db:
            st.markdown("👇 Clique para guardar")

    # Save button outside the form submission block
    if "adjusted_client_data" in st.session_state:
        st.markdown("---")
        st.subheader("💾 Guardar Dados Ajustados")

        df_preview = st.session_state.get("adjusted_client_data")
        st.caption(f"Dados prontos para guardar: {len(df_preview)} linha(s), {len(df_preview.columns)} coluna(s)")
        st.write("Colunas:", list(df_preview.columns))

        status_placeholder = st.empty()

        # Show previous status (persists across reruns)
        if "save_status" in st.session_state:
            msg = st.session_state["save_status"]
            if msg["type"] == "success":
                status_placeholder.success(msg["message"])
            elif msg["type"] == "error":
                status_placeholder.error(msg["message"])
                if "traceback" in msg:
                    st.code(msg["traceback"])

        if st.button("💿 Guardar Cliente na Base de Dados", key="save_client_db_main", type="primary"):
            df_to_save = st.session_state["adjusted_client_data"]
            db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
            id_col_save = "row_id" if "row_id" in df.columns else None

            # Reset status
            st.session_state["save_status"] = {"type": "info", "message": "🔄 A processar..."}
            status_placeholder.info("🔄 A processar...")
            try:
                import os

                if not os.path.exists(db_path):
                    st.session_state["save_status"] = {
                        "type": "error",
                        "message": f"❌ Base de dados não encontrada: {db_path}"
                    }
                    status_placeholder.error(st.session_state["save_status"]["message"])
                elif not id_col_save or id_col_save not in df_to_save.columns:
                    st.session_state["save_status"] = {
                        "type": "error",
                        "message": "❌ Não foi possível guardar: coluna row_id em falta."
                    }
                    status_placeholder.error(st.session_state["save_status"]["message"])
                else:
                    conn = sqlite3.connect(db_path)
                    try:
                        # Base: sempre ler Deployment_Output (original)
                        base_df = pd.read_sql_query("SELECT * FROM Deployment_Output", conn)
                        # Ajustado acumulado: usar tabela ajustada se existir, senão copiar base
                        try:
                            adjusted_df = pd.read_sql_query("SELECT * FROM Deployment_Output_Adjusted", conn)
                        except Exception:
                            adjusted_df = base_df.copy()

                        merged = merge_adjusted_rows(adjusted_df, df_to_save, id_col_save)
                        merged.to_sql("Deployment_Output_Adjusted", conn, if_exists="replace", index=False)
                        conn.commit()

                        st.session_state["save_status"] = {
                            "type": "success",
                            "message": f"✅ Cliente guardado com sucesso na tabela 'Deployment_Output_Adjusted'! Tabela completa com {len(merged)} linha(s)."
                        }
                        status_placeholder.success(st.session_state["save_status"]["message"])
                    finally:
                        conn.close()

            except Exception as e:
                import traceback
                st.session_state["save_status"] = {
                    "type": "error",
                    "message": f"❌ Erro ao guardar: {str(e)}",
                    "traceback": traceback.format_exc()
                }
                status_placeholder.error(st.session_state["save_status"]["message"])
                st.code(st.session_state["save_status"].get("traceback", ""))

            # Immediately show updated status without relying on rerun
            msg = st.session_state.get("save_status", {})
            if msg.get("type") == "success":
                status_placeholder.success(msg.get("message", ""))
            elif msg.get("type") == "error":
                status_placeholder.error(msg.get("message", ""))
                if msg.get("traceback"):
                    st.code(msg["traceback"])
            else:
                status_placeholder.info(msg.get("message", ""))
    else:
        st.markdown("---")
        st.info("ℹ Primeiro aplique ajustes e clique em 'Aplicar ajustes' para ativar o botão de guardar.")

# -------------------------
# Tab 3: Dados & Export
# -------------------------
with tab3:
    st.subheader("📋 Amostra de Dados")

show_cols = [c for c in [
    "row_id",
    "mw_pred_reg_lin",
    "mw_class_pred_rf",
    "mw_prob_Baixo",
    "is_risk",
    "is_high_wellness",
    "cluster_obj1_k4",
    "cluster_obj2_k4",
    "screen_time_hours",
    "sleep_hours",
    "stress_level_0_10",
] if c in df_f.columns]

st.dataframe(df_f[show_cols].head(200), use_container_width=True)

st.download_button(
    "Descarregar CSV (dados filtrados)",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="kpis_filtrado.csv",
    mime="text/csv"
)

st.caption(
    "Notas: KPI 1.1 mede clientes em risco (prob_Baixo e/ou classe 'Baixo'). "
    "KPI 1.2 mede clientes com bem-estar previsto acima do limiar. Ajuste os limiares conforme necessário."
)

# -------------------------
# Tab 4: Planejamento em Grupo
# -------------------------
with tab4:
    st.subheader("🎯 Planeamento em Grupo para um Segmento")
    st.caption("Aplique ajustes a todos os clientes de um segmento de uma vez")

    available_seg_cols = ["cluster_obj1_k4", "cluster_obj2_k4"]
    available_seg_cols = [c for c in available_seg_cols if c in df_f.columns and df_f[c].notna().any()]

    if not available_seg_cols:
        st.info("📄 Não existem colunas de segmento disponíveis.")
    else:
        seg_col = st.selectbox("Segmentação", available_seg_cols, index=0, key="group_seg_col_manter")
        seg_values = sorted(df_f[seg_col].dropna().unique().tolist())
        seg_value = st.selectbox("Selecione o Segmento", seg_values, index=0, key="group_seg_value_manter")

        df_seg_group = df_f[df_f[seg_col] == seg_value]
        
        if df_seg_group.empty:
            st.warning("📄 Nenhum cliente neste segmento.")
        else:
            # Mostrar perfil atual do segmento
            st.markdown("---")
            st.subheader("📊 Perfil Atual do Segmento")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total de Clientes", len(df_seg_group))
            col2.metric("Bem-estar Médio", f"{df_seg_group['mw_pred_reg_lin'].mean():.2f}")
            col3.metric("Ecrã Médio (h)", f"{df_seg_group['screen_time_hours'].mean():.1f}")
            col4.metric("Sono Médio (h)", f"{df_seg_group['sleep_hours'].mean():.1f}")
            col5.metric("Stress Médio", f"{df_seg_group['stress_level_0_10'].mean():.1f}")

            # Classes atuais no segmento
            if "mw_class_pred_rf" in df_seg_group.columns:
                class_dist = df_seg_group["mw_class_pred_rf"].value_counts()
                st.markdown("**Distribuição de Classes (Atual):**")
                for cls, count in class_dist.items():
                    pct = (count / len(df_seg_group)) * 100
                    st.write(f"- {cls}: {count} clientes ({pct:.1f}%)")

            # Recomendações do segmento
            label_manter, measures_manter = segment_suggestion(seg_col, seg_value)
            st.markdown("---")
            st.info(f"🎯 **Perfil do Segmento:** {label_manter}")
            if measures_manter:
                st.markdown("**📝 Recomendações de Intervenção para este Segmento:**")
                for i, m in enumerate(measures_manter, start=1):
                    st.markdown(f"{i}. {m}")
                st.caption("💡 Use os ajustes abaixo para simular a aplicação destas recomendações a todo o grupo")

            st.markdown("---")
            st.subheader("⚙️ Ajustes do Grupo")
            st.caption("Defina os ajustes que devem ser aplicados a TODOS os clientes do segmento")

            with st.form("group_adjustment_form_manter"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    work_screen_delta = st.slider(
                        "Ajuste de Ecrã de Trabalho (h/dia)",
                        -5.0, 5.0, 0.0, 0.1,
                        help="Positivo = aumentar, Negativo = diminuir",
                        key="ws_delta_manter"
                    )
                    leisure_screen_delta = st.slider(
                        "Ajuste de Ecrã de Lazer (h/dia)",
                        -5.0, 5.0, 0.0, 0.1,
                        key="ls_delta_manter"
                    )
                    sleep_delta = st.slider(
                        "Ajuste de Sono (h/dia)",
                        -3.0, 3.0, 0.0, 0.1,
                        key="sleep_delta_manter"
                    )

                with col2:
                    sleep_quality_delta = st.slider(
                        "Ajuste de Qualidade do Sono (1-5)",
                        -2, 2, 0, 1,
                        key="sq_delta_manter"
                    )
                    exercise_delta = st.slider(
                        "Ajuste de Exercício (min/semana)",
                        -150, 150, 0, 10,
                        key="ex_delta_manter"
                    )
                social_delta = st.slider(
                    "Ajuste de Horas Sociais (h/semana)",
                    -10, 10, 0, 1,
                    key="social_delta_manter"
                )

                with col3:
                    stress_delta = st.slider(
                        "Ajuste de Stress (0-10)",
                        -5, 5, 0, 1,
                        key="stress_delta_manter"
                    )
                    productivity_delta = st.slider(
                        "Ajuste de Produtividade (0-10)",
                        -5, 5, 0, 1,
                        key="prod_delta_manter"
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
                original_avg_wellness = df_seg_group["mw_pred_reg_lin"].mean()
                new_avg_wellness = adjusted_group["mw_pred_reg_lin_novo"].mean()
                wellness_delta_group = new_avg_wellness - original_avg_wellness
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Bem-estar Médio (Antes)", f"{original_avg_wellness:.2f}")
                col2.metric("Bem-estar Médio (Depois)", f"{new_avg_wellness:.2f}")
                col3.metric("Mudança Média", f"{wellness_delta_group:+.2f}", 
                           delta_color="normal" if wellness_delta_group >= 0 else "inverse")
                col4.metric("% Clientes a Melhorar", 
                           f"{(adjusted_group['mw_pred_reg_lin_novo'] > df_seg_group['mw_pred_reg_lin']).sum() / len(adjusted_group) * 100:.1f}%")

                # Distribuição de classes antes/depois
                if classifier_model is not None:
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

                    st.dataframe(class_comparison, use_container_width=True)

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
                    st.plotly_chart(fig_class, use_container_width=True)

                # Comparação de métricas
                st.markdown("**Mudança nas Métricas do Grupo:**")
                metrics_comparison = pd.DataFrame({
                    "Métrica": [
                        "Ecrã de Trabalho (h/dia)",
                        "Ecrã de Lazer (h/dia)",
                        "Sono (h/dia)",
                        "Qualidade do Sono",
                        "Exercício (min/semana)",
                        "Horas Sociais",
                        "Stress",
                        "Produtividade"
                    ],
                    "Antes": [
                        df_seg_group["work_screen_hours"].mean(),
                        df_seg_group["leisure_screen_hours"].mean(),
                        df_seg_group["sleep_hours"].mean(),
                        df_seg_group["sleep_quality_1_5"].mean(),
                        df_seg_group["exercise_minutes_per_week"].mean(),
                        df_seg_group["social_hours_per_week"].mean(),
                        df_seg_group["stress_level_0_10"].mean(),
                        df_seg_group["productivity_0_10"].mean(),
                    ],
                    "Depois": [
                        adjusted_group["work_screen_hours"].mean(),
                        adjusted_group["leisure_screen_hours"].mean(),
                        adjusted_group["sleep_hours"].mean(),
                        adjusted_group["sleep_quality_1_5"].mean(),
                        adjusted_group["exercise_minutes_per_week"].mean(),
                        adjusted_group["social_hours_per_week"].mean(),
                        adjusted_group["stress_level_0_10"].mean(),
                        adjusted_group["productivity_0_10"].mean(),
                    ]
                })
                metrics_comparison["Mudança"] = metrics_comparison["Depois"] - metrics_comparison["Antes"]

                st.dataframe(metrics_comparison, use_container_width=True)

                # Guardar dados do grupo no estado para reutilizar mesmo após rerun
                export_cols = ["row_id", "mw_pred_reg_lin", "mw_pred_reg_lin_novo", "mw_class_pred_rf", "mw_class_pred_rf_novo",
                              "mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto",
                              "work_screen_hours", "leisure_screen_hours", "sleep_hours", "sleep_quality_1_5",
                              "exercise_minutes_per_week", "social_hours_per_week", "stress_level_0_10", "productivity_0_10"]
                export_cols = [c for c in export_cols if c in adjusted_group.columns]
                st.session_state["adjusted_group_data"] = adjusted_group[export_cols]

        # Secção persistente para visualizar e guardar o grupo ajustado
        if "adjusted_group_data" in st.session_state:
            st.markdown("---")
            st.subheader("📋 Detalhes dos Clientes Ajustados (Grupo)")

            adjusted_group_view = st.session_state["adjusted_group_data"]
            st.dataframe(adjusted_group_view, use_container_width=True)
            st.caption(f"Dados prontos para guardar: {len(adjusted_group_view)} linha(s), {len(adjusted_group_view.columns)} coluna(s)")
            st.write("Colunas:", list(adjusted_group_view.columns))

            st.markdown("---")
            col_csv, col_db = st.columns(2)

            with col_csv:
                csv_data = adjusted_group_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descarregar Plano de Intervenção em Grupo (CSV)",
                    data=csv_data,
                    file_name=f"grupo_ajustado_segmento_{seg_value}.csv",
                    mime="text/csv"
                )

            with col_db:
                status_placeholder_group = st.empty()

                if "save_status_group" in st.session_state:
                    msgg = st.session_state["save_status_group"]
                    if msgg.get("type") == "success":
                        status_placeholder_group.success(msgg.get("message", ""))
                    elif msgg.get("type") == "error":
                        status_placeholder_group.error(msgg.get("message", ""))
                        if msgg.get("traceback"):
                            st.code(msgg["traceback"])
                    else:
                        status_placeholder_group.info(msgg.get("message", ""))

                if st.button("💿 Guardar Grupo na Base de Dados", key="save_group_db"):
                    db_path_group = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
                    df_to_save_group = st.session_state.get("adjusted_group_data")
                    id_col_group = "row_id" if "row_id" in df.columns else None

                    st.session_state["save_status_group"] = {"type": "info", "message": "🔄 A processar..."}
                    status_placeholder_group.info("🔄 A processar...")

                    try:
                        import os
                        if not os.path.exists(db_path_group):
                            st.session_state["save_status_group"] = {
                                "type": "error",
                                "message": f"❌ Base de dados não encontrada: {db_path_group}"
                            }
                            status_placeholder_group.error(st.session_state["save_status_group"]["message"])
                        elif not id_col_group or id_col_group not in df_to_save_group.columns:
                            st.session_state["save_status_group"] = {
                                "type": "error",
                                "message": "❌ Não foi possível guardar: coluna row_id em falta."
                            }
                            status_placeholder_group.error(st.session_state["save_status_group"]["message"])
                        else:
                            conn = sqlite3.connect(db_path_group)
                            try:
                                # Base: sempre ler Deployment_Output (original)
                                base_df = pd.read_sql_query("SELECT * FROM Deployment_Output", conn)
                                # Ajustado acumulado: usar tabela ajustada se existir, senão copiar base
                                try:
                                    adjusted_df = pd.read_sql_query("SELECT * FROM Deployment_Output_Adjusted", conn)
                                except Exception:
                                    adjusted_df = base_df.copy()

                                merged = merge_adjusted_rows(adjusted_df, df_to_save_group, id_col_group)
                                merged.to_sql("Deployment_Output_Adjusted", conn, if_exists="replace", index=False)
                                conn.commit()

                                st.session_state["save_status_group"] = {
                                    "type": "success",
                                    "message": f"✅ Grupo guardado com sucesso na tabela 'Deployment_Output_Adjusted'! Tabela completa com {len(merged)} linha(s)."
                                }
                                status_placeholder_group.success(st.session_state["save_status_group"]["message"])
                            finally:
                                conn.close()

                    except Exception as e:
                        import traceback
                        st.session_state["save_status_group"] = {
                            "type": "error",
                            "message": f"❌ Erro ao guardar grupo: {str(e)}",
                            "traceback": traceback.format_exc()
                        }
                        status_placeholder_group.error(st.session_state["save_status_group"]["message"])
                        st.code(st.session_state["save_status_group"].get("traceback", ""))

                    msgg = st.session_state.get("save_status_group", {})
                    if msgg.get("type") == "success":
                        status_placeholder_group.success(msgg.get("message", ""))
                    elif msgg.get("type") == "error":
                        status_placeholder_group.error(msgg.get("message", ""))
                        if msgg.get("traceback"):
                            st.code(msgg["traceback"])
                    else:
                        status_placeholder_group.info(msgg.get("message", ""))
        else:
            st.info("ℹ Calcule o impacto do grupo para ativar o botão de guardar.")
