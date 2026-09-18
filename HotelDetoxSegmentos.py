# medidas_segmento_db.py
# Dashboard Streamlit (SQLite) — Medidas corretas por segmento de clientes
# Fonte: tabela "Deployment_Output" (ou outra que indiques)
#
# Segmentação suportada:
# - cluster_obj2_k4 (Objetivo 2)  -> recomendações 1:1 por cluster
# - cluster_obj1_k4 (Objetivo 1)  -> recomendações 1:1 por cluster (ATUALIZADO com as tuas conclusões)
#
# Requisitos:
# pip install streamlit pandas numpy plotly

import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import os

# =========================
# Config
# =========================
st.set_page_config(page_title="Hotel Detox | Medidas por Segmento", layout="wide")
st.title("Hotel Detox — Recomendações Personalizadas por Segmento")
st.caption("Análise baseada no perfil e necessidades de cada grupo de clientes")

# =========================
# Defaults
# =========================
DB_DEFAULT = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
TABLE_DEFAULT = "Deployment_Output"

# Colunas esperadas (ajusta se o teu output usar outros nomes)
FEATURES = [
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

WELL_REAL = "mental_wellness_index_0_10"  # pode não existir no Deployment_Output (se removeres target)
WELL_PRED = "mw_pred_reg_lin"
CLASS_PRED = "mw_class_pred_rf"
PROB_ALTO = "mw_prob_Alto"
PROB_BAIXO = "mw_prob_Baixo"

CLUSTER_OBJ2 = "cluster_obj2_k4"
CLUSTER_OBJ1 = "cluster_obj1_k4"

# =========================
# OBJ2 — Medidas por cluster (1:1)
# =========================
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

# =========================
# OBJ1 — Medidas por cluster (1:1) — ATUALIZADO com as tuas conclusões
# =========================
# Nota importante:
# Os IDs dos clusters (0/1/2/3) são arbitrários do KMeans. Esta tabela assume que, no teu treino,
# o "cluster_obj1_k4" tem exatamente a interpretação que descreveste:
#   - Cluster 0: pior perfil (stress extremo, sono fraco, ecrãs altos, pouco exercício/social)
#   - Cluster 3: social alto mas stress alto e sono fraco (bem-estar baixo)
#   - Cluster 1: exercício muito alto, social baixo, sono médio, stress moderado-alto (bem-estar médio)
#   - Cluster 2: perfil ideal (sono muito bom, stress baixo, ecrãs baixos, prod alta) (bem-estar mais alto)
#
# Se mudares o treino / seed / features do Obj1, confirma se os IDs mantêm o mesmo significado.
CLUSTER_LABELS_OBJ1 = {
    0: "Crítico: stress extremo + sono fraco + ecrãs altos + baixo exercício/social",
    1: "Físico forte, social baixo: exercício alto mitiga parcialmente stress",
    2: "Perfil ideal: sono muito bom + stress baixo + ecrãs baixos + produtividade alta",
    3: "Social alto mas stress alto e sono fraco: bem-estar baixo",
}

MEASURES_BY_CLUSTER_OBJ1 = {
    # Cluster 0 (pior)
    0: [
        "Intervenção prioritária: programa anti-stress intensivo (mindfulness diário + respiração + relaxamento guiado)",
        "Plano de sono estruturado (higiene do sono + redução total de ecrãs 60–90 min antes de dormir + rotina consistente)",
        "Detox digital forte (janelas sem ecrãs; limitar lazer digital; bloqueio de notificações)",
        "Plano de atividade física progressivo (iniciar com caminhada diária + 2 sessões leves/semana)",
        "Plano de socialização assistida (atividades em grupo pequenas + refeições partilhadas sem ecrãs)",
        "Acompanhamento frequente (check-ins curtos 2–3×/semana) e monitorização de stress",
    ],
    # Cluster 3 (social alto mas stress alto e sono fraco)
    3: [
        "Foco principal: reduzir stress (protocolos de relaxamento; sauna/massagem; mindfulness diário)",
        "Reforçar sono (higiene do sono; otimização do quarto; consistência horários)",
        "Reequilibrar socialização: manter social, mas com atividades de baixa estimulação (natureza, caminhada, yoga em grupo)",
        "Limitar ecrãs à noite (regras de uso e substituição por rotinas relaxantes)",
        "Micro-pausas estruturadas e redução de multitasking",
    ],
    # Cluster 1 (exercício alto, social baixo, sono médio, stress moderado-alto)
    1: [
        "Manter exercício (é um ativo do segmento): plano de manutenção e prevenção de lesão",
        "Aumentar socialização de forma incremental (atividades em grupo 2×/semana; social hour; experiências partilhadas)",
        "Intervenções leves de stress (mindfulness curta; educação sobre gestão de agenda e carga cognitiva)",
        "Ajustes no sono (ritual noturno simples; redução de ecrãs noturnos; exposição à luz natural pela manhã)",
        "Coaching de foco e produtividade (blocos de trabalho profundo + pausas)",
    ],
    # Cluster 2 (ideal)
    2: [
        "Manutenção do estilo de vida: reforçar hábitos atuais (sono consistente + exercício + socialização saudável)",
        "Oferta premium: experiências avançadas (natureza, mindfulness, workshops de performance sustentável)",
        "Plano de prevenção: check-in semanal de stress/hábitos digitais; manutenção de limites de ecrã",
        "Fidelização: benefícios e recomendações personalizadas para manter o perfil ideal",
    ],
}

# =========================
# Helpers
# =========================
def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_mean(series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_rate(series, threshold=8.5) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(100.0 * (s >= threshold).mean()) if len(s) else np.nan


def class_rate(df: pd.DataFrame, col=CLASS_PRED, label="Alto") -> float:
    if col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(100.0 * (s == label).mean()) if len(s) else np.nan


def prob_mean_pct(df: pd.DataFrame, prob_col=PROB_ALTO) -> float:
    if prob_col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[prob_col], errors="coerce").dropna()
    return float(100.0 * s.mean()) if len(s) else np.nan


def safe_slider_value(val, min_val, max_val, step, fallback):
    """Clamp and align slider default to range/step to avoid frontend warnings."""
    try:
        v = float(val)
        if not np.isfinite(v):
            raise ValueError()
    except Exception:
        v = fallback

    v = min(max(v, min_val), max_val)
    try:
        n = round((v - min_val) / step)
        v_aligned = min(max(min_val + n * step, min_val), max_val)
        return v_aligned
    except Exception:
        return v


def segment_profile(df: pd.DataFrame, seg_col: str, seg_value, feature_cols: list[str]):
    base = df.copy()
    seg = df[df[seg_col] == seg_value].copy()

    prof_base = base[feature_cols].mean(numeric_only=True)
    prof_seg = seg[feature_cols].mean(numeric_only=True)

    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "media_segmento": [prof_seg.get(c, np.nan) for c in feature_cols],
            "media_total": [prof_base.get(c, np.nan) for c in feature_cols],
        }
    )
    out["delta"] = out["media_segmento"] - out["media_total"]
    return out, seg


def measures_for_segment(seg_col: str, seg_value):
    """Regras 1:1 por cluster, dependentes do objetivo."""
    seg_value_int = int(seg_value)

    if seg_col == CLUSTER_OBJ2:
        label = CLUSTER_LABELS_OBJ2.get(seg_value_int, f"Cluster {seg_value_int}")
        measures = MEASURES_BY_CLUSTER_OBJ2.get(seg_value_int, MEASURES_BY_CLUSTER_OBJ2[1])
        drivers = [f"Segmento (Obj2) identificado: {label}"]
        objective_label = "Objetivo 2 (contexto + comportamento)"
        return drivers, measures, label, objective_label

    if seg_col == CLUSTER_OBJ1:
        label = CLUSTER_LABELS_OBJ1.get(seg_value_int, f"Cluster {seg_value_int}")
        measures = MEASURES_BY_CLUSTER_OBJ1.get(seg_value_int, MEASURES_BY_CLUSTER_OBJ1[1])
        drivers = [f"Segmento (Obj1) identificado: {label}"]
        objective_label = "Objetivo 1 (comportamental)"
        return drivers, measures, label, objective_label

    # fallback
    return (["Segmento identificado."], ["Definir regras específicas por cluster."], f"Cluster {seg_value_int}", "—")


@st.cache_data(show_spinner=True)
def load_table(db_path: str, table_name: str) -> pd.DataFrame:
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
    """Prever classe e probabilidades de pertença."""
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


# =========================
# Sidebar: fonte de dados
# =========================
st.sidebar.header("Fonte de Dados")
db_path = st.sidebar.text_input("Caminho da Base de Dados", value=DB_DEFAULT)
table_name = st.sidebar.text_input("Tabela de Dados", value=TABLE_DEFAULT)

try:
    df0 = load_table(db_path, table_name).copy()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if df0.empty:
    st.warning("⚠ Não há dados disponíveis.")
    st.stop()

# Tipos numéricos + derivados
df0 = to_numeric(df0, FEATURES + [WELL_REAL, WELL_PRED, PROB_ALTO, PROB_BAIXO])
if "sleep_hours" in df0.columns and "sleep_quality_1_5" in df0.columns:
    df0["sleep_index"] = df0["sleep_hours"] + df0["sleep_quality_1_5"]

# =========================
# Sidebar: escolher segmentação
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Perfil de Clientes")

available_clusters = [c for c in [CLUSTER_OBJ2, CLUSTER_OBJ1] if c in df0.columns and df0[c].notna().any()]
if not available_clusters:
    st.error("⚠ Não foram encontrados perfis de segmentação nos dados.")
    st.stop()

default_seg_col = CLUSTER_OBJ2 if CLUSTER_OBJ2 in available_clusters else available_clusters[0]
seg_col = st.sidebar.selectbox(
    "Tipo de Segmentação",
    options=available_clusters,
    index=available_clusters.index(default_seg_col),
    help="Escolha o critério de segmentação de clientes"
)

seg_values = sorted(df0[seg_col].dropna().unique().tolist())
seg_value = st.sidebar.selectbox(
    "Segmento de Cliente",
    options=seg_values,
    index=0,
    help="Selecione o grupo de clientes para análise"
)

# =========================
# Sidebar: filtros opcionais
# =========================
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
st.sidebar.subheader("Filtros Adicionais")

# filtro por nível de bem-estar (opcional)
if CLASS_PRED in df0.columns and df0[CLASS_PRED].notna().any():
    classes = sorted(df0[CLASS_PRED].dropna().unique().tolist())
    sel_classes = st.sidebar.multiselect(
        "Nível de Bem-estar",
        options=classes,
        default=classes,
        help="Filtrar por classificação de bem-estar"
    )
else:
    sel_classes = None

threshold = st.sidebar.slider(
    "Limiar de bem-estar elevado (0–10)",
    0.0, 10.0, 8.5, 0.1,
    help="Valores acima deste limiar são considerados bem-estar elevado"
)

# Business Objective
st.sidebar.markdown("---")
st.sidebar.subheader("Objetivo de Negócio")
st.sidebar.markdown("**Meta por segmento:** Taxa de sucesso")

target_wellness_pct = st.sidebar.number_input(
    "Meta: % mínima de bem-estar elevado",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=1.0,
    help="Objetivo: pelo menos esta % do segmento com índice de bem-estar acima do limiar"
)

target_high_class_pct = st.sidebar.number_input(
    "Meta: % mínima de clientes com elevado bem-estar",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=1.0,
    help="Objetivo: pelo menos esta % do segmento classificado com bem-estar elevado"
)

df = df0.copy()
if sel_classes is not None:
    df = df[df[CLASS_PRED].isin(sel_classes)]

if "screen_time_hours" in df.columns and df["screen_time_hours"].notna().any():
    st_min = float(df["screen_time_hours"].min())
    st_max = float(df["screen_time_hours"].max())
    a, b = st.sidebar.slider(
        "Tempo de ecrã (horas/dia)",
        min_value=st_min,
        max_value=st_max,
        value=(st_min, st_max),
        help="Filtrar por tempo diário de uso de ecrãs"
    )
    df = df[(df["screen_time_hours"] >= a) & (df["screen_time_hours"] <= b)]

if df.empty:
    st.warning("⚠ Nenhum dado corresponde aos filtros selecionados. Ajuste os critérios.")
    st.stop()

# =========================
# Perfil do segmento + medidas
# =========================
feature_cols = [c for c in FEATURES if c in df.columns]
profile_df, df_seg = segment_profile(df, seg_col, seg_value, feature_cols)

drivers, measures, cluster_label, objective_label = measures_for_segment(seg_col, seg_value)

# =========================
# Header + KPIs
# =========================
st.caption(f"👥 Segmento: {cluster_label} | {objective_label}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clientes no Segmento", f"{len(df_seg)}")
k2.metric("Total de Clientes", f"{len(df)}")

# KPI 3: Taxa bem-estar previsto (deployment)
if WELL_PRED in df.columns and df_seg[WELL_PRED].notna().any():
    wellness_rate = safe_rate(df_seg[WELL_PRED], threshold)
    wellness_meets_target = wellness_rate >= target_wellness_pct
    wellness_delta = wellness_rate - target_wellness_pct
    k3.metric(
        f"Taxa bem-estar ≥ {threshold:.1f}",
        f"{wellness_rate:.1f}%",
        delta=f"{wellness_delta:+.1f}pp vs meta ({target_wellness_pct:.0f}%)",
        delta_color="normal",
        help=f"{'✓ Meta atingida' if wellness_meets_target else '✗ Abaixo da meta'}"
    )
    wellness_available = True
else:
    k3.metric("Taxa bem-estar", "Indisponível")
    wellness_available = False
    wellness_meets_target = False
    wellness_rate = np.nan

# KPI 4: Classificação de bem-estar elevado
if CLASS_PRED in df.columns:
    high_class_rate = class_rate(df_seg)
    high_class_meets_target = high_class_rate >= target_high_class_pct
    high_class_delta = high_class_rate - target_high_class_pct
    k4.metric(
        "% Bem-estar Elevado",
        f"{high_class_rate:.1f}%",
        delta=f"{high_class_delta:+.1f}pp vs meta ({target_high_class_pct:.0f}%)",
        delta_color="normal",
        help=f"Clientes classificados como bem-estar elevado | {'✓ Meta atingida' if high_class_meets_target else '✗ Abaixo da meta'}"
    )
    class_available = True
else:
    k4.metric("% Bem-estar Elevado", "Indisponível")
    class_available = False
    high_class_meets_target = False
    high_class_rate = np.nan

k5.metric("Probabilidade de Sucesso", f"{prob_mean_pct(df_seg):.1f}%" if PROB_ALTO in df.columns else "Indisponível")

# Status do objetivo de negócio
st.markdown("---")

if wellness_available and class_available:
    both_met = wellness_meets_target and high_class_meets_target
    if both_met:
        st.success(
            f"✓ 🎯 Objetivos atingidos neste segmento! {wellness_rate:.1f}% atingem bem-estar elevado (meta: {target_wellness_pct:.0f}%) "
            f"e {high_class_rate:.1f}% classificados com bem-estar elevado (meta: {target_high_class_pct:.0f}%)"
        )
    elif wellness_meets_target:
        st.warning(
            f"⚠ Progresso parcial: Índice de bem-estar bom ({wellness_rate:.1f}% ≥ {target_wellness_pct:.0f}%), "
            f"mas percentual de bem-estar elevado precisa melhorar ({high_class_rate:.1f}% < {target_high_class_pct:.0f}%)"
        )
    elif high_class_meets_target:
        st.warning(
            f"⚠ Progresso parcial: Percentual de bem-estar elevado bom ({high_class_rate:.1f}% ≥ {target_high_class_pct:.0f}%), "
            f"mas índice geral precisa melhorar ({wellness_rate:.1f}% < {target_wellness_pct:.0f}%)"
        )
    else:
        gap_wellness = target_wellness_pct - wellness_rate
        gap_class = target_high_class_pct - high_class_rate
        st.error(
            f"✗ Segmento necessita intervenção: Apenas {wellness_rate:.1f}% com índice elevado (meta: {target_wellness_pct:.0f}%, faltam {gap_wellness:.1f}pp) "
            f"e {high_class_rate:.1f}% classificados como elevado (meta: {target_high_class_pct:.0f}%, faltam {gap_class:.1f}pp)"
        )
elif wellness_available:
    if wellness_meets_target:
        st.success(f"✓ Meta de bem-estar atingida: {wellness_rate:.1f}% dos clientes com índice elevado (meta: {target_wellness_pct:.0f}%)")
    else:
        gap_wellness = target_wellness_pct - wellness_rate
        st.error(f"✗ Segmento abaixo da meta: {wellness_rate:.1f}% com bem-estar elevado (meta: {target_wellness_pct:.0f}%, faltam {gap_wellness:.1f}pp)")
elif class_available:
    if high_class_meets_target:
        st.success(f"✓ Meta de classificação atingida: {high_class_rate:.1f}% com bem-estar elevado (meta: {target_high_class_pct:.0f}%)")
    else:
        gap_class = target_high_class_pct - high_class_rate
        st.error(f"✗ Segmento abaixo da meta: {high_class_rate:.1f}% com bem-estar elevado (meta: {target_high_class_pct:.0f}%, faltam {gap_class:.1f}pp)")
else:
    st.warning("⚠ Não há dados suficientes para avaliar os objetivos deste segmento")

# =========================
# KPI 2: High-Risk Intervention Alignment
# =========================
st.markdown("---")
st.subheader("🎯 KPI: Alinhamento de Intervenções para Clientes de Risco")
st.caption("Validação de que clientes de alto risco estão em segmentos com intervenções apropriadas")

# Configuration
high_risk_threshold = 6.0  # Clients with wellness < 6.0 are considered high-risk
intervention_segments = [0, 1, 2]  # Segments with active interventions (not maintenance)

if WELL_PRED in df.columns and df[WELL_PRED].notna().any():
    # Calculate overall (across all data, not just current segment view)
    high_risk_clients = df[df[WELL_PRED] < high_risk_threshold]
    
    if len(high_risk_clients) > 0:
        # How many high-risk clients are in intervention segments?
        in_intervention = high_risk_clients[high_risk_clients[seg_col].isin(intervention_segments)]
        kpi2_alignment = (len(in_intervention) / len(high_risk_clients)) * 100
        
        # Target
        kpi2_target = 85.0
        kpi2_meets_target = kpi2_alignment >= kpi2_target
        kpi2_delta = kpi2_alignment - kpi2_target
        
        # Display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Clientes de Risco", f"{len(high_risk_clients)}", 
                    help=f"Clientes com bem-estar previsto < {high_risk_threshold:.1f}")
        col2.metric("Em Segmentos de Intervenção", f"{len(in_intervention)}", 
                    help=f"Clientes de risco nos segmentos {intervention_segments}")
        col3.metric(
            "Taxa de Alinhamento",
            f"{kpi2_alignment:.1f}%",
            delta=f"{kpi2_delta:+.1f}pp vs meta ({kpi2_target:.0f}%)",
            delta_color="normal",
            help="% de clientes de risco corretamente atribuídos a segmentos com intervenções ativas"
        )
        
        if kpi2_meets_target:
            col4.markdown("### ✅")
            col4.caption("Meta atingida")
        else:
            col4.markdown("### ⚠️")
            col4.caption("Abaixo da meta")
        
        # Visual breakdown
        st.markdown("**Distribuição de Clientes de Risco por Segmento:**")
        risk_by_segment = high_risk_clients.groupby(seg_col).size().reset_index(name='count')
        risk_by_segment['segment_type'] = risk_by_segment[seg_col].apply(
            lambda x: 'Intervenção Ativa' if x in intervention_segments else 'Manutenção'
        )
        risk_by_segment['percentage'] = (risk_by_segment['count'] / len(high_risk_clients) * 100).round(1)
        
        fig_kpi2 = px.bar(
            risk_by_segment,
            x=seg_col,
            y='count',
            color='segment_type',
            title=f"Clientes de risco (bem-estar < {high_risk_threshold}) por segmento",
            labels={seg_col: 'Segmento', 'count': 'Número de Clientes'},
            color_discrete_map={'Intervenção Ativa': '#e74c3c', 'Manutenção': '#2ecc71'}
        )
        fig_kpi2.update_layout(height=400)
        st.plotly_chart(fig_kpi2, use_container_width=True)
        
        # Summary message
        if kpi2_meets_target:
            st.success(
                f"✓ KPI Atingido: {kpi2_alignment:.1f}% dos clientes de risco estão em segmentos com intervenções ativas. "
                f"Isto valida que a segmentação está a identificar corretamente quem precisa de ajuda."
            )
        else:
            gap = kpi2_target - kpi2_alignment
            st.warning(
                f"⚠ Atenção: Apenas {kpi2_alignment:.1f}% dos clientes de risco estão em segmentos de intervenção (meta: {kpi2_target:.0f}%). "
                f"Faltam {gap:.1f}pp. Considere rever os critérios de segmentação."
            )
    else:
        st.info("✓ Não há clientes de risco identificados nos dados filtrados (todos têm bem-estar ≥ 6.0)")
else:
    st.warning("⚠ Dados de bem-estar previsto não disponíveis para calcular este KPI")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Recomendações", "📊 Perfil do Segmento", "🔍 Análise & Comparações", "🧪 Ajustar Cliente", "🎯 Planeamento em Grupo"])

# -------------------------
# Tab 1: recomendações
# -------------------------
with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Características Principais do Segmento")
        for d in drivers:
            st.write(f"• {d}")

        st.subheader("Programa de Intervenção Recomendado")
        for i, m in enumerate(measures, start=1):
            st.write(f"**{i}.** {m}")

        st.markdown("---")
        st.caption(
            "💡 As recomendações são personalizadas para cada segmento com base na análise dos padrões "
            "de comportamento e bem-estar identificados."
        )

    with right:
        st.subheader("Distribuição do Nível de Bem-estar")
        if CLASS_PRED in df_seg.columns and df_seg[CLASS_PRED].notna().any():
            cnt = df_seg[CLASS_PRED].value_counts().reset_index()
            cnt.columns = ["Nível", "Quantidade"]
            fig = px.pie(cnt, names="Nível", values="Quantidade", title="Níveis de bem-estar no segmento")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Informação de classificação não disponível.")

        st.subheader("Probabilidade de Bem-estar Elevado")
        if PROB_ALTO in df_seg.columns and df_seg[PROB_ALTO].notna().any():
            fig2 = px.histogram(df_seg, x=PROB_ALTO, nbins=20, title="Distribuição da probabilidade de bem-estar elevado")
            fig2.update_layout(height=380, xaxis_title="Probabilidade (0-1)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Informação de probabilidade não disponível.")

# -------------------------
# Tab 2: perfil do segmento
# -------------------------
with tab2:
    st.subheader("📊 Características do Segmento")
    st.caption("Comparação das características médias deste segmento com a média geral")
    plot_df = profile_df.sort_values("delta", ascending=False).copy()

    fig = px.bar(
        plot_df,
        x="feature",
        y="delta",
        title="Diferença em relação à média geral",
    )
    fig.update_layout(height=420, xaxis_title="Característica", yaxis_title="Diferença")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Tabela Detalhada")
    st.caption("Médias do segmento comparadas com a média total")
    st.dataframe(profile_df.sort_values("delta", ascending=False), use_container_width=True)

    st.subheader("✅ Principais Indicadores do Segmento")
    st.caption("Médias das variáveis-chave neste grupo de clientes")
    cols_check = [c for c in [
        "screen_time_hours",
        "sleep_hours",
        "sleep_quality_1_5",
        "sleep_index",
        "exercise_minutes_per_week",
        "social_hours_per_week",
        "stress_level_0_10",
        "productivity_0_10",
        WELL_PRED,
        WELL_REAL,
    ] if c in df_seg.columns]

    if cols_check:
        check = df_seg[cols_check].mean(numeric_only=True).to_frame("media_segmento")
        st.dataframe(check, use_container_width=True)
    else:
        st.info("Não há colunas suficientes para validação automática.")

# -------------------------
# Tab 3: evidência e comparações
# -------------------------
with tab3:
    st.subheader("🔍 Análise Exploratória")
    st.caption("Explore as relações entre variáveis e bem-estar")

    y_options = []
    if WELL_REAL in df.columns and df[WELL_REAL].notna().any():
        y_options.append(WELL_REAL)
    if WELL_PRED in df.columns and df[WELL_PRED].notna().any():
        y_options.append(WELL_PRED)

    if not y_options:
        st.info("📄 Dados de bem-estar não disponíveis para análise gráfica.")
    else:
        y_choice = st.selectbox("Eixo Y (Bem-estar)", options=y_options, index=0)
        x_choice = st.selectbox("Eixo X (Variável)", options=feature_cols + (["sleep_index"] if "sleep_index" in df.columns else []), index=0)

        fig = px.scatter(
            df,
            x=x_choice,
            y=y_choice,
            color=seg_col,
            title=f"Relação entre {x_choice} e bem-estar por segmento",
            opacity=0.65
        )
        fig.update_layout(height=520, xaxis_title=x_choice, yaxis_title="Índice de Bem-estar")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Comparação entre Segmentos")
    st.caption("Taxas de sucesso e indicadores médios por segmento")

    agg_dict = {
        "n": (seg_col, "size"),
    }

    if CLASS_PRED in df.columns and df[CLASS_PRED].notna().any():
        agg_dict["rf_alto_pct"] = (
            CLASS_PRED,
            lambda s: 100.0 * (s.dropna() == "Alto").mean() if s.dropna().shape[0] else np.nan,
        )

    if PROB_ALTO in df.columns and df[PROB_ALTO].notna().any():
        agg_dict["p_alto_pct"] = (
            PROB_ALTO,
            lambda s: 100.0 * pd.to_numeric(s, errors="coerce").mean(),
        )

    if WELL_REAL in df.columns and df[WELL_REAL].notna().any():
        agg_dict["bem_real_media"] = (WELL_REAL, "mean")

    if WELL_PRED in df.columns and df[WELL_PRED].notna().any():
        agg_dict["bem_prev_media"] = (WELL_PRED, "mean")

    g = df.groupby(seg_col, dropna=False).agg(**agg_dict).reset_index()

    # Gráfico principal (preferência: p_alto_pct; depois rf_alto_pct)
    if "p_alto_pct" in g.columns:
        fig2 = px.bar(g, x=seg_col, y="p_alto_pct", title="Probabilidade de bem-estar elevado por segmento")
        fig2.update_layout(height=420, yaxis_title="Probabilidade Média (%)")
        st.plotly_chart(fig2, use_container_width=True)
    elif "rf_alto_pct" in g.columns:
        fig2 = px.bar(g, x=seg_col, y="rf_alto_pct", title="% de clientes com bem-estar elevado por segmento")
        fig2.update_layout(height=420, yaxis_title="Percentual (%)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📄 Dados insuficientes para comparação entre segmentos.")

    st.dataframe(g, use_container_width=True)

# -------------------------
# Tab 4: Ajustar Cliente
# -------------------------
with tab4:
    st.subheader("🧪 Ajustar um cliente com base no segmento")

    id_col = "row_id" if "row_id" in df_seg.columns else None
    client_options = df_seg[id_col].tolist() if id_col else df_seg.index.tolist()

    if client_options:
        client_id = st.selectbox(
            "Selecione um cliente",
            client_options,
            format_func=lambda x: f"Cliente {x}"
        )

        if id_col:
            client_row = df_seg[df_seg[id_col] == client_id].iloc[0]
        else:
            client_row = df_seg.loc[client_id]

        st.success(f"Sugestão do segmento: {cluster_label}")
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
                    0.0, 14.0, safe_slider_value(client_row.get("work_screen_hours", 0.0), 0.0, 14.0, 0.1, 0.0), 0.1
                )
                leisure_screen = st.slider(
                    "Tempo de ecrã no lazer (h/dia)",
                    0.0, 14.0, safe_slider_value(client_row.get("leisure_screen_hours", 0.0), 0.0, 14.0, 0.1, 0.0), 0.1
                )
                screen_total = work_screen + leisure_screen
                st.caption(f"Tempo de ecrã total calculado: {screen_total:.1f} h/dia")

            with col2:
                sleep_hours = st.slider(
                    "Horas de sono", 3.0, 10.0, safe_slider_value(client_row.get("sleep_hours", 7.0), 3.0, 10.0, 0.1, 7.0), 0.1
                )
                sleep_quality = st.slider(
                    "Qualidade do sono (1–5)", 1, 5, int(safe_slider_value(client_row.get("sleep_quality_1_5", 3), 1, 5, 1, 3)), 1
                )
                exercise = st.slider(
                    "Exercício (min/semana)", 0, 600, int(safe_slider_value(client_row.get("exercise_minutes_per_week", 150), 0, 600, 10, 150)), 10
                )

            with col3:
                social = st.slider(
                    "Horas sociais/semana", 0, 40, int(safe_slider_value(client_row.get("social_hours_per_week", 8), 0, 40, 1, 8)), 1
                )
                stress = st.slider(
                    "Stress (0–10)", 0, 10, int(safe_slider_value(client_row.get("stress_level_0_10", 5), 0, 10, 1, 5)), 1
                )
                productivity = st.slider(
                    "Produtividade (0–10)", 0, 10, int(safe_slider_value(client_row.get("productivity_0_10", 6), 0, 10, 1, 6)), 1
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

            # Calcular previsão de bem-estar e classe para os valores ajustados
            adjusted_wellness = predict_wellness(regression_model, adjusted)
            adjusted_class, prob_cols = predict_class_with_proba(classifier_model, adjusted)
            if adjusted_wellness is not None:
                adjusted["mw_pred_reg_lin"] = adjusted_wellness
            if adjusted_class is not None:
                adjusted["mw_class_pred_rf"] = adjusted_class
            if prob_cols:
                adjusted.update(prob_cols)

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
            st.dataframe(comparison_df, use_container_width=True)

            st.markdown("---")
            st.markdown("**Pré-visualização do cliente ajustado**")
            df_out = pd.DataFrame([adjusted])
            st.dataframe(df_out, use_container_width=True)

            # Store adjusted client in session state for persistent save button
            st.session_state["adjusted_client_data"] = df_out
            st.session_state["adjusted_client_id"] = client_id

        # Persistent save section (outside form submit block)
        if "adjusted_client_data" in st.session_state:
            st.markdown("---")
            col_csv, col_db = st.columns(2)
            
            with col_csv:
                df_out_persist = st.session_state["adjusted_client_data"]
                csv_data = df_out_persist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descarregar cliente ajustado (CSV)",
                    data=csv_data,
                    file_name="cliente_ajustado.csv",
                    mime="text/csv",
                    key="download_client_segmentos"
                )
            
            with col_db:
                save_status_client = st.empty()
                
                if "save_status_client" in st.session_state:
                    msg = st.session_state["save_status_client"]
                    if msg.get("type") == "success":
                        save_status_client.success(msg.get("message", ""))
                    elif msg.get("type") == "error":
                        save_status_client.error(msg.get("message", ""))
                        if msg.get("traceback"):
                            st.code(msg["traceback"])
                
                if st.button("💿 Guardar cliente na Base de Dados", key=f"save_client_persistent"):
                    df_out = st.session_state["adjusted_client_data"]
                    db_path_client = db_path
                    id_col_client = "row_id" if "row_id" in df_out.columns else None

                    st.session_state["save_status_client"] = {"type": "info", "message": "🔄 A processar..."}
                    save_status_client.info("🔄 A processar...")

                    try:
                        import os
                        if not os.path.exists(db_path_client):
                            st.session_state["save_status_client"] = {
                                "type": "error",
                                "message": f"❌ Base de dados não encontrada: {db_path_client}"
                            }
                            save_status_client.error(st.session_state["save_status_client"]["message"])
                        elif not id_col_client:
                            st.session_state["save_status_client"] = {
                                "type": "error",
                                "message": "❌ Não foi possível guardar: coluna row_id em falta."
                            }
                            save_status_client.error(st.session_state["save_status_client"]["message"])
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
                                st.session_state["save_status_client"] = {
                                    "type": "success",
                                    "message": f"✅ Cliente guardado na tabela 'Deployment_Output_Adjusted'. Tabela completa com {len(merged)} linha(s)."
                                }
                                save_status_client.success(st.session_state["save_status_client"]["message"])
                            finally:
                                conn.close()
                    except Exception as e:
                        import traceback
                        st.session_state["save_status_client"] = {
                            "type": "error",
                            "message": f"❌ Erro ao guardar cliente: {e}",
                            "traceback": traceback.format_exc()
                        }
                        save_status_client.error(st.session_state["save_status_client"]["message"])
                        st.code(st.session_state["save_status_client"]["traceback"])
    else:
        st.info("📄 Nenhum cliente neste segmento com os filtros atuais.")

st.markdown("---")
st.subheader("📋 Dados do Segmento")
st.caption("Amostra dos registros deste grupo de clientes")
show_cols = [c for c in [
    "row_id", seg_col, WELL_PRED, WELL_REAL, CLASS_PRED, PROB_ALTO, PROB_BAIXO,
    *FEATURES, "sleep_index",
] if c in df_seg.columns]

st.dataframe(df_seg[show_cols].head(200), use_container_width=True)

st.write("Global % real >= 8.5:", safe_rate(df0[WELL_REAL], 8.5))
st.write("Global % pred >= 8.5:", safe_rate(df0[WELL_PRED], 8.5))
st.write("Global classes:", df0[CLASS_PRED].value_counts())
st.write("Global P(Alto) mean:", df0[PROB_ALTO].mean())


st.download_button(
    "Descarregar CSV (segmento filtrado)",
    data=df_seg.to_csv(index=False).encode("utf-8"),
    file_name=f"segmento_{seg_col}_{seg_value}.csv",
    mime="text/csv"
)

# -------------------------
# Tab 5: Planeamento em Grupo
# -------------------------
with tab5:
    st.subheader("🎯 Planeamento em Grupo para o Segmento")
    st.caption("Aplique ajustes a todos os clientes deste segmento e visualize o impacto agregado")

    # Mostrar perfil atual do segmento
    st.markdown("---")
    st.subheader("📊 Perfil Atual do Segmento")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total de Clientes", len(df_seg))
    col2.metric("Bem-estar Médio", f"{df_seg[WELL_PRED].mean():.2f}")
    col3.metric("Ecrã Médio (h)", f"{df_seg['screen_time_hours'].mean():.1f}")
    col4.metric("Sono Médio (h)", f"{df_seg['sleep_hours'].mean():.1f}")
    col5.metric("Stress Médio", f"{df_seg['stress_level_0_10'].mean():.1f}")

    # Classes atuais no segmento
    if CLASS_PRED in df_seg.columns:
        class_dist = df_seg[CLASS_PRED].value_counts()
        st.markdown("**Distribuição de Classes (Atual):**")
        for cls, count in class_dist.items():
            pct = (count / len(df_seg)) * 100
            st.write(f"- {cls}: {count} clientes ({pct:.1f}%)")

    # Recomendações do segmento
    drivers_grp, measures_grp, cluster_label_grp, objective_label_grp = measures_for_segment(seg_col, seg_value)
    st.markdown("---")
    st.info(f"🎯 **Perfil do Segmento:** {cluster_label_grp} | {objective_label_grp}")
    if measures_grp:
        st.markdown("**📝 Recomendações de Intervenção para este Segmento:**")
        for i, m in enumerate(measures_grp, start=1):
            st.markdown(f"{i}. {m}")
        st.caption("💡 Use os ajustes abaixo para simular a aplicação destas recomendações a todo o grupo")

    st.markdown("---")
    st.subheader("⚙️ Ajustes do Grupo")
    st.caption("Defina os ajustes que devem ser aplicados a TODOS os clientes deste segmento")

    with st.form("group_adjustment_form_seg"):
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
        adjusted_group = df_seg.copy()
        
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
        original_avg_wellness = df_seg[WELL_PRED].mean()
        new_avg_wellness = adjusted_group["mw_pred_reg_lin_novo"].mean()
        wellness_delta_group = new_avg_wellness - original_avg_wellness
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bem-estar Médio (Antes)", f"{original_avg_wellness:.2f}")
        col2.metric("Bem-estar Médio (Depois)", f"{new_avg_wellness:.2f}")
        col3.metric("Mudança Média", f"{wellness_delta_group:+.2f}", 
                   delta_color="normal" if wellness_delta_group >= 0 else "inverse")
        col4.metric("% Clientes a Melhorar", 
                   f"{(adjusted_group['mw_pred_reg_lin_novo'] > df_seg[WELL_PRED]).sum() / len(adjusted_group) * 100:.1f}%")

        # Distribuição de classes antes/depois
        if CLASS_PRED in adjusted_group.columns:
            st.markdown("**Mudança na Distribuição de Classes de Bem-estar:**")
            
            original_classes = df_seg[CLASS_PRED].value_counts()
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
                df_seg["work_screen_hours"].mean(),
                df_seg["leisure_screen_hours"].mean(),
                df_seg["sleep_hours"].mean(),
                df_seg["sleep_quality_1_5"].mean(),
                df_seg["exercise_minutes_per_week"].mean(),
                df_seg["social_hours_per_week"].mean(),
                df_seg["stress_level_0_10"].mean(),
                df_seg["productivity_0_10"].mean(),
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

        # Dados detalhados dos clientes ajustados
        st.markdown("---")
        st.subheader("📋 Detalhes dos Clientes Ajustados")
        
        export_cols = [
            "row_id", WELL_PRED, "mw_pred_reg_lin_novo", CLASS_PRED, "mw_class_pred_rf_novo",
            "mw_prob_Baixo", "mw_prob_Medio", "mw_prob_Alto",
            "work_screen_hours", "leisure_screen_hours", "sleep_hours", "sleep_quality_1_5",
            "exercise_minutes_per_week", "social_hours_per_week", "stress_level_0_10", "productivity_0_10",
        ]
        export_cols = [c for c in export_cols if c in adjusted_group.columns]
        
        st.dataframe(adjusted_group[export_cols], use_container_width=True)

        # Store in session state for persistent save button
        st.session_state["adjusted_group_segmentos"] = adjusted_group[export_cols]

    # Persistent save section (outside form submit block)
    if "adjusted_group_segmentos" in st.session_state:
        st.markdown("---")
        col_csv, col_db = st.columns(2)
        
        with col_csv:
            df_group_persist = st.session_state["adjusted_group_segmentos"]
            csv_data = df_group_persist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Descarregar Plano de Intervenção em Grupo (CSV)",
                data=csv_data,
                file_name=f"grupo_ajustado_segmento.csv",
                mime="text/csv",
                key="download_group_segmentos"
            )
        
        with col_db:
            status_placeholder_group = st.empty()
            
            if "save_status_segmentos_group" in st.session_state:
                msg = st.session_state["save_status_segmentos_group"]
                if msg.get("type") == "success":
                    status_placeholder_group.success(msg.get("message", ""))
                elif msg.get("type") == "error":
                    status_placeholder_group.error(msg.get("message", ""))
                    if msg.get("traceback"):
                        st.code(msg["traceback"])
            
            if st.button("💿 Guardar Grupo na Base de Dados", key=f"save_group_segmentos_persistent"):
                df_to_save_group = st.session_state["adjusted_group_segmentos"]
                id_col_group = "row_id" if "row_id" in df_to_save_group.columns else None

                st.session_state["save_status_segmentos_group"] = {"type": "info", "message": "🔄 A processar..."}
                status_placeholder_group.info("🔄 A processar...")

                try:
                    import os
                    if not os.path.exists(db_path):
                        st.session_state["save_status_segmentos_group"] = {
                            "type": "error",
                            "message": f"❌ Base de dados não encontrada: {db_path}"
                        }
                        status_placeholder_group.error(st.session_state["save_status_segmentos_group"]["message"])
                    elif not id_col_group:
                        st.session_state["save_status_segmentos_group"] = {
                            "type": "error",
                            "message": "❌ Não foi possível guardar: coluna row_id em falta."
                        }
                        status_placeholder_group.error(st.session_state["save_status_segmentos_group"]["message"])
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

                            st.session_state["save_status_segmentos_group"] = {
                                "type": "success",
                                "message": f"✅ Grupo guardado na tabela 'Deployment_Output_Adjusted'. Tabela completa com {len(merged)} linha(s)."
                            }
                            status_placeholder_group.success(st.session_state["save_status_segmentos_group"]["message"])
                        finally:
                            conn.close()
                except Exception as e:
                    import traceback
                    st.session_state["save_status_segmentos_group"] = {
                        "type": "error",
                        "message": f"❌ Erro ao guardar: {e}",
                        "traceback": traceback.format_exc()
                    }
                    status_placeholder_group.error(st.session_state["save_status_segmentos_group"]["message"])
                    st.code(st.session_state["save_status_segmentos_group"]["traceback"])
