# app2.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Hotel Detox - Simulação (What-if)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS leve
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      .hd-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(49,51,63,0.15);
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
      }
      .hd-muted { opacity: 0.82; }
      .hd-small { font-size: 0.92rem; }
      .hd-hr { margin: 0.8rem 0 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hotel Detox — Simulador de Cenários (What-if)")
st.caption("Introduza valores e veja o impacto previsto no bem-estar e risco com os modelos finais (.pkl).")
st.markdown("<div class='hd-hr'></div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar: artefactos
# -----------------------------
st.sidebar.header("Modelos e artefactos (.pkl)")
base_dir = st.sidebar.text_input("Pasta dos .pkl", value=".", help="Diretório onde guardou os modelos exportados do treino")

reg_path = st.sidebar.text_input("Regressão (.pkl)", value="model_regressao.pkl")
clf_path = st.sidebar.text_input("Classificação (.pkl)", value="model_classificacao.pkl")

k1_path = st.sidebar.text_input("KMeans Obj1 (.pkl)", value="kmeans_obj1_k4.pkl")
s1_path = st.sidebar.text_input("Scaler Obj1 (.pkl)", value="scaler_obj1.pkl")

k2_path = st.sidebar.text_input("KMeans Obj2 (.pkl)", value="kmeans_obj2_k4.pkl")
s2_path = st.sidebar.text_input("Scaler Obj2 (.pkl)", value="scaler_obj2.pkl")


def pjoin(folder: str, filename: str) -> str:
    return os.path.abspath(os.path.join(folder, filename))


REG_MODEL_PATH = pjoin(base_dir, reg_path)
CLF_MODEL_PATH = pjoin(base_dir, clf_path)
KMEANS1_PATH = pjoin(base_dir, k1_path)
SCALER1_PATH = pjoin(base_dir, s1_path)
KMEANS2_PATH = pjoin(base_dir, k2_path)
SCALER2_PATH = pjoin(base_dir, s2_path)

required_files = [REG_MODEL_PATH, CLF_MODEL_PATH, KMEANS1_PATH, SCALER1_PATH, KMEANS2_PATH, SCALER2_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]

st.sidebar.markdown("---")
st.sidebar.subheader("Estado dos ficheiros")
if missing_files:
    st.sidebar.error("Ficheiros em falta")
    for f in missing_files:
        st.sidebar.write(f"- {f}")
else:
    st.sidebar.success("Tudo OK — ficheiros encontrados")

with st.sidebar.expander("Ajuda rápida", expanded=False):
    st.write("1) Confirme se os .pkl estão na pasta indicada.")
    st.write("2) Ajuste os sliders e clique em 'Prever'.")
    st.write("3) Consulte 'Auditoria & Diagnóstico' para ver os inputs usados e as distâncias de cluster.")

if missing_files:
    st.error("Faltam ficheiros .pkl. Verifique a pasta e os nomes na sidebar.")
    st.stop()

# -----------------------------
# Features
# -----------------------------
NEEDED_BASE = [
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

CLUSTER_COLS_OBJ1 = [
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "sleep_index",
    "stress_level_0_10",
    "productivity_0_10",
    "screen_time_hours",
]

CLUSTER_COLS_OBJ2_MAX = [
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
    "sleep_hours",
    "sleep_quality_1_5",
    "exercise_minutes_per_week",
    "social_hours_per_week",
    "stress_level_0_10",
    "productivity_0_10",
    "occupation_Desempregado",
    "occupation_Empregado",
    "occupation_Estudante",
    "occupation_Reformado",
    "work_mode_Hybrid",
    "work_mode_In-person",
    "work_mode_Remote",
    "age_Jovem_16-25",
    "age_Adulto_25-50",
    "age_Senior_50-60",
]

CLASS_LABELS = ["Baixo", "Medio", "Alto"]

# -----------------------------
# Carregar modelos
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models(reg_p: str, clf_p: str, k1_p: str, s1_p: str, k2_p: str, s2_p: str):
    reg = joblib.load(reg_p)
    clf = joblib.load(clf_p)
    k1 = joblib.load(k1_p)
    s1 = joblib.load(s1_p)
    k2 = joblib.load(k2_p)
    s2 = joblib.load(s2_p)
    return reg, clf, k1, s1, k2, s2


try:
    reg_model, clf_model, kmeans1, scaler1, kmeans2, scaler2 = load_models(
        REG_MODEL_PATH, CLF_MODEL_PATH, KMEANS1_PATH, SCALER1_PATH, KMEANS2_PATH, SCALER2_PATH
    )
except Exception as e:
    st.error(f"Erro ao carregar modelos/scalers: {e}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def ensure_cols(df_in: pd.DataFrame, cols: list, fill: float = 0.0) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = fill
    return df_out[cols]


def build_user_df(inputs: dict) -> pd.DataFrame:
    df_user = pd.DataFrame([inputs]).copy()

    # Converter base numérica e preencher
    for c in NEEDED_BASE:
        if c in df_user.columns:
            df_user[c] = pd.to_numeric(df_user[c], errors="coerce").fillna(0)

    # Forçar consistência: total = trabalho + lazer
    if "work_screen_hours" in df_user.columns and "leisure_screen_hours" in df_user.columns:
        df_user["screen_time_hours"] = df_user["work_screen_hours"] + df_user["leisure_screen_hours"]

    # Derivada
    df_user["sleep_index"] = (
        pd.to_numeric(df_user.get("sleep_hours", 0), errors="coerce").fillna(0)
        + pd.to_numeric(df_user.get("sleep_quality_1_5", 0), errors="coerce").fillna(0)
    )

    return df_user


def predict_all(df_user: pd.DataFrame):
    # Regressão
    X_reg = ensure_cols(df_user, NEEDED_BASE)
    pred_reg_raw = float(reg_model.predict(X_reg)[0])
    pred_reg = float(np.clip(pred_reg_raw, 0.0, 10.0))

    # Classificação
    X_clf = ensure_cols(df_user, NEEDED_BASE)
    pred_class = clf_model.predict(X_clf)[0]

    proba_vec = clf_model.predict_proba(X_clf)[0]
    cls_order = list(clf_model.classes_)
    proba_map = {cls: float(proba_vec[i]) for i, cls in enumerate(cls_order)}
    proba_ordered = {cls: proba_map.get(cls, np.nan) for cls in CLASS_LABELS}

    # Obj1 + distâncias
    X1 = ensure_cols(df_user, CLUSTER_COLS_OBJ1).apply(pd.to_numeric, errors="coerce").fillna(0)
    X1_scaled = scaler1.transform(X1)
    cluster1 = int(kmeans1.predict(X1_scaled)[0])

    centers1 = kmeans1.cluster_centers_
    x_vec1 = X1_scaled[0]
    distances_obj1 = {f"cluster_{i}": float(np.linalg.norm(x_vec1 - centers1[i])) for i in range(centers1.shape[0])}

    # Obj2 (colunas do treino) + distâncias
    if hasattr(scaler2, "feature_names_in_"):
        cols_obj2_train = list(scaler2.feature_names_in_)
    else:
        cols_obj2_train = CLUSTER_COLS_OBJ2_MAX

    X2 = ensure_cols(df_user, cols_obj2_train, fill=0.0).apply(pd.to_numeric, errors="coerce")

    dummy_cols = [c for c in X2.columns if c.startswith(("occupation_", "work_mode_", "age_"))]
    cont_cols = [c for c in X2.columns if c not in dummy_cols]
    if dummy_cols:
        X2[dummy_cols] = X2[dummy_cols].fillna(0)
    if cont_cols:
        X2[cont_cols] = X2[cont_cols].fillna(0)

    X2_scaled = scaler2.transform(X2)
    cluster2 = int(kmeans2.predict(X2_scaled)[0])

    centers2 = kmeans2.cluster_centers_
    x_vec2 = X2_scaled[0]
    distances_obj2 = {f"cluster_{i}": float(np.linalg.norm(x_vec2 - centers2[i])) for i in range(centers2.shape[0])}

    return pred_reg, pred_class, proba_ordered, cluster1, cluster2, distances_obj1, distances_obj2


# -----------------------------
# UI: Tabs
# -----------------------------
tabs = st.tabs(["Inputs", "Resultados", "Auditoria & Diagnóstico"])

with tabs[0]:
    st.markdown(
        "<div class='hd-card'><b>Perfil do cliente</b><br>"
        "<span class='hd-muted hd-small'>O tempo de ecrã total é calculado automaticamente: trabalho + lazer.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("sim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Hábitos digitais**")
            work_screen = st.slider("Tempo de ecrã no trabalho (h/dia)", 0.0, 12.0, 4.0, 0.1)
            leisure_screen = st.slider("Tempo de ecrã no lazer (h/dia)", 0.0, 12.0, 2.0, 0.1)
            screen_time_total = float(work_screen + leisure_screen)
            st.metric("Screen time total (calculado)", f"{screen_time_total:.1f} h/dia")

        with col2:
            st.markdown("**Sono & Exercício**")
            sleep_hours = st.slider("Horas de sono", 3.0, 10.0, 7.0, 0.1)
            sleep_quality = st.slider("Qualidade do sono (1–5)", 1, 5, 3, 1)
            exercise = st.slider("Exercício (min/semana)", 0, 600, 150, 10)

        with col3:
            st.markdown("**Stress & Rotina**")
            social = st.slider("Horas sociais/semana", 0, 40, 8, 1)
            stress = st.slider("Stress (0–10)", 0, 10, 5, 1)
            productivity = st.slider("Produtividade (0–10)", 0, 10, 6, 1)

        st.markdown("<div class='hd-hr'></div>", unsafe_allow_html=True)

        # CONTEXTO bem destacado
        st.markdown(
            "<div class='hd-card'><b>Contexto (para Clustering Obj2)</b><br>"
            "<span class='hd-muted hd-small'>Estas variáveis são convertidas em dummies para o modelo de clustering.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        cx1, cx2, cx3 = st.columns(3)
        with cx1:
            occ = st.selectbox("Ocupação", ["Empregado", "Estudante", "Desempregado", "Reformado"], index=0)
        with cx2:
            work_mode = st.selectbox("Modo de trabalho", ["Remote", "Hybrid", "In-person"], index=0)
        with cx3:
            age_group = st.selectbox("Faixa etária", ["Jovem_16-25", "Adulto_25-50", "Senior_50-60"], index=1)

        submitted = st.form_submit_button("Prever", use_container_width=False)

# Executar previsão e guardar estado
if submitted:
    inputs = {
        "screen_time_hours": float(work_screen + leisure_screen),
        "work_screen_hours": float(work_screen),
        "leisure_screen_hours": float(leisure_screen),
        "sleep_hours": float(sleep_hours),
        "sleep_quality_1_5": int(sleep_quality),
        "exercise_minutes_per_week": float(exercise),
        "social_hours_per_week": float(social),
        "stress_level_0_10": float(stress),
        "productivity_0_10": float(productivity),
    }

    dummies = {
        "occupation_Empregado": 1 if occ == "Empregado" else 0,
        "occupation_Estudante": 1 if occ == "Estudante" else 0,
        "occupation_Desempregado": 1 if occ == "Desempregado" else 0,
        "occupation_Reformado": 1 if occ == "Reformado" else 0,
        "work_mode_Remote": 1 if work_mode == "Remote" else 0,
        "work_mode_Hybrid": 1 if work_mode == "Hybrid" else 0,
        "work_mode_In-person": 1 if work_mode == "In-person" else 0,
        "age_Jovem_16-25": 1 if age_group == "Jovem_16-25" else 0,
        "age_Adulto_25-50": 1 if age_group == "Adulto_25-50" else 0,
        "age_Senior_50-60": 1 if age_group == "Senior_50-60" else 0,
    }

    df_user = build_user_df({**inputs, **dummies})
    pred_reg, pred_class, proba_ordered, cluster1, cluster2, distances_obj1, distances_obj2 = predict_all(df_user)

    st.session_state["last_df_user"] = df_user
    st.session_state["last_pred"] = {
        "pred_reg": pred_reg,
        "pred_class": pred_class,
        "proba_ordered": proba_ordered,
        "cluster1": cluster1,
        "cluster2": cluster2,
        "dist1": distances_obj1,
        "dist2": distances_obj2,
    }

with tabs[1]:
    if "last_pred" not in st.session_state:
        st.info("Preencha os inputs e carregue em 'Prever' para ver os resultados.")
    else:
        out = st.session_state["last_pred"]

        st.markdown("<div class='hd-card'><b>Resultados da simulação</b></div>", unsafe_allow_html=True)

        a, b, c, d, e = st.columns(5)
        a.metric("Bem-estar previsto (0–10)", f"{out['pred_reg']:.2f}")
        b.metric("Classe de risco prevista", str(out["pred_class"]))
        c.metric("Prob. de risco 'Baixo'", f"{float(out['proba_ordered']['Baixo']):.3f}")
        d.metric("Cluster Obj1", f"{out['cluster1']}")
        e.metric("Cluster Obj2", f"{out['cluster2']}")

        st.markdown("<div class='hd-hr'></div>", unsafe_allow_html=True)

        st.subheader("Probabilidades das classes de risco")
        prob_df = pd.DataFrame(
            {
                "Classe": ["Baixo", "Medio", "Alto"],
                "Probabilidade": [
                    float(out["proba_ordered"]["Baixo"]) if not np.isnan(out["proba_ordered"]["Baixo"]) else 0.0,
                    float(out["proba_ordered"]["Medio"]) if not np.isnan(out["proba_ordered"]["Medio"]) else 0.0,
                    float(out["proba_ordered"]["Alto"]) if not np.isnan(out["proba_ordered"]["Alto"]) else 0.0,
                ],
            }
        )
        st.bar_chart(prob_df.set_index("Classe"), width="stretch")
        st.caption(f"Soma das probabilidades (controlo): {prob_df['Probabilidade'].sum():.3f}")

        st.markdown(
            "<div class='hd-card'><b>Nota</b><br>"
            "<span class='hd-muted hd-small'>Os resultados refletem os modelos finais guardados. "
            "Para análise histórica, utilize o dashboard baseado em Deployment_Output.</span></div>",
            unsafe_allow_html=True,
        )

with tabs[2]:
    if "last_pred" not in st.session_state:
        st.info("Depois de prever, aqui ficará disponível a auditoria e o diagnóstico.")
    else:
        out = st.session_state["last_pred"]
        df_user = st.session_state["last_df_user"]

        st.markdown("<div class='hd-card'><b>Auditoria — inputs usados</b></div>", unsafe_allow_html=True)
        st.dataframe(df_user, width="stretch")

        st.markdown("<div class='hd-hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hd-card'><b>Diagnóstico — distâncias aos centroides</b></div>", unsafe_allow_html=True)

        with st.expander("Obj1: distâncias aos centroides (ordenado)", expanded=True):
            distances1_sorted = dict(sorted(out["dist1"].items(), key=lambda kv: kv[1]))
            st.write(distances1_sorted)
            st.caption(
                "O cluster atribuído (Obj1) é o centroide mais próximo após normalização (StandardScaler)."
            )

        with st.expander("Obj2: distâncias aos centroides (ordenado)", expanded=False):
            distances2_sorted = dict(sorted(out["dist2"].items(), key=lambda kv: kv[1]))
            st.write(distances2_sorted)
            st.caption(
                "O cluster atribuído (Obj2) é o centroide mais próximo após normalização (StandardScaler)."
            )
