# HotelDetox_KPI_Tracker.py
# Dashboard para comparar KPIs: Dados Originais vs Clientes Ajustados


import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Hotel Detox - KPI Tracker", layout="wide")
st.title("📊 Hotel Detox — Tracker de KPIs: Antes vs Depois")
st.caption("Monitorização do impacto das intervenções nos clientes")

# -----------------------------
# Sidebar: Configuração
# -----------------------------
st.sidebar.header("⚙️ Configuração")

# Fonte de dados
db_path = st.sidebar.text_input(
    "📁 Base de Dados",
    value=r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db",
    help="Caminho para a base de dados com Deployment_Output e Deployment_Output_Adjusted"
)

base_table = st.sidebar.text_input(
    "Tabela original",
    value="Deployment_Output",
    help="Tabela base (original) usada como referência"
)

# Thresholds
risk_threshold = st.sidebar.slider(
    "Limiar de risco (mw_prob_Baixo ≥)",
    0.0, 1.0, 0.5, 0.05,
    help="Probabilidade acima da qual o cliente é considerado em risco"
)

high_wellness_cutoff = st.sidebar.slider(
    "Limiar para bem-estar elevado (mw_pred_reg_lin ≥)",
    0.0, 10.0, 8.5, 0.5,
    help="Bem-estar previsto acima do qual é considerado elevado"
)

# -----------------------------
# Carregar dados
# -----------------------------
@st.cache_data
def load_original_data(db_file, table_name):
    """Carregar dados originais da base de dados"""
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados originais: {e}")
        return None

def load_adjusted_data():
    """Carregar dados ajustados da base de dados SQLite"""
    try:
        conn = sqlite3.connect(r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db")
        df = pd.read_sql("SELECT * FROM Deployment_Output_Adjusted", conn)
        conn.close()
        return df
    except Exception as e:
        return None

# Carregar dados
df_original = load_original_data(db_path, base_table)
df_adjusted = load_adjusted_data()

# -----------------------------
# Verificação de dados
# -----------------------------
if df_original is None:
    st.error("❌ Não foi possível carregar os dados originais. Verifique o nome do ficheiro.")
    st.stop()

if df_adjusted is None or len(df_adjusted) == 0:
    st.warning("⚠️ Não existem dados ajustados na base de dados ainda.")
    st.info("💡 Use o painel 'Manter Clientes' para ajustar clientes e guardá-los na base de dados.")
    st.stop()

st.success(f"✓ Dados carregados: {len(df_original)} clientes originais | {len(df_adjusted)} clientes ajustados")

# -----------------------------
# Calcular KPIs
# -----------------------------
def calculate_kpis(df, risk_thresh, wellness_thresh):
    """Calcular KPIs para um dataframe"""
    kpis = {}
    
    # Total de clientes
    kpis['total'] = len(df)
    
    # KPI 1: Clientes em risco
    if 'mw_prob_Baixo' in df.columns:
        risk_by_prob = df['mw_prob_Baixo'] >= risk_thresh
    else:
        risk_by_prob = pd.Series([False] * len(df))
    
    if 'mw_class_pred_rf' in df.columns:
        risk_by_class = df['mw_class_pred_rf'] == 'Baixo'
    else:
        risk_by_class = pd.Series([False] * len(df))
    
    kpis['risk_count'] = (risk_by_prob | risk_by_class).sum()
    kpis['risk_pct'] = (kpis['risk_count'] / kpis['total'] * 100) if kpis['total'] > 0 else 0
    
    # KPI 2: Bem-estar elevado
    if 'mw_pred_reg_lin' in df.columns:
        high_wellness = df['mw_pred_reg_lin'] >= wellness_thresh
        kpis['high_wellness_count'] = high_wellness.sum()
        kpis['high_wellness_pct'] = (kpis['high_wellness_count'] / kpis['total'] * 100) if kpis['total'] > 0 else 0
        kpis['avg_wellness'] = df['mw_pred_reg_lin'].mean()
    else:
        kpis['high_wellness_count'] = 0
        kpis['high_wellness_pct'] = 0
        kpis['avg_wellness'] = 0
    
    # Distribuição de classes
    if 'mw_class_pred_rf' in df.columns:
        kpis['class_dist'] = df['mw_class_pred_rf'].value_counts().to_dict()
    else:
        kpis['class_dist'] = {}
    
    # Métricas comportamentais
    behavioral_cols = [
        'work_screen_hours', 'leisure_screen_hours', 'screen_time_hours',
        'sleep_hours', 'sleep_quality_1_5', 'exercise_minutes_per_week',
        'social_hours_per_week', 'stress_level_0_10', 'productivity_0_10'
    ]
    
    kpis['behavioral'] = {}
    for col in behavioral_cols:
        if col in df.columns:
            kpis['behavioral'][col] = df[col].mean()
    
    return kpis

# Calcular KPIs originais e ajustados
kpi_original = calculate_kpis(df_original, risk_threshold, high_wellness_cutoff)
kpi_adjusted = calculate_kpis(df_adjusted, risk_threshold, high_wellness_cutoff)

# -----------------------------
# Dashboard Principal
# -----------------------------
st.header("📈 Visão Geral dos KPIs")

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total de Clientes",
        f"{kpi_adjusted['total']}",
        delta=f"{kpi_adjusted['total'] - kpi_original['total']}" if kpi_adjusted['total'] != kpi_original['total'] else None
    )

with col2:
    risk_delta = kpi_adjusted['risk_pct'] - kpi_original['risk_pct']
    st.metric(
        "Clientes em Risco",
        f"{kpi_adjusted['risk_pct']:.1f}%",
        delta=f"{risk_delta:.1f}pp",
        delta_color="inverse",
        help="Diminuição é melhor"
    )

with col3:
    wellness_delta = kpi_adjusted['high_wellness_pct'] - kpi_original['high_wellness_pct']
    st.metric(
        "Bem-estar Elevado",
        f"{kpi_adjusted['high_wellness_pct']:.1f}%",
        delta=f"{wellness_delta:.1f}pp",
        delta_color="normal",
        help="Aumento é melhor"
    )

with col4:
    avg_wellness_delta = kpi_adjusted['avg_wellness'] - kpi_original['avg_wellness']
    st.metric(
        "Bem-estar Médio",
        f"{kpi_adjusted['avg_wellness']:.2f}",
        delta=f"{avg_wellness_delta:+.2f}",
        delta_color="normal"
    )

st.markdown("---")

# Impacto Summary
st.subheader("🎯 Resumo do Impacto")

impact_col1, impact_col2, impact_col3 = st.columns(3)

with impact_col1:
    risk_reduction = kpi_original['risk_count'] - kpi_adjusted['risk_count']
    st.markdown(f"""
    **Redução de Risco**
    - Clientes em risco: {kpi_original['risk_count']} → {kpi_adjusted['risk_count']}
    - **Redução:** {risk_reduction} clientes ({risk_delta:.1f}pp)
    """)

with impact_col2:
    wellness_increase = kpi_adjusted['high_wellness_count'] - kpi_original['high_wellness_count']
    st.markdown(f"""
    **Melhoria de Bem-estar**
    - Bem-estar elevado: {kpi_original['high_wellness_count']} → {kpi_adjusted['high_wellness_count']}
    - **Aumento:** {wellness_increase} clientes (+{wellness_delta:.1f}pp)
    """)

with impact_col3:
    if avg_wellness_delta > 0:
        impact_icon = "📈"
        impact_text = "Melhoria"
    elif avg_wellness_delta < 0:
        impact_icon = "📉"
        impact_text = "Deterioração"
    else:
        impact_icon = "➡️"
        impact_text = "Sem mudança"
    
    st.markdown(f"""
    **Bem-estar Geral**
    - Médio: {kpi_original['avg_wellness']:.2f} → {kpi_adjusted['avg_wellness']:.2f}
    - **{impact_icon} {impact_text}:** {avg_wellness_delta:+.2f} pontos
    """)

st.markdown("---")

# -----------------------------
# Comparação de Distribuições
# -----------------------------
st.subheader("📊 Distribuição de Classes de Bem-estar")

# Preparar dados para comparação
classes_orig = kpi_original.get('class_dist', {})
classes_adj = kpi_adjusted.get('class_dist', {})

all_classes = ["Baixo", "Medio", "Alto"]
display_names = ["Baixo", "Médio", "Alto"]

comparison_df = pd.DataFrame({
    'Classe': display_names,
    'Original': [classes_orig.get(c, 0) for c in all_classes],
    'Ajustado': [classes_adj.get(c, 0) for c in all_classes]
})
comparison_df['Mudança'] = comparison_df['Ajustado'] - comparison_df['Original']

col_table, col_chart = st.columns([1, 2])

with col_table:
    st.dataframe(comparison_df, use_container_width=True)

with col_chart:
    fig_classes = px.bar(
        comparison_df,
        x='Classe',
        y=['Original', 'Ajustado'],
        barmode='group',
        title='Distribuição de Classes: Original vs Ajustado',
        labels={'value': 'Número de Clientes', 'variable': 'Estado'}
    )
    st.plotly_chart(fig_classes, use_container_width=True)

st.markdown("---")

# -----------------------------
# Comparação de Métricas Comportamentais
# -----------------------------
st.subheader("🔄 Mudanças nas Métricas Comportamentais")

behavioral_orig = kpi_original.get('behavioral', {})
behavioral_adj = kpi_adjusted.get('behavioral', {})

metric_labels = {
    'work_screen_hours': 'Ecrã de Trabalho (h/dia)',
    'leisure_screen_hours': 'Ecrã de Lazer (h/dia)',
    'screen_time_hours': 'Ecrã Total (h/dia)',
    'sleep_hours': 'Sono (h/dia)',
    'sleep_quality_1_5': 'Qualidade do Sono (1-5)',
    'exercise_minutes_per_week': 'Exercício (min/semana)',
    'social_hours_per_week': 'Horas Sociais/semana',
    'stress_level_0_10': 'Stress (0-10)',
    'productivity_0_10': 'Produtividade (0-10)'
}

behavioral_comparison = []
for metric, label in metric_labels.items():
    if metric in behavioral_orig and metric in behavioral_adj:
        orig_val = behavioral_orig[metric]
        adj_val = behavioral_adj[metric]
        delta = adj_val - orig_val
        
        behavioral_comparison.append({
            'Métrica': label,
            'Original': f"{orig_val:.2f}",
            'Ajustado': f"{adj_val:.2f}",
            'Mudança': f"{delta:+.2f}",
            'Mudança %': f"{(delta/orig_val*100):+.1f}%" if orig_val != 0 else "N/A"
        })

if behavioral_comparison:
    behavioral_df = pd.DataFrame(behavioral_comparison)
    st.dataframe(behavioral_df, use_container_width=True)
    
    # Visualização de mudanças
    st.markdown("**Visualização das Mudanças:**")
    
    # Criar gráfico de barras horizontal para mudanças
    behavioral_chart_df = pd.DataFrame(behavioral_comparison)
    behavioral_chart_df['Mudança_num'] = behavioral_chart_df['Mudança'].str.replace('+', '').astype(float)
    
    fig_behavioral = px.bar(
        behavioral_chart_df,
        y='Métrica',
        x='Mudança_num',
        orientation='h',
        title='Mudanças nas Métricas Comportamentais',
        labels={'Mudança_num': 'Mudança (valor absoluto)'},
        color='Mudança_num',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig_behavioral.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_behavioral, use_container_width=True)
else:
    st.info("Não há dados comportamentais suficientes para comparar.")

st.markdown("---")

# -----------------------------
# Comparação de Distribuições de Bem-estar
# -----------------------------
st.subheader("📈 Distribuição de Bem-estar Previsto")

if 'mw_pred_reg_lin' in df_original.columns and 'mw_pred_reg_lin' in df_adjusted.columns:
    fig_wellness_dist = go.Figure()
    
    fig_wellness_dist.add_trace(go.Histogram(
        x=df_original['mw_pred_reg_lin'],
        name='Original',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_wellness_dist.add_trace(go.Histogram(
        x=df_adjusted['mw_pred_reg_lin'],
        name='Ajustado',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig_wellness_dist.add_vline(
        x=high_wellness_cutoff,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Limiar Elevado ({high_wellness_cutoff})"
    )
    
    fig_wellness_dist.update_layout(
        title='Distribuição de Bem-estar: Original vs Ajustado',
        xaxis_title='Bem-estar Previsto (mw_pred_reg_lin)',
        yaxis_title='Número de Clientes',
        barmode='overlay'
    )
    
    st.plotly_chart(fig_wellness_dist, use_container_width=True)

st.markdown("---")

# -----------------------------
# Dados Brutos
# -----------------------------
with st.expander("📋 Ver Dados Ajustados (Amostra)"):
    st.dataframe(df_adjusted.head(50), use_container_width=True)
    
    st.download_button(
        "💾 Descarregar Dados Ajustados (CSV)",
        data=df_adjusted.to_csv(index=False).encode("utf-8"),
        file_name="clientes_ajustados_export.csv",
        mime="text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("💡 Este painel compara automaticamente os dados originais (Deployment_Output) com os clientes ajustados guardados na base de dados (Deployment_Output_Adjusted).")
st.sidebar.caption("🔄 Os dados ajustados são atualizados sempre que guarda novos ajustes no painel 'Manter Clientes'.")