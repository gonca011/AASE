import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



# =========================================================
# 1) Ler dados e criar variável alvo (mw_class)
# =========================================================

db_path = r"C:\BaseDadosTeste\ScreenTimevsMentalWellness.db"
main_table = "ScreenTimevsMentalWellness"

conn = sqlite3.connect(db_path)
df = pd.read_sql_query(f"SELECT * FROM {main_table}", conn)
conn.close()

print("Dimensão original:", df.shape)

target_cont = "mental_wellness_index_0_10"
if target_cont not in df.columns:
    raise ValueError(f"A coluna '{target_cont}' não existe na tabela.")

y_cont = df[target_cont]

# Remover linhas sem valor de bem-estar
mask_valid = y_cont.notna()
df = df[mask_valid].copy()
y_cont = y_cont[mask_valid]

# Definir classes: 0–3  -> Baixo | 4–7  -> Medio | 8–10 -> Alto
bins = [-0.1, 3.0, 7.0, 10.0]
labels = ["Baixo", "Medio", "Alto"]

df["mw_class"] = pd.cut(
    y_cont,
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)

print("Distribuição das classes de bem-estar:")
print(df["mw_class"].value_counts())

# =========================================================
# 2) Definir lista completa de features disponíveis
# =========================================================

all_feature_cols = [
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

# Garantir que só usamos colunas que existem
all_feature_cols = [c for c in all_feature_cols if c in df.columns]
print("Colunas totais disponíveis como features:", all_feature_cols)

# =========================================================
# 3) Definir os 5 cenários de atributos
# =========================================================
# Cenário 1 – Só hábitos digitais
scenario1 = [
    "screen_time_hours",
    "work_screen_hours",
    "leisure_screen_hours",
]

# Cenário 2 – Hábitos digitais + sono
scenario2 = scenario1 + [
    "sleep_hours",
    "sleep_quality_1_5",
]

# Cenário 3 – Cenário 2 + atividade física/social
scenario3 = scenario2 + [
    "exercise_minutes_per_week",
    "social_hours_per_week",
]

# Cenário 4 – Cenário 3 + stress e produtividade
scenario4 = scenario3 + [
    "stress_level_0_10",
    "productivity_0_10",
]

# Cenário 5 – Cenário 4 + contexto demográfico (ocupação, modo de trabalho, idade)
scenario5 = scenario4 + [
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

scenarios = {
    "C1_Digital": scenario1,
    "C2_Digital_Sono": scenario2,
    "C3_Digital_Sono_Atividade": scenario3,
    "C4_Comportamental_Completo": scenario4,
    "C5_Comportamental_Demografico": scenario5,
}

# Garantir que cada cenário só tem colunas existentes
for name in scenarios:
    scenarios[name] = [c for c in scenarios[name] if c in df.columns]

print("\nCenários definidos (features por cenário):")
for name, feats in scenarios.items():
    print(f"{name}: {feats}")

# =========================================================
# 4) Definir modelos (técnicas de Data Mining)
# =========================================================

models = {
    "DecisionTree": DecisionTreeClassifier(
        random_state=42,
        max_depth=None
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    ),
    "MLP_ANN": MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0005,
        alpha=0.0005,
        max_iter=2000,
        random_state=42
    ),
}

# =========================================================
# Função para especificidade média (macro)
# =========================================================

def specificity_macro(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    for i in range(len(labels)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specs.append(spec)
    return np.mean(specs)

# =========================================================
# 5) Avaliar cada modelo em cada cenário com Stratified K-Fold
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
class_labels = ["Baixo", "Medio", "Alto"]

results_rows = []

for scen_name, feat_list in scenarios.items():
    print(f"\n==============================")
    print(f"Cenário: {scen_name}")
    print(f"Features: {feat_list}")
    print(f"==============================")

    # Construir X e y para este cenário
    X_scen = df[feat_list].copy()
    y = df["mw_class"].copy()

    # Tratar NaNs
    X_scen = X_scen.fillna(X_scen.mean())

    for model_name, model in models.items():
        acc_list = []
        f1_list = []
        rec_list = []
        spec_list = []
        auc_list = []

        print(f"\n--- Modelo: {model_name} ---")
        fold_idx = 1

        for train_index, test_index in skf.split(X_scen, y):
            X_train_k = X_scen.iloc[train_index]
            X_test_k = X_scen.iloc[test_index]
            y_train_k = y.iloc[train_index]
            y_test_k = y.iloc[test_index]

            # Normalização (mantida para todos por consistência)
            scaler_k = StandardScaler()
            X_train_k_scaled = scaler_k.fit_transform(X_train_k)
            X_test_k_scaled = scaler_k.transform(X_test_k)

            model.fit(X_train_k_scaled, y_train_k)
            y_pred_k = model.predict(X_test_k_scaled)

            acc_k = accuracy_score(y_test_k, y_pred_k)
            f1_k = f1_score(y_test_k, y_pred_k, average="macro")
            rec_k = recall_score(y_test_k, y_pred_k, average="macro", zero_division=0)
            spec_k = specificity_macro(y_test_k, y_pred_k, labels=class_labels)

            # AUC macro (multi-classe)
            try:
                if hasattr(model, "predict_proba"):
                    y_score_raw = model.predict_proba(X_test_k_scaled)
                    # alinhar colunas à ordem das classes
                    class_order = list(model.classes_)
                    idx = [class_order.index(lbl) for lbl in class_labels]
                    y_score = y_score_raw[:, idx]
                    y_test_bin = label_binarize(y_test_k, classes=class_labels)
                    auc_k = roc_auc_score(
                        y_test_bin, y_score,
                        average="macro",
                        multi_class="ovr"
                    )
                else:
                    auc_k = np.nan
            except ValueError:
                # casos extremos (p.ex., falta absoluta de uma classe no fold)
                auc_k = np.nan

            acc_list.append(acc_k)
            f1_list.append(f1_k)
            rec_list.append(rec_k)
            spec_list.append(spec_k)
            auc_list.append(auc_k)

            print(
                f"Fold {fold_idx}: "
                f"Acc={acc_k:.3f}, "
                f"F1={f1_k:.3f}, "
                f"Rec={rec_k:.3f}, "
                f"Spec={spec_k:.3f}, "
                f"AUC={auc_k:.3f}"
            )
            fold_idx += 1

        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        f1_mean = np.mean(f1_list)
        f1_std = np.std(f1_list)
        rec_mean = np.mean(rec_list)
        rec_std = np.std(rec_list)
        spec_mean = np.mean(spec_list)
        spec_std = np.std(spec_list)
        auc_mean = np.nanmean(auc_list)
        auc_std = np.nanstd(auc_list)

        results_rows.append({
            "Cenario": scen_name,
            "Modelo": model_name,
            "Acc_media": acc_mean,
            "Acc_std": acc_std,
            "Recall_media": rec_mean,
            "Recall_std": rec_std,
            "Especificidade_media": spec_mean,
            "Especificidade_std": spec_std,
            "F1_media": f1_mean,
            "F1_std": f1_std,
            "AUC_media": auc_mean,
            "AUC_std": auc_std,
        })

# =========================================================
# 6) Resumo comparativo final
# =========================================================

results_df = pd.DataFrame(results_rows)
results_df = results_df.sort_values(by=["Cenario", "F1_media"], ascending=[True, False])

print("\n=== Resumo comparativo por cenário e modelo (ordenado por F1_media) ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Opcional: guardar em ficheiro CSV para usar no relatório
results_df.to_csv("resultados_classificacao_cenarios.csv", index=False)
print("\nResultados guardados em 'resultados_classificacao_cenarios.csv'.")

# Ordenar por cenário e, dentro de cada cenário, pelas melhores métricas
results_sorted = results_df.sort_values(
    by=["Cenario", "F1_media", "AUC_media", "Acc_media"],
    ascending=[True, False, False, False]
)

# Melhor modelo por cenário = a primeira linha de cada grupo de cenário
best_per_scenario = results_sorted.groupby("Cenario").head(1)

print("\n=== Melhor modelo por cenário (critério: F1_media, depois AUC, depois Accuracy) ===")
print(best_per_scenario.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Opcional: guardar para consulta
best_per_scenario.to_csv("melhor_modelo_por_cenario.csv", index=False)
print("\nTabela 'melhor_modelo_por_cenario.csv' criada.")


# =========================================================
# 7) Curvas ROC usando input do utilizador (cenário + modelo)
# =========================================================

print("\n=== Escolha do Cenário ===")
for i, name in enumerate(scenarios.keys(), start=1):
    print(f"{i}. {name}")

# Ler cenário
while True:
    try:
        escolha_cenario = int(input("\nDigite o número do cenário desejado: "))
        if 1 <= escolha_cenario <= len(scenarios):
            cenario_escolhido = list(scenarios.keys())[escolha_cenario - 1]
            break
        else:
            print("Número inválido. Tente novamente.")
    except ValueError:
        print("Por favor introduza um número válido.")

print(f"\nCenário escolhido: {cenario_escolhido}")

# Mostrar modelos
print("\n=== Escolha do Modelo ===")
for i, name in enumerate(models.keys(), start=1):
    print(f"{i}. {name}")

# Ler modelo
while True:
    try:
        escolha_modelo = int(input("\nDigite o número do modelo desejado: "))
        if 1 <= escolha_modelo <= len(models):
            modelo_escolhido = list(models.keys())[escolha_modelo - 1]
            break
        else:
            print("Número inválido. Tente novamente.")
    except ValueError:
        print("Por favor introduza um número válido.")

print(f"\nModelo escolhido: {modelo_escolhido}")
print("\nA gerar curva ROC...\n")

# ---------------------------------------------------------
#  Lógica igual à anterior, só substitui nomes por input
# ---------------------------------------------------------

feat_list = scenarios[cenario_escolhido]
X_scen = df[feat_list].copy()
y = df["mw_class"].copy()
X_scen = X_scen.fillna(X_scen.mean())

model = models[modelo_escolhido]

skf_roc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_true_all = []
y_score_all = []

for train_index, test_index in skf_roc.split(X_scen, y):
    X_train_k = X_scen.iloc[train_index]
    X_test_k = X_scen.iloc[test_index]
    y_train_k = y.iloc[train_index]
    y_test_k = y.iloc[test_index]

    scaler_k = StandardScaler()
    X_train_k_scaled = scaler_k.fit_transform(X_train_k)
    X_test_k_scaled = scaler_k.transform(X_test_k)

    model.fit(X_train_k_scaled, y_train_k)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"O modelo '{modelo_escolhido}' não suporta predict_proba; ROC impossível."
        )

    y_score_raw = model.predict_proba(X_test_k_scaled)

    class_order = list(model.classes_)
    idx = [class_order.index(lbl) for lbl in class_labels]
    y_score = y_score_raw[:, idx]

    y_true_all.append(y_test_k.values)
    y_score_all.append(y_score)

# Concatenar previsões de todos os folds
y_true_all = np.concatenate(y_true_all, axis=0)
y_score_all = np.concatenate(y_score_all, axis=0)

y_true_bin = label_binarize(y_true_all, classes=class_labels)
n_classes = y_true_bin.shape[1]

# ROC por classe
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k--", label="Baseline (AUC = 0.50)")

cores = ["tab:blue", "tab:orange", "tab:green"]
for i, cls in enumerate(class_labels):
    plt.plot(
        fpr[i],
        tpr[i],
        lw=2,
        color=cores[i],
        label=f"{cls} (AUC = {roc_auc[i]:.2f})"
    )

plt.plot(
    fpr["macro"],
    tpr["macro"],
    lw=2,
    linestyle=":",
    color="tab:red",
    label=f"Macro-média (AUC = {roc_auc['macro']:.2f})"
)

plt.xlabel("False Positive Rate (1 - Especificidade)")
plt.ylabel("True Positive Rate (Sensibilidade)")
plt.title(f"ROC - {modelo_escolhido} | {cenario_escolhido}")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

output_file = f"ROC_{modelo_escolhido}_{cenario_escolhido}.png"
plt.savefig(output_file, dpi=300)
plt.show()

print(f"Curva ROC gravada como '{output_file}'")

