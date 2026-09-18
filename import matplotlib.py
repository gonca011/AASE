import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Definir os dados (Posicionamento no Gráfico)
# Eixo X: 0 = Plataforma Curada (Estruturada); 10 = Gerado pelo Utilizador (UGC)
# Eixo Y: 0 = Individual/Autoestudo; 10 = Colaborativo/Gestão de Turma

data = {
    'Plataforma': [
        'Khan Academy', 'Duolingo', 'Estudo em Casa',
        'Quizlet', 'Knowunity', 'Studocu',
        'Classroom', 'Blackboard', 'Brainly',
        'LABRATS'
    ],
    'X': [
        1, 1.5, 0.5,
        8, 9, 9.5,
        5, 4.5, 9,
        5,  # Posicionamento Estratégico Híbrido (X)
    ],
    'Y': [
        1.5, 1, 0.5,
        4, 3, 5,
        9, 8.5, 8,
        7,  # Posicionamento Elevado em Colaboração/Gestão (Y)
    ],
    'Cor': [
        'blue', 'blue', 'blue',
        'green', 'green', 'green',
        'red', 'red', 'red',
        'gold' # Destaque para LABRATS
    ]
}

df = pd.DataFrame(data)

# 2. Configuração do Gráfico
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='X', y='Y', data=df, s=200, color=df['Cor'], ax=ax, zorder=3)

# 3. Desenhar e Rotular os Quadrantes
# Linhas de Divisão (Média em 5)
ax.axvline(5, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(5, color='gray', linestyle='--', linewidth=0.8)

# Configurar Limites e Títulos
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Matriz de Posicionamento Competitivo', fontsize=16, fontweight='bold')
ax.set_xlabel('Foco no Conteúdo (Eixo X)', fontsize=12)
ax.set_ylabel('Uso Principal da Plataforma (Eixo Y)', fontsize=12)

# Rótulos dos Eixos (Descrição)
ax.text(0.1, -0.7, 'Curado pela Plataforma (Estruturado)', fontsize=10, color='darkgray')
ax.text(7.5, -0.7, 'Gerado pelo Utilizador (UGC/Comunitário)', fontsize=10, color='darkgray')
ax.text(-0.7, 0.5, 'Individual/Autoestudo', rotation=90, fontsize=10, color='darkgray')
ax.text(-0.7, 7.5, 'Colaborativo/Gestão de Turma', rotation=90, fontsize=10, color='darkgray')

# 4. Rotular os Quadrantes
# Q1: Baixo Y, Baixo X
ax.text(2.5, 0.5, 'Q1: Consumo Estruturado', fontsize=11, fontweight='bold', ha='center', color='blue')
# Q3: Baixo Y, Alto X
ax.text(7.5, 0.5, 'Q3: Comunidade UGC (Notas/Flashcards)', fontsize=11, fontweight='bold', ha='center', color='green')
# Q2: Alto Y, Baixo X
ax.text(2.5, 9.5, 'Q2: Gestão de Aprendizagem Curada', fontsize=11, fontweight='bold', ha='center', color='blue')
# Q4: Alto Y, Alto X
ax.text(7.5, 9.5, 'Q4: Ambiente de Gestão e Colaboração', fontsize=11, fontweight='bold', ha='center', color='red')


# 5. Adicionar Nomes das Plataformas aos Pontos
for i in range(len(df)):
    plt.text(
        df['X'][i] + 0.2, df['Y'][i], # Posição do texto ligeiramente deslocada
        df['Plataforma'][i],
        fontsize=9,
        color='black' if df['Plataforma'][i] != 'LABRATS' else 'darkorange',
        fontweight='normal' if df['Plataforma'][i] != 'LABRATS' else 'bold'
    )

# 6. Destaque LABRATS
# Adicionar um círculo extra para destacar LABRATS se desejar
ax.scatter(df.loc[df['Plataforma'] == 'LABRATS', 'X'],
           df.loc[df['Plataforma'] == 'LABRATS', 'Y'],
           s=400, facecolors='none', edgecolors='gold', linewidths=2, zorder=5)

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()