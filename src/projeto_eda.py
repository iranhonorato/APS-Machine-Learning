#!/usr/bin/env python
# coding: utf-8

# # **Projeto de Machine Learning**
# 
# ## **APS1: Projeto EDA**

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt


# ### 1. Carregamento e inspeção inicial dos dados:

# In[3]:
def read_adult_data(path:int="../data/adult.data") -> None:
    with open(path, "r") as file:
        for i in range(5):
            print(file.readline())
    return 


# In[4]:

def read_adult_names(path:int="../data/adult.names") -> None:

    with open(path, "r") as file:
        for i, linha in enumerate(file):
            print(i, linha.strip())
    return

# Após uma análise dos arquivos "adult.data" e "adult.names" foi possível descobrir o nome das colunas em ordem para a estruturação do dataframe: 
# 
# **Variável Alvo (Target)**
# ---
# `income`: Indica se a renda anual é maior ou menor que 50 mil dólares.
# 
# **Características Demográficas e Sociais**
# ---
# * `age` Idade da pessoa em anos. (Contínua)
# 
# * `race` Raça declarada.(Categorias: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.)
# 
# * `sex` Sexo biológico. (Categorias: Female, Male.)
# 
# * `native-country` País de origem da pessoa. (Exemplos: United-States, Mexico, Philippines, Germany, etc. (Mais de 40 países).)
# 
# **Educação**
# ---
# * `education` Nível educacional da pessoa. (Exemplos: Bachelors, Some-college, 11th, HS-grad, Masters, Doctorate, etc.)
# 
# * `education-num` Representação numérica do nível educacional. (contínua. Geralmente, quanto maior o número, maior o nível de escolaridade.)
# 
# **Vida Civil e Relacionamentos**
# ---
# * `marital-status` Estado civil da pessoa. (Categorias: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, etc.)
# 
# * `relationship` Papel da pessoa na estrutura familiar. (Categorias: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.)
# 
# 
# **Trabalho e Ocupação**
# ---
# * `hours-per-week` Número de horas trabalhadas por semana. (Contínua)
# 
# * `workclass` Tipo de empregador ou setor de trabalho. (Categorias: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.)
# 
# * `occupation` Profissão ou ocupação da pessoa. (Categorias: Tech-support, Craft-repair, Exec-managerial, Prof-specialty, Sales, etc.)
# 
# 
# **Variáveis Financeiras e Técnicas**
# ---
# * `capital-gain` Ganhos de capital (Contínua. Lucro com venda de ativos).
# 
# * `capital-loss` Perdas de capital (Contínua. Prejuízos financeiros).
# 
# * `fnlwgt` Peso amostral atribuído pelo Census Bureau. (Explicação: É uma estimativa de quantas pessoas na população real aquele registro representa, baseada em características socio-demográficas semelhantes.)
# 
# 
# 

# In[5]:

def get_dataframe(path:str = "../data/adult.data") -> pd.DataFrame:
    colunas = ["age", "workclass", "fnlwgt", "education", 
            "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", 
            "capital-loss", "hours-per-week", "native-country", "income"]

    df = pd.read_csv(path, names=colunas, sep=",")

    for c in df.columns:
        if df[c].dtype == "str":
            df[c] = df[c].str.strip() 

    return df 
df = get_dataframe()

# In[6]:


print(df.info())


# In[7]:


rows_all_null = df.isna().all(axis=1).sum()
rows_some_null = df.isna().any(axis=1) & ~df.isna().all(axis=1)

print("Número de linhas com todas as features são nulas: ", rows_all_null.sum())
print("Número de linhas com algumas (não todas) features nulas: ", rows_some_null.sum())
print("\n")

#-----------------------------------------------------------------------------------------------------------------
print("Campos com valores ausentes do tipo: '?'")

for c in df.columns:
    qtd = (df[c] == "?").sum()
    if qtd > 0:
        print(f"{c}: {qtd}")

print("\n")

#------------------------------------------------------------------------------------------------------------------
info_df = pd.DataFrame({
    "Column": df.columns,
    "Dtype": df.dtypes.values
})

num = [c for c in df.columns if df[c].dtype == "int64"]
cat = [c for c in df.columns if df[c].dtype == "str"]

print(f"Dados numéricos: {len(num)} -> {num}")
print(f"Dados categóricos: {len(cat)} -> {cat}")
print("\n")


#------------------------------------------------------------------------------------------------------------------
menor = len(df[df["income"] == "<=50K"])
maior = len(df[df["income"] == ">50K"])
print("Distribuição do target:")
print(f"income <= 50k: {menor} ({100*menor/(len(df)):.2f}%)")
print(f"income >50k: {maior} ({100*maior/(len(df)):.2f}%)")


# ### 2. Análise univariada

# In[8]:


print(df[num].describe())


# In[9]:

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, c in enumerate(num):
    axes[i].boxplot(df[c], vert=False)
    axes[i].set_title(f"Boxplot de {c}")
    axes[i].set_xlabel(c)

plt.tight_layout()
plt.show()


for c in num:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[c] < lower_bound) | (df[c] > upper_bound)]
    print(f"{c}: {len(outliers)} outliers")


# In[10]:


fig, axes = plt.subplots(3, 3, figsize=(15, 12))

axes = axes.flatten()

for i, col in enumerate(cat):
    if col == "native-country":
        df[col].value_counts().head(12).plot(kind='bar', ax=axes[i], title=col, logy=True)
        axes[i].set_ylabel('Frequência em log')
        continue
    if col == "race":
        df[col].value_counts().head(12).plot(kind='bar', ax=axes[i], title=col, logy=True)
        axes[i].set_ylabel('Frequência em log')
        continue 
    df[col].value_counts().plot(kind='bar', ax=axes[i], title=col)
    axes[i].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

