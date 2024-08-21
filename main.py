import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('census.csv')
print(df)

#Contagem dos valores distintos coluna 'income'
print(np.unique(df['income'], return_counts=True)) #Base de dados desbalanceda

#Visualização da quantidade de valores da coluna 'income'
sns.countplot(x=df['income'])
plt.show()

#Selecionando as colunas da base com excesão do 'income'
X_censu = df.iloc[:,0:14].values

print(X_censu)

#Selecionando a coluna 'income'
Y_censu = df.iloc[:,14].values

print(Y_censu)

#Criar label encoder p/ cada atributo categórico
from sklearn.preprocessing import LabelEncoder

#Transformar variaveis categóricos em números
Label_encoder_workclass = LabelEncoder()
Label_encoder_education = LabelEncoder()
Label_encoder_marital = LabelEncoder()
Label_encoder_ocupation = LabelEncoder()
Label_encoder_relationship = LabelEncoder()
Label_encoder_raca = LabelEncoder()
Label_encoder_sexo = LabelEncoder()
Label_encoder_country = LabelEncoder()

X_censu[:,1] = Label_encoder_workclass.fit_transform(X_censu[:,1])
X_censu[:,3] = Label_encoder_education.fit_transform(X_censu[:,3])
X_censu[:,5] = Label_encoder_marital.fit_transform(X_censu[:,5])
X_censu[:,6] = Label_encoder_ocupation.fit_transform(X_censu[:,6])
X_censu[:,7] = Label_encoder_relationship.fit_transform(X_censu[:,7])
X_censu[:,8] = Label_encoder_raca.fit_transform(X_censu[:,8])
X_censu[:,9] = Label_encoder_sexo.fit_transform(X_censu[:,9])
X_censu[:,13] = Label_encoder_country.fit_transform(X_censu[:,13])

#Aplicando Sobreamostragem SMOT - aumentar os dados

from imblearn.over_sampling import SMOTE

print('X celsu', X_censu.shape)

smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X_censu, Y_censu)

print('X_over', X_over.shape)
print('Y_censu', np.unique(Y_censu, return_counts=True)) #Qtd de valores antes aplicação SMOTE
print('Y_over', np.unique(y_over, return_counts=True)) #Qtd de valores após aplicação SMOTE

from sklearn.model_selection import train_test_split

X_censu_treinamento_over, X_censu_teste_over, y_censu_treinamento_over, y_teste_treinamento_over = train_test_split(X_over, y_over, test_size=0.15)