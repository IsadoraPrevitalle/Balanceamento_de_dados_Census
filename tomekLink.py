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

# Subamostragem 
from sklearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')

# Tinha 32561 - passa a ter 30162
X_under, Y_under = tl.fit_resample(X_censu, Y_censu)

print(X_under.shape, Y_under.shape)

print(np.unique(Y_censu, return_counts=True))
print(np.unique(Y_under, return_counts=True))

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,13])], remainder='passthrough')

X_censu = onehotencoder.fit_transform(X_under).toarray()
print(X_censu)
print(X.census.shape, X_under.shape) #verificando como os dados ficaram após a tecnica vs como eram

from sklearn.model_selection import train_test_split
X_censu_treinamento_under, X_censu_teste_under, y_census_treinamento_under, y_censu_teste_under = train_test_split(X_under, test_size=0.15, random_state=0)

# A maioria 'vence'
from sklearn.ensemble import RandomForestClassifier 
random_forest_census = RandomForestClassifier(criterion='entropy', n_estimators=100)

# Aplicação do algoritmo
random_forest_census.fit(X_censu_treinamento_under, y_census_treinamento_under)

from sklearn.metrics import accuracy_score, classification_report
previsoes = random_forest_census.predict(X_censu_teste_under)
print(accuracy_score(y_censu_teste_under, previsoes))