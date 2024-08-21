# Aplicando SMOTE na Base de Dados do Censo

Este projeto visa aplicar a técnica de oversampling **SMOTE (Synthetic Minority Over-sampling Technique)** para balancear uma base de dados desbalanceada. O foco é ajustar a distribuição da variável `income`, que classifica indivíduos como ganhando `<=50K` ou `>50K` por ano, com predominância da classe `<=50K`.

## Problema de Desbalanceamento
A base de dados apresenta uma distribuição desigual entre as classes de `income`, o que prejudica a capacidade de modelos preditivos de identificar corretamente a classe minoritária (`>50K`). O desbalanceamento reduz a eficácia dos algoritmos, levando a previsões enviesadas.

## Técnica Utilizada: SMOTE
O **SMOTE** cria exemplos sintéticos da classe minoritária para igualar sua quantidade à da classe majoritária, permitindo que os modelos aprendam de maneira mais equilibrada. Assim, a técnica melhora a capacidade preditiva de modelos de machine learning em bases de dados desbalanceadas.

## Aplicação e Resultados
Após transformar variáveis categóricas em numéricas, o SMOTE foi aplicado, equilibrando as classes de `income`. Com isso, os modelos de machine learning passaram a ter melhor desempenho ao classificar indivíduos de ambas as classes. A precisão das previsões para a classe minoritária aumentou consideravelmente.
