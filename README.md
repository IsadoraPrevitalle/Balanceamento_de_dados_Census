# Aplicando Técnicas de Balanceamento na Base de Dados do Census

Este projeto visa aplicar diferentes técnicas de balanceamento para tratar o desbalanceamento da base de dados do **Census**, com foco na variável income, que classifica indivíduos como ganhando <=50K ou >50K por ano. As técnicas abordadas incluem o **SMOTE (Synthetic Minority Over-sampling Technique)** e a subamostragem por **Tomek Link**.

## Problema de Desbalanceamento
A base de dados do Census apresenta um desbalanceamento acentuado entre as classes de income, com a classe <=50K representando a maioria. Esse desequilíbrio dificulta o desempenho dos modelos preditivos, que tendem a classificar incorretamente a classe minoritária (>50K). Como resultado, os algoritmos tendem a enviesar suas previsões, prejudicando a capacidade de identificar corretamente indivíduos com rendimentos maiores.

## Técnica Utilizada: Tomek Link
A subamostragem utilizando **Tomek Link** visa remover amostras redundantes e sobrepostas entre as classes. Tomek Links são pares de amostras, uma de cada classe, que são os vizinhos mais próximos entre si, mas pertencem a classes diferentes. Quando esses pares são identificados, a amostra pertencente à classe majoritária é removida, diminuindo o desequilíbrio entre as classes e ajudando a melhorar a distinção entre elas.

## Técnica Utilizada: SMOTE
O **SMOTE** cria exemplos sintéticos da classe minoritária para aumentar sua representatividade na base de dados, igualando-a à classe majoritária. Isso permite que os modelos de machine learning aprendam de forma mais equilibrada, resultando em uma melhoria no desempenho preditivo para a classe menos representada.

## Aplicação e Resultados
Após transformar as variáveis categóricas em numéricas, tanto o **Tomek Link** quanto o **SMOTE** foram aplicados para balancear as classes de income. Os resultados obtidos com os modelos de machine learning foram comparados em três cenários:

- **Sem balanceamento**: A precisão geral do modelo ficou em **85%**.
- **Com balanceamento usando Tomek Link**: Houve uma leve melhora, com a precisão chegando a **86%**.
- **Com balanceamento usando SMOTE**: A técnica de oversampling trouxe um avanço significativo, aumentando a precisão para **91%**.

## Conclusão
Os resultados demonstram que técnicas de balanceamento são essenciais para melhorar o desempenho de modelos em bases de dados desbalanceadas. Embora o **Tomek Link** tenha trazido uma melhoria moderada na precisão, o **SMOTE** foi particularmente eficaz, resultando em um aumento substancial na acurácia preditiva. Isso indica que, para esse conjunto de dados específico, o uso de oversampling (como o SMOTE) pode ser uma abordagem mais eficaz para lidar com desbalanceamento de classes.
