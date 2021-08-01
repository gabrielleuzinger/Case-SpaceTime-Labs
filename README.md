Este repositório contém a solução do case do processo seletivo para Cientista de Dados na SpaceTime Labs. O objetivo deste projeto desenvolver um modelo capaz de prever o TCH dos blocos do dataset para a safra de 2020 usando os dados disponibilizados.

Este projeto foi montado seguindo as premissas de [Pesquisas Reprodutíveis](https://pt.coursera.org/learn/reproducible-research), de modo que qualquer pessoa consiga chegar aos mesmos resultados que eu utilizando os passos que segui no Jupyter Notebook.

# Dependências do projeto

Todas as dependências podem ser encontradas no arquivo `requirements.txt`, mas abaixo estão listadas:
* pandas
* numpy
* matplolib
* seaborn
* scikit-learn
* Jupyter Notebook

Para instalar as dependências execute na pasta raiz do projeto: `pip install -r requirements.txt`. 

Para acessar o Jupyter Notebook que criei, execute na pasta raiz do projeto `jupyter notebook`. Logo em seguida seu browser será aberto e basta selecionar o arquivo `Case SpaceTime Labs - colheita-2020.ipynb`. 

É importante frisar que os dados utilizados para este desafio não foram adicionados a este projeto. 

# Estrutura do projeto

```{sh}
  .
  |-data
  |-report
  |  |- markdown
  |  |  |- Case SpaceTime Labs - colheita-2020.md
  |- Case SpaceTime Labs - colheita-2020.ipynb
  |- requirements.txt
```

A pasta `report` contém um arquivo markdown com uma versão do relatório gerado a partir do estudo feito nesse projeto. Esse arquivo contém **todos os insights e estudos feitos, bem como uma descrição detalhada de como foi elaborado o projeto**.
