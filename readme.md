# FT Linear Regression

## Documentação Técnica — Implementação Monolítica de Regressão Linear

**Projecto:** FT Linear Regression  
**Tipo de implementação:** Monolítica  
**Abordagem:** Machine Learning implementado manualmente  
**Algoritmo:** Simple Linear Regression + Batch Gradient Descent  
**Linguagem:** Python  
**Dataset:** `data.csv`  
**Feature:** `km` — quilometragem do automóvel  
**Target:** `price` — preço do automóvel  

---

## 1. Visão geral do projecto

O objectivo deste projecto foi desenvolver um modelo de regressão linear capaz de estimar o preço de um automóvel com base na sua quilometragem.

A implementação foi desenvolvida de forma **monolítica**, dentro de um único projecto Python. Não construímos uma arquitectura distribuída, não utilizámos microserviços nem separamos o sistema em aplicações independentes.

O projecto concentra as responsabilidades de preparação dos dados, treino, avaliação, persistência dos parâmetros e previsão em diferentes módulos e ficheiros, mas todos pertencem à mesma aplicação.

### Implementação manual, sem bibliotecas de Machine Learning

Uma das decisões centrais foi implementar o algoritmo manualmente, utilizando Python puro.

Não utilizámos bibliotecas como:

- Scikit-learn;
- TensorFlow;
- PyTorch;
- NumPy.

A função de previsão, a função de custo, o cálculo dos gradientes, as actualizações dos parâmetros e o processo de Gradient Descent foram implementados directamente.

No notebook, utilizámos Pandas para explorar os dados e Matplotlib para construir gráficos. Estas bibliotecas serviram como ferramentas de análise e visualização, não como mecanismos de treino do modelo.

O objectivo foi compreender o funcionamento interno de um algoritmo de Machine Learning, em vez de depender de uma implementação pronta.

## 2. Arquitectura e organização

A aplicação segue uma organização monolítica simples:

```text
ft_linear_regression/
│
├── data.csv
│
├── ft_linear_regrassion.ipynb
│
├── train.py
│
├── predict.py
│
└── theta.json
```

### Responsabilidade dos ficheiros

| Ficheiro | Responsabilidade |
|---|---|
| `data.csv` | Armazenar os dados históricos dos automóveis. |
| `ft_linear_regrassion.ipynb` | Exploração, experiências, cálculos e visualização. |
| `train.py` | Preparação dos dados, treino, conversão dos parâmetros e avaliação. |
| `theta.json` | Persistência dos parâmetros aprendidos. |
| `predict.py` | Carregamento dos parâmetros e previsão de preços. |

Embora o treino e a previsão estejam em ficheiros diferentes, não são serviços independentes. Fazem parte da mesma aplicação e trabalham sobre os mesmos artefactos locais.

O ficheiro JSON funciona como armazenamento dos parâmetros do modelo, não como uma base de dados tradicional.

## 3. Conhecer os dados

O dataset utilizado contém 24 observações e duas colunas:

| Coluna | Significado | Papel |
|---|---|---|
| `km` | Quilometragem do automóvel | Feature (`X`) |
| `price` | Preço do automóvel | Target (`y`) |

A feature é a informação fornecida ao modelo. O target é o valor que pretendemos estimar.

### Estatísticas do dataset

| Estatística | Quilometragem (`km`) | Preço (`price`) |
|---|---:|---:|
| Contagem | 24 | 24 |
| Média | 101 066,25 | 6 331,83 |
| Desvio-padrão | 52 674,25 | 1 319,65 |
| Mínimo | 22 899 | 3 650 |
| Mediana | 83 014,50 | 6 495 |
| Máximo | 240 000 | 8 290 |

Os dados apresentam uma tendência geral: automóveis com maior quilometragem tendem a apresentar preços inferiores.

Contudo, a relação não é perfeita. O preço de um automóvel também pode depender de outros factores, como marca, modelo, ano, estado de conservação e histórico de manutenção.

Como o dataset contém apenas quilometragem e preço, o modelo não consegue considerar essas outras características.

## 4. Exploração e visualização dos dados

### 4.1 Matplotlib

Matplotlib é uma biblioteca Python utilizada para criar gráficos e visualizar dados.

No notebook, utilizámos:

```python
import matplotlib.pyplot as plt
```

O alias `plt` permite chamar as funções da biblioteca de forma mais simples.

A visualização foi importante para observar a relação entre quilometragem e preço e perceber como a recta de regressão se ajusta aos dados.

### 4.2 Scatter Plot

Um Scatter Plot é um gráfico de dispersão. Cada observação é representada por um ponto.

Neste projecto:

- eixo horizontal (`X`): quilometragem;
- eixo vertical (`Y`): preço.

Exemplo:

```python
plt.scatter(x_train, y_train, label="Treino")
```

Cada ponto representa um automóvel do conjunto de treino.

O gráfico permite observar a distribuição dos automóveis e a tendência geral dos preços.

### 4.3 Diferença entre `scatter()` e `plot()`

| Função | Representação | Utilização |
|---|---|---|
| `plt.scatter()` | Pontos individuais | Mostrar os dados observados. |
| `plt.plot()` | Linha entre coordenadas | Mostrar a recta de regressão. |

Os pontos representam observações reais. A linha representa as previsões produzidas pelo modelo.

### 4.4 Como `x_line` e `y_line` são representados?

Para desenhar a recta, construímos duas sequências de valores:

```python
x_line = sorted(x_train + x_test)

y_line = [
    b0 + b1 * ((km - mean_x) / std_x)
    for km in x_line
]
```

`x_line` contém os valores de quilometragem ordenados.

`y_line` contém a previsão correspondente a cada valor de quilometragem.

A associação é feita pela posição dos elementos:

```text
x_line[0]  →  y_line[0]
x_line[1]  →  y_line[1]
x_line[2]  →  y_line[2]
...
```

Cada par forma uma coordenada:

$$
(x_{\text{line}}[i], y_{\text{line}}[i])
$$

A ordenação de `x_line` é importante. Sem ela, o Matplotlib poderia ligar pontos numa sequência que avança e recua no eixo horizontal, produzindo uma linha visualmente confusa.

Como os parâmetros `b0` e `b1` foram aprendidos com a quilometragem normalizada, cada valor de `km` também é normalizado durante o cálculo de `y_line`.

### 4.5 Gráfico completo

```python
plt.scatter(x_train, y_train, label="Treino")
plt.scatter(x_test, y_test, label="Teste", marker="x")
plt.plot(x_line, y_line, label="Regressão")

plt.xlabel("Quilometragem (km)")
plt.ylabel("Preço")
plt.title("REGRESSÃO LINEAR: PREÇO VS QUILOMETRAGEM")
plt.legend()
plt.grid()
plt.show()
```

Elementos do gráfico:

- pontos de treino: dados utilizados para aprender os parâmetros;
- cruzes: dados reservados para teste;
- linha: previsões da regressão;
- legenda: identificação dos elementos;
- grelha: facilita a leitura das coordenadas;
- título e eixos: identificam o conteúdo representado.

A distância vertical entre um ponto observado e a linha representa o erro da previsão para aquela observação.

## 5. Separação entre treino e teste

O dataset foi embaralhado utilizando uma semente fixa:

```python
random.seed(42)
random.shuffle(data)
```

Depois foi dividido numa proporção aproximada de 80/20:

```python
index = int(len(data) * 0.8)

train_data = data[:index]
test_data = data[index:]
```

Resultado:

| Conjunto | Observações |
|---|---:|
| Treino | 19 |
| Teste | 5 |
| Total | 24 |

### Porquê esta divisão?

O conjunto de treino é utilizado para aprender os parâmetros.

O conjunto de teste permanece fora do processo de aprendizagem e é utilizado para avaliar o comportamento do modelo em exemplos que não participaram no treino.

A utilização de `random.seed(42)` permite reproduzir a mesma divisão em execuções futuras.

**Trade-off:** a divisão 80/20 é simples e adequada para compreender o processo, mas cinco observações de teste são insuficientes para uma avaliação estatística robusta.

## 6. Normalização da feature

A quilometragem apresenta valores bastante superiores aos valores do preço.

Para melhorar a escala da optimização, aplicámos a normalização Z-score à feature `X`.

### Fórmula

$$
z_i = \frac{x_i-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Onde:

- $x_i$: quilometragem original;
- $\mu_{\text{train}}$: média das quilometragens de treino;
- $\sigma_{\text{train}}$: desvio-padrão das quilometragens de treino;
- $z_i$: quilometragem normalizada.

### Estatísticas utilizadas

| Estatística | Valor |
|---|---:|
| Média de X no treino | 90 134,736842 |
| Desvio-padrão de X no treino | 40 301,992222 |

A média e o desvio-padrão foram calculados exclusivamente a partir dos dados de treino.

O conjunto de teste foi transformado utilizando exactamente as mesmas estatísticas.

### Porquê?

Para evitar Data Leakage, ou seja, a utilização indevida de informação do conjunto de teste durante a preparação dos dados.

Também decidimos não normalizar o target `price`. Assim, as previsões e os erros permaneceram na escala original dos preços.

**Trade-off:** a normalização melhora a escala da optimização, mas exige converter os parâmetros antes de os utilizar com quilometragens reais.

## 7. Modelo de regressão linear

A regressão linear simples procura encontrar uma relação linear entre a quilometragem e o preço.

### Função de previsão

$$
\hat{y} = \theta_0 + \theta_1 x
$$

Onde:

- $\hat{y}$: preço estimado;
- $x$: quilometragem;
- $\theta_0$: intercepto;
- $\theta_1$: inclinação da recta.

O intercepto representa o valor previsto quando a quilometragem é zero.

A inclinação representa a variação estimada do preço por cada quilómetro adicional.

Como a inclinação aprendida é negativa, o modelo representa uma associação decrescente entre quilometragem e preço.

### Função de custo utilizada no treino

O código calcula os gradientes utilizando a função de custo com o factor
$\frac{1}{2m}$. Este factor $1/2$ é uma convenção matemática que
simplifica as derivadas. A métrica MSE utilizada na avaliação é calculada
separadamente e não altera as actualizações do treino.

$$
J(\theta_0,\theta_1)
=
\frac{1}{2m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)^2
$$

Substituindo a função de previsão:

$$
J(\theta_0,\theta_1)
=
\frac{1}{2m}
\sum_{i=1}^{m}
(\theta_0+\theta_1x_i-y_i)^2
$$

O erro é elevado ao quadrado para impedir que erros positivos e negativos se anulem e para penalizar mais os erros de maior magnitude.

## 8. Gradient Descent

Gradient Descent é o algoritmo utilizado para ajustar os parâmetros do modelo, procurando minimizar a função de custo.

O processo consiste em:

1. inicializar os parâmetros;
2. calcular previsões;
3. calcular os erros;
4. calcular os gradientes;
5. actualizar os parâmetros;
6. repetir o processo.

### Erro individual

$$
e_i = \hat{y}_i-y_i
$$

### Derivada parcial em relação ao intercepto

$$
\frac{\partial J}{\partial \theta_0}
=
\frac{1}{m}
\sum_{i=1}^{m}(\hat{y}_i-y_i)
$$

### Derivada parcial em relação à inclinação

$$
\frac{\partial J}{\partial \theta_1}
=
\frac{1}{m}
\sum_{i=1}^{m}(\hat{y}_i-y_i)x_i
$$

### Actualização dos parâmetros

$$
\theta_0 :=
\theta_0 -
\alpha
\frac{\partial J}{\partial \theta_0}
$$

$$
\theta_1 :=
\theta_1 -
\alpha
\frac{\partial J}{\partial \theta_1}
$$

Onde $\alpha$ representa a taxa de aprendizagem (*learning rate*).

### Configuração utilizada

| Parâmetro | Valor |
|---|---:|
| Learning rate | 0,1 |
| Epochs | 500 |
| Inicialização no script | 0 para ambos os parâmetros |
| Tipo de Gradient Descent | Batch |

Uma epoch corresponde a uma passagem completa por todos os exemplos de treino.

A implementação utiliza Batch Gradient Descent porque acumula os gradientes de todas as observações antes de actualizar os parâmetros.

### Acumulação dos gradientes

A estrutura correcta utilizada foi:

```python
for epoch in range(iterations):
    aux_theta_0 = 0
    aux_theta_1 = 0

    for i in range(m):
        y_pred = theta_0 + theta_1 * x[i]
        error = y_pred - y[i]

        aux_theta_0 += error
        aux_theta_1 += error * x[i]

    aux_theta_0 /= m
    aux_theta_1 /= m

    theta_0 -= learning_rate * aux_theta_0
    theta_1 -= learning_rate * aux_theta_1
```

Os acumuladores têm de ser inicializados uma vez por epoch, antes do ciclo interno.

Se forem inicializados dentro do ciclo das observações, os valores anteriores são apagados a cada iteração e o algoritmo deixa de acumular o gradiente de todos os exemplos.

### Convergência observada

| Epoch | b0 | b1 |
|---:|---:|---:|
| 0 | 0 | 0 |
| 100 | 6 520,089971 | -926,813805 |
| 200 | 6 520,263153 | -926,838423 |
| 300 | 6 520,263158 | -926,838423 |
| 400 | 6 520,263158 | -926,838423 |

Os parâmetros praticamente deixaram de mudar depois da epoch 200.

Isto demonstra convergência numérica para a configuração utilizada e para o conjunto de treino disponível.

## 9. Conversão dos parâmetros

O treino foi realizado com a quilometragem normalizada:

$$
z = \frac{x-\mu}{\sigma}
$$

A função aprendida foi:

$$
\hat{y}=b_0+b_1z
$$

Para utilizar quilometragens reais no `predict.py`, foi necessário converter os parâmetros.

Substituindo a normalização:

$$
\hat{y}
=
b_0+b_1\left(\frac{x-\mu}{\sigma}\right)
$$

Reorganizando:

$$
\hat{y}
=
\left(b_0-\frac{b_1\mu}{\sigma}\right)
+
\left(\frac{b_1}{\sigma}\right)x
$$

Assim:

$$
\theta_1 = \frac{b_1}{\sigma}
$$

$$
\theta_0 = b_0-\frac{b_1\mu}{\sigma}
$$

### Parâmetros obtidos

| Parâmetro | Valor |
|---|---:|
| b0 normalizado | 6 520,263157894733 |
| b1 normalizado | -926,8384234032482 |
| Média de X | 90 134,73684210527 |
| Desvio-padrão de X | 40 301,99222167573 |
| theta_0 | 8 593,121912099761 |
| theta_1 | -0,02299733517651676 |

### Modelo final

$$
\boxed{\widehat{price}=8593.1219121 - 0.02299733518 \times km}
$$

A inclinação indica uma redução estimada de aproximadamente 0,023 unidades de preço por quilómetro adicional.

Esta relação representa uma associação aprendida no dataset, não uma relação causal universal.

## 10. Persistência dos parâmetros e previsão

Depois do treino, decidimos guardar os parâmetros num ficheiro JSON.

```json
{
    "theta_0": 8593.121912099761,
    "theta_1": -0.02299733517651676
}
```

O JSON funciona como armazenamento dos parâmetros aprendidos.

É importante distinguir:

- `data.csv`: contém os dados utilizados para treinar;
- `theta.json`: contém os parâmetros aprendidos pelo modelo.

Não estamos a guardar uma base de dados de automóveis no JSON. Estamos a guardar o resultado do processo de aprendizagem.

### Carregamento dos parâmetros

O `predict.py` lê o JSON e recupera os valores de `theta_0` e `theta_1`.

As chaves utilizadas no código têm de corresponder exactamente às chaves presentes no ficheiro.

### Função de previsão

$$
\widehat{price}
=
\theta_0+\theta_1 \times km
$$

A previsão é calculada com os parâmetros já convertidos para a escala original.

### Validação da entrada

O programa solicita a quilometragem ao utilizador, converte a entrada para um número e trata entradas não numéricas.

Também foi adicionada uma verificação para quilometragens negativas e um aviso para valores fora do intervalo observado no treino.

### Exemplos de previsão

| Quilometragem | Preço estimado |
|---:|---:|
| 100 000 km | 6 293,3884 |
| 150 000 km | 5 143,52 |
| 300 km | 8 586,22 |
| 500 000 km | -2 905,55 |

O resultado negativo para 500 000 km é matematicamente coerente com a recta, mas não é uma estimativa comercialmente plausível.

Isto demonstra uma limitação importante: a regressão linear continua a extrapolar a recta mesmo fora do intervalo dos dados observados.

**Nota:** a execução `python predict.py 300` não passa automaticamente o valor `300` para o programa, porque a implementação utiliza `input()`. A leitura de argumentos posicionais através de `argparse` ficou como possível melhoria futura.

## 11. Avaliação do modelo

O modelo foi avaliado com as cinco observações reservadas para teste.

### Mean Squared Error

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

### Root Mean Squared Error

$$
RMSE = \sqrt{MSE}
$$

O MSE mede a média dos erros ao quadrado. O RMSE é a sua raiz quadrada e volta à unidade original do preço.

### Resultados

| Métrica | Resultado |
|---|---:|
| MSE de treino | 491 966,52 |
| RMSE de treino | aproximadamente 701,40 |
| MSE de teste | 319 253,78 |
| RMSE de teste | 565,0254707729614 |

Resultado observado:

```text
Test MSE: 319253.78262220667
Test RMSE: 565.0254707729614
```

O RMSE de teste indica a escala do erro quadrático médio neste pequeno conjunto de avaliação.

Não significa que todas as previsões tenham um erro exacto de 565 unidades.

O facto de o RMSE de teste ser inferior ao de treino não demonstra que o modelo generaliza melhor. O conjunto de teste contém apenas cinco observações e, por isso, a métrica é sensível aos exemplos seleccionados.

## 12. Resultados gerais

| Elemento | Resultado |
|---|---|
| Observações | 24 |
| Features | 1 |
| Target | 1 |
| Treino | 19 |
| Teste | 5 |
| Seed | 42 |
| Learning rate | 0,1 |
| Epochs | 500 |
| theta_0 | 8 593,121912 |
| theta_1 | -0,02299733518 |
| MSE treino | 491 966,52 |
| RMSE treino | 701,40 |
| MSE teste | 319 253,78 |
| RMSE teste | 565,03 |

### Intervalo observado no treino

- Mínimo: 22 899 km.
- Máximo: 176 000 km.

O dataset completo contém um automóvel com 240 000 km, mas esse exemplo ficou no conjunto de teste.

O aviso do programa utiliza o intervalo observado no treino. Estar dentro desse intervalo não garante que uma previsão seja necessariamente fiável.

## 13. Problemas encontrados e correcções

### 13.1 Acumuladores dos gradientes

Inicialmente, os acumuladores dos gradientes estavam posicionados dentro do ciclo das observações.

Isso fazia com que os valores fossem reinicializados para cada automóvel.

Corrigimos a indentação, colocando os acumuladores no início de cada epoch.

### 13.2 Factor 2 nas derivadas

A implementação divide os acumuladores por $m$:

```python
aux_theta_0 /= m
aux_theta_1 /= m
```

Isto corresponde às derivadas da função de custo
$\frac{1}{2m}\sum(\hat{y}_i-y_i)^2$. Não é utilizado o factor $2/m$,
porque o factor $1/2$ da função de custo cancela o $2$ resultante da
derivação. A MSE apresentada na avaliação continua a ser calculada como
$\frac{1}{n}\sum(\hat{y}_i-y_i)^2$.

### 13.3 Parâmetros normalizados

Os parâmetros aprendidos com `X` normalizado não podiam ser aplicados directamente a quilómetros reais.

Implementámos a conversão para `theta_0` e `theta_1`.

### 13.4 Persistência JSON

Garantimos que as chaves utilizadas para guardar os parâmetros correspondem às chaves utilizadas no carregamento.

### 13.5 Previsões fora do intervalo

A regressão linear pode produzir resultados pouco plausíveis quando recebe quilometragens muito superiores às observadas.

Foi acrescentado um aviso para tornar essa limitação visível.

## 14. Decisões e trade-offs

| Decisão | Vantagem | Limitação |
|---|---|---|
| Arquitectura monolítica | Simplicidade e facilidade de execução | Menor separação operacional do que uma arquitectura distribuída |
| Python puro para o algoritmo | Compreensão dos mecanismos internos | Exige implementar e validar manualmente os cálculos |
| Uma única feature | Modelo simples e interpretável | Ignora outros factores do preço |
| Divisão 80/20 | Permite avaliação independente | Apenas cinco exemplos de teste |
| Seed 42 | Reprodutibilidade | Uma única divisão pode não representar todos os casos |
| Normalizar apenas X | Melhora a escala da optimização | Exige converter os parâmetros |
| JSON | Persistência simples e legível | Não substitui uma base de dados nem uma plataforma de gestão de modelos |
| Aviso fora do intervalo | Torna a extrapolação visível | Não impede todas as previsões pouco plausíveis |

## 15. Limitações e melhorias futuras

### Limitações

1. Dataset pequeno: apenas 24 observações.
2. Pouca diversidade de exemplos para avaliar generalização.
3. Apenas uma feature: quilometragem.
4. Modelo linear: pode não representar relações mais complexas.
5. Conjunto de teste muito pequeno.
6. Extrapolação pode produzir previsões negativas ou pouco realistas.
7. Não existe informação suficiente para avaliar factores como marca, modelo, ano ou estado do automóvel.

### Melhorias possíveis

- recolher mais observações e maior diversidade;
- analisar os resíduos;
- comparar com um baseline que prevê sempre a média do treino;
- utilizar validação cruzada quando houver dados suficientes;
- experimentar outras features, caso o âmbito do projecto permita;
- testar diferentes taxas de aprendizagem;
- validar valores especiais como `NaN` e infinito;
- tratar ficheiros JSON inexistentes ou malformados;
- implementar argumentos posicionais com `argparse`.

Estas melhorias são possibilidades futuras e não devem ser confundidas com funcionalidades já implementadas.

## 16. Execução do projecto

### Treino

```bash
python train.py
```

O processo deve carregar os dados, separar treino e teste, normalizar a feature, executar Gradient Descent, converter os parâmetros, avaliar o modelo e guardar o JSON.

### Previsão

```bash
python predict.py
```

O programa carrega os parâmetros e solicita a quilometragem para calcular o preço estimado.

## 17. Conclusão

Este projecto permitiu implementar manualmente uma regressão linear simples, desde a preparação dos dados até à previsão.

Foram trabalhados os seguintes conceitos:

- Feature e Target;
- exploração de dados;
- Scatter Plot e visualização com Matplotlib;
- divisão treino/teste;
- normalização Z-score;
- Data Leakage;
- função de previsão;
- Mean Squared Error;
- derivadas parciais;
- Gradient Descent;
- Batch Gradient Descent;
- conversão de parâmetros;
- persistência JSON;
- avaliação com MSE e RMSE;
- limitações de generalização e extrapolação.

O modelo final aprendeu a seguinte relação:

$$
\boxed{\widehat{price}=8593.1219121 - 0.02299733518 \times km}
$$

O RMSE de teste foi aproximadamente **565,03**, calculado sobre apenas cinco observações.

O resultado demonstra que o pipeline foi implementado e executado, mas não permite concluir que o modelo generaliza muito bem. Para isso, seria necessário um dataset maior e mais diversificado, além de uma avaliação mais robusta.

A aprendizagem principal foi compreender como os dados, a normalização, a função de custo, os gradientes e a avaliação se relacionam na construção de um modelo de Machine Learning — sem depender de uma biblioteca que esconda a implementação do algoritmo.

---

**Nota de manutenção:** esta documentação representa as decisões e os resultados obtidos durante o desenvolvimento. Deve ser actualizada se o código, os parâmetros ou os resultados forem alterados.