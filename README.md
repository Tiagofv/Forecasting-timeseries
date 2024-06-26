
# Forecasting-timeseries
O projeto compara duas abordagens para solucionar o problema de forecasting timeseries, comparando uma implementação naive e outra utilizando RNN (LSTM).

# Dataset
O dataset utilizado foi o DEOK_hourly.csv, que contém dados de consumo de energia por hora em Ohio, EUA. O dataset contém as seguintes colunas:
- Datetime: timestamp
- DEOK_MW: demanda de energia em megawatts
dataset https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=DOM_hourly.csv

# Implementação ingenua

A abordagem ingênua implementada neste script é um método simples de previsão baseado em médias históricas. Aqui está uma breve explicação:

Agrupamento de Dados: A abordagem agrupa os dados históricos de demanda de energia por dois fatores:

Dia da semana (0-6, representando segunda a domingo)
Hora do dia (0-23)


Cálculo de Médias: Para cada combinação única de dia e hora, calcula a demanda média de energia (DEOK_MW).
Previsão: Quando solicitado a prever a demanda de energia para uma determinada data e hora:

Determina o dia da semana e a hora para essa data e hora.
Em seguida, retorna a demanda média pré-calculada para essa combinação específica de dia e hora.



Por exemplo, se estiver prevendo para terça-feira às 15h, retornaria a demanda histórica média de todas as terças-feiras às 15h nos dados de treinamento.
Este método captura padrões básicos semanais e diários na demanda de energia, assumindo que esses padrões se repetem consistentemente. É considerado "ingênuo" porque não leva em conta outros fatores potencialmente importantes como clima, tendências de longo prazo ou flutuações recentes na demanda. Sua simplicidade o torna uma boa linha de base para comparação com modelos mais sofisticados.

# Implementação RNN (LSTM)

Foi utilizado uma LSTM treinada por 3 épocas com o framework keras.

#  Resultados

MAE RNN: 861,34
MAE Naive: 7441,36

O modelo RNN (Rede Neural Recorrente) supera significativamente a abordagem ingênua (Naive). O erro do RNN é cerca de 11,6% do erro da abordagem ingênua, representando uma redução de 88,4% no erro.

O MAE (Erro Médio Absoluto) menor do RNN indica que suas previsões estão, em média, muito mais próximas dos valores reais.

A grande diferença no desempenho justifica o uso de técnicas mais avançadas de aprendizado de máquina para este problema.
A abordagem RNN parece capturar padrões e relações que a abordagem naive não captura.