
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.dates import date2num
from sklearn.preprocessing import StandardScaler
import neurolab as nl
# Загрузка данных
data_2020 = pd.read_csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2020\DAT_ASCII_EURUSD_M1_2020_new.csv')
data_2021 = pd.read_csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2021\DAT_ASCII_EURUSD_M1_2021_new.csv')
data_2022 = pd.read_csv(r'C:\Users\ext17\Downloads\DAT_ASCII_EURUSD_M1_2022\DAT_ASCII_EURUSD_M1_2022_new.csv')

# Объединить три фрейма данных в один
data = pd.concat([data_2020, data_2021, data_2022], ignore_index=True)
data['Close-Open'] = data['Close'] - data['Open']
data['High-Low'] = data['High'] - data['Low']
data['High-Close'] = data['High'] - data['Close']
data['High-Open'] = data['High'] - data['Open']
data['Low-Close'] = data['Low'] - data['Close']
data['Low-Open'] = data['Low'] - data['Open']

# Стандартизация данных
# ... (остальной код загрузки данных и предобработки остается без изменений)

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Создание и обучение сети Кохонена
n_clusters = 4000
dim1 = min(scaled_data[:, 0])
dim2 = max(scaled_data[:, 0])
dims = [[dim1, dim2] for _ in range(scaled_data.shape[1])]
net = nl.net.newc(dims, n_clusters)
error = net.train(scaled_data, epochs=500, show=100)

# Кластеризация
output = net.sim(scaled_data)
labels = np.argmax(output, axis=1)
data['Cluster'] = labels

# ... (остальной код для вычисления средних значений, построения графиков и визуализации остается без изменений)

with open(r'C:\Users\ext17\Downloads\cluster888.txt', "w") as f:
    for idx, label in enumerate(data['Cluster']):
        if idx != 0:
            f.write(" ")
        f.write(str(label))
pd.set_option('display.max_rows', None)
print(data['Cluster'].value_counts())

# ... (предыдущий код)

# Вычисление средних арифметических моделей свеч для каждого класса
# Вычисление средних арифметических моделей свеч для каждого класса
mean_values = data.groupby('Cluster').mean()[['Close-Open', 'High-Low', 'High-Close', 'High-Open', 'Low-Close', 'Low-Open']]
mean_values1 = mean_values.rename(columns={'Close-Open': 'Mean_Close_Open'})
mean_values1.to_csv(r'C:\Users\ext17\Downloads\mean.csv')
data[['Cluster', 'Close-Open']].groupby('Cluster').mean().rename(columns={'Close-Open': 'Mean_Close_Open'}).to_csv(r'C:\Users\ext17\Downloads\means8888.csv')
# Отображение графиков
n_plots = 20
plot_counter = 0
bar_width = 0.15

while plot_counter < n_plots:
    cluster_label = plot_counter % n_clusters
    mean_data = mean_values.loc[cluster_label]

    fig, ax = plt.subplots()
    index = np.arange(len(mean_data))
    rects1 = ax.bar(index, mean_data, bar_width, label=f'Cluster {cluster_label}')

    ax.set_title(f'Средние значения атрибутов свечей для кластера {cluster_label}')
    ax.set_xticks(index)
    ax.set_xticklabels(['Close-Open', 'High-Low', 'High-Close', 'High-Open', 'Low-Close', 'Low-Open'])
    ax.legend()

    plt.show()
    plot_counter += 1



# Определение параметров визуализации
mc = mpf.make_marketcolors(up='g', down='r')
s = mpf.make_mpf_style(marketcolors=mc)

# Количество графиков и котировок в каждом
n_plots = 20
quotes_per_plot = 10

# Отображение графиков
plot_counter = 0
while plot_counter < n_plots:
    cluster_label = plot_counter % n_clusters
    plot_data = data[data['Cluster'] == cluster_label].iloc[:quotes_per_plot]

    if len(plot_data) == quotes_per_plot:
        # Визуализация котировок в виде свечей
        fig, axes = mpf.plot(plot_data, type='candle', title=f'Котировки с метками классификации {cluster_label}', ylabel='Цена', returnfig=True, style=s)
        # Добавление меток классификации над каждой свечой
        for idx, (index, row) in enumerate(plot_data.iterrows()):
            offset = 0.01 * (row['High'] - row['Low'])  # Отступ составляет 1% от диапазона свечи
            label_position = row['High'] + offset
            axes[0].annotate(str(cluster_label), xy=(idx, label_position), fontsize=8, color='black', backgroundcolor='yellow')

    plt.show()
    plot_counter += 1

data = data.drop(plot_data.index)
