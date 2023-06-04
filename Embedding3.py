
# Загрузка данных из файла Excel
file_path = r"C:\Users\ext17\Downloads\aud.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Переименование столбца '1 AUD=' в 'AUD_value'
df.columns.values[1] = 'AUD_value'

# Преобразование колонки 'Date' в тип данных datetime
df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')

# Вычисление разности между значениями колонки 'AUD_value' для последующего и текущего дня
df['Difference'] = df['AUD_value'].diff().shift(-1)

# Сохранение измененного файла
df.to_excel("output.xlsx", index=False)
import pandas as pd

# Загрузка данных из файла CSV
file_path = r"C:\Users\ext17\Downloads\abcnews-date-text.csv"
news_df = pd.read_csv(file_path)

# Преобразование колонки 'publish_date' в тип данных datetime
news_df['publish_date'] = pd.to_datetime(news_df['publish_date'], format='%Y%m%d')

# Переименование колонки 'publish_date' в 'Date'
news_df.rename(columns={'publish_date': 'Date'}, inplace=True)

# Группировка по столбцу 'Date' и объединение текстов в столбце 'headline_text'
grouped_news_df = news_df.groupby('Date')['headline_text'].apply(' '.join).reset_index()

# Сохранение измененного файла
grouped_news_df.to_csv("grouped_abcnews.csv", index=False)
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model

# Загрузка файла сгруппированных новостей
file_path = "grouped_abcnews.csv"
grouped_news_df = pd.read_csv(file_path)

# Инициализация токенизатора и модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Функция для получения эмбеддингов
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Преобразование текстовых строк в эмбеддинги и добавление их в DataFrame
embeddings = grouped_news_df["headline_text"].apply(get_embedding)
grouped_news_df["embedding"] = embeddings

# Сохранение измененного файла
grouped_news_df.to_csv("grouped_abcnews_with_embeddings.csv", index=False)

import pandas as pd

# Чтение файла output.xlsx
output_path = r'C:\Users\ext17\output.xlsx'
output_df = pd.read_excel(output_path, engine='openpyxl')

# Чтение файла embedded_abc.csv
embedded_path = r'C:\Users\ext17\embedded_abc.csv'
embedded_df = pd.read_csv(embedded_path)

# Преобразование строки с вектором в список чисел
embedded_df['ada_vector'] = embedded_df['ada_vector'].apply(lambda x: [float(num) for num in x.strip('[]').split(', ')])

# Преобразование столбца 'Date' во втором датафрейме к типу datetime64[ns]
embedded_df['Date'] = pd.to_datetime(embedded_df['Date'])

# Объединение датафреймов на основе сходства столбца Date
merged_df = output_df.merge(embedded_df, on='Date', how='left')

print(merged_df)