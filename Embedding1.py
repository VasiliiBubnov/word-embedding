import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

model_name = 'gpt2'
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(model_name)
model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

def get_word_embeddings(word, model, tokenizer):
    input_ids = tokenizer.encode(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs[0][0][-1].numpy()

vocab = list(tokenizer_gpt2.get_vocab().keys())  # Список всех токенов в словаре GPT-2

# Фильтрация словаря и удаление специальных токенов
filtered_vocab = [word for word in vocab if not word in tokenizer_gpt2.all_special_tokens]

# Создание прогресс-бара
progress_bar = tqdm(total=len(filtered_vocab), desc="Вычисление векторов слов")

# Вычисление векторов слов с использованием прогресс-бара
word_vectors = []
for word in filtered_vocab:
    word_vectors.append(get_word_embeddings(word, model_gpt2, tokenizer_gpt2))
    progress_bar.update(1)

# Закрытие прогресс-бара
progress_bar.close()

# Остальной код
your_model_directory = './model_bn_custom/'
tokenizer_custom = GPT2Tokenizer.from_pretrained(your_model_directory)
model_custom = GPT2LMHeadModel.from_pretrained(your_model_directory)

digits = [str(i) for i in range(2000)]

def get_digit_embeddings(digit, model, tokenizer):
    input_ids = tokenizer.encode(digit, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs[0][0][-1].numpy()

digit_vectors = [get_digit_embeddings(digit, model_custom, tokenizer_custom) for digit in digits]

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(digit_vectors, word_vectors)

import numpy as np

closest_words_indices = [np.argmax(similarity_row) for similarity_row in similarity_matrix]
closest_words = [vocab[index] for index in closest_words_indices]
print(closest_words)

import pandas as pd

# Создайте DataFrame с данными
closest_words_df = pd.DataFrame({'closest_words': closest_words})

# Сохраните DataFrame в файл CSV
closest_words_df.to_csv('closest_words.csv', index=False)
