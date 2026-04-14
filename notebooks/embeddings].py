# %% [markdown]
# # Установка необходимых библиотек

# %%
!pip install -r requirements.txt

# %%
import pandas as pd
import numpy as np
import kagglehub
import os
import pickle
import faiss
import asyncio
import nest_asyncio
from sentence_transformers import SentenceTransformer
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import (ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters)

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage

# %% [markdown]
# # Знакомство с данными

# %%
# Скачиваем датасет с KaggleHub
path = kagglehub.dataset_download("coolonce/recipes-and-interpretation-dim")

# Собираем путь к нужному файлу
file_path = os.path.join(path, 'all_recepies_inter.csv')

# Загружаем CSV-файл
df = pd.read_csv(file_path, sep='\t')
print(df.head())

# %%
df.info() # Проверяем наличие пропусков и знакомимся с полями и типами данных

# %% [markdown]
# # Предобработка

# %%
df = df.drop(columns=['Unnamed: 0', 'Дата', 'source', 'composition_inter', 'cooking_type']) # Удаляем лишние колонки

# %%
df.columns

# %%
df.columns = ['Название блюда', 'Ингредиенты', 'Приготовление', 'Тип блюда', 'Фото'] # Переименовываем колонки для удобства на русский язык

# %%
# Удаление дубликатов, сохраняя только первое вхождение
df_cleaned = df.drop_duplicates()

# %% [markdown]
# Вывод: датасет из 27 тысяч рецептов, содержащий в себе название, ингридиенты, процесс приготовления, фото и ссылку на рецепт. Для дальнейшей обработки оставляем только ключевые поля, содержащие важную информацию и убираем дубликаты.

# %% [markdown]
# # Векторизация и индексация

# %%
# Загружаем энкодер для преобразования текста в эмбеддинги
encoder = SentenceTransformer("intfloat/multilingual-e5-base")

# Формируем тексты из DataFrame: название, ингредиенты, шаги, тип блюда
texts = (
    'Название блюда: ' + df['Название блюда'] +
    '. Ингредиенты: ' + df['Ингредиенты'] +
    '. Приготовление: ' + df['Приготовление'] +
    '. Тип блюда: ' + df['Тип блюда']
).tolist()

# Добавляем префикс "passage: " — формат, ожидаемый моделью E5
texts_for_index = ["passage: " + t for t in texts]

# Преобразуем тексты в эмбеддинги с помощью модели
embeddings = encoder.encode(texts_for_index, show_progress_bar=True, batch_size=32)

# Определяем размерность эмбеддингов
dim = embeddings[0].shape[0]

# Создаём FAISS-индекс для поиска по L2 (евклидовому) расстоянию
index = faiss.IndexFlatL2(dim)

# Добавляем эмбеддинги в индекс
index.add(np.array(embeddings))

# Сохраняем FAISS-индекс в бинарный файл
faiss.write_index(index, "faiss_index.bin")

# Сохраняем тексты, к которым относятся эмбеддинги (например, для вывода найденных рецептов)
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Индекс и тексты успешно сохранены.")

# %% [markdown]
# Модель эмбеддингов: intfloat/multilingual-e5-base
# * Оптимизирована под Retrieval (поиск): обучена на задаче поиска (retrieval) с парами query и passage. Это значительно повышает точность поиска.
# * Поддерживает русский язык.
# * Легко запускается на локальной машине.
# * Метод индексирования: FAISS.IndexFlatL2
# 
# FAISS — стандарт в задачах семантического поиска, совместим с NumPy и легко расширяется.
# * IndexFlatL2 — точный индекс (exhaustive search), не использует приближённые методы, а проводит полный перебор по всем векторам. Подходит, если база рецептов относительно небольшая.
# * Не требует сложной настройки, можно сразу использовать и легко переключиться на ускоренные варианты  при необходимости.
# 


