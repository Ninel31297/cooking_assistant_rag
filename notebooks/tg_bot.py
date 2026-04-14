# %% [markdown]
# # Установка окружения

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
# # Загрузка данных и преобразование

# %%
df = pd.read_csv("all_recepies_inter.csv", sep="\t")
df = df.drop(columns=['Unnamed: 0', 'Дата', 'photo', 'composition_inter', 'cooking_type']) # Удаляем лишние колонки

# %%
df.columns

# %%
df.columns = ['Название блюда', 'Ингредиенты', 'Приготовление', 'Тип блюда', 'Ссылка'] # Переименовываем колонки для удобства на русский язык
# Удаление дубликатов, сохраняя только первое вхождение
df_cleaned = df.drop_duplicates()

# %%
df['Ссылка']

# %% [markdown]
# # Загрузка векторизатора и индексов векторов

# %%
# Загрузка сохранённых объектов
index = faiss.read_index("faiss_index.bin")

# Загрузка модели и датафрейма
encoder = SentenceTransformer("intfloat/multilingual-e5-base")

# %% [markdown]
# #Доступы для OpenRouter API, Telegram

# %%
# OpenRouter ключ
OPENROUTER_API_KEY = "токен"

# Настройка LLM через OpenRouter
llm = OpenRouter(
    model="deepseek/deepseek-chat-v3-0324:free",
    api_key=OPENROUTER_API_KEY
)

# Telegram token
BOT_TOKEN = "токен"

# %% [markdown]
# # Сборка бота

# %% [markdown]
# ## Стартовый интерфейс

# %%
# Словарь для хранения пользовательских сессий (по user_id)
user_sessions = {}

# Обработчик команды /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получаем идентификатор пользователя (чата)
    user_id = update.message.chat_id

    # Удаляем старые кнопки с клавиатуры (если были)
    await update.message.reply_text("👋", reply_markup=ReplyKeyboardRemove())

    # Текст приветствия, который отправит бот
    welcome_text = (
        "👨‍🍳 Привет-привет!\n\n"
        "Я — шеф Борис, твой кулинарный помощник 😺🍽️\n"
        "Напиши, что хочешь приготовить — я предложу блюда!\n\n"
        "⚙️ Если у тебя есть аллергии или ты чего-то не любишь — нажми кнопку «⚙️ Настройки».\n"
        "🔁 Можно также менять варианты или сбросить поиск 🔴."
    )

    # Отправляем изображение-приветствие (например, логотип или аватар шефа)
    with open("Welcome.png", "rb") as photo:
        await update.message.reply_photo(photo=photo)

    # Кнопка, которая появится после приветствия ("Настройки")
    buttons = [[KeyboardButton("⚙️ Настройки")]]
    await update.message.reply_text(
        welcome_text,
        reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True)  # Отображаем кнопки с автоподбором размера
    )

# %% [markdown]
# ## Логика работы и обработка сообщений пользователя

# %% [markdown]
# # Обработчик сообщений

# %%
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Главный обработчик входящих сообщений пользователя.

    Аргументы:
        update: объект обновления Telegram с информацией о сообщении.
        context: контекст обработки (не используется явно).

    Логика:
        - Получает ID пользователя и текст сообщения.
        - Загружает сессию пользователя.
        - По состоянию сессии и тексту сообщения вызывает соответствующую функцию обработки.
    """
    user_id = update.message.chat_id
    user_input = update.message.text.strip()
    session = user_sessions.get(user_id, {})

    # Проверка: ожидает ли пользователь подтверждения сброса
    if session.get("awaiting_reset_confirmation"):
        return await handle_reset_confirmation(update, user_id, user_input)

    # Проверка: пользователь хочет открыть настройки исключений
    if user_input == "⚙️ Настройки":
        return await handle_enter_settings(update, user_id, session)

    # Проверка: пользователь вводит список исключённых ингредиентов
    if session.get("awaiting_exclusions"):
        return await handle_exclusions_input(update, user_id, user_input, session)

    # Проверка: пользователь запросил сброс поиска и настроек
    if user_input == "🔴 Сбросить поиск":
        return await handle_reset_request(update, user_id, session)

    # Проверка: пользователь выбрал одно из предложенных блюд
    if user_input in session.get("options", {}):
        return await handle_recipe_selection(update, user_input, session)

    # Иначе — новый запрос или запрос "хочу что-то другое"
    return await handle_new_query(update, user_input, session, user_id)

# %% [markdown]
# ## Обработчик предпочтений

# %%
async def handle_enter_settings(update, user_id, session):
    """
    Инициирует режим ввода исключённых ингредиентов (например аллергий, нелюбимых продуктов).

    Отправляет сообщение с просьбой указать ингредиенты,
    удаляет клавиатуру для удобства ввода текста.

    В сессии выставляет флаг 'awaiting_exclusions' = True,
    чтобы следующий ввод был интерпретирован как список исключений.
    """
    await update.message.reply_text(
        "✍️ Напиши, какие ингредиенты тебе нельзя или не нравятся (например: «аллергия на орехи, не люблю грибы»):",
        reply_markup=ReplyKeyboardRemove()
    )
    session["awaiting_exclusions"] = True
    user_sessions[user_id] = session


async def handle_exclusions_input(update, user_id, user_input, session):
    """
    Обрабатывает пользовательский ввод исключённых ингредиентов.

    Парсит строку, удаляя слова "аллергия на" и "не люблю",
    разделяет по запятым и очищает от пробелов.

    Сохраняет список исключённых ингредиентов в сессию,
    снимает флаг ожидания ввода исключений.

    Отправляет подтверждение и приглашение к следующему запросу.
    """
    excluded = [
        e.strip()
        for e in user_input.lower()
        .replace("аллергия на", "")
        .replace("не люблю", "")
        .split(",")
        if e.strip()
    ]
    session["excluded_ingredients"] = excluded
    session["awaiting_exclusions"] = False
    user_sessions[user_id] = session

    await update.message.reply_text(
        f"👌 Понял! Буду избегать: {', '.join(excluded)}.\n\nТеперь напиши, что хочешь приготовить!",
        reply_markup=ReplyKeyboardMarkup([], resize_keyboard=True)
    )

# %% [markdown]
# ## Сброс настроек

# %%
async def handle_reset_request(update, user_id, session):
    """
    Обрабатывает запрос пользователя на сброс всех настроек и текущего поиска.

    Устанавливает флаг 'awaiting_reset_confirmation' в сессии,
    чтобы следующий ввод ожидался как подтверждение.

    Отправляет предупреждающее сообщение с инструкцией.
    """
    session["awaiting_reset_confirmation"] = True
    user_sessions[user_id] = session
    await update.message.reply_text(
        "⚠️ Ты собираешься сбросить все настройки и поиск. Напиши: «Подтверждаю сброс».",
        reply_markup=ReplyKeyboardMarkup([[KeyboardButton("⚙️ Настройки")]], resize_keyboard=True)
    )

# %%
async def handle_reset_confirmation(update, user_id, user_input):
    """
    Обрабатывает подтверждение или отмену сброса пользовательских настроек.

    Если пользователь написал "подтверждаю сброс", очищает сессию,
    отправляет уведомление об успешном сбросе.

    Иначе отправляет сообщение об отмене сброса и предлагает продолжить работу.
    """
    if user_input.lower() == "подтверждаю сброс":
        # Удаляем сессию пользователя, тем самым сбрасываем все настройки
        user_sessions.pop(user_id, None)
        await update.message.reply_text(
            "✅ Поиск и настройки сброшены! Напиши, что хочешь приготовить.",
            reply_markup=ReplyKeyboardMarkup([[KeyboardButton("⚙️ Настройки")]], resize_keyboard=True)
        )
    else:
        # Пользователь отменил сброс — продолжаем работу без изменений
        await update.message.reply_text(
            "❌ Сброс отменён. Продолжай вводить запрос или нажми «🔴 Сбросить поиск».",
            reply_markup=ReplyKeyboardMarkup([[KeyboardButton("⚙️ Настройки")]], resize_keyboard=True)
        )

# %% [markdown]
# ## Генератор рецепта приготовления

# %%
import html  # Для экранирования ссылки

async def handle_recipe_selection(update, user_input, session):
    """
    Обрабатывает выбор блюда пользователем.

    🔸 Находит рецепт по названию (user_input) в df.
    🔸 Учитывает исключённые ингредиенты.
    🔸 Запрашивает у LLM пошаговый рецепт.
    🔸 Добавляет HTML-ссылку в конце.
    🔸 Отправляет сообщение в чат, разбивая при необходимости.
    """
    # Ищем строку в df по названию блюда
    row_match = df[df['Название блюда'].str.strip().str.lower() == user_input.strip().lower()]
    if row_match.empty:
        await update.message.reply_text("❌ Не удалось найти рецепт в базе данных.")
        return

    row = row_match.iloc[0]
    excluded = session.get("excluded_ingredients", [])
    allergy_note = f"⚠️ У пользователя аллергия: {', '.join(excluded)}." if excluded else ""

    # Получаем ссылку из колонки "Ссылка" и экранируем
    recipe_link = row.get("Ссылка", "").strip()
    if recipe_link:
        safe_link = html.escape(recipe_link)
        html_link = f'\n\n🔗 <a href="{safe_link}">Смотреть рецепт на сайте</a>'
    else:
        html_link = "\n\n❗ Ссылка на оригинальный рецепт недоступна."

    # Промпт для генерации рецепта
    prompt = (
        f"Ты — кулинарный помощник. Пользователь выбрал: {row['Название блюда']}.\n"
        f"Ингредиенты: {row['Ингредиенты']}\n"
        f"Инструкция: {row['Приготовление']}\n"
        f"{allergy_note}\n"
        f"Сформулируй необходимые ингредиенты и краткой рецепт с эмоджи на русском языке."
    )

    try:
        messages = [
            ChatMessage(role="system", content="Ты — вдохновляющий кулинарный шеф."),
            ChatMessage(role="user", content=prompt),
        ]
        response = llm.chat(messages)
        full_reply = response.message.content.strip()
    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка генерации рецепта: {e}")
        return

    # Добавляем ссылку
    full_reply += html_link

    # Отправка ответа по частям (ограничение Telegram — 4096 символов)
    chunk_size = 4000
    for i in range(0, len(full_reply), chunk_size):
        chunk = full_reply[i:i + chunk_size]
        await update.message.reply_text(chunk, parse_mode="HTML")

# %% [markdown]
# ## Обработчик новых запросов

# %%
async def handle_new_query(update, user_input, session, user_id):
    """
    Обрабатывает новый запрос на рецепт или запрос "хочу что-то другое".

    Формирует embedding запроса, ищет подходящие блюда в FAISS индексе,
    фильтрует блюда с учётом исключённых ингредиентов.

    Для каждого найденного блюда генерирует короткое описание через LLM.

    Ограничивает количество вариантов тремя.

    Обновляет сессию: запоминает запрос, использованные индексы, новые варианты.

    Отправляет пользователю список блюд с описаниями и кнопками для выбора,
    а также служебные кнопки "хочу что-то другое", "сбросить поиск" и "настройки".

    Если вариантов нет — уведомляет пользователя.
    """
    # Если пользователь хочет "что-то другое", используем предыдущий запрос и уже использованные индексы
    if user_input == "🔁 Хочу что-то другое" and "query" in session:
        query = session["query"]
        used_indices = session.get("used_indices", set())
    else:
        query = user_input
        used_indices = set()

    # Генерируем эмбеддинг запроса и ищем ближайшие рецепты
    query_embedding = encoder.encode([f"query: {query}"])
    D, I = index.search(np.array(query_embedding), k=20)

    excluded = session.get("excluded_ingredients", [])
    new_options = {}
    buttons = []
    descriptions = []
    count = 0

    for idx in I[0]:
        # Пропускаем блюда, уже предложенные ранее
        if idx in used_indices:
            continue
        row = df.iloc[idx]
        # Пропускаем блюда, содержащие исключённые ингредиенты
        if any(ex in row["Ингредиенты"].lower() for ex in excluded):
            continue

        title = row["Название блюда"]
        try:
            prompt = (
                f"Ты — шеф-повар. Пользователь интересуется: {query}.\n"
                f"Название блюда: {title}\n"
                f"Ингредиенты: {row['Ингредиенты']}\n"
                f"Приготовление: {row['Приготовление']}\n\n"
                "Сформулируй короткое, аппетитное, дружелюбное описание этого блюда на русском языке (1-2 предложения, не повторяй название)."
            )
            messages = [
                ChatMessage(role="system", content="Ты — вдохновляющий кулинарный шеф."),
                ChatMessage(role="user", content=prompt),
            ]
            response = llm.chat(messages, max_tokens=100)
            short_desc = response.message.content.strip()
        except Exception:
            short_desc = "Описание временно недоступно."

        new_options[title] = idx
        buttons.append([KeyboardButton(title)])
        descriptions.append(f"🍽️ *{title}*\n_{short_desc}_")
        used_indices.add(idx)
        count += 1
        if count == 3:
            break

    # Если нет ни одного подходящего варианта
    if not new_options:
        await update.message.reply_text("😔 Больше нет подходящих вариантов. Попробуй другой запрос или сбрось поиск.")
        return

    # Добавляем нижние служебные кнопки
    buttons += [
        [KeyboardButton("🔁 Хочу что-то другое")],
        [KeyboardButton("🔴 Сбросить поиск")],
        [KeyboardButton("⚙️ Настройки")],
    ]

    # Обновляем сессию пользователя
    user_sessions[user_id] = {
        "query": query,
        "used_indices": used_indices,
        "options": new_options,
        "excluded_ingredients": excluded,
    }

    # Отправляем пользователю список с описаниями блюд
    await update.message.reply_text(
        "Вот что могу предложить 👇\n\n" + "\n\n".join(descriptions),
        parse_mode="Markdown",
    )
    # Отправляем клавиатуру с кнопками для выбора
    await update.message.reply_text(
        "Выбери блюдо или нажми «🔁 Хочу что-то другое», «🔴 Сбросить поиск» или «⚙️ Настройки»:",
        reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True),
    )

# %% [markdown]
# # Запуск

# %%
# 🔁 Основная функция запуска
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен!")
    await app.run_polling()

# 👟 Запуск (для Jupyter/Colab)
nest_asyncio.apply()
await main()


