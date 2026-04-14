{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Установка окружения"
      ],
      "metadata": {
        "id": "gh08WTkFm82E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -r requirements.txt"
      ],
      "metadata": {
        "id": "R_Nttou1nH3J"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import kagglehub\n",
        "import os\n",
        "import pickle\n",
        "import faiss\n",
        "import asyncio\n",
        "import nest_asyncio\n",
        "from sentence_transformers import SentenceTransformer\n",
        "from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove\n",
        "from telegram.ext import (ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters)\n",
        "\n",
        "from llama_index.llms.openrouter import OpenRouter\n",
        "from llama_index.core.llms import ChatMessage"
      ],
      "metadata": {
        "id": "G0GSShrSnEqD"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Загрузка данных и преобразование"
      ],
      "metadata": {
        "id": "3BrlMtrTtJNM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"all_recepies_inter.csv\", sep=\"\\t\")\n",
        "df = df.drop(columns=['Unnamed: 0', 'Дата', 'photo', 'composition_inter', 'cooking_type']) # Удаляем лишние колонки"
      ],
      "metadata": {
        "id": "Bz0ZP-6XIpMF"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DKStKuI7Iqnb",
        "outputId": "ce2ec110-0afa-4498-e084-f0e7d2fe61f7"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['name', 'composition', 'Инструкции', 'dish_type', 'source'], dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns = ['Название блюда', 'Ингредиенты', 'Приготовление', 'Тип блюда', 'Ссылка'] # Переименовываем колонки для удобства на русский язык\n",
        "# Удаление дубликатов, сохраняя только первое вхождение\n",
        "df_cleaned = df.drop_duplicates()"
      ],
      "metadata": {
        "id": "fwLxDTW_tNav"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df['Ссылка']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 458
        },
        "id": "vUo1S7WleFt9",
        "outputId": "8e312fec-9846-4f2e-d537-694bc5950c1d"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0        https://1000.menu/cooking/33395-rassolnik-s-pe...\n",
              "1        https://1000.menu/cooking/25399-sup-pure-iz-be...\n",
              "2             https://1000.menu/cooking/5159-postnje-shchi\n",
              "3                     https://1000.menu/cooking/5085-turya\n",
              "4        https://1000.menu/cooking/38765-fasolevyi-sup-...\n",
              "                               ...                        \n",
              "27879    https://eda.ru/recepty/salaty/salat-s-file-tun...\n",
              "27880    https://eda.ru/recepty/salaty/teplyy-salat-s-k...\n",
              "27881    https://eda.ru/recepty/salaty/samyy-zelenyy-sa...\n",
              "27882    https://eda.ru/recepty/salaty/teplyy-salat-s-t...\n",
              "27883    https://eda.ru/recepty/salaty/salat-iz-rakovyh...\n",
              "Name: Ссылка, Length: 27884, dtype: object"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Ссылка</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>https://1000.menu/cooking/33395-rassolnik-s-pe...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>https://1000.menu/cooking/25399-sup-pure-iz-be...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>https://1000.menu/cooking/5159-postnje-shchi</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>https://1000.menu/cooking/5085-turya</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>https://1000.menu/cooking/38765-fasolevyi-sup-...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27879</th>\n",
              "      <td>https://eda.ru/recepty/salaty/salat-s-file-tun...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27880</th>\n",
              "      <td>https://eda.ru/recepty/salaty/teplyy-salat-s-k...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27881</th>\n",
              "      <td>https://eda.ru/recepty/salaty/samyy-zelenyy-sa...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27882</th>\n",
              "      <td>https://eda.ru/recepty/salaty/teplyy-salat-s-t...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27883</th>\n",
              "      <td>https://eda.ru/recepty/salaty/salat-iz-rakovyh...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>27884 rows × 1 columns</p>\n",
              "</div><br><label><b>dtype:</b> object</label>"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Загрузка векторизатора и индексов векторов"
      ],
      "metadata": {
        "id": "oPQ4Ld2vpgrA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Загрузка сохранённых объектов\n",
        "index = faiss.read_index(\"faiss_index.bin\")\n",
        "\n",
        "# Загрузка модели и датафрейма\n",
        "encoder = SentenceTransformer(\"intfloat/multilingual-e5-base\")"
      ],
      "metadata": {
        "id": "GmTVeopCphEd"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Доступы для OpenRouter API, Telegram"
      ],
      "metadata": {
        "id": "eEwVp2GHm2DB"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "lg7ExxFsmwAi"
      },
      "outputs": [],
      "source": [
        "# OpenRouter ключ\n",
        "OPENROUTER_API_KEY = \"токен\"\n",
        "\n",
        "# Настройка LLM через OpenRouter\n",
        "llm = OpenRouter(\n",
        "    model=\"deepseek/deepseek-chat-v3-0324:free\",\n",
        "    api_key=OPENROUTER_API_KEY\n",
        ")\n",
        "\n",
        "# Telegram token\n",
        "BOT_TOKEN = \"токен\""
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Сборка бота"
      ],
      "metadata": {
        "id": "FOZ_XNw1neO7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Стартовый интерфейс"
      ],
      "metadata": {
        "id": "XLMqNgQCnkla"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Словарь для хранения пользовательских сессий (по user_id)\n",
        "user_sessions = {}\n",
        "\n",
        "# Обработчик команды /start\n",
        "async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):\n",
        "    # Получаем идентификатор пользователя (чата)\n",
        "    user_id = update.message.chat_id\n",
        "\n",
        "    # Удаляем старые кнопки с клавиатуры (если были)\n",
        "    await update.message.reply_text(\"👋\", reply_markup=ReplyKeyboardRemove())\n",
        "\n",
        "    # Текст приветствия, который отправит бот\n",
        "    welcome_text = (\n",
        "        \"👨‍🍳 Привет-привет!\\n\\n\"\n",
        "        \"Я — шеф Борис, твой кулинарный помощник 😺🍽️\\n\"\n",
        "        \"Напиши, что хочешь приготовить — я предложу блюда!\\n\\n\"\n",
        "        \"⚙️ Если у тебя есть аллергии или ты чего-то не любишь — нажми кнопку «⚙️ Настройки».\\n\"\n",
        "        \"🔁 Можно также менять варианты или сбросить поиск 🔴.\"\n",
        "    )\n",
        "\n",
        "    # Отправляем изображение-приветствие (например, логотип или аватар шефа)\n",
        "    with open(\"Welcome.png\", \"rb\") as photo:\n",
        "        await update.message.reply_photo(photo=photo)\n",
        "\n",
        "    # Кнопка, которая появится после приветствия (\"Настройки\")\n",
        "    buttons = [[KeyboardButton(\"⚙️ Настройки\")]]\n",
        "    await update.message.reply_text(\n",
        "        welcome_text,\n",
        "        reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True)  # Отображаем кнопки с автоподбором размера\n",
        "    )"
      ],
      "metadata": {
        "id": "xWqYjjNqnhFy"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Логика работы и обработка сообщений пользователя"
      ],
      "metadata": {
        "id": "vtoqc-WMnqmT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Обработчик сообщений"
      ],
      "metadata": {
        "id": "tJkt4jAAoAoc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):\n",
        "    \"\"\"\n",
        "    Главный обработчик входящих сообщений пользователя.\n",
        "\n",
        "    Аргументы:\n",
        "        update: объект обновления Telegram с информацией о сообщении.\n",
        "        context: контекст обработки (не используется явно).\n",
        "\n",
        "    Логика:\n",
        "        - Получает ID пользователя и текст сообщения.\n",
        "        - Загружает сессию пользователя.\n",
        "        - По состоянию сессии и тексту сообщения вызывает соответствующую функцию обработки.\n",
        "    \"\"\"\n",
        "    user_id = update.message.chat_id\n",
        "    user_input = update.message.text.strip()\n",
        "    session = user_sessions.get(user_id, {})\n",
        "\n",
        "    # Проверка: ожидает ли пользователь подтверждения сброса\n",
        "    if session.get(\"awaiting_reset_confirmation\"):\n",
        "        return await handle_reset_confirmation(update, user_id, user_input)\n",
        "\n",
        "    # Проверка: пользователь хочет открыть настройки исключений\n",
        "    if user_input == \"⚙️ Настройки\":\n",
        "        return await handle_enter_settings(update, user_id, session)\n",
        "\n",
        "    # Проверка: пользователь вводит список исключённых ингредиентов\n",
        "    if session.get(\"awaiting_exclusions\"):\n",
        "        return await handle_exclusions_input(update, user_id, user_input, session)\n",
        "\n",
        "    # Проверка: пользователь запросил сброс поиска и настроек\n",
        "    if user_input == \"🔴 Сбросить поиск\":\n",
        "        return await handle_reset_request(update, user_id, session)\n",
        "\n",
        "    # Проверка: пользователь выбрал одно из предложенных блюд\n",
        "    if user_input in session.get(\"options\", {}):\n",
        "        return await handle_recipe_selection(update, user_input, session)\n",
        "\n",
        "    # Иначе — новый запрос или запрос \"хочу что-то другое\"\n",
        "    return await handle_new_query(update, user_input, session, user_id)"
      ],
      "metadata": {
        "id": "a_zRxwEu4KuJ"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Обработчик предпочтений"
      ],
      "metadata": {
        "id": "0jCeg_sk5SGH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "async def handle_enter_settings(update, user_id, session):\n",
        "    \"\"\"\n",
        "    Инициирует режим ввода исключённых ингредиентов (например аллергий, нелюбимых продуктов).\n",
        "\n",
        "    Отправляет сообщение с просьбой указать ингредиенты,\n",
        "    удаляет клавиатуру для удобства ввода текста.\n",
        "\n",
        "    В сессии выставляет флаг 'awaiting_exclusions' = True,\n",
        "    чтобы следующий ввод был интерпретирован как список исключений.\n",
        "    \"\"\"\n",
        "    await update.message.reply_text(\n",
        "        \"✍️ Напиши, какие ингредиенты тебе нельзя или не нравятся (например: «аллергия на орехи, не люблю грибы»):\",\n",
        "        reply_markup=ReplyKeyboardRemove()\n",
        "    )\n",
        "    session[\"awaiting_exclusions\"] = True\n",
        "    user_sessions[user_id] = session\n",
        "\n",
        "\n",
        "async def handle_exclusions_input(update, user_id, user_input, session):\n",
        "    \"\"\"\n",
        "    Обрабатывает пользовательский ввод исключённых ингредиентов.\n",
        "\n",
        "    Парсит строку, удаляя слова \"аллергия на\" и \"не люблю\",\n",
        "    разделяет по запятым и очищает от пробелов.\n",
        "\n",
        "    Сохраняет список исключённых ингредиентов в сессию,\n",
        "    снимает флаг ожидания ввода исключений.\n",
        "\n",
        "    Отправляет подтверждение и приглашение к следующему запросу.\n",
        "    \"\"\"\n",
        "    excluded = [\n",
        "        e.strip()\n",
        "        for e in user_input.lower()\n",
        "        .replace(\"аллергия на\", \"\")\n",
        "        .replace(\"не люблю\", \"\")\n",
        "        .split(\",\")\n",
        "        if e.strip()\n",
        "    ]\n",
        "    session[\"excluded_ingredients\"] = excluded\n",
        "    session[\"awaiting_exclusions\"] = False\n",
        "    user_sessions[user_id] = session\n",
        "\n",
        "    await update.message.reply_text(\n",
        "        f\"👌 Понял! Буду избегать: {', '.join(excluded)}.\\n\\nТеперь напиши, что хочешь приготовить!\",\n",
        "        reply_markup=ReplyKeyboardMarkup([], resize_keyboard=True)\n",
        "    )"
      ],
      "metadata": {
        "id": "SZPkwAfU5NUX"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Сброс настроек"
      ],
      "metadata": {
        "id": "2zjtY5j24__X"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "async def handle_reset_request(update, user_id, session):\n",
        "    \"\"\"\n",
        "    Обрабатывает запрос пользователя на сброс всех настроек и текущего поиска.\n",
        "\n",
        "    Устанавливает флаг 'awaiting_reset_confirmation' в сессии,\n",
        "    чтобы следующий ввод ожидался как подтверждение.\n",
        "\n",
        "    Отправляет предупреждающее сообщение с инструкцией.\n",
        "    \"\"\"\n",
        "    session[\"awaiting_reset_confirmation\"] = True\n",
        "    user_sessions[user_id] = session\n",
        "    await update.message.reply_text(\n",
        "        \"⚠️ Ты собираешься сбросить все настройки и поиск. Напиши: «Подтверждаю сброс».\",\n",
        "        reply_markup=ReplyKeyboardMarkup([[KeyboardButton(\"⚙️ Настройки\")]], resize_keyboard=True)\n",
        "    )"
      ],
      "metadata": {
        "id": "35woKLg55AtD"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "async def handle_reset_confirmation(update, user_id, user_input):\n",
        "    \"\"\"\n",
        "    Обрабатывает подтверждение или отмену сброса пользовательских настроек.\n",
        "\n",
        "    Если пользователь написал \"подтверждаю сброс\", очищает сессию,\n",
        "    отправляет уведомление об успешном сбросе.\n",
        "\n",
        "    Иначе отправляет сообщение об отмене сброса и предлагает продолжить работу.\n",
        "    \"\"\"\n",
        "    if user_input.lower() == \"подтверждаю сброс\":\n",
        "        # Удаляем сессию пользователя, тем самым сбрасываем все настройки\n",
        "        user_sessions.pop(user_id, None)\n",
        "        await update.message.reply_text(\n",
        "            \"✅ Поиск и настройки сброшены! Напиши, что хочешь приготовить.\",\n",
        "            reply_markup=ReplyKeyboardMarkup([[KeyboardButton(\"⚙️ Настройки\")]], resize_keyboard=True)\n",
        "        )\n",
        "    else:\n",
        "        # Пользователь отменил сброс — продолжаем работу без изменений\n",
        "        await update.message.reply_text(\n",
        "            \"❌ Сброс отменён. Продолжай вводить запрос или нажми «🔴 Сбросить поиск».\",\n",
        "            reply_markup=ReplyKeyboardMarkup([[KeyboardButton(\"⚙️ Настройки\")]], resize_keyboard=True)\n",
        "        )"
      ],
      "metadata": {
        "id": "g1aVXncz5WgY"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Генератор рецепта приготовления"
      ],
      "metadata": {
        "id": "HdZx3xeJ45tb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import html  # Для экранирования ссылки\n",
        "\n",
        "async def handle_recipe_selection(update, user_input, session):\n",
        "    \"\"\"\n",
        "    Обрабатывает выбор блюда пользователем.\n",
        "\n",
        "    🔸 Находит рецепт по названию (user_input) в df.\n",
        "    🔸 Учитывает исключённые ингредиенты.\n",
        "    🔸 Запрашивает у LLM пошаговый рецепт.\n",
        "    🔸 Добавляет HTML-ссылку в конце.\n",
        "    🔸 Отправляет сообщение в чат, разбивая при необходимости.\n",
        "    \"\"\"\n",
        "    # Ищем строку в df по названию блюда\n",
        "    row_match = df[df['Название блюда'].str.strip().str.lower() == user_input.strip().lower()]\n",
        "    if row_match.empty:\n",
        "        await update.message.reply_text(\"❌ Не удалось найти рецепт в базе данных.\")\n",
        "        return\n",
        "\n",
        "    row = row_match.iloc[0]\n",
        "    excluded = session.get(\"excluded_ingredients\", [])\n",
        "    allergy_note = f\"⚠️ У пользователя аллергия: {', '.join(excluded)}.\" if excluded else \"\"\n",
        "\n",
        "    # Получаем ссылку из колонки \"Ссылка\" и экранируем\n",
        "    recipe_link = row.get(\"Ссылка\", \"\").strip()\n",
        "    if recipe_link:\n",
        "        safe_link = html.escape(recipe_link)\n",
        "        html_link = f'\\n\\n🔗 <a href=\"{safe_link}\">Смотреть рецепт на сайте</a>'\n",
        "    else:\n",
        "        html_link = \"\\n\\n❗ Ссылка на оригинальный рецепт недоступна.\"\n",
        "\n",
        "    # Промпт для генерации рецепта\n",
        "    prompt = (\n",
        "        f\"Ты — кулинарный помощник. Пользователь выбрал: {row['Название блюда']}.\\n\"\n",
        "        f\"Ингредиенты: {row['Ингредиенты']}\\n\"\n",
        "        f\"Инструкция: {row['Приготовление']}\\n\"\n",
        "        f\"{allergy_note}\\n\"\n",
        "        f\"Сформулируй необходимые ингредиенты и краткой рецепт с эмоджи на русском языке.\"\n",
        "    )\n",
        "\n",
        "    try:\n",
        "        messages = [\n",
        "            ChatMessage(role=\"system\", content=\"Ты — вдохновляющий кулинарный шеф.\"),\n",
        "            ChatMessage(role=\"user\", content=prompt),\n",
        "        ]\n",
        "        response = llm.chat(messages)\n",
        "        full_reply = response.message.content.strip()\n",
        "    except Exception as e:\n",
        "        await update.message.reply_text(f\"⚠️ Ошибка генерации рецепта: {e}\")\n",
        "        return\n",
        "\n",
        "    # Добавляем ссылку\n",
        "    full_reply += html_link\n",
        "\n",
        "    # Отправка ответа по частям (ограничение Telegram — 4096 символов)\n",
        "    chunk_size = 4000\n",
        "    for i in range(0, len(full_reply), chunk_size):\n",
        "        chunk = full_reply[i:i + chunk_size]\n",
        "        await update.message.reply_text(chunk, parse_mode=\"HTML\")"
      ],
      "metadata": {
        "id": "X4pqwKZs44lt"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Обработчик новых запросов"
      ],
      "metadata": {
        "id": "_RPVQ0p64tld"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "async def handle_new_query(update, user_input, session, user_id):\n",
        "    \"\"\"\n",
        "    Обрабатывает новый запрос на рецепт или запрос \"хочу что-то другое\".\n",
        "\n",
        "    Формирует embedding запроса, ищет подходящие блюда в FAISS индексе,\n",
        "    фильтрует блюда с учётом исключённых ингредиентов.\n",
        "\n",
        "    Для каждого найденного блюда генерирует короткое описание через LLM.\n",
        "\n",
        "    Ограничивает количество вариантов тремя.\n",
        "\n",
        "    Обновляет сессию: запоминает запрос, использованные индексы, новые варианты.\n",
        "\n",
        "    Отправляет пользователю список блюд с описаниями и кнопками для выбора,\n",
        "    а также служебные кнопки \"хочу что-то другое\", \"сбросить поиск\" и \"настройки\".\n",
        "\n",
        "    Если вариантов нет — уведомляет пользователя.\n",
        "    \"\"\"\n",
        "    # Если пользователь хочет \"что-то другое\", используем предыдущий запрос и уже использованные индексы\n",
        "    if user_input == \"🔁 Хочу что-то другое\" and \"query\" in session:\n",
        "        query = session[\"query\"]\n",
        "        used_indices = session.get(\"used_indices\", set())\n",
        "    else:\n",
        "        query = user_input\n",
        "        used_indices = set()\n",
        "\n",
        "    # Генерируем эмбеддинг запроса и ищем ближайшие рецепты\n",
        "    query_embedding = encoder.encode([f\"query: {query}\"])\n",
        "    D, I = index.search(np.array(query_embedding), k=20)\n",
        "\n",
        "    excluded = session.get(\"excluded_ingredients\", [])\n",
        "    new_options = {}\n",
        "    buttons = []\n",
        "    descriptions = []\n",
        "    count = 0\n",
        "\n",
        "    for idx in I[0]:\n",
        "        # Пропускаем блюда, уже предложенные ранее\n",
        "        if idx in used_indices:\n",
        "            continue\n",
        "        row = df.iloc[idx]\n",
        "        # Пропускаем блюда, содержащие исключённые ингредиенты\n",
        "        if any(ex in row[\"Ингредиенты\"].lower() for ex in excluded):\n",
        "            continue\n",
        "\n",
        "        title = row[\"Название блюда\"]\n",
        "        try:\n",
        "            prompt = (\n",
        "                f\"Ты — шеф-повар. Пользователь интересуется: {query}.\\n\"\n",
        "                f\"Название блюда: {title}\\n\"\n",
        "                f\"Ингредиенты: {row['Ингредиенты']}\\n\"\n",
        "                f\"Приготовление: {row['Приготовление']}\\n\\n\"\n",
        "                \"Сформулируй короткое, аппетитное, дружелюбное описание этого блюда на русском языке (1-2 предложения, не повторяй название).\"\n",
        "            )\n",
        "            messages = [\n",
        "                ChatMessage(role=\"system\", content=\"Ты — вдохновляющий кулинарный шеф.\"),\n",
        "                ChatMessage(role=\"user\", content=prompt),\n",
        "            ]\n",
        "            response = llm.chat(messages, max_tokens=100)\n",
        "            short_desc = response.message.content.strip()\n",
        "        except Exception:\n",
        "            short_desc = \"Описание временно недоступно.\"\n",
        "\n",
        "        new_options[title] = idx\n",
        "        buttons.append([KeyboardButton(title)])\n",
        "        descriptions.append(f\"🍽️ *{title}*\\n_{short_desc}_\")\n",
        "        used_indices.add(idx)\n",
        "        count += 1\n",
        "        if count == 3:\n",
        "            break\n",
        "\n",
        "    # Если нет ни одного подходящего варианта\n",
        "    if not new_options:\n",
        "        await update.message.reply_text(\"😔 Больше нет подходящих вариантов. Попробуй другой запрос или сбрось поиск.\")\n",
        "        return\n",
        "\n",
        "    # Добавляем нижние служебные кнопки\n",
        "    buttons += [\n",
        "        [KeyboardButton(\"🔁 Хочу что-то другое\")],\n",
        "        [KeyboardButton(\"🔴 Сбросить поиск\")],\n",
        "        [KeyboardButton(\"⚙️ Настройки\")],\n",
        "    ]\n",
        "\n",
        "    # Обновляем сессию пользователя\n",
        "    user_sessions[user_id] = {\n",
        "        \"query\": query,\n",
        "        \"used_indices\": used_indices,\n",
        "        \"options\": new_options,\n",
        "        \"excluded_ingredients\": excluded,\n",
        "    }\n",
        "\n",
        "    # Отправляем пользователю список с описаниями блюд\n",
        "    await update.message.reply_text(\n",
        "        \"Вот что могу предложить 👇\\n\\n\" + \"\\n\\n\".join(descriptions),\n",
        "        parse_mode=\"Markdown\",\n",
        "    )\n",
        "    # Отправляем клавиатуру с кнопками для выбора\n",
        "    await update.message.reply_text(\n",
        "        \"Выбери блюдо или нажми «🔁 Хочу что-то другое», «🔴 Сбросить поиск» или «⚙️ Настройки»:\",\n",
        "        reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True),\n",
        "    )"
      ],
      "metadata": {
        "id": "fEirzGic4qp3"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Запуск"
      ],
      "metadata": {
        "id": "pXnI_Ofc4loa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 🔁 Основная функция запуска\n",
        "async def main():\n",
        "    app = ApplicationBuilder().token(BOT_TOKEN).build()\n",
        "\n",
        "    app.add_handler(CommandHandler(\"start\", start_command))\n",
        "    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))\n",
        "\n",
        "    print(\"🤖 Бот запущен!\")\n",
        "    await app.run_polling()\n",
        "\n",
        "# 👟 Запуск (для Jupyter/Colab)\n",
        "nest_asyncio.apply()\n",
        "await main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tjha93sooAEQ",
        "outputId": "e2ae1557-5587-4214-bdb9-4f1020f517c0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "🤖 Бот запущен!\n"
          ]
        }
      ]
    }
  ]
}
