# Chief Boris 🍳🤖  
**Personal Recipe Assistant powered by RAG + LLMs**

![Bot preview](Welcome.png)

🔗 Telegram bot: https://t.me/Cook_for_youu_bot  

---

## 📌 Overview

**Chief Boris** is a conversational AI recipe assistant built as a Telegram chatbot.  
It uses a **Retrieval-Augmented Generation (RAG)** architecture combined with **prompt engineering** to provide personalized cooking recommendations.

The system retrieves relevant recipes using semantic search and generates structured, step-by-step cooking instructions via a large language model (LLM).

Key capabilities:
- semantic retrieval with vector search
- multilingual embeddings
- LLM-based recipe generation
- personalization via constraints (allergies, exclusions)

---

## 🧠 System Architecture

- Recipe dataset (~27,000 entries) is loaded and preprocessed
- Text is encoded into dense embeddings using SentenceTransformers
- Embeddings are indexed with FAISS for fast similarity search
- User queries are matched to top-k relevant recipes
- LLM generates structured cooking instructions
- Personal constraints (e.g., allergens) are applied during retrieval and filtering

---

## 📊 Dataset

The dataset contains 27,000+ recipes with:
- recipe title  
- ingredients  
- cooking steps  
- images  
- external recipe links  

🔗 Dataset: https://drive.google.com/file/d/1UeYkiCsNkTpOF8qS-OUpfZ-uTLrunnb5/view?usp=sharing  

---

## 🔢 Embeddings

- Model: `intfloat/multilingual-e5-base`
- Library: SentenceTransformers
- Indexing: FAISS (Facebook AI Similarity Search)

Embeddings enable semantic matching (e.g., “something cheesy and quick” → relevant pasta dishes).

🔗 Embeddings file: https://drive.google.com/file/d/1izJImuphXig22T_o4Rkhg5RRE572Bytx/view?usp=sharing  

---

## ⚙️ Features

- 🔎 **Semantic recipe search**
  - Natural language queries (e.g., "pasta with cheese")

- ⚠️ **Personalized filtering**
  - Excludes allergens and unwanted ingredients

- 🧠 **LLM-generated cooking instructions**
  - Step-by-step structured recipes

- 🔁 **Alternative recommendations**
  - “Show me something else” returns non-repetitive results

- 🔗 **Source traceability**
  - Each result includes a link to the original recipe

---

## 🔧 Tech Stack

- Python: `asyncio`, `pandas`, `numpy`
- NLP: `sentence-transformers`
- Vector search: `FAISS`
- LLM orchestration: `LlamaIndex`, OpenRouter API
- Bot framework: `python-telegram-bot`

---

## 🏗️ How it works

1. Load dataset (`all_recepies_inter.csv`)
2. Generate embeddings for all recipes
3. Build FAISS index for similarity search
4. Receive user query via Telegram bot
5. Retrieve top-k semantically similar recipes
6. Apply user constraints (allergies, exclusions)
7. Generate final response using LLM
8. Return structured recipe output

---

## 🚀 Setup Instructions

1. Run `embedding.ipynb` to create or load:
   - `faiss_index.bin`

2. Load into `tg_bot.ipynb`:
   - FAISS index
   - recipe dataset (`all_recepies_inter.csv`)
   - `Welcome.png`

3. Create a bot via BotFather:
   - set `BOT_TOKEN`

4. Add OpenRouter API key:
   - `OPENROUTER_API_KEY`

5. Run the bot from the notebook entry point

---

## 🎛️ Bot Interface

| Button | Function |
|--------|----------|
| ⚙️ Settings | Set excluded ingredients (allergies/preferences) |
| 🔁 Show alternatives | Retrieve different recipes for the same query |
| 🔴 Reset search | Clear current session and filters |
| 🍽️ Recipe buttons | Select a specific recipe from results |

---

## 💡 Key Engineering Highlights

- Hybrid RAG architecture (vector search + LLM generation)
- Real-world semantic retrieval system using FAISS
- Constraint-aware recommendation pipeline
- Multilingual embedding model for robust matching
- Chat-based interface via Telegram bot API
