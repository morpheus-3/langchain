# LangChain Powered AI Application

A modular, data-aware AI application built using the [LangChain](https://github.com/langchain-ai/langchain) framework. This project leverages Large Language Models (LLMs), external data sources, and autonomous agents to deliver context-aware, intelligent interactions.

---

## 🚀 Features

* **Retrieval-Augmented Generation (RAG):** Connects LLMs to custom documents (PDFs, TXT, Web Markdown) for grounded, accurate question-answering.
* **Autonomous Agents:** Uses LLMs as reasoning engines to dynamically select and execute external tools (e.g., Google Search, Web Scraping, Calculators).
* **Memory Management:** Seamlessly maintains conversational history across user sessions.
* **Modular Components:** Easily switch out LLM providers (OpenAI, Anthropic, Ollama) and Vector Databases (Chroma, Pinecone, FAISS).

---

## 🛠️ Tech Stack

* **Framework:** LangChain 
* **Language:** Python 3.10+
* **LLM Engine:** OpenAI GPT-4o / Claude 3.5 Sonnet
* **Vector Store:** ChromaDB (for local embeddings) / Pinecone (for cloud)
* **Environment Management:** `python-dotenv`

---

## 📦 Installation & Setup

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
