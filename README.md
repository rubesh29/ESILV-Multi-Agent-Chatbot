# ESILV-Multi-Agent-Chatbot (GenAI Project)


A Retrieval-Augmented Generation (RAG) system designed to answer student queries about college regulations, internships, exams, and international programs. 

This project uses **Google Gemini 1.5 Flash** as the LLM and **ChromaDB** for vector storage, allowing users to chat with official college PDFs and websites in both French and English.

## 🚀 Features

* **Multi-Source Data Ingestion:** Loads data from multiple PDF files (Rules, Calendars, Internship guides) and official college websites.
* **Multilingual Support:** Understands documents in French and answers questions in either English or French based on the user's input.
* **Smart Rate Limiting:** Includes a robust ingestion script that handles Google API rate limits automatically.
* **Interactive UI:** Built with **Streamlit** for a clean, chat-like user experience.
* **Vector Search:** Uses **ChromaDB** and **Google Embeddings (text-embedding-004)** for accurate semantic search.

## 🛠️ Tech Stack

* **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Frontend:** Streamlit
* **Embeddings:** Google Generative AI Embeddings (`models/text-embedding-004`)

## 📂 Project Structure

```bash
├── home.py                  # Main Streamlit application file
├── data_ingestion.ipynb    # Jupyter Notebook to process PDFs/URLs and save to DB
├── requirements.txt        # List of dependencies
├── .env                    # API Keys (Not uploaded to GitHub)
└── chroma_db/              # Generated Vector Database (Created by the notebook)
