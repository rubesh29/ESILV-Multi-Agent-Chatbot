# 🏫 ESILV Smart Assistant: Local RAG & Multi-Agent System

> *AI using Llama 3.1 & Ollama*

![Status](https://img.shields.io/badge/Status-Completed-success)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green)
![Stack](https://img.shields.io/badge/Tech-LangChain%20|%20Ollama%20|%20Streamlit-orange)

## 📖 Project Overview

This project implements an **Offline Retrieval-Augmented Generation (RAG)** system integrated with **Agentic Workflows**. Designed for the ESILV engineering school context, it allows students to query internal institutional documents (PDFs) without relying on external cloud APIs.

Unlike standard chatbots, this system features a **Multi-Agent Architecture** that can distinguish between simple information retrieval tasks and administrative actions (such as lead generation), performing them securely on local hardware.

## 🚀 Key Features

### 🔒 1. Privacy-First & Offline
* **Zero Data Leakage:** Powered by **Ollama**, all processing happens locally. No student data or internal documents are sent to third-party servers (OpenAI/Google).
* **Cost-Free:** Eliminates API token costs and rate limits.

### 🧠 2. Intelligent Multi-Agent Orchestration
The system utilizes **LangChain Agents** to dynamically select the correct tool:
* **Retrieval Tool:** For semantic queries (e.g., *"What is the passing grade for Year 4?"*). Uses **ChromaDB** vector storage.
* **Lead Capture Tool:** For intent detection (e.g., *"I want to register, my email is..."*). Extracts entities and saves them to a CSV log.

### 📊 3. Admin & Operations Dashboard
A comprehensive **Streamlit** interface that allows administrators to:
* **Dynamic Ingestion:** Drag-and-drop PDF upload to update the knowledge base instantly.
* **Lead Management:** View and download student contact logs (`student_leads.csv`).

## 🛠️ Technical Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | **Ollama (Llama 3.1)** | 8B parameter model quantized for local execution. |
| **Orchestrator** | **LangChain** | Manages the "ReAct" (Reasoning + Acting) loop for the agent. |
| **Vector Database** | **ChromaDB** | Stores document embeddings locally for semantic search. |
| **Embeddings** | **Nomic-Embed-Text** | High-performance open-source embedding model. |
| **Frontend** | **Streamlit** | Interactive web UI for chat and admin tasks. |

## 📂 Repository Structure

```bash
├── app.py                  # Entry point: Streamlit UI + Agent Logic
├── backend.py              # (Optional) Separated logic for tools/chains
├── requirements.txt        # Python dependencies
├── student_leads.csv       # Persistent storage for captured leads
├── chroma_db/              # Local vector store (generated at runtime)
└── README.md               # Documentation
