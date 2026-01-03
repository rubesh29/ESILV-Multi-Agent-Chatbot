import streamlit as st
import pandas as pd
import os
import csv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
LEADS_FILE = "student_leads.csv"

st.set_page_config(page_title="ESILV Local AI", page_icon="🦙", layout="wide")

# Ensure leads file exists
if not os.path.exists(LEADS_FILE):
    with open(LEADS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Email", "Interest", "Timestamp"])

# --- BACKEND LOGIC (Tools & Agent) ---

@tool
def retrieve_school_info(query: str):
    """Finds information about ESILV programs, courses, and rules."""
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        if not docs:
            return "No info found in documents."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error: {e}"

@tool
def save_contact_info(name: str, email: str, interest: str = "General"):
    """Saves student contact details for follow-up."""
    import datetime
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LEADS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, email, interest, timestamp])
        return f"Success! Saved contact for {name}."
    except Exception as e:
        return f"Error saving: {e}"

@st.cache_resource
def get_agent():
    # Initialize Ollama Model
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    tools = [retrieve_school_info, save_contact_info]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the AI Assistant for ESILV. "
         "1. Use 'retrieve_school_info' for questions. "
         "2. Use 'save_contact_info' if user gives Name/Email to register. "
         "3. Be concise and polite."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_file(uploaded_file):
    """Ingests a PDF using Ollama Embeddings"""
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    vectorstore.add_documents(splits)
    
    os.remove(temp_path)
    return len(splits)

# --- FRONTEND INTERFACE ---

# Initialize Agent
if "agent" not in st.session_state:
    st.session_state.agent = get_agent()

st.sidebar.title("🦙 Local Agent")
page = st.sidebar.radio("Navigation", ["Chat", "Admin Dashboard"])

if page == "Chat":
    st.title("🏫 ESILV AI Assistant (Ollama)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question or register..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking (Llama 3.1)..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    answer = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "Admin Dashboard":
    st.title("🛠️ Admin Panel")
    
    tab1, tab2 = st.tabs(["Upload PDFs", "View Leads"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.button("Ingest"):
            with st.spinner("Processing..."):
                count = process_file(uploaded_file)
                st.success(f"Added {count} chunks to knowledge base!")
                
    with tab2:
        if os.path.exists(LEADS_FILE):
            df = pd.read_csv(LEADS_FILE)
            st.dataframe(df)
        else:
            st.info("No leads yet.")