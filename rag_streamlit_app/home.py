import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in .env file.")
    st.stop()

# ---------------------------------------------------
# Streamlit UI setup
# ---------------------------------------------------
st.set_page_config(
    page_title="ESILV RAG Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 ESILV Engineering Cycle Assistant")
st.write(
    "Ask questions about **engineering cycle, internships, calendar, scolarité, and documents**."
)

# ---------------------------------------------------
# Load RAG chain (no caching to avoid stale UI)
# ---------------------------------------------------
def load_rag(api_key):
    try:
        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

        # Vectorstore (update path if needed)
        vectorstore = Chroma(
            persist_directory="/Users/rubesh/Documents/genAi_project/chroma_db",
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.3,
            google_api_key=api_key
        )

        # System prompt
        system_prompt = (
            "You are a helpful assistant for ESILV students. "
            "Use ONLY the provided context to answer. "
            "If the answer is not in the context, say you don't know. "
            "Answer in French if the question is in French, otherwise English.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        return rag_chain

    except Exception as e:
        st.error(f"⚠️ Failed to load RAG chain: {e}")
        st.stop()

# Load the chain
rag_chain = load_rag(GOOGLE_API_KEY)

# ---------------------------------------------------
# User input
# ---------------------------------------------------
question = st.text_input(
    "💬 Ask your question:",
    placeholder="e.g. What is the structure of the engineering cycle?"
)

if question:
    with st.spinner("Thinking... 🤔"):
        try:
            response = rag_chain.invoke({"input": question})
            st.subheader("📌 Answer")
            st.write(response.get("answer", "No answer returned."))
        except Exception as e:
            st.error(f"⚠️ Error while getting answer: {e}")

# Optional footer
st.markdown("---")
st.write("🛠 Built with Streamlit and Google Generative AI")
