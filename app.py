import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import GPT4All
from langchain.prompts import PromptTemplate

# ------------------------------
# Load Text Files
# ------------------------------
def load_text_files(data_dir="data"):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"filename": filename, "text": text})
    return docs

# ------------------------------
# Create Vector Stores
# ------------------------------
def create_vectorstores(docs):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    stores = {}
    for doc in docs:
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(doc["text"])
        stores[doc["filename"].split(".")[0]] = FAISS.from_texts(chunks, embeddings)
    return stores

# ------------------------------
# Initialize LLM
# ------------------------------
def init_llm(model_path="models/gpt4all-lora-quantized.bin"):
    return GPT4All(model=model_path, n_ctx=512, backend="gptj")

# ------------------------------
# Create Agents
# ------------------------------
class Agent:
    def __init__(self, name, vectorstore, llm):
        self.name = name
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2}),
            return_source_documents=False
        )

    def answer(self, query):
        return self.qa.run(query)

# ------------------------------
# Coordinator Agent
# ------------------------------
class Coordinator:
    def __init__(self, salary_agent, insurance_agent):
        self.salary_agent = salary_agent
        self.insurance_agent = insurance_agent

    def route_query(self, query):
        query_lower = query.lower()
        if "salary" in query_lower or "pay" in query_lower or "deduction" in query_lower or "annual" in query_lower:
            return self.salary_agent.answer(query)
        elif "insurance" in query_lower or "premium" in query_lower or "claim" in query_lower or "coverage" in query_lower:
            return self.insurance_agent.answer(query)
        else:
            return "Sorry, I can only answer salary or insurance questions."

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Multi-Agent RAG System Offline", layout="wide")
st.title("💬 Multi-Agent RAG Chatbot (Offline)")

# Load data & vectorstores
docs = load_text_files("data")
vectorstores = create_vectorstores(docs)

# Initialize local LLM
llm = init_llm()

# Create agents
salary_agent = Agent("Salary Agent", vectorstores["salary"], llm)
insurance_agent = Agent("Insurance Agent", vectorstores["insurance"], llm)
coordinator = Coordinator(salary_agent, insurance_agent)

# User query
query = st.text_input("Ask a question about salary or insurance:")

if query:
    with st.spinner("Thinking..."):
        response = coordinator.route_query(query)
    st.markdown(f"**Answer:** {response}")
