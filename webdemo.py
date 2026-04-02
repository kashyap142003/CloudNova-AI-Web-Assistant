import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from config import OPENAI_API_KEY as ENV_OPENAI_API_KEY
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="CloudNova AI Assistant")

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        return ENV_OPENAI_API_KEY


OPENAI_API_KEY = get_api_key()

if not OPENAI_API_KEY:
    st.error(
        "Missing API key. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` "
        "or set `OPENAI_API_KEY` / `OPENROUTER_API_KEY` in `.env`."
    )
    st.stop()

st.title("CloudNova AI Web Assistant")
st.write("Ask Anything to our CloudNova AI Assistant")

user_input = st.text_input("Please feel free to ask your question here")
folder_path = os.path.dirname(os.path.abspath(__file__))

llm = ChatOpenAI(
    api_key = OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0
)

@st.cache_resource(show_spinner=False)
def build_retriever(api_key, docs_folder):
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small"
    )

    texts = []
    for file in sorted(os.listdir(docs_folder)):
        if file.endswith(".txt"):
            with open(os.path.join(docs_folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())

    if not texts:
        raise ValueError("No .txt knowledge-base files were found for retrieval.")

    text_chunk = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = text_chunk.create_documents(texts)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="cloudnova_docs"
    )

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 6,
            "lambda_mult": 0.7
        }
    )


try:
    retriever = build_retriever(OPENAI_API_KEY, folder_path)
except Exception as exc:
    st.error(f"Unable to build the knowledge base: {exc}")
    st.stop()

prompt_template = PromptTemplate.from_template("""
You are the official AI assistant for CloudNova CRM.

Your responsibilities:
- Answer questions about CloudNova products and services
- Help users navigate CRM features
- Provide onboarding guidance
- Explain pricing and plans
- Assist with troubleshooting based on documentation

Rules:
- Only answer based on CloudNova documentation provided in the context
- If the question is not related to CloudNova CRM or cannot be answered from the context, politely say: "I'm sorry! I don't have idea about that. Please contact to CloudNova Support Team."
- Do not claim CloudNova is a real operating company
- Do not provide legal, financial, or medical advice
- Maintain a professional and friendly tone
- If user says "exit" or "goodbye" then says "Thank you! Have a great day!"

Context:
{context}

Question:
{question}

Answer:""")

rag_chain = (
    {
        "context" : retriever, "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

if user_input:
    with st.spinner("Getting the Information"):
        answer = rag_chain.invoke(user_input)
        st.write(answer)
