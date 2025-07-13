import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import chromadb
from utils.process_document import chunk_text, embed_text


load_dotenv()

chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = os.getenv("CHROMA_PORT", "8000")

groq_api_key = os.getenv("GROQ_API_KEY")

client = chromadb.HttpClient(
    host=chroma_host,
    port=chroma_port,
)

if not (ret := client.heartbeat()):
    st.error("ChromaDB is not reachable. Please check your configuration.")
else:
    st.success("ChromaDB is reachable.")

with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        st.text_area("Document content", value=content, height=300)

        if st.button("Chunk Document"):
            chunks = chunk_text(content)
            st.session_state.chunks = chunks
            st.success(f"Document split into {len(chunks)} chunks.")
        
        if st.button("Embed Document"):
            if "chunks" in st.session_state:
                embedded_chunks = [embed_text(chunk) for chunk in st.session_state.chunks]
                st.session_state.embedded_chunks = embedded_chunks
                st.success("Document chunks embedded successfully.")
            else:
                st.error("Please chunk the document first.")


        if st.button("Save to ChromaDB"):
            collection = client.get_or_create_collection("documents")

            num_chunks = len(st.session_state.embedded_chunks)

            ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(num_chunks)]
            metadatas = [{"source": uploaded_file.name, "chunk_index": i} for i in range(num_chunks)]

            try:
                collection.add(
                    documents=st.session_state.chunks,
                    embeddings=st.session_state.embedded_chunks,
                    metadatas=metadatas,
                    ids=ids,
                )
                st.success("Document saved to ChromaDB.")
            except Exception as e:
                st.error(f"Error saving document to ChromaDB: {e}")

if "relevant_docs" not in st.session_state:
    st.session_state.relevant_docs = []

if st.session_state.relevant_docs:
    with st.sidebar:
        st.write("**Relevant Documents**")
        for doc in st.session_state.relevant_docs:
            st.divider()
            st.write(doc)
        st.divider()

collection = client.get_or_create_collection(
    "documents",
)

system_message = SystemMessage(
    content="You are a helpful assistant that answers questions based on the provided documents. If you don't know the answer, just say you don't know."
)

if "latest_msgs_sent" not in st.session_state:
    st.session_state.latest_msgs_sent = []
if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "llm" not in st.session_state:
    llm = ChatGroq(
        model = os.getenv("GROQ_MODEL", "groq/groq-llama-3.1-70b"),
        api_key=groq_api_key,
        temperature=0.1,
        max_tokens=1024,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_relevant_docs(message: HumanMessage) -> list[str]:
    query = message.content
    results = collection.query(
        query_texts=[query],
        n_results=5,
    )
    docs = [doc for doc in results["documents"][0]]
    st.session_state.relevant_docs = docs
    return docs
def generate_response(msg:str):
    res = collection.query(
        query_texts=[msg],
        n_results=5,
    )
    docs = res["documents"][0]
    st.session_state.relevant_docs = docs
    combined_sys_message = f'''{system_message} Use the following documents to answer the question: {"\n".join(docs)}'''
    messages = [SystemMessage(content=combined_sys_message)] + st.session_state.messages
    response = llm.invoke(messages)
    st.session_state.messages.append(response)
    st.session_state.latest_msgs_sent = messages
    return response

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if msg := st.chat_input("Enter a message"):
    st.session_state.messages.append(HumanMessage(content=msg))
    response = generate_response(msg)
    st.rerun()