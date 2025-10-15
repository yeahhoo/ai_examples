import os
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate


# === Configuration ===
DATA_DIR = "/home/sasha/rag_example/knowledge_base"
PERSIST_DIR = "vector_index"


# === Step 1: Load local text files ===
def load_documents(data_dir: str):
    """
    Loads all .txt files under `data_dir` (recursively).
    If a file contains multiple entries separated by "===========",
    each section is treated as a separate document.
    """
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".txt"):
                continue

            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read().strip()

            # Split the file content into sections if separator exists
            parts = [part.strip() for part in content.split("===========") if part.strip()]

            for i, part in enumerate(parts):
                docs.append(
                    Document(
                        page_content=part,
                        metadata={
                            "source": path,
                            "section": i + 1,
                            "filename": os.path.basename(path),
                        },
                    )
                )
    print(f"Loaded {len(docs)} document sections from {data_dir}")
    return docs

# === Step 2: Build or load Chroma HNSW index ===
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


# === Step 3: Set up a local LLM ===
def get_local_llm():
    model_name = "facebook/bart-large-cnn"
    hf_pipeline = pipeline("text2text-generation", model=model_name, device_map="auto")
    return HuggingFacePipeline(pipeline=hf_pipeline)


# === Step 4: Create RAG chain ===
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = get_local_llm()

    # Custom prompt template — no instructional prefix
    template = """Context:
{context}

Question:
{question}

Answer in a concise and direct way:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

# === Main run loop ===
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_DIR)

    print("Building or loading vector index (HNSW)...")
    vectorstore = get_vectorstore(docs)

    print("Creating RAG QA chain...")
    qa_chain = create_rag_chain(vectorstore)

    result_ag14 = qa_chain.invoke({"query": "what is ag14?"})
    print("\nanswer_ag14:", result_ag14["result"])

    print("\n===================================")
    result_btx4 = qa_chain.invoke({"query": "Suggest an algorithm for encryption of IoT devices"})
    print("\nanswer_btx4:", result_btx4["result"])
    
     # === Test without vectorDB ===
    print("\n>>> TEST 2: Without vectorDB (no retrieval)")
    llm = get_local_llm()  # just raw LLM
    raw_answer = llm.invoke("Suggest an algorithm for encryption of IoT devices")
    print("LLM-only answer (no retrieval):", raw_answer)

