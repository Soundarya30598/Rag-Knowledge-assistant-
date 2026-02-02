import os
import subprocess
import re
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Suppress transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

TEXT_FOLDER = r"C:\Rag\rag files"
OLLAMA_PATH = r"C:\Users\DLP-I516-156\AppData\Local\Programs\Ollama\Ollama.exe"
LLAMA_MODEL = "llama3.2:1b"
TOP_K = 3
CHUNK_SIZE = 500

# Load documents
all_documents = []
text_files = [f for f in os.listdir(TEXT_FOLDER) if f.lower().endswith(".txt")]
for file_name in text_files:
    with open(os.path.join(TEXT_FOLDER, file_name), "r", encoding="utf-8") as f:
        all_documents.append({"file_name": file_name, "text": f.read()})

# Chunking
def chunk_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    para_no = 1
    for s in sentences:
        if len(current) + len(s) <= CHUNK_SIZE:
            current += " " + s
        else:
            chunks.append((current.strip(), para_no))
            para_no += 1
            current = s
    if current.strip():
        chunks.append((current.strip(), para_no))
    return chunks

chunked_documents = []
for doc in all_documents:
    for idx, (chunk, para_no) in enumerate(chunk_text(doc["text"])):
        chunked_documents.append({"file_name": doc["file_name"], "para_no": para_no, "text": chunk})

# Vector DB
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
chroma_client = chromadb.Client(chromadb.config.Settings(persist_directory="chroma_store"))
collection = chroma_client.get_or_create_collection("knowledge_base")

if collection.count() == 0:
    documents = [d["text"] for d in chunked_documents]
    metadatas = [{"file_name": d["file_name"], "para_no": d["para_no"]} for d in chunked_documents]
    ids = [f"doc_{i}" for i in range(len(documents))]
    embeddings = embedding_model.encode(documents, normalize_embeddings=True).tolist()
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

# Retrieval: returns exact KB chunk if found, else None
def retrieve_context(query):
    q_emb = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K)

    for d, m, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        sim = 1 - dist
        if sim > 0:  # Return top chunk if any similarity exists
            return d
    return None

# Prompt for exact extraction
def build_prompt(query, doc):
    return (
        "Extract the exact sentence from the context that answers the question.\n"
        "Do not rephrase or explain.\n"
        "Return only the sentence. If the answer is not present, return nothing.\n\n"
        "Context:\n" + doc + "\n\nQuestion:\n" + query + "\n\nAnswer:\n"
    )

# Call Ollama
def call_ollama(prompt):
    result = subprocess.run(
        [OLLAMA_PATH, "run", LLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore",
        timeout=60
    )
    return result.stdout.strip()

# Chat loop
def main():
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if not query or query.lower() == "exit":
            break
        doc = retrieve_context(query)
        if not doc:
            print("Information not found")
            continue
        prompt = build_prompt(query, doc)
        answer = call_ollama(prompt)
        if not answer.strip():
            print("Information not found")
        else:
            print(answer)

if __name__ == "__main__":
    main()
