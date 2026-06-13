import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
    return _embedder

def get_embeddings(chunks: list[str]) -> list:
    return _get_embedder().embed_documents(chunks)

if __name__ == "__main__":
    test_chunks = ["Hello world", "Classroom audio RAG pipeline"]
    vectors = get_embeddings(test_chunks)
    
    print(f"Number of vectors generated: {len(vectors)}")
    print(f"Length of a single vector (dimensions): {len(vectors[0])}")
    print(f"Sample look at the first vector: {vectors[0][:5]}...") # print first 5 numbers