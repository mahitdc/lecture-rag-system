from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from core.embeddings import _get_embedder

_vector_store_cache: dict[str, FAISS] = {}

def _load_vector_store(folder_name: str) -> FAISS:
    if folder_name not in _vector_store_cache:
        _vector_store_cache[folder_name] = FAISS.load_local(
            folder_name, _get_embedder(), allow_dangerous_deserialization=True
        )
    return _vector_store_cache[folder_name]

def create_vector_store(chunks: list[str], folder_name: str = "faiss_index"):
    vector_store = FAISS.from_texts(
        chunks, _get_embedder(),
        distance_strategy=DistanceStrategy.COSINE
    )
    vector_store.save_local(folder_name)
    print(f"FAISS index saved to {folder_name}")

def get_relevant_chunks(query: str, folder_name: str = "faiss_index") -> str:
    vector_store = _load_vector_store(folder_name)
    results = vector_store.similarity_search_with_score(query, k=4)
    filtered = [doc for doc, score in results if score >= 0.2]

    if not filtered:
        return ""

    context = "\n\n---\n\n".join(
        [f"[Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(filtered)]
    )
    return context
