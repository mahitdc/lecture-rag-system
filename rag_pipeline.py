import os
from core.transcriber import transcribe_audio
from core.chunker import chunk_text, chunk_segments
from core.retriever import create_vector_store, get_relevant_chunks
from core.generator import generate_answer


def process_audio(file_path: str) -> str:
    full_text, segment_texts = transcribe_audio(file_path)
    chunks = chunk_segments(segment_texts)
    folder_name = f"faiss_{os.path.splitext(os.path.basename(file_path))[0]}"
    create_vector_store(chunks, folder_name=folder_name)
    return folder_name

def answer_question(query: str, folder_name: str) -> str:
    context = get_relevant_chunks(query, folder_name=folder_name)
    answer = generate_answer(query, context)
    return answer