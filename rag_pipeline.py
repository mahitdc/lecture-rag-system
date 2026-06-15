import os
from core.transcriber import transcribe_audio
from core.chunker import chunk_text, chunk_segments
from core.retriever import create_vector_store, get_relevant_chunks
from core.generator import (classify_intent, generate_answer, generate_from_transcript, explain_with_context, handle_conversational)

def process_audio(file_path: str) -> str:
    full_text, segment_texts = transcribe_audio(file_path)
    print(f"[DEBUG] Full text length: {len(full_text)}")
    print(f"[DEBUG] Number of segments: {len(segment_texts)}")
    
    if not segment_texts or not full_text.strip():
        raise ValueError("No speech detected. Please ensure the audio contains clear spoken words.")
    
    chunks = chunk_segments(segment_texts)
    print(f"[DEBUG] Number of chunks: {len(chunks)}")
    
    if not chunks:
        raise ValueError("The transcript was too short to process into database chunks.")
    
    folder_name = f"faiss_{os.path.splitext(os.path.basename(file_path))[0]}"
    create_vector_store(chunks, folder_name=folder_name)
    
    transcript_path = f"{folder_name}/transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    return folder_name
def handle_query(query: str, folder_name: str, chat_history: list = []) -> str:
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history[-6:]
        ])

    intent = classify_intent(query, history_text)

    if intent == "conversational":
        return handle_conversational(query, history_text)

    if intent == "generative":
        transcript_path = f"{folder_name}/transcript.txt"
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                full_transcript = f.read()
            return generate_from_transcript(query, full_transcript)
        else:
            return "Transcript not found. Please reprocess the audio."

    context = get_relevant_chunks(query, folder_name=folder_name)

    if intent == "comprehension":
        if not context.strip():
            return "I couldn't find relevant content in the audio to explain that."
        return explain_with_context(query, context)
    
    if not context.strip():
        return "I couldn't find that specific information in the recorded material."
    return generate_answer(query, context)