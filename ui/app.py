import sys
import os
import tempfile
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_pipeline import process_audio, handle_query

st.set_page_config(page_title="Lecture RAG", page_icon="🎧", layout="centered")

st.title("🎧 Lecture RAG")
st.caption("Upload an audio file, transcribe it, and chat with it.")

# ── Session state ──────────────────────────────────────────────────────────────
if "folder" not in st.session_state:
    st.session_state.folder = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar — existing indexes ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Processed Lectures")
    folders = [f for f in os.listdir(ROOT_DIR) if f.startswith("faiss_") and os.path.isdir(ROOT_DIR / f)]
    if folders:
        choice = st.selectbox("Load existing:", ["— select —"] + folders)
        if choice != "— select —" and st.button("Load"):
            st.session_state.folder = choice
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No processed lectures yet.")

    if st.session_state.folder:
        st.success(f"Active: `{st.session_state.folder}`")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ── Upload & process ───────────────────────────────────────────────────────────
st.subheader("1. Upload Audio")
uploaded = st.file_uploader("MP3, WAV, M4A, FLAC, OGG", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded:
    if st.button("▶ Transcribe & Index"):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.status("Processing audio...", expanded=True) as status:
            st.write("🎙️ Transcribing...")
            try:
                folder = process_audio(tmp_path)
                st.write("✅ Transcribed and indexed.")
                status.update(label="Done!", state="complete")
                st.session_state.folder = folder
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))

# ── Chat ───────────────────────────────────────────────────────────────────────
if st.session_state.folder:
    st.divider()
    st.subheader("2. Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the lecture...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = handle_query(question, st.session_state.folder, st.session_state.messages)
                except Exception as e:
                    answer = f"Error: {e}"
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload and process an audio file to start chatting.")