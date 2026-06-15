import os
from rag_pipeline import process_audio, handle_query

def process_new_audio(file_path: str):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_name = f"faiss_{base_name}"

    if os.path.isdir(folder_name):
        overwrite = input(f"Index '{folder_name}' already exists. Re-process? (y/n): ")
        if overwrite.lower() != 'y':
            return folder_name

    print(f"1. Transcribing audio from {file_path}...")
    print(f"2. Slicing transcript into semantic chunks...")
    print(f"3. Embedding chunks and creating vector store in '{folder_name}'...")
    folder_name = process_audio(file_path)
    print("Vector store created successfully. Your audio is now searchable.")
    return folder_name

def chat_with_audio(folder_name: str):
    print(f"Audio Database '{folder_name}' Loaded. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == 'exit':
            print("Shutting down.")
            break

        print("Thinking...")
        answer = handle_query(query, folder_name=folder_name)
        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    action = input("Type '1' to process a new audio file, or '2' to chat with an existing database: ")

    if action == '1':
        audio_path = input("Enter the full path to your audio file (e.g., C:/Downloads/lecture.mp3): ")
        audio_path = audio_path.strip('\"')
        active_folder = process_new_audio(audio_path)
        chat_with_audio(active_folder)

    elif action == '2':
        available_dbs = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith("faiss_")]

        if len(available_dbs) > 0:
            print("\nAvailable Lecture Databases:")
            for idx, db in enumerate(available_dbs):
                print(f"[{idx}] {db.replace('faiss_', '')}")

            choice = input("\nSelect the database number you want to chat with: ")
            try:
                selected_folder = available_dbs[int(choice)]
                chat_with_audio(selected_folder)
            except (ValueError, IndexError):
                print("Invalid selection.")
        else:
            print("No FAISS database found. Please process an audio file first (Option 1).")
    else:
        print("Invalid input.")
