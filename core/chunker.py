from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

def chunk_text (text : str) -> list[str]:
    chunks = text_splitter.split_text(text)
    return chunks

def chunk_segments(segment_texts: list[str], max_words: int = 200) -> list[str]:
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for segment in segment_texts:
        word_count = len(segment.split())
        if current_word_count + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [segment]
            current_word_count = word_count
        else:
            current_chunk.append(segment)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks