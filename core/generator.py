import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
    return _llm

def classify_intent(query: str, history: str = "") -> str:
    prompt = f"""
    Classify this user query into one of four categories:

    1. "retrieval" - Factual question about audio content
       Examples: "What is RAG?", "What did the speaker say about X?"

    2. "generative" - User wants something CREATED from content
       Examples: "Quiz me", "Mock interview", "Summarize", "Flashcards"

    3. "comprehension" - Explain or define something from the audio
       Examples: "What does X mean?", "Explain this concept"

    4. "conversational" - Greetings, short replies, reactions, OR questions about the conversation history itself.
       Examples: "Hello", "Thanks", "What did you just ask me?", "Did I get that right?", "I don't know"

    Recent conversation history:
    {history if history else "No history yet."}

    Current Query: "{query}"

    Reply with ONLY one word: retrieval, generative, comprehension, or conversational
    """
    result = _get_llm().invoke(prompt)

    if isinstance(result.content, list):
        intent = result.content[0]['text'].strip().lower()
    else:
        intent = result.content.strip().lower()

    if intent not in ["retrieval", "generative", "comprehension", "conversational"]:
        intent = "retrieval"

    return intent

def generate_answer(query: str, context: str) -> str:
    prompt = f"""
    You are an advanced analytical assistant specializing in processing audio recordings and lecture transcripts.
    Use the following pieces of retrieved context from the recorded audio to answer the user's question accurately.

    If the answer cannot be found in the provided context, state: "I couldn't find that specific information in the recorded material." Do not attempt to extrapolate or invent details outside the context.

    Context:
    {context}

    User Question: {query}

    Answer:
    """
    answer = _get_llm().invoke(prompt)
    if isinstance(answer.content, list):
        return answer.content[0]['text'].strip()
    return answer.content.strip()

def generate_from_transcript(query: str, full_transcript: str) -> str:
    prompt = f"""
    You are an assistant helping users learn from audio content.
    Based on the following transcript, fulfill the user's request.

    Transcript:
    {full_transcript}

    User Request: {query}

    Response:
    """
    answer = _get_llm().invoke(prompt)
    if isinstance(answer.content, list):
        return answer.content[0]['text'].strip()
    return answer.content.strip()

def explain_with_context(query: str, context: str) -> str:
    prompt = f"""
    You are an assistant helping users understand audio content.
    Use the retrieved context AND your general knowledge to answer.
    If the question asks for explanation or meaning — explain it
    using both the context and what you know.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    answer = _get_llm().invoke(prompt)
    if isinstance(answer.content, list):
        return answer.content[0]['text'].strip()
    return answer.content.strip()

def handle_conversational(query: str, history: str) -> str:
    prompt = f"""
    You are a helpful assistant for a lecture audio analysis app.
    Respond naturally to the user's message using the conversation history for context.
    Keep responses short and friendly.

    Recent conversation:
    {history if history else "No history yet."}

    User: {query}

    Response:
    """
    answer = _get_llm().invoke(prompt)
    if isinstance(answer.content, list):
        return answer.content[0]['text'].strip()
    return answer.content.strip()