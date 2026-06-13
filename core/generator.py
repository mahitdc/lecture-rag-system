import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_llm = None
def _get_llm() :
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
    return _llm

def generate_answer(query : str, context : str) -> str :
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
