from faster_whisper import WhisperModel

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path: str) -> tuple[str, list[str]]:
    model = _get_model()
    segments, info = model.transcribe(file_path, beam_size=5)
    
    segment_texts = []
    full_text = ""
    
    for segment in segments:
        segment_texts.append(segment.text.strip())
        full_text += segment.text + " "
    
    return full_text.strip(), segment_texts