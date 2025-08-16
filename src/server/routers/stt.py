from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import tempfile

from src.core.speech_to_text import SpeechToTextProcessor

router = APIRouter()

_stt = None

def _get_stt():
    global _stt
    if _stt is None:
        _stt = SpeechToTextProcessor()
    return _stt

class STTResponse(BaseModel):
    text: str
    language: Optional[str] = None
    method: Optional[str] = None
    model: Optional[str] = None

@router.post("/stt", response_model=STTResponse)
async def transcribe(file: UploadFile = File(...), language: Optional[str] = None):
    stt = _get_stt()
    # Save to temp and run STT
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = stt.process(str(tmp_path), language=language)
        return STTResponse(
            text=result.get("text", ""),
            language=result.get("language"),
            method=result.get("method"),
            model=result.get("model"),
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
