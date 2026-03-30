import os
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "0") == "1"

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "requests.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

tts_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    from qwen_tts import Qwen3TTSModel

    kwargs = dict(
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    if FLASH_ATTN:
        kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Loading Qwen3TTS with flash_attention_2")
    else:
        logger.info("Loading Qwen3TTS without flash attention")

    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        **kwargs,
    )
    logger.info("Model loaded successfully")
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str
    instruction: str = ""


@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    logger.info(
        "Request %s | text=%r | instruction=%r",
        request_id,
        req.text,
        req.instruction,
    )

    try:
        wavs, sr = tts_model.generate_custom_voice(
            text=req.text,
            language="Russian",
            instruct=req.instruction if req.instruction else None,
        )
    except Exception as exc:
        logger.error("Request %s failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    output_path = os.path.join(
        "outputs", f"{timestamp.replace(':', '-')}_{request_id}.wav"
    )

    sf.write(output_path, wavs[0], sr)
    logger.info("Request %s | saved audio to %s", request_id, output_path)

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request_id}.wav",
    )
