import os
import time
import logging
import re
print(f"Initial logging._nameToLevel: {logging._nameToLevel}")
from pathlib import Path
from typing import List, Dict, Any, Optional

import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure sensevoice_rknn.py is in the same directory or PYTHONPATH
# Add the directory of this script to sys.path if sensevoice_rknn is not found directly
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from sensevoice_rknn import WavFrontend, SenseVoiceInferenceSession, FSMNVad, languages
except ImportError as e:
    logging.error(f"Error importing from sensevoice_rknn.py: {e}")
    logging.error("Please ensure sensevoice_rknn.py is in the same directory as server.py or in your PYTHONPATH.")
    # Fallback for critical components if import fails, to allow FastAPI to at least start and show an error
    class WavFrontend:
        def __init__(self, *args, **kwargs): raise NotImplementedError("WavFrontend not loaded")
        def get_features(self, *args, **kwargs): raise NotImplementedError("WavFrontend not loaded")
    class SenseVoiceInferenceSession:
        def __init__(self, *args, **kwargs): raise NotImplementedError("SenseVoiceInferenceSession not loaded")
        def __call__(self, *args, **kwargs): raise NotImplementedError("SenseVoiceInferenceSession not loaded")
    class FSMNVad:
        def __init__(self, *args, **kwargs): raise NotImplementedError("FSMNVad not loaded")
        def segments_offline(self, *args, **kwargs): raise NotImplementedError("FSMNVad not loaded")
        class Vad:
            def all_reset_detection(self, *args, **kwargs): raise NotImplementedError("FSMNVad not loaded")
        vad = Vad()

    languages = {"en": 4} # Default fallback

app = FastAPI()

# Logging will be handled by Uvicorn's default configuration or a custom log_config if provided to uvicorn.run
# Get a logger instance for application-specific logs if needed
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set level for this specific logger

# --- Model Configuration & Loading ---
MODEL_BASE_PATH = Path(__file__).resolve().parent

# These paths should match those used in sensevoice_rknn.py's main function
# or be configurable if they differ.
MVN_PATH = MODEL_BASE_PATH / "am.mvn"
EMBEDDING_NPY_PATH = MODEL_BASE_PATH / "embedding.npy"
ENCODER_RKNN_PATH = MODEL_BASE_PATH / "sense-voice-encoder.rknn"
BPE_MODEL_PATH = MODEL_BASE_PATH / "chn_jpn_yue_eng_ko_spectok.bpe.model"
VAD_CONFIG_DIR = MODEL_BASE_PATH # Assuming fsmn-config.yaml and fsmnvad-offline.onnx are here

# Global model instances
w_frontend: Optional[WavFrontend] = None
asr_model: Optional[SenseVoiceInferenceSession] = None
vad_model: Optional[FSMNVad] = None

@app.on_event("startup")
def load_models():
    global w_frontend, asr_model, vad_model
    logging.info("Loading models...")
    start_time = time.time()
    try:
        if not MVN_PATH.exists():
            raise FileNotFoundError(f"CMVN file not found: {MVN_PATH}")
        w_frontend = WavFrontend(cmvn_file=str(MVN_PATH))

        if not EMBEDDING_NPY_PATH.exists() or not ENCODER_RKNN_PATH.exists() or not BPE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"One or more ASR model files not found: "
                f"Embedding: {EMBEDDING_NPY_PATH}, Encoder: {ENCODER_RKNN_PATH}, BPE: {BPE_MODEL_PATH}"
            )
        asr_model = SenseVoiceInferenceSession(
            embedding_model_file=str(EMBEDDING_NPY_PATH),
            encoder_model_file=str(ENCODER_RKNN_PATH),
            bpe_model_file=str(BPE_MODEL_PATH),
            # Assuming default device_id and num_threads as in sensevoice_rknn.py's main
            device_id=-1, 
            intra_op_num_threads=4 
        )

        # Check for VAD model files (fsmn-config.yaml, fsmnvad-offline.onnx)
        if not (VAD_CONFIG_DIR / "fsmn-config.yaml").exists() or not (VAD_CONFIG_DIR / "fsmnvad-offline.onnx").exists():
             raise FileNotFoundError(f"VAD config or model not found in {VAD_CONFIG_DIR}")
        vad_model = FSMNVad(config_dir=str(VAD_CONFIG_DIR))
        
        logging.info(f"Models loaded successfully in {time.time() - start_time:.2f} seconds.")
    except FileNotFoundError as e:
        logging.error(f"Model loading failed: {e}")
        # Keep models as None, endpoints will raise errors
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading: {e}")
        # Keep models as None

class TranscribeRequest(BaseModel):
    audio_file_path: str
    language: str = "en"  # Default to English
    use_itn: bool = False

class Segment(BaseModel):
    start_time_s: float
    end_time_s: float
    text: str

class TranscribeResponse(BaseModel):
    full_transcription: str
    segments: List[Segment]

@app.post("/transcribe", response_model=str)
async def transcribe_audio(request: TranscribeRequest):
    if w_frontend is None or asr_model is None or vad_model is None:
        logging.error("Models not loaded. Transcription cannot proceed.")
        raise HTTPException(status_code=503, detail="Models are not loaded. Please check server logs.")

    audio_path = Path(request.audio_file_path)
    if not audio_path.exists() or not audio_path.is_file():
        logging.error(f"Audio file not found: {audio_path}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True
        )
    except Exception as e:
        logging.error(f"Error reading audio file {audio_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    if sample_rate != 16000:
        # Basic resampling could be added here if needed, or just raise an error
        logging.warning(f"Audio sample rate is {sample_rate}Hz, expected 16000Hz. Results may be suboptimal.")
        # For now, we proceed but log a warning. For critical applications, convert or reject.

    logging.info(f"Processing audio: {audio_path}, Duration: {len(waveform) / sample_rate:.2f}s, Channels: {waveform.shape[1]}")

    lang_code = languages.get(request.language.lower())
    if lang_code is None:
        logging.warning(f"Unsupported language: {request.language}. Defaulting to 'en'. Supported: {list(languages.keys())}")
        lang_code = languages.get("en", 0) # Fallback to 'en' or 'auto' if 'en' isn't in languages

    all_segments_text: List[str] = []
    detailed_segments: List[Segment] = []
    processing_start_time = time.time()

    for channel_id in range(waveform.shape[1]):
        channel_data = waveform[:, channel_id]
        logging.info(f"Processing channel {channel_id + 1}/{waveform.shape[1]}")
        
        try:
            # Ensure channel_data is 1D for VAD if it expects that
            speech_segments = vad_model.segments_offline(channel_data) # segments_offline expects 1D array
        except Exception as e:
            logging.error(f"VAD processing failed for channel {channel_id}: {e}")
            # Optionally skip this channel or raise an error for the whole request
            continue # Skip to next channel

        for part_idx, part in enumerate(speech_segments):
            start_sample = int(part[0] * 16)  # VAD returns ms, convert to samples (16 samples/ms for 16kHz)
            end_sample = int(part[1] * 16)
            segment_audio = channel_data[start_sample:end_sample]

            if len(segment_audio) == 0:
                logging.info(f"Empty audio segment for channel {channel_id}, part {part_idx}. Skipping.")
                continue
            
            try:
                # Ensure get_features expects 1D array
                audio_feats = w_frontend.get_features(segment_audio) 
                # ASR model expects batch dimension, add [None, ...]
                asr_result_text_raw = asr_model(
                    audio_feats[None, ...],
                    language=lang_code,
                    use_itn=request.use_itn,
                )
                # Remove tags like <|en|>, <|HAPPY|>, etc.
                asr_result_text_cleaned = re.sub(r"<\|[^\|]+\|>", "", asr_result_text_raw).strip()
                
                segment_start_s = part[0] / 1000.0
                segment_end_s = part[1] / 1000.0
                logging.info(f"[Ch{channel_id}] [{segment_start_s:.2f}s - {segment_end_s:.2f}s] Raw: {asr_result_text_raw} Cleaned: {asr_result_text_cleaned}")
                all_segments_text.append(asr_result_text_cleaned)
                detailed_segments.append(Segment(start_time_s=segment_start_s, end_time_s=segment_end_s, text=asr_result_text_cleaned))
            except Exception as e:
                logging.error(f"ASR processing failed for segment {part_idx} in channel {channel_id}: {e}")
                # Optionally add a placeholder or skip this segment's text
                detailed_segments.append(Segment(start_time_s=part[0]/1000.0, end_time_s=part[1]/1000.0, text="[ASR_ERROR]"))

        vad_model.vad.all_reset_detection() # Reset VAD state for next channel or call

    full_transcription = " ".join(all_segments_text).strip()
    logging.info(f"Transcription complete in {time.time() - processing_start_time:.2f}s. Result: {full_transcription}")

    return full_transcription

if __name__ == "__main__":
    import uvicorn

    MINIMAL_LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False, # Let other loggers (like our app logger) exist
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": { # Uvicorn's own operational logs
                "handlers": ["default"],
                "level": logging.INFO, # Explicitly use integer
                "propagate": False,
            },
            "uvicorn.error": { # Logs for errors within Uvicorn
                "handlers": ["default"],
                "level": logging.INFO, # Explicitly use integer
                "propagate": False,
            },
            # We are deliberately not configuring uvicorn.access here for simplicity
            # It might default to INFO or be silent if not configured and no parent handler catches it.
        },
        # Ensure our application logger also works if needed
        __name__: {
            "handlers": ["default"],
            "level": logging.INFO,
            "propagate": False,
        }
    }

    logger.info(f"Attempting to run Uvicorn with minimal explicit log_config.")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=MINIMAL_LOGGING_CONFIG)
