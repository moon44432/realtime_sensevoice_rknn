# -*- coding:utf-8 -*-
# @FileName  :audio_processor.py

import logging
import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce
import requests
from collections import deque
from typing import Optional
from datetime import datetime

from sensevoice_rknn import WavFrontend, FSMNVad, SenseVoiceInferenceSession
from token_parser import SpecialTokenParser
from speaker_diarization import SpeakerDiarization


class AudioProcessor:
    """Audio processing and inference handler"""
    
    def __init__(
        self,
        model_path: str,
        device_id: int = -1,
        num_threads: int = 4,
        language: int = 12,  # Korean by default
        use_itn: bool = False,
        sample_rate: int = 16000,
        chunk_duration: float = 0.3,
        silence_threshold: float = 0.5,
        max_sentence_duration: float = 7.0,
        save_recordings: bool = True,
        recordings_dir: str = "./recordings",
        backend_url: str = "http://localhost:8080",
        enable_speaker_diarization: bool = True
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.language = language
        self.use_itn = use_itn
        self.silence_threshold = silence_threshold
        self.max_sentence_duration = max_sentence_duration
        self.save_recordings = save_recordings
        self.recordings_dir = recordings_dir
        self.backend_url = backend_url
        self.enable_speaker_diarization = enable_speaker_diarization

        # Initialize speaker diarization
        if self.enable_speaker_diarization:
            self.speaker_diarization = SpeakerDiarization()

        # Create recordings directory
        if self.save_recordings:
            os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Sentence buffer management
        self.sentence_buffer = []
        self.silence_duration = 0.0
        self.last_inference_length = 0
        self.inference_interval = 1.25 # in seconds
        self.in_sentence = False

        # Audio input queue
        self.audio_queue = queue.Queue()
        
        # Model initialization
        self._initialize_models(model_path, device_id, num_threads)

        # Token Parser initialization
        self.token_parser = SpecialTokenParser()

        # Threading control
        self.is_running = False
        self.processing_thread = None

        self.prev_chunks = deque(maxlen=1)
        self.last_result = None
        
        # Record start time for subtitle transmission
        self.start_time = time.time()
        
    def _initialize_models(self, model_path: str, device_id: int, num_threads: int):
        """Initialize models."""
        try:
            # Initialize Frontend
            self.frontend = WavFrontend(os.path.join(model_path, "am.mvn"))
            
            # Initialize SenseVoice model
            self.model = SenseVoiceInferenceSession(
                os.path.join(model_path, "embedding.npy"),
                os.path.join(model_path, "sense-voice-encoder.rknn"),
                os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
                device_id,
                num_threads,
            )
            
            # Initialize VAD model
            self.vad = FSMNVad(model_path)
            
            logging.info("All models initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            raise
    
    def send_subtitle_to_backend(self, clean_text: str, token_info: dict, is_final: bool = False):
        """Send subtitle and emotion info to backend."""
        try:
            # Emoji mapping by emotion
            emotion_emoji_map = {
                "HAPPY": "😊",
                "SAD": "😢", 
                "ANGRY": "😠",
                "NEUTRAL": "😐",
                "FEARFUL": "😨",
                "DISGUSTED": "🤢",
                "SURPRISED": "😲",
                "EMO_UNKNOWN": "😐"
            }
            
            # Language code mapping
            lang_code_map = {
                "zh": "🇨🇳 CN",
                "en": "🇬🇧 EN",
                "yue": "🇨🇳 YUE", 
                "ja": "🇯🇵 JP",
                "ko": "🇰🇷 KR",
                None: "🇰🇷 KR"  # Default
            }
            
            # Compose subtitle data
            subtitle_data = {
                "text": clean_text,
                "emotion": token_info.get("emotion", "NEUTRAL"),
                "language": token_info.get("lang", "ko"),
                "timestamp": token_info.get("timestamp", datetime.now().strftime("%H:%M:%S")),
                "is_final": is_final,
                "emoji": emotion_emoji_map.get(token_info.get("emotion", "NEUTRAL"), "😐"),
                "lang_code": lang_code_map.get(token_info.get("lang"), "KR"),
                "speaker": int(token_info.get("speaker", -1)),
            }
            # Send to backend
            response = requests.post(
                f"{self.backend_url}/subtitle",
                json=subtitle_data,
                timeout=1.0
            )
            
            if response.status_code == 200:
                logging.debug(f"Subtitle sent successfully: {clean_text[:50]}...")
            else:
                logging.warning(f"Failed to send subtitle: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            # logging.warning(f"Failed to send subtitle to backend: {e}")
            pass
        except Exception as e:
            logging.error(f"Error sending subtitle: {e}")
    
    def save_audio_file(self, audio_data: np.ndarray, transcription: str = "") -> str:
        """Save audio data as WAV file."""
        if not self.save_recordings:
            return None
        
        try:
            # Validate audio data
            if len(audio_data) == 0:
                logging.warning("Empty audio data, skipping save")
                return None
            
            # Include current time in filename (date_time_milliseconds)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}.wav"
            
            filepath = os.path.join(self.recordings_dir, filename)
            
            # Normalize audio data (-1.0 ~ 1.0 range)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if volume is too high
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Save as WAV file
            sf.write(filepath, audio_data, self.sample_rate, subtype='PCM_16')
            
            logging.info(f"Audio saved: {filepath} (duration: {len(audio_data)/self.sample_rate:.2f}s)")
            return filepath
            
        except ImportError:
            logging.error("soundfile library not installed. Please install: pip install soundfile")
            return None
        except Exception as e:
            logging.error(f"Failed to save audio file: {e}")
            return None
    
    def has_speech_in_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Check if chunk contains speech using VAD."""
        try:
            # Check minimum length required by VAD model
            min_length = int(self.sample_rate * 0.2)
            if len(audio_chunk) < min_length:
                # Pad to minimum length
                padded_chunk = np.pad(audio_chunk, (0, min_length - len(audio_chunk)), 'constant')
            else:
                padded_chunk = audio_chunk
            
            # Run VAD
            segments = self.vad.segments_offline(padded_chunk)
            
            # Validate result
            if segments is None:
                return False
            
            return len(segments) > 0
            
        except IndexError as e:
            return False
        except Exception as e:
            logging.error(f"Error in VAD detection: {e}")
            return False
    
    def perform_inference(self, audio_data: np.ndarray) -> Optional[str]:
        """Perform inference on audio data."""
        try:
            if len(audio_data) < self.sample_rate * 0.3:  # Skip if less than 0.3 seconds
                return None

            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Denoise
            audio_data = noisereduce.reduce_noise(y=audio_data, sr=self.sample_rate)
            
            # Feature extraction
            audio_feats = self.frontend.get_features(audio_data)

            # Run ASR
            asr_result = self.model(
                audio_feats[None, ...],
                language=self.language,
                use_itn=self.use_itn,
            )

            if asr_result and asr_result.strip():
                self.last_result = asr_result.strip()
                return self.last_result

            return None
            
        except Exception as e:
            logging.error(f"Error in inference: {e}")
            return self.last_result
    
    def should_perform_inference(self) -> bool:
        """Determine if inference should be performed."""
        current_length = len(self.sentence_buffer) / self.sample_rate
        return current_length >= self.last_inference_length + self.inference_interval

    def should_finalize_sentence(self) -> bool:
        """Determine if sentence should be finalized."""
        buffer_duration = len(self.sentence_buffer) / self.sample_rate
        return (self.silence_duration >= self.silence_threshold or 
                buffer_duration >= self.max_sentence_duration)

    def get_speaker_id(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Get speaker ID using speaker diarization."""
        if not self.enable_speaker_diarization:
            return -1
        return self.speaker_diarization.identify_speaker(audio_data, sample_rate)
    
    def finalize_sentence(self):
        """Finalize the sentence."""
        # Perform final inference
        final_buffer = np.array(self.sentence_buffer)
        final_result = self.perform_inference(final_buffer)

        if final_result is None:
            final_result = self.last_result

        # Save WAV file
        saved_file = self.save_audio_file(final_buffer)
        if saved_file:
            print(f"Saved: {os.path.basename(saved_file)}")
        
        if final_result:
            # Parse special tokens
            clean_text, token_info = self.token_parser.parse_result(final_result)

            if len(clean_text.strip()) == 0:
                print(f"Sentence is too short. ({clean_text})")
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                buffer_duration = len(self.sentence_buffer) / self.sample_rate
            
                token_info['speaker'] = self.get_speaker_id(final_buffer, self.sample_rate)
                token_info['timestamp'] = timestamp

                print(f"\r[{timestamp}] Finished ({buffer_duration:.1f}s): {clean_text}")
                print(f"Token Info: {token_info}")
                
                # Send final subtitle to backend
                self.send_subtitle_to_backend(clean_text, token_info, is_final=True)
                
                # Call result callback
                if hasattr(self, 'result_callback'):
                    self.result_callback(clean_text, token_info)

        else:
            print(f"Final result is empty.")

        print("End of sentence")
        # Reset buffer
        self.sentence_buffer = []
        self.silence_duration = 0.0
        self.last_inference_length = 0
        
        # Reset VAD status
        self.frontend.reset_status()
        self.vad.vad.all_reset_detection()
    
    def process_audio_stream(self):
        """Continuously process audio stream."""
        logging.info("Starting audio processing thread")
        
        current_chunk_buffer = []
        
        while self.is_running:
            try:
                # Get audio data from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                current_chunk_buffer.extend(audio_chunk)
                
                # Wait until chunk size is reached
                if len(current_chunk_buffer) < self.chunk_size:
                    continue
                
                # Extract chunk to process
                process_chunk = np.array(current_chunk_buffer[:self.chunk_size])
                current_chunk_buffer = current_chunk_buffer[self.chunk_size:]
                
                # Check for speech using VAD
                has_speech = self.has_speech_in_chunk(process_chunk)

                if has_speech:
                    if not self.in_sentence:
                        self.in_sentence = True
                    self.silence_duration = 0.0  # Reset silence counter
                else:
                    self.silence_duration += self.chunk_duration
                
                if self.in_sentence:
                    # If speech, add chunk to sentence buffer
                    if len(self.sentence_buffer) == 0 and len(self.prev_chunks) > 0:
                        # Add previous chunks to sentence buffer (preserve speech start)
                        for prev_chunk in self.prev_chunks:
                            self.sentence_buffer.extend(prev_chunk)
                    self.sentence_buffer.extend(process_chunk)
                    
                    # Perform periodic inference
                    if self.should_perform_inference():
                        buffer_array = np.array(self.sentence_buffer)
                        partial_result = self.perform_inference(buffer_array)
                        
                        if partial_result:
                            clean_text = self.token_parser.extract_clean_text(partial_result)
                            current_time = time.strftime('%H:%M:%S')
                            buffer_duration = len(self.sentence_buffer) / self.sample_rate
                            print(f"\r[{current_time}] ({buffer_duration:.1f}s) {clean_text}", end='', flush=True)
                            
                            # Send partial subtitle to backend (for real-time update)
                            if clean_text.strip():
                                clean_text_full, token_info_full = self.token_parser.parse_result(partial_result)
                                token_info_full['speaker'] = self.get_speaker_id(buffer_array, self.sample_rate)
                                self.send_subtitle_to_backend(clean_text_full, token_info_full, is_final=False)

                        # Update next inference time
                        self.last_inference_length = len(self.sentence_buffer) / self.sample_rate

                    # Check sentence finalization condition
                    if self.sentence_buffer and self.should_finalize_sentence():
                        self.finalize_sentence()
                        self.in_sentence = False

                self.prev_chunks.append(process_chunk)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                continue
                
    def set_result_callback(self, callback):
        """Set result callback function."""
        self.result_callback = callback
        
    def start_processing(self):
        """Start audio processing thread."""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop audio processing."""
        self.is_running = False
        
        # Process last sentence if exists
        if self.sentence_buffer:
            self.finalize_sentence()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Clear buffer
        self.sentence_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break