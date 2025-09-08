# -*- coding:utf-8 -*-
# @FileName  :realtime_sensevoice.py
# @Time      :2024/8/31 16:45
# @Author    :modified for real-time processing

import argparse
import logging
import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce
from collections import deque
from typing import Optional, List
from datetime import datetime

# 기존 모듈들 import (파일에서 가져온 클래스들)
from sensevoice_rknn import WavFrontend, FSMNVad, SenseVoiceInferenceSession

# 로깅 설정
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

# 언어 설정
languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}

class SpecialTokenParser:
    """SenseVoice의 special token을 파싱하는 클래스"""
    
    def __init__(self):
        self.language_tokens = {
            "<|zh|>": "zh",
            "<|en|>": "en", 
            "<|yue|>": "yue",
            "<|ja|>": "ja",
            "<|ko|>": "ko"
        }
        
        self.emotion_tokens = {
            "<|HAPPY|>": "HAPPY",
            "<|SAD|>": "SAD",
            "<|ANGRY|>": "ANGRY", 
            "<|NEUTRAL|>": "NEUTRAL",
            "<|FEARFUL|>": "FEARFUL",
            "<|DISGUSTED|>": "DISGUSTED",
            "<|SURPRISED|>": "SURPRISED",
            "<|EMO_UNKNOWN|>": "EMO_UNKNOWN"
        }
        
        self.event_tokens = {
            "<|BGM|>": "BGM",
            "<|Speech|>": "Speech",
            "<|Applause|>": "Applause",
            "<|Laughter|>": "Laughter",
            "<|Cry|>": "Cry",
            "<|Sneeze|>": "Sneeze",
            "<|Breath|>": "Breath",
            "<|Cough|>": "Cough"
        }
        
        self.itn_tokens = {
            "<|withitn|>": "with_itn",
            "<|woitn|>": "without_itn"
        }
        
        # 모든 special token 패턴
        self.all_tokens = {}
        self.all_tokens.update(self.language_tokens)
        self.all_tokens.update(self.emotion_tokens)
        self.all_tokens.update(self.event_tokens)
        self.all_tokens.update(self.itn_tokens)
    
    def parse_result(self, text: str) -> tuple:
        """
        텍스트에서 special token을 파싱합니다.
        
        Returns:
            tuple: (clean_text, token_info_dict)
        """
        if not text or not text.strip():
            return "", {}
        
        original_text = text
        token_info = {
            "lang": None,
            "emotion": None, 
            "event": None,
            "itn": None,
            "transcript": ""
        }
        
        # 각 카테고리별로 토큰 추출
        for token, value in self.language_tokens.items():
            if token in text:
                token_info["lang"] = value
                text = text.replace(token, "")
        
        for token, value in self.emotion_tokens.items():
            if token in text:
                token_info["emotion"] = value
                text = text.replace(token, "")
        
        for token, value in self.event_tokens.items():
            if token in text:
                token_info["event"] = value
                text = text.replace(token, "")
        
        for token, value in self.itn_tokens.items():
            if token in text:
                token_info["itn"] = value
                text = text.replace(token, "")
        
        # 남은 텍스트가 실제 전사 결과
        clean_text = text.strip()
        token_info["transcript"] = clean_text
        
        return clean_text, token_info
    
    def extract_clean_text(self, text: str) -> str:
        """텍스트에서 special token을 제거하고 깨끗한 텍스트만 반환"""
        clean_text, _ = self.parse_result(text)
        return clean_text

class RealTimeSenseVoice:
    def __init__(
        self, 
        model_path: str, 
        device_id: int = -1, 
        num_threads: int = 4,
        language: str = "auto",
        use_itn: bool = False,
        sample_rate: int = 16000,
        chunk_duration: float = 0.2,  # 0.2초씩 처리 (VAD 안정성 향상)
        silence_threshold: float = 0.3,  # 0.3초 침묵 임계값
        max_sentence_duration: float = 10.0,  # 최대 문장 길이 10초
        save_recordings: bool = True,  # WAV 파일 저장 여부
        recordings_dir: str = "./recordings"  # 녹음 파일 저장 디렉토리
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.language = languages[language]
        self.use_itn = use_itn
        self.silence_threshold = silence_threshold
        self.max_sentence_duration = max_sentence_duration
        self.save_recordings = save_recordings
        self.recordings_dir = recordings_dir

        # 녹음 저장 디렉토리 생성
        if self.save_recordings:
            os.makedirs(self.recordings_dir, exist_ok=True)
        
        # 문장 단위 버퍼 관리
        self.sentence_buffer = []  # 현재 문장을 담는 버퍼
        self.silence_duration = 0.0  # 현재 침묵 지속 시간
        self.last_inference_length = 0  # 마지막 추론 시점의 버퍼 길이
        self.inference_interval = 1.0  # 1.0초마다 추론
        self.in_sentence = False  # 현재 문장 중인지 여부

        # 오디오 입력 큐
        self.audio_queue = queue.Queue()
        
        # 모델 초기화
        self._initialize_models(model_path, device_id, num_threads)

        # Token Parser 초기화
        self.token_parser = SpecialTokenParser()

        # 스레딩 제어
        self.is_running = False
        self.processing_thread = None

        self.prev_chunks = deque(maxlen=1)
        self.last_result = None
        
    def _initialize_models(self, model_path: str, device_id: int, num_threads: int):
        """모델들을 초기화합니다."""
        try:
            # Frontend 초기화
            self.frontend = WavFrontend(os.path.join(model_path, "am.mvn"))
            
            # SenseVoice 모델 초기화
            self.model = SenseVoiceInferenceSession(
                os.path.join(model_path, "embedding.npy"),
                os.path.join(model_path, "sense-voice-encoder.rknn"),
                os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
                device_id,
                num_threads,
            )
            
            # VAD 모델 초기화
            self.vad = FSMNVad(model_path)
            
            logging.info("All models initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize models: {e}")
            raise
    
    def audio_callback(self, indata, frames, time, status):
        """마이크 입력 콜백 함수"""
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # 모노 채널로 변환
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            audio_data = indata[:, 0]
        else:
            audio_data = indata.flatten()
        
        # 큐에 오디오 데이터 추가
        self.audio_queue.put(audio_data.copy())

    def save_audio_file(self, audio_data: np.ndarray, transcription: str = "") -> str:
        """오디오 데이터를 WAV 파일로 저장합니다."""
        if not self.save_recordings:
            return None
        
        try:
            # 오디오 데이터 검증
            if len(audio_data) == 0:
                logging.warning("Empty audio data, skipping save")
                return None
            
            # 현재 시각을 파일명에 포함 (날짜_시각_밀리초)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            filename = f"{timestamp}.wav"
            
            filepath = os.path.join(self.recordings_dir, filename)
            
            # 오디오 데이터 정규화 (-1.0 ~ 1.0 범위로)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 볼륨이 너무 큰 경우 정규화
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # WAV 파일로 저장
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
        """청크에 음성이 있는지 VAD로 확인"""
        try:
            # VAD 모델이 요구하는 최소 길이 확인
            min_length = int(self.sample_rate * 0.2)
            if len(audio_chunk) < min_length:
                # 패딩으로 최소 길이 맞추기
                padded_chunk = np.pad(audio_chunk, (0, min_length - len(audio_chunk)), 'constant')
            else:
                padded_chunk = audio_chunk
            
            # VAD 수행
            segments = self.vad.segments_offline(padded_chunk)
            
            # 결과 검증
            if segments is None:
                return False
            
            return len(segments) > 0
            
        except IndexError as e:
            # logging.warning(f"VAD IndexError - likely empty audio: {e}")
            return False
        except Exception as e:
            logging.error(f"Error in VAD detection: {e}")
            return False
    
    def perform_inference(self, audio_data: np.ndarray) -> Optional[str]:
        """오디오 데이터에 대해 추론을 수행합니다."""
        try:
            if len(audio_data) < self.sample_rate * 0.3:  # 0.3초 미만은 건너뛰기
                return None

            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            # denoise
            audio_data = noisereduce.reduce_noise(y=audio_data, sr=self.sample_rate)
            
            # 특징 추출
            audio_feats = self.frontend.get_features(audio_data)

            # ASR 수행
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
        """추론을 수행해야 하는지 판단"""
        current_length = len(self.sentence_buffer) / self.sample_rate
        return current_length >= self.last_inference_length + self.inference_interval

    def should_finalize_sentence(self) -> bool:
        """문장을 완료해야 하는지 판단"""
        buffer_duration = len(self.sentence_buffer) / self.sample_rate
        return (self.silence_duration >= self.silence_threshold or 
                buffer_duration >= self.max_sentence_duration)
    
    def finalize_sentence(self):
        """문장을 최종적으로 완료합니다."""
        # 최종 추론 수행
        final_buffer = np.array(self.sentence_buffer)
        final_result = self.perform_inference(final_buffer)

        if final_result is None:
            final_result = self.last_result

        # WAV 파일 저장
        saved_file = self.save_audio_file(final_buffer)
        if saved_file:
            print(f"💾 저장됨: {os.path.basename(saved_file)}")
                    
        if final_result:
            # Special Token 파싱
            clean_text, token_info = self.token_parser.parse_result(final_result)

            if len(clean_text.strip()) == 0:
                print(f"Sentence is too short. ({clean_text})")
            else:
                current_time = time.strftime('%H:%M:%S')
                buffer_duration = len(self.sentence_buffer) / self.sample_rate
                print(f"\r✅ [{current_time}] 완료 ({buffer_duration:.1f}s): {clean_text}")
                print(f"   Token Info: {token_info}")
                            
                # 결과 콜백 호출
                if hasattr(self, 'result_callback'):
                    self.result_callback(clean_text, token_info)

        else:
            print(f"Final result is empty.")

        print("End of sentence")
        # 버퍼 초기화
        self.sentence_buffer = []
        self.silence_duration = 0.0
        self.last_inference_length = 0
                    
        # VAD 상태 리셋
        self.frontend.reset_status()
        self.vad.vad.all_reset_detection()
    
    def process_audio_stream(self):
        """오디오 스트림을 지속적으로 처리합니다."""
        logging.info("Starting audio processing thread")
        
        current_chunk_buffer = []
        
        while self.is_running:
            try:
                # 큐에서 오디오 데이터 가져오기
                audio_chunk = self.audio_queue.get(timeout=1.0)
                current_chunk_buffer.extend(audio_chunk)
                
                # 지정된 청크 크기가 될 때까지 대기
                if len(current_chunk_buffer) < self.chunk_size:
                    continue
                
                # 처리할 청크 추출
                process_chunk = np.array(current_chunk_buffer[:self.chunk_size])
                current_chunk_buffer = current_chunk_buffer[self.chunk_size:]
                
                # VAD로 음성 구간 확인
                has_speech = self.has_speech_in_chunk(process_chunk)

                if has_speech:
                    if not self.in_sentence:
                        self.in_sentence = True
                    self.silence_duration = 0.0  # 침묵 카운터 리셋
                else:
                    self.silence_duration += self.chunk_duration
                
                if self.in_sentence:
                    # 음성이 있으면 문장 버퍼에 청크 전체를 추가
                    if len(self.sentence_buffer) == 0 and len(self.prev_chunks) > 0:
                        # 이전 청크들도 문장 버퍼에 추가 (음성 시작 부분 보존)
                        for prev_chunk in self.prev_chunks:
                            self.sentence_buffer.extend(prev_chunk)
                    self.sentence_buffer.extend(process_chunk)
                    
                    # 주기적 추론 수행
                    if self.should_perform_inference():
                        buffer_array = np.array(self.sentence_buffer)
                        partial_result = self.perform_inference(buffer_array)
                        
                        if partial_result:
                            clean_text = self.token_parser.extract_clean_text(partial_result)
                            current_time = time.strftime('%H:%M:%S')
                            buffer_duration = len(self.sentence_buffer) / self.sample_rate
                            print(f"\r🎤 [{current_time}] ({buffer_duration:.1f}s) {clean_text}", end='', flush=True)

                        # 다음 추론 시점 업데이트
                        self.last_inference_length = len(self.sentence_buffer) / self.sample_rate

                    # 문장 완료 조건 확인
                    if self.sentence_buffer and self.should_finalize_sentence():
                        self.finalize_sentence()
                        self.in_sentence = False

                self.prev_chunks.append(process_chunk)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                continue
    
    def start_recording(self):
        """실시간 녹음 및 처리를 시작합니다."""
        if self.is_running:
            logging.warning("Recording is already running")
            return
        
        self.is_running = True
        
        # 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self.process_audio_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 시작 메시지
        print("=" * 60)
        print("🎤 실시간 음성인식이 시작되었습니다")
        print("🗣️  마이크에 대고 말씀해 주세요")
        print(f"⚙️  설정: 침묵 임계값 {self.silence_threshold}초, 최대 문장 길이 {self.max_sentence_duration}초")
        print("⏹️  중지하려면 Ctrl+C를 누르세요")
        print("=" * 60)
        
        # 오디오 스트림 시작
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.chunk_duration),
                dtype=np.float32
            ):
                # 메인 루프
                while self.is_running:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logging.info("Recording stopped by user")
        except Exception as e:
            logging.error(f"Error in audio recording: {e}")
        finally:
            self.stop_recording()
    
    def stop_recording(self):
        """녹음을 중지합니다."""
        self.is_running = False
        
        # 마지막 문장이 있다면 처리
        if self.sentence_buffer:
            self.finalize_sentence()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 버퍼 클리어
        self.sentence_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print("\n🛑 녹음이 중지되었습니다.")
        logging.info("Recording stopped and resources cleaned up")
    
    def set_result_callback(self, callback):
        """결과 콜백 함수를 설정합니다."""
        self.result_callback = callback


def main():
    arg_parser = argparse.ArgumentParser(description="Real-time SenseVoice Speech-to-Text")
    arg_parser.add_argument(
        "-dp",
        "--download_path",
        default=os.path.dirname(__file__),
        type=str,
        help="dir path of resource downloaded",
    )
    arg_parser.add_argument("-d", "--device", default=-1, type=int, help="Device ID")
    arg_parser.add_argument(
        "-n", "--num_threads", default=4, type=int, help="Number of threads"
    )
    arg_parser.add_argument(
        "-l",
        "--language",
        choices=languages.keys(),
        default="ko",
        type=str,
        help="Language",
    )
    arg_parser.add_argument("--use_itn", action="store_true", help="Use ITN")
    arg_parser.add_argument(
        "--chunk_duration", 
        # default=0.3,
        default=0.2, 
        type=float, 
        help="Audio chunk duration in seconds"
    )
    arg_parser.add_argument(
        "--silence_threshold",
        default=0.4,
        type=float,
        help="Silence threshold in seconds to finalize sentence"
    )
    arg_parser.add_argument(
        "--max_sentence_duration",
        default=8.0,
        type=float,
        help="Maximum sentence duration in seconds"
    )
    arg_parser.add_argument(
        "--save_recordings",
        default=False,
        action="store_true",
        help="Save audio recordings as WAV files"
    )
    arg_parser.add_argument(
        "--recordings_dir",
        default="./recordings",
        type=str,
        help="Directory to save audio recordings"
    )
    
    args = arg_parser.parse_args()
    
    try:
        # 실시간 음성인식 시스템 초기화
        stt_system = RealTimeSenseVoice(
            model_path=args.download_path,
            device_id=args.device,
            num_threads=args.num_threads,
            language=args.language,
            use_itn=args.use_itn,
            chunk_duration=args.chunk_duration,
            silence_threshold=args.silence_threshold,
            max_sentence_duration=args.max_sentence_duration,
            save_recordings=args.save_recordings,
            recordings_dir=args.recordings_dir
        )
        
        # 결과 처리 콜백 (선택사항)
        def on_result(clean_text, token_info, audio_file=""):
            # 여기서 결과를 파일로 저장하거나 다른 처리를 할 수 있습니다
            with open("transcription_log.txt", "a", encoding="utf-8") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {clean_text}\n")
                f.write(f"    Token Info: {token_info}\n")
                if audio_file:
                    f.write(f"    Audio: {audio_file}\n")
        
        stt_system.set_result_callback(on_result)
        
        # 실시간 녹음 시작
        stt_system.start_recording()
        
    except Exception as e:
        logging.error(f"Failed to start real-time STT: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())