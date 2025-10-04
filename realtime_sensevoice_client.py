# -*- coding:utf-8 -*-
# @FileName  :realtime_sensevoice_client.py

import argparse
import logging
import os
import time
import numpy as np
import sounddevice as sd

from audio_processor import AudioProcessor

# Logging configuration
formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

# Language settings
languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}

class RealTimeSenseVoice:
    """Main class for real-time speech recognition interface"""
    
    def __init__(
        self, 
        model_path: str, 
        device_id: int = -1, 
        num_threads: int = 4,
        language: str = "auto",
        use_itn: bool = False,
        sample_rate: int = 16000,
        chunk_duration: float = 0.2,  # Process every 0.2 seconds (improves VAD stability)
        silence_threshold: float = 0.3,  # 0.3 seconds silence threshold
        max_sentence_duration: float = 10.0,  # Max sentence length 10 seconds
        save_recordings: bool = True,  # Whether to save WAV files
        recordings_dir: str = "./recordings",  # Directory to save recordings
        backend_url: str = "http://localhost:8080",  # Backend URL
        enable_speaker_diarization: bool = True
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            model_path=model_path,
            device_id=device_id,
            num_threads=num_threads,
            language=languages[language],
            use_itn=use_itn,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            silence_threshold=silence_threshold,
            max_sentence_duration=max_sentence_duration,
            save_recordings=save_recordings,
            recordings_dir=recordings_dir,
            backend_url=backend_url,
            enable_speaker_diarization=enable_speaker_diarization
        )
        
        # Audio stream control
        self.is_running = False
        
    def audio_callback(self, indata, frames, time, status):
        """Microphone input callback function"""
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # Convert to mono channel
        if len(indata.shape) > 1 and indata.shape[1] > 1:
            audio_data = indata[:, 0]
        else:
            audio_data = indata.flatten()
        
        # Add audio data to queue
        self.audio_processor.audio_queue.put(audio_data.copy())

    def start_recording(self):
        """Start real-time recording and processing."""
        if self.is_running:
            logging.warning("Recording is already running")
            return
        
        self.is_running = True
        
        # Start processing thread
        self.audio_processor.start_processing()
        
        # Start message
        print("=" * 60)
        print("Real-time speech recognition started.")
        print("Please speak into the microphone.")
        print(f"Settings: Silence threshold {self.audio_processor.silence_threshold}s, Max sentence length {self.audio_processor.max_sentence_duration}s")
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        
        # Start audio stream
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * self.chunk_duration),
                dtype=np.float32
            ):
                # Main loop
                while self.is_running:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            logging.info("Recording stopped by user")
        except Exception as e:
            logging.error(f"Error in audio recording: {e}")
        finally:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording."""
        self.is_running = False
        self.audio_processor.stop_processing()
        
        print("\nRecording stopped.")
        logging.info("Recording stopped and resources cleaned up")
    
    def set_result_callback(self, callback):
        """Set result callback function."""
        self.audio_processor.set_result_callback(callback)


def main():
    arg_parser = argparse.ArgumentParser(description="Real-time SenseVoice Speech-to-Text")
    arg_parser.add_argument(
        "-dp",
        "--download_path",
        default=os.path.dirname(__file__),
        type=str,
        help="Directory path of downloaded resources",
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
        default=0.2, 
        type=float, 
        help="Audio chunk duration in seconds"
    )
    arg_parser.add_argument(
        "--silence_threshold",
        default=0.5,
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
    arg_parser.add_argument(
        "--backend_url",
        default="http://localhost:8080",
        type=str,
        help="Backend server URL for subtitle transmission"
    )
    arg_parser.add_argument(
        "--enable_speaker_diarization",
        default=True,
        action="store_true",
        help="Enable speaker diarization"
    )
    
    args = arg_parser.parse_args()
    
    try:
        # Initialize real-time speech recognition system
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
            recordings_dir=args.recordings_dir,
            backend_url=args.backend_url,
            enable_speaker_diarization=args.enable_speaker_diarization
        )
        
        # Result callback (optional)
        def on_result(clean_text, token_info, audio_file=""):
            # You can save results to a file or perform other actions here
            with open("transcription_log.txt", "a", encoding="utf-8") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {clean_text}\n")
                f.write(f"    Token Info: {token_info}\n")
                if audio_file:
                    f.write(f"    Audio: {audio_file}\n")
        
        stt_system.set_result_callback(on_result)
        
        # Start real-time recording
        stt_system.start_recording()
        
    except Exception as e:
        logging.error(f"Failed to start real-time STT: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())