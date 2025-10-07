# -*- coding:utf-8 -*-
# @FileName  :speaker_diarization.py

import logging
import os
import numpy as np
import torch

# Patch logging before importing SpeechBrain to prevent ValueError
import sys
_original_logging_getLogger = logging.getLogger

def _patched_getLogger(name=None):
    logger = _original_logging_getLogger(name)
    # Prevent SpeechBrain from setting invalid log levels
    original_setLevel = logger.setLevel
    def safe_setLevel(level):
        try:
            if isinstance(level, str):
                level = level.upper()
                valid_levels = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'}
                if level not in valid_levels:
                    level = 'INFO'
            original_setLevel(level)
        except (ValueError, AttributeError):
            original_setLevel(logging.INFO)
    logger.setLevel = safe_setLevel
    return logger

logging.getLogger = _patched_getLogger

from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

class SpeakerDiarization:
    """Handles speaker embedding extraction and identification."""
    
    def __init__(self, similarity_threshold: float = 0.5, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", max_speakers: int = 100):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.speaker_embeddings = {}  # Dictionary with speaker_id as key
        self.speaker_order = []  # List to track access order (most recent first)
        self.next_speaker_id = 0  # Counter for assigning new speaker IDs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load speaker embedding model
        self._load_model()

    def _load_model(self):
        """Load the speaker embedding model, downloading if necessary."""
        try:
            logging.info(f"Loading speaker embedding model: {self.model_name}")
            # The model will be downloaded to the local cache automatically
            self.embedding_model = EncoderClassifier.from_hparams(
                source=self.model_name, 
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", self.model_name.replace("/", "_")),
                run_opts={"device": self.device}
            )
            logging.info("Speaker embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load speaker embedding model: {e}")
            raise

    def get_embedding(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract speaker embedding from audio data."""
        try:
            with torch.no_grad():
                audio_tensor = torch.tensor(audio_data).float().to(self.device)
                embedding = self.embedding_model.encode_batch(audio_tensor.unsqueeze(0))
                embedding = embedding.squeeze().cpu().numpy()
                return embedding
        except Exception as e:
            logging.error(f"Failed to extract speaker embedding: {e}")
            return None

    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int) -> int:
        """Identify speaker from audio data or add a new one."""
        embedding = self.get_embedding(audio_data, sample_rate)
        if embedding is None:
            return -1  # Error case

        # normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        if not self.speaker_embeddings:
            # First speaker
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speaker_embeddings[speaker_id] = embedding
            self.speaker_order.append(speaker_id)
            return speaker_id

        # Calculate similarity with existing speakers
        speaker_ids = list(self.speaker_embeddings.keys())
        embeddings_list = [self.speaker_embeddings[sid] for sid in speaker_ids]
        similarities = cosine_similarity([embedding], embeddings_list)[0]
        
        # Debugging: Print similarity scores
        similarity_scores = {f"Speaker {speaker_ids[i]}": sim for i, sim in enumerate(similarities)}
        print(f"Similarity Scores: {similarity_scores}")

        max_similarity = np.max(similarities)
        most_similar_idx = np.argmax(similarities)
        most_similar_speaker_id = speaker_ids[most_similar_idx]

        if max_similarity >= self.similarity_threshold:
            # Existing speaker, update embedding with a weighted average
            # This helps to create a more robust speaker profile over time
            self.speaker_embeddings[most_similar_speaker_id] = (
                0.8 * self.speaker_embeddings[most_similar_speaker_id] + 0.2 * embedding
            )
            # normalize
            self.speaker_embeddings[most_similar_speaker_id] /= np.linalg.norm(self.speaker_embeddings[most_similar_speaker_id])
            
            # Move this speaker to the front of the order list (most recently used)
            self.speaker_order.remove(most_similar_speaker_id)
            self.speaker_order.insert(0, most_similar_speaker_id)
            
            return most_similar_speaker_id
        else:
            # New speaker
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speaker_embeddings[speaker_id] = embedding
            self.speaker_order.insert(0, speaker_id)  # Add to front (most recent)
            
            # Check if we exceeded max speakers limit
            if len(self.speaker_embeddings) > self.max_speakers:
                # Remove the least recently used speaker (last in order)
                removed_speaker_id = self.speaker_order.pop()
                del self.speaker_embeddings[removed_speaker_id]
                logging.info(f"Removed least recently used speaker {removed_speaker_id} (limit: {self.max_speakers})")
            
            return speaker_id
