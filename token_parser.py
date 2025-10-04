# -*- coding:utf-8 -*-
# @FileName  :token_parser.py

class SpecialTokenParser:
    """Class for parsing SenseVoice special tokens"""
    
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
        
        # All special token patterns
        self.all_tokens = {}
        self.all_tokens.update(self.language_tokens)
        self.all_tokens.update(self.emotion_tokens)
        self.all_tokens.update(self.event_tokens)
        self.all_tokens.update(self.itn_tokens)
    
    def parse_result(self, text: str) -> tuple:
        """
        Parse special tokens from text.
        
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
        
        # Extract tokens by category
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

        text = text.replace("<|nospeech|>", "")
        text = text.replace("<|Event_UNK|>", "")

        # Remaining text is the actual transcript
        clean_text = text.strip()
        token_info["transcript"] = clean_text
        
        return clean_text, token_info
    
    def extract_clean_text(self, text: str) -> str:
        """Remove special tokens and return clean text only"""
        clean_text, _ = self.parse_result(text)
        return clean_text