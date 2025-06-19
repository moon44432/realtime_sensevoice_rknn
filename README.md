---
license: agpl-3.0
language:
- en
- zh
- ja
- ko
base_model: lovemefan/SenseVoice-onnx
tags:
- rknn
---

# SenseVoiceSmall-RKNN2

SenseVoice is an audio foundation model with audio understanding capabilities, including Automatic Speech Recognition (ASR), Language Identification (LID), Speech Emotion Recognition (SER), and Acoustic Event Classification (AEC) or Acoustic Event Detection (AED).

Currently, SenseVoice-small supports multilingual speech recognition, emotion recognition, and event detection for Chinese, Cantonese, English, Japanese, and Korean, with extremely low inference latency.

- Inference speed (RKNN2): About 20x real-time on a single NPU core of RK3588 (processing 20 seconds of audio per second), approximately 6 times faster than the official whisper model provided in the rknn-model-zoo.
- Memory usage (RKNN2): About 1.1GB

## Usage

1. Clone the project to your local machine

2. Install dependencies

```bash
pip install kaldi_native_fbank onnxruntime sentencepiece soundfile pyyaml numpy<2

pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```
[Source](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl) of the .whl file:

3. Copy librknnt.so to /usr/lib/

Source of librknnt.so: https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

4. Run

```bash
python ./sensevoice_rknn.py --audio_file english.wav
```

If you find that recognition is not working correctly when testing with your own audio files, you may need to convert them to 16kHz, 16-bit, mono WAV format in advance.

```bash
ffmpeg -i input.mp3 -f wav -acodec pcm_s16le -ac 1 -ar 16000 output.wav
```

## RKNN Model Conversion

You need to install rknn-toolkit2 v2.1.0 or higher in advance.

1. Download or convert the ONNX model

You can download the ONNX model from https://huggingface.co/lovemefan/SenseVoice-onnx.
It should also be possible to convert from a PyTorch model to an ONNX model according to the documentation at https://github.com/FunAudioLLM/SenseVoice.

The model file should be named 'sense-voice-encoder.onnx' and placed in the same directory as the conversion script.

2. Convert to RKNN model
```bash
python convert_rknn.py 
```

## Known Issues

- When using fp16 inference with RKNN2, overflow may occur, resulting in inf values. You can try modifying the scaling ratio of the input data to resolve this.  
  Set `SPEECH_SCALE` to a smaller value in `sensevoice_rknn.py`.

## References
- [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [lovemefan/SenseVoice-python](https://github.com/lovemefan/SenseVoice-python)


## FastAPI Transcription Server

This project includes a FastAPI server (`server.py`) that provides an HTTP endpoint for speech-to-text transcription.

### Running the Server

1.  Ensure all dependencies for `sensevoice_rknn.py` and the server are installed. This includes `fastapi` and `uvicorn`:
    ```bash
    pip install fastapi uvicorn
    ```
2.  Place the required model files (`*.rknn`, `*.onnx`, `spm.model`) in the same directory as `server.py`.
3.  Run the server:
    ```bash
    python server.py
    ```
    The server will start on `http://0.0.0.0:8000` by default.

### API Endpoint: `/transcribe`

*   **Method:** `POST`
*   **Description:** Transcribes the audio file specified in the request.
*   **Request Body:** JSON object with the following fields:
    *   `audio_file_path` (string, required): The absolute path to the WAV audio file on the server's filesystem.
    *   `language` (string, optional, default: `"en"`): The language code for transcription. Supported codes depend on the model (e.g., "en", "zh", "ja", "ko").
    *   `use_itn` (boolean, optional, default: `false`): Whether to apply Inverse Text Normalization to the transcription output.

*   **Example Request (`curl`):**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"audio_file_path": "/path/to/your/audio.wav", "language": "en", "use_itn": false}' \
    http://0.0.0.0:8000/transcribe
    ```

*   **Response Body:** JSON object with the following fields:
    *   `full_transcription` (string): The complete transcribed text, including any special tokens from the model.
    *   `segments` (list of objects): A list where each object represents a transcribed audio segment and contains:
        *   `start_time_s` (float): Start time of the segment in seconds.
        *   `end_time_s` (float): End time of the segment in seconds.
        *   `text` (string): Transcribed text for the segment.

*   **Example Response:**
    ```json
    {
      "full_transcription": "<|en|><|HAPPY|><|Speech|><|woitn|>the stale smell of old beer lingers <|en|><|NEUTRAL|><|Speech|><|woitn|>it takes heat to bring out the odor but <|en|><|HAPPY|><|Speech|><|woitn|>a cold dip restores health and zest a salt pickle tastes fine with ham tacos al pastor are my favorite <|en|><|EMO_UNKNOWN|><|Speech|><|woitn|>a zestful food is the hot cross bun",
      "segments": [
        {
          "start_time_s": 1.01,
          "end_time_s": 3.93,
          "text": "<|en|><|HAPPY|><|Speech|><|woitn|>the stale smell of old beer lingers"
        },
        {
          "start_time_s": 4.21,
          "end_time_s": 6.59,
          "text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>it takes heat to bring out the odor but"
        },
        {
          "start_time_s": 6.87,
          "end_time_s": 14.68,
          "text": "<|en|><|HAPPY|><|Speech|><|woitn|>a cold dip restores health and zest a salt pickle tastes fine with ham tacos al pastor are my favorite"
        },
        {
          "start_time_s": 14.96,
          "end_time_s": 18.34,
          "text": "<|en|><|EMO_UNKNOWN|><|Speech|><|woitn|>a zestful food is the hot cross bun"
        }
      ]
    }
    ```

