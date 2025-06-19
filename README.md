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
