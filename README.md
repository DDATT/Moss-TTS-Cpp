# Moss-TTS-Cpp

## 1. Project Introduction
A C++ inference runtime implementation for the [**MOSS-TTS-Nano**](https://github.com/OpenMOSS/MOSS-TTS-Nano) Multilingual Open-Source TTS system.

## 2. To-do List
- ✅ Port core inference processing logic from Python to C++ using the ONNX Runtime API.
- ⬜ SentencePiece implementation on windows (Linux work but I messed up with building on Windows).
- ⬜ Support Streaming mode (returning audio chunks immediately to reduce latency).

## 3. References
- [**ONNX Runtime C++ API**](https://onnxruntime.ai/docs/api/c/) - Core library for running ML model inference.
- [**nlohmann/json**](https://github.com/nlohmann/json) - C++ library used to parse JSON configuration files (`browser_poc_manifest.json`).
- [**OpenMOSS**](https://github.com/OpenMOSS/MOSS-TTS) for the great tiny multilingual TTS model.
