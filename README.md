# Moss-TTS-Cpp

## 1. Project Introduction
A C++ inference runtime implementation for the [**MOSS-TTS-Nano**](https://github.com/OpenMOSS/MOSS-TTS-Nano) Multilingual Open-Source TTS system.

## 2. To-do List
- ✅ Port core inference processing logic from Python to C++ using the ONNX Runtime API.
- ✅ SentencePiece implementation on windows
- ⬜ Support Streaming mode (returning audio chunks immediately to reduce latency).
## 3. Build and Run

### Building SentencePiece
Since this project depends on **SentencePiece** for text tokenization, you need to build it first.

**Linux:**
Follow the build instructions at the [original source](https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source).

**Windows:**
Run the following commands in your terminal to build and install SentencePiece locally:
```bat
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=".\root" -DSPM_DISABLE_EMBEDDED_DATA=ON
cmake --build . --config Release --target install
```

## 4. References
- [**ONNX Runtime C++ API**](https://onnxruntime.ai/docs/api/c/) - Core library for running ML model inference.
- [**nlohmann/json**](https://github.com/nlohmann/json) - C++ library used to parse JSON configuration files (`browser_poc_manifest.json`).
- [**OpenMOSS**](https://github.com/OpenMOSS/MOSS-TTS) for the great tiny multilingual TTS model.
- [**SentencePiece**](https://github.com/google/sentencepiece) for text tokenizer.
