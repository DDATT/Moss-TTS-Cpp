#include "moss.h"

Moss::Moss(const std::string modelDir, bool useCuda)
    : env_(nullptr),
      sessionOptions_(),
      conditionalDecoder(nullptr),
      embedTokens(nullptr),
      languageModel(nullptr) {
    
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Moss-TTS-Nano");
    env_.DisableTelemetryEvents();                       

    if (useCuda) {
        // Use CUDA provider
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
    }
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    sessionOptions_.DisableCpuMemArena();
    sessionOptions_.DisableMemPattern();
    sessionOptions_.DisableProfiling();

    Ort::Session codecEncode;
    Ort::Session codecDecode;
    Ort::Session codecDecodeStep;
    std::string tokenizerPathString = modelDir + "/MOSS-TTS-Nano/tokenizer.model";
    std::string prefillPathString = modelDir + "/MOSS-TTS-Nano-100M-ONNX/moss_tts_prefill.onnx";
    std::string decodePathString = modelDir + "/MOSS-TTS-Nano-100M-ONNX/moss_tts_decode_step.onnx";
    std::string localDecoderGreedyPathString = modelDir + "/MOSS-TTS-Nano-100M-ONNX/local_greedy_frame.onnx";
    std::string codecEncodePathString = modelDir + "/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_encode.onnx";
    std::string codecDecodeString = modelDir + "/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_full.onnx";
    std::string codecDecodeStepPathString = modelDir + "/MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_step.onnx";
    #ifdef _WIN32
    std::wstring prefillPath_wstr = std::wstring(prefillPathString.begin(), prefillPathString.end());
    std::wstring decodePath_wstr = std::wstring(decodePathString.begin(), decodePathString.end());
    std::wstring localDecoderGreedyPath_wstr = std::wstring(localDecoderGreedyPathString.begin(), localDecoderGreedyPathString.end());
    auto prefillPath = prefillPath_wstr.c_str();
    auto decodePathPath = decodePathPath_wstr.c_str();
    auto localDecoderGreedyPath = localDecoderGreedyPath_wstr.c_str();
    std::wstring codecEncodePath_wstr = std::wstring(codecEncodePathString.begin(), codecEncodePathString.end());
    std::wstring codecDecodePath_wstr = std::wstring(codecDecodeString.begin(), codecDecodeString.end());
    std::wstring codecDecodeStepPath_wstr = std::wstring(codecDecodeStepPathString.begin(), codecDecodeStepPathString.end());
    auto codecEncodePath = codecEncodePath_wstr.c_str();
    auto codecDecodePath = codecDecodePath_wstr.c_str();
    auto codecDecodeStepPath = codecDecodeStepPath_wstr.c_str();
    #else
    auto prefillPath = conditionalDecoderPathString.c_str();
    auto decodePathPath = decodePathPathString.c_str();
    auto localDecoderGreedyPath = localDecoderGreedyPathString.c_str();
        auto codecEncodePath = codecEncodePathString.c_str();
    auto codecDecodePath = codecDecodePathString.c_str();
    auto codecDecodeStepPath = codecDecodeStepPathString.c_str();
    #endif

    prefill = Ort::Session(env_, prefillPath, sessionOptions_);
    decode = Ort::Session(env_, decodePathPath, sessionOptions_);
    localDecoderGreedy = Ort::Session(env_, localDecoderGreedyPath, sessionOptions_);
    const auto status = processor.Load("/home/datbq@kaopiz.local/DatBQ/Moss-CPP-ONNX/models/MOSS-TTS-Nano/tokenizer.model");
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        // error
    }
}

Moss::~Moss() {}

std::vector<int64_t> Moss::SynthesizeSpeechTokens(std::vector<int64_t> inputIds) {
    std::vector<int64_t> generatedTokens;
    generatedTokens.push_back(START_SPEECH_TOKEN);

    std::vector<int64_t> embedTokensInputsDim{1, static_cast<int64_t>(inputIds.size())};
    std::vector<Ort::Value> embedTokensInputTensors;
    embedTokensInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, inputIds.data(), inputIds.size(),
            embedTokensInputsDim.data(), embedTokensInputsDim.size()));
    int64_t condEmbLength = int64_t(condEmb.size() / 1024);
    int64_t currentSeqLen = static_cast<int64_t>(inputIds.size()) + condEmbLength;
    int64_t currentPosition = currentSeqLen - 1; 

    std::vector<Ort::Value> languageModelInputTensors;
    std::vector<Ort::Value> pastKeyValues; 
    
    int64_t nextTokenId = 0;
    for (int i = 0; i < 1024; i++) {
        std::vector<float> currentEmbedsData;
        std::vector<int64_t> currentEmbedsShape;

        if (i == 0) {
            // Get embedding from input text
            std::vector<int64_t> embedTokensInputsDim{1, static_cast<int64_t>(inputIds.size())};
            Ort::Value embedTokensInput = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, inputIds.data(), inputIds.size(),
                embedTokensInputsDim.data(), embedTokensInputsDim.size());
            
            auto inputsEmbedsOutput = embedTokens.Run(Ort::RunOptions{nullptr}, 
                embedTokensInputNames.data(), &embedTokensInput, 1, 
                bertEncoderOutputNames.data(), bertEncoderOutputNames.size());

            const float* promptEmbedsData = inputsEmbedsOutput.front().GetTensorData<float>();
            size_t promptEmbedsSize = inputsEmbedsOutput.front().GetTensorTypeAndShapeInfo().GetElementCount();

            currentEmbedsData = condEmb;
            currentEmbedsData.insert(currentEmbedsData.end(), promptEmbedsData, promptEmbedsData + promptEmbedsSize);
            
            currentEmbedsShape = {1, static_cast<int64_t>(inputIds.size()) + condEmbLength, 1024};

        } 
        else {
            // Get embedding for the next generated token            
            std::vector<int64_t> nextInputIdVec = {nextTokenId};
            std::vector<int64_t> embedTokensInputsDim{1, 1}; // Batch 1, Seq 1
            
            Ort::Value embedTokensInput = Ort::Value::CreateTensor<int64_t>(
                memoryInfo, nextInputIdVec.data(), nextInputIdVec.size(),
                embedTokensInputsDim.data(), embedTokensInputsDim.size());

            auto inputsEmbedsOutput = embedTokens.Run(Ort::RunOptions{nullptr}, 
                embedTokensInputNames.data(), &embedTokensInput, 1, 
                bertEncoderOutputNames.data(), bertEncoderOutputNames.size());
            
            const float* newEmbedData = inputsEmbedsOutput.front().GetTensorData<float>();
            size_t newEmbedSize = inputsEmbedsOutput.front().GetTensorTypeAndShapeInfo().GetElementCount();

            currentEmbedsData.assign(newEmbedData, newEmbedData + newEmbedSize);
            currentEmbedsShape = {1, 1, 1024}; // Each iteration just generated one token
        }

        // prepare inputs for language model
        languageModelInputTensors.clear();

        // Input 0: inputs_embeds
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, currentEmbedsData.data(), currentEmbedsData.size(),
            currentEmbedsShape.data(), currentEmbedsShape.size()));

        // Input 1: attention_mask
        std::vector<int64_t> currentMask(currentSeqLen, 1);
        std::vector<int64_t> currentMaskShape{1, currentSeqLen};
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, currentMask.data(), currentMask.size(),
            currentMaskShape.data(), currentMaskShape.size()));

        // Input 2: position_ids
        std::vector<int64_t> currentPosIds;
        std::vector<int64_t> currentPosIdsShape;
        
        if (i == 0) {
            // i=0: 0, 1, 2, ..., seqLen-1
            currentPosIds.resize(currentSeqLen);
            for(int k=0; k<currentSeqLen; ++k) currentPosIds[k] = k;
            currentPosIdsShape = {1, currentSeqLen};
        } else {
            currentPosition++;
            currentPosIds.push_back(currentPosition);
            currentPosIdsShape = {1, 1};
        }
        languageModelInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo, currentPosIds.data(), currentPosIds.size(),
            currentPosIdsShape.data(), currentPosIdsShape.size()));

        // Input 3..50: past_key_values
        if (i == 0) {
            // At first iteration, past KV is setted to zeros
            for (int j = 0; j < 48; j++) {
                std::vector<int64_t> pastShape = {1, 16, 0, 64};
                std::vector<float> pastData(0.0); 
                pastKeyValues.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, pastData.data(), pastData.size(),
                    pastShape.data(), pastShape.size()));
            }
        }
        
        // Push KV Cache into input list
        for (auto& kv : pastKeyValues) {
            languageModelInputTensors.push_back(std::move(kv)); // Move to avoid copy
        }

        // Run language model
        auto languageModelOutput = languageModel.Run(Ort::RunOptions{nullptr},
            languageModelInputNames.data(), languageModelInputTensors.data(), languageModelInputTensors.size(),
            languageModelOutputNames.data(), languageModelOutputNames.size());

        // Output 0: Logits [Batch, Seq, Vocab]
        float* logitsRaw = languageModelOutput[0].GetTensorMutableData<float>();
        auto logitsShape = languageModelOutput[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocabSize = logitsShape[2];
        int64_t seqDim = logitsShape[1];
        
        float* lastTokenLogits = logitsRaw + 1*(seqDim-1) * vocabSize;

        applyRepetitionPenalty(lastTokenLogits, vocabSize, generatedTokens, repetitionPenalty);
        // ----------------------------------

        // Greedy Search (Argmax) - Find the token with the highest score after penalty
        int64_t bestTokenId = 0;
        float maxScore = -std::numeric_limits<float>::infinity();

        for (int64_t v = 0; v < vocabSize; v++) {
            if (lastTokenLogits[v] > maxScore) {
                maxScore = lastTokenLogits[v];
                bestTokenId = v;
            }
        }

        nextTokenId = bestTokenId;
        if (nextTokenId == STOP_SPEECH_TOKEN) {
            std::cout << "\nStop token reached at step " << i << std::endl;
            break;
        }
        generatedTokens.push_back(nextTokenId);
        
        // Update KV Cache for next iteration
        pastKeyValues.clear(); // Clear old
        for (size_t k = 1; k < languageModelOutput.size(); k++) {
            pastKeyValues.push_back(std::move(languageModelOutput[k]));
        }
        currentSeqLen++;
    }
    return generatedTokens;
}

std::vector<int16_t> ChatterBox::synthesizeSpeech(std::vector<int64_t> generatedTokens) {
    // Run audio decoder model
    std::vector<int64_t> speechTokens;
    speechTokens.insert(speechTokens.end(), promptToken.begin(), promptToken.end());    
    speechTokens.insert(speechTokens.end(), generatedTokens.begin()+1, generatedTokens.end());
    speechTokens.insert(speechTokens.end(), {4299, 4299, 4299}); // Add silence at the end
    std::vector<Ort::Value> conditionalDecoderInputTensors;
    std::vector<int64_t> speechTokensDim{1, static_cast<int64_t>(speechTokens.size())};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, speechTokens.data(), speechTokens.size(),
        speechTokensDim.data(), speechTokensDim.size()));
    std::vector<int64_t> speakerEmbeddingsDim{1, 192};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speakerEmbeddings.data(), speakerEmbeddings.size(),
        speakerEmbeddingsDim.data(), speakerEmbeddingsDim.size()));
    std::vector<int64_t> speakerFeaturesDim{1, 500, 80};
    conditionalDecoderInputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speakerFeatures.data(), speakerFeatures.size(),
        speakerFeaturesDim.data(), speakerFeaturesDim.size()));
    auto audioOutput = conditionalDecoder.Run(
        Ort::RunOptions{nullptr},
        conditionalDecoderInputNames.data(), conditionalDecoderInputTensors.data(), conditionalDecoderInputTensors.size(),
        conditionalDecoderOutputNames.data(), conditionalDecoderOutputNames.size());
    
    std::vector<int16_t> audioBuffer;
    const float *audioOutputData = audioOutput.front().GetTensorData<float>();

    std::vector<int64_t> audioOutputShape = audioOutput.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t audioOutputCount = audioOutputShape[audioOutputShape.size() - 1];
    audioBuffer.reserve(audioOutputCount);

    // Convert float audio to int16
    for (int64_t i = 0; i < audioOutputCount; i++) {
        int16_t intAudioValue = static_cast<int16_t>(
            std::clamp(audioOutputData[i] * MAX_WAV_VALUE,
                        static_cast<float>(std::numeric_limits<int16_t>::min()),
                        static_cast<float>(std::numeric_limits<int16_t>::max())));
        audioBuffer.push_back(intAudioValue);
    }
    return audioBuffer;
}

void ChatterBox::applyRepetitionPenalty(float* logits, int64_t vocabSize, const std::vector<int64_t>& generatedTokens, float penalty) {
    std::unordered_set<int64_t> seenTokens;
    
    for (auto id : generatedTokens) {
        seenTokens.insert(id);
    }

    for (int64_t id : seenTokens) {
        float& score = logits[id];

        if (score < 0) {
            score *= penalty;
        } else {
            score /= penalty;
        }
    }
}