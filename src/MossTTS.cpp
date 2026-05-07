#include "MossTTS.h"

MossTTS::MossTTS(const std::string modelDir, json manifest)
    : env_(nullptr),
    sessionOptions_(),
    //sessionDMLOptions_(),
    prefill(nullptr),
    decode(nullptr),
    localDecoderFixed(nullptr),
    codecEncode(nullptr),
    codecDecode(nullptr),
    codecDecodeStep(nullptr)
{
    tts_cfg = TtsConfig::from_json(manifest.at("tts_config"));
    prompt_tmpl = PromptTemplates::from_json(manifest.at("prompt_templates"));

    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Moss-TTS-Nano");
    env_.DisableTelemetryEvents();

    //if (useCuda) {
    //    // Use CUDA provider
    //    OrtCUDAProviderOptions cuda_options{};
    //    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    //    sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
    //}
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    sessionOptions_.DisableCpuMemArena();
    sessionOptions_.DisableMemPattern();
    sessionOptions_.DisableProfiling();
    //OrtSessionOptionsAppendExecutionProvider_DML(sessionDMLOptions_, 0); // Append the DML EP
    //sessionDMLOptions_.SetGraphOptimizationLevel(
    //    GraphOptimizationLevel::ORT_DISABLE_ALL);

    //sessionDMLOptions_.DisableCpuMemArena();
    //sessionDMLOptions_.DisableMemPattern();
    //sessionDMLOptions_.DisableProfiling();

    std::string tokenizerPathString = modelDir + "\\MOSS-TTS-Nano-100M-ONNX\\tokenizer.model";
    std::string prefillPathString = modelDir + "\\MOSS-TTS-Nano-100M-ONNX\\moss_tts_prefill.onnx";
    std::string decodePathString = modelDir + "\\MOSS-TTS-Nano-100M-ONNX\\moss_tts_decode_step.onnx";
    std::string localDecoderFixedPathString = modelDir + "\\MOSS-TTS-Nano-100M-ONNX\\moss_tts_local_fixed_sampled_frame.onnx";
    std::string codecEncodePathString = modelDir + "\\MOSS-Audio-Tokenizer-Nano-ONNX\\moss_audio_tokenizer_encode.onnx";
    std::string codecDecodeString = modelDir + "\\MOSS-Audio-Tokenizer-Nano-ONNX\\moss_audio_tokenizer_decode_full.onnx";
    std::string codecDecodeStepPathString = modelDir + "\\MOSS-Audio-Tokenizer-Nano-ONNX\\moss_audio_tokenizer_decode_step.onnx";
#ifdef _WIN32
    std::wstring prefillPath_wstr = std::wstring(prefillPathString.begin(), prefillPathString.end());
    std::wstring decodePath_wstr = std::wstring(decodePathString.begin(), decodePathString.end());
    std::wstring localDecoderFixedPath_wstr = std::wstring(localDecoderFixedPathString.begin(), localDecoderFixedPathString.end());
    auto prefillPath = prefillPath_wstr.c_str();
    auto decodePathPath = decodePath_wstr.c_str();
    auto localDecoderFixedPath = localDecoderFixedPath_wstr.c_str();
    std::wstring codecEncodePath_wstr = std::wstring(codecEncodePathString.begin(), codecEncodePathString.end());
    std::wstring codecDecodePath_wstr = std::wstring(codecDecodeString.begin(), codecDecodeString.end());
    std::wstring codecDecodeStepPath_wstr = std::wstring(codecDecodeStepPathString.begin(), codecDecodeStepPathString.end());
    auto codecEncodePath = codecEncodePath_wstr.c_str();
    auto codecDecodePath = codecDecodePath_wstr.c_str();
    auto codecDecodeStepPath = codecDecodeStepPath_wstr.c_str();
#else
    auto prefillPath = conditionalDecoderPathString.c_str();
    auto decodePathPath = decodePathPathString.c_str();
    auto localDecoderFixedPath = localDecoderFixedPathString.c_str();
    auto codecEncodePath = codecEncodePathString.c_str();
    auto codecDecodePath = codecDecodePathString.c_str();
    auto codecDecodeStepPath = codecDecodeStepPathString.c_str();
#endif
	std::cout << "Prefill model path: " << prefillPathString << std::endl;
    prefill = Ort::Session(env_, prefillPath, sessionOptions_);
	std::cout << "Decode model path: " << decodePathString << std::endl;
    decode = Ort::Session(env_, decodePathPath, sessionOptions_);
	std::cout << "Local Decoder Greedy model path: " << localDecoderFixedPathString << std::endl;
    localDecoderFixed = Ort::Session(env_, localDecoderFixedPath, sessionOptions_);
	std::cout << "Codec Encode model path: " << codecEncodePathString << std::endl;
    codecEncode = Ort::Session(env_, codecEncodePath, sessionOptions_);
    codecDecode = Ort::Session(env_, codecDecodePath, sessionOptions_);
    codecDecodeStep = Ort::Session(env_, codecDecodeStepPath, sessionOptions_);
    const auto status = processor.Load(tokenizerPathString);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        // error
    }
}

MossTTS::~MossTTS() {}

void MossTTS::SynthesizeSpeech(const std::vector<std::vector<int>>& promptAudioCodes, std::string textInput) {
    //place holder for Text preprocessing
    std::vector<int> TextTokenIDs;
    processor.Encode(textInput, &TextTokenIDs);
	VoiceCloneRequest request = build_voice_clone_request_rows(promptAudioCodes, TextTokenIDs);

    std::vector<Ort::Value> prefillInputTensors;

    Ort::Value prefillInputs = Ort::Value::CreateTensor<int32_t>(
        memoryInfo, request.inputIdsFlat.data(), request.inputIdsFlat.size(),
        request.inputIdsShape.data(), request.inputIdsShape.size());

    std::vector<int64_t> maskDims = { 1, static_cast<int64_t>(request.attentionMask.size()) };

    Ort::Value prefillMaskInputs = Ort::Value::CreateTensor<int32_t>(
        memoryInfo, request.attentionMask.data(), request.attentionMask.size(),
        maskDims.data(), maskDims.size());
	prefillInputTensors.push_back(std::move(prefillInputs));
	prefillInputTensors.push_back(std::move(prefillMaskInputs));

	auto prefillOutputs = prefill.Run(Ort::RunOptions{ nullptr },
        prefillInputNames.data(), prefillInputTensors.data(), prefillInputTensors.size(),
        prefillOutputNames.data(), prefillOutputNames.size()
    );
    //float* globalHidden = prefillOutputs[0].GetTensorMutableData<float>();
    //auto globalHiddenShape = prefillOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto globalHiddenPrefill = extract_last_hidden(prefillOutputs[0]);
    std::vector<float> global_hidden_buf(globalHiddenPrefill.data, globalHiddenPrefill.data + globalHiddenPrefill.numel());

    std::vector<Ort::Value> decoderInputPast;
    for (int j = 1; j < prefillOutputs.size(); j++) {
        decoderInputPast.push_back(std::move(prefillOutputs[j]));
    }

    // Allocate persistent buffers for loop
    std::vector<float> assistant_u_buf(1, 0.0f);
    std::vector<float> audio_u_buf(16, 0.0f);
    std::vector<int32_t> repetition_seen_mask_buf(16 * 1024, 0);
    std::vector<int32_t> nextRowData(17, 1024);
    nextRowData[0] = tts_cfg.audio_assistant_slot_token_id;
    int past_valid_length = 178 + 5; //Hard code for testing, should be calculated from inputIDs length
    std::vector<int32_t> pastValidLengthData = { past_valid_length };

    std::vector<int64_t> aru_shape = { 1 };
    std::vector<int64_t> auru_shape = { 1, 16 };
    std::vector<int64_t> mask_shape = { 1, 16, 1024 };
    std::vector<int64_t> nextRowDims = { 1, 1, 17 };
    std::vector<int64_t> pastValidLengthDims = { 1 };

    std::vector<Ort::Value> localDecoderFixedInputs;
    localDecoderFixedInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, global_hidden_buf.data(), global_hidden_buf.size(),
        globalHiddenPrefill.shape.data(), globalHiddenPrefill.shape.size()));
    localDecoderFixedInputs.push_back(Ort::Value::CreateTensor<int32_t>(
        memoryInfo, repetition_seen_mask_buf.data(), repetition_seen_mask_buf.size(),
        mask_shape.data(), mask_shape.size()));
    localDecoderFixedInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, assistant_u_buf.data(), assistant_u_buf.size(),
        aru_shape.data(), aru_shape.size()));
    localDecoderFixedInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, audio_u_buf.data(), audio_u_buf.size(),
        auru_shape.data(), auru_shape.size()));

    std::vector<int32_t> generatedFramesFlat;
    generatedFramesFlat.reserve(375 * 16);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::mt19937_64 rng(1234);
	int audio_code_lengths = 0;

    for (int i = 0; i < 375; i++) {
        assistant_u_buf[0] = std::min(0.99999994f, std::max(0.0f, dist(rng)));
        for (int u = 0; u < 16; ++u){
            audio_u_buf[u] = std::min(0.99999994f, std::max(0.0f, dist(rng)));
        }

        auto localFixedSampledFrameOutput = localDecoderFixed.Run(Ort::RunOptions{ nullptr },
            localFixedSampledFrameInputNames.data(), localDecoderFixedInputs.data(), localDecoderFixedInputs.size(),
            localFixedSampledFrameOutputNames.data(), localFixedSampledFrameOutputNames.size()
		);

        std::unordered_map<std::string, Ort::Value> result;
        result.reserve(2);
        for (size_t k = 0; k < 2; ++k)
            result.emplace(localFixedSampledFrameOutputNames[k], std::move(localFixedSampledFrameOutput[k]));

		LocalFrameResult frameResult = parse_frame_result(result);
        if (!frameResult.should_continue){
			audio_code_lengths = i; // include the current frame
			std::cout << "Generation finished at step " << i << ", total frames: " << audio_code_lengths << std::endl;
            break;
        }

        for (int ch = 0; ch < 16 && ch < static_cast<int>(frameResult.frame_token_ids.size()); ++ch) {
            int token = frameResult.frame_token_ids[ch];
            if (token >= 0 && token < 1024) {
                repetition_seen_mask_buf[static_cast<size_t>(ch) * 1024 + token] = 1;
            }
            generatedFramesFlat.push_back(token);
            nextRowData[ch + 1] = token;
        }

        Ort::Value decoderInputsIDs = Ort::Value::CreateTensor<int32_t>(
            memoryInfo, nextRowData.data(), nextRowData.size(),
            nextRowDims.data(), nextRowDims.size());

        pastValidLengthData[0] = past_valid_length;
        Ort::Value decoderPastValidLength = Ort::Value::CreateTensor<int32_t>(
            memoryInfo, pastValidLengthData.data(), pastValidLengthData.size(),
            pastValidLengthDims.data(), pastValidLengthDims.size());

		std::vector<Ort::Value> decodeInputs;
        decodeInputs.reserve(2 + decoderInputPast.size());
		decodeInputs.push_back(std::move(decoderInputsIDs));
        decodeInputs.push_back(std::move(decoderPastValidLength));
        for (size_t j = 0; j < decoderInputPast.size(); j++) {
            decodeInputs.push_back(std::move(decoderInputPast[j]));
		}

        auto decodeOutputs = decode.Run(Ort::RunOptions{ nullptr },
            decodeInputNames.data(), decodeInputs.data(), decodeInputs.size(),
            decodeOutputNames.data(), decodeOutputNames.size()
		);

		decoderInputPast.clear();
        decoderInputPast.reserve(decodeOutputs.size() - 1);
        for (size_t j = 1; j < decodeOutputs.size(); j++) {
            decoderInputPast.push_back(std::move(decodeOutputs[j]));
        }

        auto decoderview = extract_last_hidden(decodeOutputs[0]);
        std::copy(decoderview.data, decoderview.data + decoderview.numel(), global_hidden_buf.begin());

        past_valid_length++;
		std::cout << "Step " << i << " done." << std::endl;
    }

    std::vector<int64_t> dims = { 1, audio_code_lengths, 16 };
    Ort::Value audioCodes = Ort::Value::CreateTensor<int32_t>(
        memoryInfo, generatedFramesFlat.data(), generatedFramesFlat.size(),
        dims.data(), dims.size());
	std::vector<int32_t> audioCodeLengths = { audio_code_lengths };
    Ort::Value audioCodeLengthsTensor = Ort::Value::CreateTensor<int32_t>(
        memoryInfo, audioCodeLengths.data(), audioCodeLengths.size(),
		std::vector<int64_t>{1}.data(), 1);
	std::vector<Ort::Value> codecDecodeStepInputs;
	codecDecodeStepInputs.push_back(std::move(audioCodes));
	codecDecodeStepInputs.push_back(std::move(audioCodeLengthsTensor));
    auto codecDecodeStepOutput = codecDecode.Run(Ort::RunOptions{ nullptr },
        codecDecodeStepInputNames.data(), codecDecodeStepInputs.data(), codecDecodeStepInputs.size(),
        codecDecodeStepOutputNames.data(), codecDecodeStepOutputNames.size()
	);

    auto channels = slice_channel_major_audio(
        codecDecodeStepOutput[0],  // audio [1, C, T]
        codecDecodeStepOutput[1],  // audio_lengths
        /*start_sample=*/ 0
    );

    Waveform chunk_waveform = merge_audio_channels(channels);
    std::vector<Waveform> all_parts;
    all_parts.push_back(std::move(chunk_waveform));
    //int pause_samples = /* sample_rate * pause_seconds */;
    //if (pause_samples > 0)
    //    all_parts.push_back(make_silence(pause_samples, num_channels));

    Waveform final_waveform = concat_waveforms(all_parts);
    write_waveform_to_wav("output.wav", final_waveform, 48000);
}
