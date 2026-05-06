#define NOMINMAX
#include <iostream>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <random>

#include "nlohmann/json.hpp"
//#include <sentencepiece_processor.h> #Not work on windows, need to build from source, currently skip tokenizer part
// #include <dml_provider_factory.h> #Sometimes it works, sometimes it doesn't, need further investigation, currently skip DML provider part

using json = nlohmann::json;

// ============================================================
// PromptTemplates: mirrors manifest["prompt_templates"]
// ============================================================
struct PromptTemplates {
    std::vector<int> user_prompt_prefix_token_ids;
    std::vector<int> user_prompt_after_reference_token_ids;
    std::vector<int> assistant_prompt_prefix_token_ids;

    static PromptTemplates from_json(const json& j) {
        PromptTemplates t;
        for (auto& v : j.at("user_prompt_prefix_token_ids"))
            t.user_prompt_prefix_token_ids.push_back(v.get<int>());
        for (auto& v : j.at("user_prompt_after_reference_token_ids"))
            t.user_prompt_after_reference_token_ids.push_back(v.get<int>());
        for (auto& v : j.at("assistant_prompt_prefix_token_ids"))
            t.assistant_prompt_prefix_token_ids.push_back(v.get<int>());
        return t;
    }
};

struct TtsConfig {
    int n_vq = 16;
    int audio_pad_token_id = 1024;
    int audio_start_token_id = 6;
    int audio_end_token_id = 7;
    int audio_user_slot_token_id = 8;
    int audio_assistant_slot_token_id = 9;

    static TtsConfig from_json(const json& j) {
        TtsConfig c;
        c.n_vq = j.value("n_vq", 16);
        c.audio_pad_token_id = j.value("audio_pad_token_id", 1024);
        c.audio_start_token_id = j.value("audio_start_token_id", 6);
        c.audio_end_token_id = j.value("audio_end_token_id", 7);
        c.audio_user_slot_token_id = j.value("audio_user_slot_token_id", 8);
        c.audio_assistant_slot_token_id = j.value("audio_assistant_slot_token_id", 9);
        return c;
    }
};

struct LocalFrameResult {
    bool             should_continue = false;
    std::vector<int> frame_token_ids;
};

struct TensorView {
    const float* data;
    std::vector<int64_t> shape;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

struct ChannelView {
    const float* data;
    int64_t      length;
};

struct VoiceCloneRequest {
    std::vector<int32_t> inputIdsFlat;
    std::vector<int64_t> inputIdsShape;
    std::vector<int32_t> attentionMask;
};

struct Waveform {
    std::vector<float> data;  // interleaved: [s0_ch0, s0_ch1, s1_ch0, s1_ch1, ...]
    int64_t num_samples;
    int     num_channels;
};

class MossTTS {
public:
    TtsConfig       tts_cfg;
    PromptTemplates prompt_tmpl;

    MossTTS() = delete;
    MossTTS(const std::string modelDir, json manifest);
    virtual ~MossTTS();
    void SynthesizeSpeech(const std::vector<std::vector<int>>& promptAudioCodes, const std::vector<int>& TextTokenIDs);

    //sentencepiece::SentencePieceProcessor processor;

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
	//Ort::SessionOptions sessionDMLOptions_; Messing up when using GPU, sometimes it works, sometimes it doesn't, need further investigation
    Ort::Session prefill;
    Ort::Session decode;
    Ort::Session localDecoderFixed;
    Ort::Session codecEncode;
    Ort::Session codecDecode;
    Ort::Session codecDecodeStep;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<float>condEmb;
    std::vector<int64_t>promptToken;
    std::vector<float>speakerEmbeddings;
    std::vector<float>speakerFeatures;

	std::array<const char*, 2> prefillInputNames = { 
        "input_ids", 
        "attention_mask" 
    };
    std::array<const char*, 25> prefillOutputNames = {
      "global_hidden",
      "present_key_0",
      "present_value_0",
      "present_key_1",
      "present_value_1",
      "present_key_2",
      "present_value_2",
      "present_key_3",
      "present_value_3",
      "present_key_4",
      "present_value_4",
      "present_key_5",
      "present_value_5",
      "present_key_6",
      "present_value_6",
      "present_key_7",
      "present_value_7",
      "present_key_8",
      "present_value_8",
      "present_key_9",
      "present_value_9",
      "present_key_10",
      "present_value_10",
      "present_key_11",
      "present_value_11" 
    };

    std::array<const char*, 26> decodeInputNames = {
      "input_ids",
      "past_valid_lengths",
      "past_key_0",
      "past_value_0",
      "past_key_1",
      "past_value_1",
      "past_key_2",
      "past_value_2",
      "past_key_3",
      "past_value_3",
      "past_key_4",
      "past_value_4",
      "past_key_5",
      "past_value_5",
      "past_key_6",
      "past_value_6",
      "past_key_7",
      "past_value_7",
      "past_key_8",
      "past_value_8",
      "past_key_9",
      "past_value_9",
      "past_key_10",
      "past_value_10",
      "past_key_11",
      "past_value_11" 
    };
    std::array<const char*, 25> decodeOutputNames = {
      "global_hidden",
      "present_key_0",
      "present_value_0",
      "present_key_1",
      "present_value_1",
      "present_key_2",
      "present_value_2",
      "present_key_3",
      "present_value_3",
      "present_key_4",
      "present_value_4",
      "present_key_5",
      "present_value_5",
      "present_key_6",
      "present_value_6",
      "present_key_7",
      "present_value_7",
      "present_key_8",
      "present_value_8",
      "present_key_9",
      "present_value_9",
      "present_key_10",
      "present_value_10",
      "present_key_11",
      "present_value_11"
    };

    std::array<const char*, 4> localFixedSampledFrameInputNames = {
      "global_hidden",
      "repetition_seen_mask",
      "assistant_random_u",
      "audio_random_u"
    };
    std::array<const char*, 2> localFixedSampledFrameOutputNames = {
      "should_continue",
      "frame_token_ids"
    };

    std::array<const char*, 2> codecDecodeStepInputNames = {
      "audio_codes",
      "audio_code_lengths"
    };
    std::array<const char*, 2> codecDecodeStepOutputNames = {
      "audio",
      "audio_lengths"
    };

    VoiceCloneRequest build_voice_clone_request_rows(
        const std::vector<std::vector<int>>& prompt_audio_codes,
        const std::vector<int>& text_token_ids) const
    {
        size_t prefix_len = prompt_tmpl.user_prompt_prefix_token_ids.size() + 1;
        size_t audio_len = prompt_audio_codes.size();
        size_t suffix_len = 1 + prompt_tmpl.user_prompt_after_reference_token_ids.size() + 
                            text_token_ids.size() + prompt_tmpl.assistant_prompt_prefix_token_ids.size() + 1;
        
        size_t total_seq_len = prefix_len + audio_len + suffix_len;
        size_t row_width = static_cast<size_t>(tts_cfg.n_vq) + 1;

        VoiceCloneRequest req;
        req.inputIdsShape = { 1, static_cast<int64_t>(total_seq_len), static_cast<int64_t>(row_width) };
        req.inputIdsFlat.assign(total_seq_len * row_width, tts_cfg.audio_pad_token_id);
        req.attentionMask.assign(total_seq_len, 1);

        size_t row_idx = 0;

        auto write_row = [&](int token_id, const std::vector<int>& vq_codes = {}) {
            size_t offset = row_idx * row_width;
            req.inputIdsFlat[offset] = token_id;
            for (size_t i = 0; i < std::min<size_t>(vq_codes.size(), static_cast<size_t>(tts_cfg.n_vq)); ++i) {
                req.inputIdsFlat[offset + 1 + i] = vq_codes[i];
            }
            row_idx++;
        };

        for (int id : prompt_tmpl.user_prompt_prefix_token_ids) write_row(id);
        write_row(tts_cfg.audio_start_token_id);

        for (const auto& code_row : prompt_audio_codes) {
            write_row(tts_cfg.audio_user_slot_token_id, code_row);
        }

        write_row(tts_cfg.audio_end_token_id);
        for (int id : prompt_tmpl.user_prompt_after_reference_token_ids) write_row(id);
        for (int id : text_token_ids) write_row(id);
        for (int id : prompt_tmpl.assistant_prompt_prefix_token_ids) write_row(id);
        write_row(tts_cfg.audio_start_token_id);

        return req;
    }

    static TensorView extract_last_hidden(const Ort::Value& hidden_states) {
        auto info = hidden_states.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        size_t ndim = shape.size();
        const float* data = hidden_states.GetTensorData<float>();
        // Case 1: 2D [seq, D]
        if (ndim == 2)
            return { data, shape };
        // Case 2: 3D [1, T, D]
        if (ndim != 3 || shape[0] != 1)
            throw std::runtime_error(
                "Unexpected global_hidden shape: ndim=" + std::to_string(ndim) +
                ", shape[0]=" + std::to_string(ndim > 0 ? shape[0] : -1));
        int64_t T = shape[1], D = shape[2];
        return { data + (T - 1) * D, {1, D} };
    }

    static LocalFrameResult parse_frame_result(
        std::unordered_map<std::string, Ort::Value>& named_outputs)
    {
        LocalFrameResult res;

        // --- frame_token_ids ---
        auto& ft = named_outputs.at("frame_token_ids");
        auto ft_info = ft.GetTensorTypeAndShapeInfo();
        size_t ft_len = ft_info.GetElementCount();
        const int32_t* ft_data = ft.GetTensorMutableData<int32_t>();
        res.frame_token_ids.reserve(ft_len);
        for (size_t i = 0; i < ft_len; ++i){
            res.frame_token_ids.push_back(static_cast<int>(ft_data[i]));
        }
        // --- should_continue ---
        auto& sc = named_outputs.at("should_continue");
        const int32_t* sc_data = sc.GetTensorMutableData<int32_t>();
        res.should_continue = (sc_data[0] != 0);
        return res;
    }

    static std::vector<ChannelView> slice_channel_major_audio(
        const Ort::Value& audio,
        const Ort::Value& audio_lengths,
        int64_t start_sample = 0)
    {
        // --- validate audio shape [1, C, T] ---
        auto shape = audio.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[0] != 1)
            throw std::runtime_error(
                "Unexpected audio tensor shape: ndim=" + std::to_string(shape.size()));
        const int64_t C = shape[1];             // channels
        const int64_t T = shape[2];             // total samples
        // --- get end_sample from audio_lengths ---
        auto len_info = audio_lengths.GetTensorTypeAndShapeInfo();
        int64_t end_sample;
        if (len_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
            end_sample = static_cast<int64_t>(audio_lengths.GetTensorData<int32_t>()[0]);
        else
            end_sample = audio_lengths.GetTensorData<int64_t>()[0];
        // --- clamp ---
        const int64_t start = std::max<int64_t>(0, start_sample);
        const int64_t end = std::max(start, std::min(end_sample, T));
        const int64_t len = end - start;
        // --- create views, zero-copy ---
        // audio[0, ch, start:end] → base + ch*T + start
        const float* base = audio.GetTensorData<float>();
        std::vector<ChannelView> result;
        result.reserve(static_cast<size_t>(C));
        for (int64_t ch = 0; ch < C; ++ch)
            result.push_back({ base + ch * T + start, len });
        return result;
    }
    static Waveform merge_audio_channels(const std::vector<ChannelView>& channels) {
        // Empty → zeros (0, 1)
        if (channels.empty())
            return { {}, 0, 1 };
        // Single channel → reshape to [-1, 1]
        if (channels.size() == 1) {
            int64_t n = channels[0].length;
            return {
                std::vector<float>(channels[0].data, channels[0].data + n),
                n, 1
            };
        }
        // Multi-channel: trim to min_length then interleave
        int64_t min_len = channels[0].length;
        for (const auto& ch : channels)
            min_len = std::min(min_len, ch.length);
        const int C = static_cast<int>(channels.size());
        std::vector<float> data(static_cast<size_t>(min_len) * C);
        // data[s * C + ch] = channels[ch].data[s]
        for (int ch = 0; ch < C; ++ch)
            for (int64_t s = 0; s < min_len; ++s)
                data[static_cast<size_t>(s) * C + ch] = channels[ch].data[s];
        return { std::move(data), min_len, C };
    }
    static Waveform concat_waveforms(const std::vector<Waveform>& parts) {
        if (parts.empty()) return { {}, 0, 1 };
        const int C = parts[0].num_channels;
        int64_t total = 0;
        for (const auto& p : parts) total += p.num_samples;
        std::vector<float> data;
        data.reserve(static_cast<size_t>(total) * C);
        for (const auto& p : parts)
            data.insert(data.end(), p.data.begin(), p.data.end());
        return { std::move(data), total, C };
    }
    /**
     * create silence padding
     */
    static Waveform make_silence(int64_t num_samples, int num_channels) {
        return {
            std::vector<float>(static_cast<size_t>(num_samples) * num_channels, 0.0f),
            num_samples, num_channels
        };
    }

    static void write_waveform_to_wav(const std::string& path,
        const Waveform& waveform,
        int                sample_rate)
    {
        const uint16_t C = static_cast<uint16_t>(waveform.num_channels);
        const uint32_t sr = static_cast<uint32_t>(sample_rate);
        const uint16_t bits_per_sample = 32;
        const uint16_t block_align = C * (bits_per_sample / 8);  // bytes per frame
        const uint32_t byte_rate = sr * block_align;
        const uint32_t data_size = static_cast<uint32_t>(waveform.data.size()) * sizeof(float);
        const uint32_t riff_size = 36 + data_size;
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot open output file: " + path);
        // Helper lambdas
        auto w2 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
        auto w4 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
        auto ws = [&](const char* s, size_t n) { fwrite(s, 1, n, f); };
        ws("RIFF", 4);  w4(riff_size);
        ws("WAVE", 4);
        // fmt chunk
        ws("fmt ", 4);  w4(16);
        w2(3);              // IEEE float (≠ 1 PCM int)
        w2(C);              // num_channels
        w4(sr);             // sample_rate
        w4(byte_rate);      // byte_rate
        w2(block_align);    // block_align
        w2(bits_per_sample);
        // data chunk
        ws("data", 4);  w4(data_size);
        fwrite(waveform.data.data(), sizeof(float), waveform.data.size(), f);
        fclose(f);
    }
};
