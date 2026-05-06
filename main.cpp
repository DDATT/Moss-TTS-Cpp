// Moss-TTS-ONNX.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "MossTTS.h"
#include <chrono>

using json = nlohmann::json;

int main()
{
    std::ifstream file("path_to/models/MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json");
    if (!file.is_open()) {
        std::cerr << "Can't open JSON!" << std::endl;
        return 1;
    }
    json manifest;
    file >> manifest;
	MossTTS tts("path_to/models", manifest);

    if (manifest.contains("builtin_voices") && !manifest["builtin_voices"].empty()) {
        auto first_voice = manifest["builtin_voices"][0];
        if (first_voice.contains("prompt_audio_codes")) {
            std::vector<std::vector<int>> prompt_audio_codes =
                first_voice["prompt_audio_codes"].get<std::vector<std::vector<int>>>();

			//Hard code text token ids for testing, which say "Hello world! This is a test".
            std::vector<int> text_token_ids = { 7026, 1177, 11449, 994, 343, 273, 2422, 10380 };
            // Measure the execution time of the SynthesizeSpeechTokens function
            auto start = std::chrono::high_resolution_clock::now();
            tts.SynthesizeSpeech(prompt_audio_codes, text_token_ids);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Execution time of SynthesizeSpeechTokens: " << elapsed.count() << " seconds" << std::endl;

        }
    }

    std::cout << "Hello World!\n";
}
