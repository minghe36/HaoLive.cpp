#include "haolive/haolive.h"
#include "haolive/haolive.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: haolive-vad-tests MODEL.gguf SPEECH.wav [SPEECH.wav...]\n"; return 2; }
    hl_vad* detector = nullptr;
    try {
        int checks = 0;
        auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error(std::string("VAD check failed: ") + hl_last_error()); };
        auto options = hl_default_vad_options(); options.model_path = argv[1];
        detector = hl_vad_create(&options); require(detector != nullptr);
        hl_vad_result result{};
        require(hl_vad_detect(detector, nullptr, 0, nullptr, &result) == 0 && !result.sufficient_audio && !result.is_speech);
        std::vector<float> silence(16000, 0);
        require(hl_vad_detect(detector, silence.data(), 399, nullptr, &result) == 0 && !result.sufficient_audio);
        require(hl_vad_detect(detector, silence.data(), silence.size(), nullptr, &result) == 0 && !result.is_speech && result.sufficient_audio);
        auto silent = result;
        float invalid = std::numeric_limits<float>::quiet_NaN();
        require(hl_vad_detect(detector, &invalid, 1, nullptr, &result) == -1 && !result.sufficient_audio);
        require(hl_vad_detect(detector, nullptr, 1, nullptr, &result) == -1);
        require(hl_vad_detect(detector, silence.data(), 1, nullptr, nullptr) == -1);
        require(hl_vad_detect(detector, &invalid, std::numeric_limits<size_t>::max(), nullptr, &result) == -1);
        auto params = hl_default_vad_params(); params.threshold = 1;
        require(hl_vad_detect(detector, nullptr, 0, &params, &result) == -1);
        for (int i = 2; i < argc; ++i) {
            std::ifstream file(argv[i], std::ios::binary);
            if (!file) throw std::runtime_error("cannot read speech fixture");
            std::string bytes((std::istreambuf_iterator<char>(file)), {});
            auto speech = haolive::decode_wav(bytes.data(), bytes.size());
            speech.resize(std::min(speech.size(), size_t(80000)));
            require(hl_vad_detect(detector, speech.data(), speech.size(), nullptr, &result) == 0 && result.is_speech);
            auto original = result;
            // The same detector can safely serve simultaneous independent windows.
            auto task = std::async(std::launch::async, [&] {
                hl_vad_result r{};
                return hl_vad_detect(detector, silence.data(), silence.size(), nullptr, &r) == 0 && !r.is_speech && r.max_probability == silent.max_probability;
            });
            require(hl_vad_detect(detector, speech.data(), speech.size(), nullptr, &result) == 0 && result.is_speech && result.max_probability == original.max_probability);
            require(task.get());
            std::cout << argv[i] << ": speech=true max=" << result.max_probability << " frames=" << result.frames << '\n';
        }
        hl_vad_free(detector); detector = nullptr;
        options.api_version = 99;
        require(!hl_vad_create(&options));
        std::cout << checks << " real-model VAD C ABI checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        hl_vad_free(detector); std::cerr << error.what() << '\n'; return 1;
    }
}
