#include "haolive/haolive.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: haolive-xasr-streaming-tests MODEL.gguf BACKEND SPEECH.wav\n";
        return 2;
    }
    try {
        std::ifstream file(argv[3], std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(file)), {});
        auto audio = haolive::decode_wav(bytes.data(), bytes.size());
        audio.resize(std::min(audio.size(), static_cast<size_t>(8 * 16000)));
        if (audio.size() < 3 * 16000) throw std::runtime_error("need three seconds of speech");
        haolive::Options options;
        options.model_path = argv[1]; options.backend = argv[2];
        auto engine = haolive::load(options);
        if (engine->architecture() != "xasr") throw std::runtime_error("X-ASR model required");
        auto reference = engine->create_session();
        const auto expected = reference->accept(audio.data(), audio.size(), true).text;
        if (expected.empty()) throw std::runtime_error("speech fixture produced no reference text");
        auto stream = engine->create_session();
        auto irregular = engine->create_session();
        auto low_latency = engine->create_session();
        for (size_t at = 0; at < audio.size(); at += 320)
            low_latency->accept(audio.data() + at, std::min(static_cast<size_t>(320), audio.size() - at), false);
        if (low_latency->accept(nullptr, 0, true).text != expected)
            throw std::runtime_error("20 ms packet boundaries changed final transcript");
        size_t irregular_at = 0;
        int updates = 0, first_audio_ms = -1;
        double first_compute_ms = 0, total_compute_ms = 0;
        for (size_t at = 0; at < audio.size(); at += 2560) {
            const size_t count = std::min(static_cast<size_t>(2560), audio.size() - at);
            const auto start = std::chrono::steady_clock::now();
            const auto result = stream->accept(audio.data() + at, count, false);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            total_compute_ms += elapsed;
            if (result.final || result.audio_ms != static_cast<int>((at + count) / 16))
                throw std::runtime_error("invalid streaming progress");
            if (result.updated) {
                updates++;
                if (first_audio_ms < 0) { first_audio_ms = result.audio_ms; first_compute_ms = total_compute_ms; }
            }
            // Interleave another session with different transport boundaries to expose shared caches.
            const size_t other = std::min(static_cast<size_t>(3711), audio.size() - irregular_at);
            if (other) {
                irregular->accept(audio.data() + irregular_at, other, false);
                irregular_at += other;
            }
        }
        if (updates < 2) throw std::runtime_error("expected multiple partial updates before final: count=" + std::to_string(updates) + " reference=" + expected);
        const auto final = stream->accept(nullptr, 0, true);
        const auto other_final = irregular->accept(nullptr, 0, true);
        if (final.text != expected || other_final.text != expected)
            throw std::runtime_error("offline, 160 ms and irregular interleaved sessions differ");
        std::cout << "backend=" << argv[2] << " audio_ms=" << audio.size() / 16
                  << " updates=" << updates << " first_update_audio_ms=" << first_audio_ms
                  << " first_update_compute_ms=" << first_compute_ms
                  << " total_stream_compute_ms=" << total_compute_ms
                  << "\ntext=" << final.text << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
