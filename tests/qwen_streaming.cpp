// Real-model regression: partial budgets must yield, while final remains strict.
#include "haolive/haolive.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: haolive-qwen-streaming-tests MODEL.gguf BACKEND SPEECH.wav\n";
        return 2;
    }
    try {
        std::ifstream file(argv[3], std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(file)), {});
        auto audio = haolive::decode_wav(bytes.data(), bytes.size());
        if (audio.size() < 6 * 16000) throw std::runtime_error("need at least six seconds of speech");
        haolive::Options options;
        options.model_path = argv[1]; options.backend = argv[2]; options.max_tokens = 1;
        {
            auto normal = options;
            normal.max_tokens = 512;
            auto model = haolive::load(normal);
            auto bounded = model->create_session();
            int callbacks = 0;
            const auto draft = bounded->accept(audio.data(), 3 * 16000, false,
                [&](const std::string&) { ++callbacks; });
            if (draft.text.empty() || callbacks < 1 || callbacks > 32)
                throw std::runtime_error("normal partial must yield within 32 token callbacks");
            const auto complete = bounded->accept(audio.data() + 3 * 16000, 3 * 16000, true);
            if (!complete.final || complete.audio_ms != 6000 || complete.text.empty())
                throw std::runtime_error("bounded partial must not prevent full finalization");
        }
        auto engine = haolive::load(options);
        if (engine->architecture() != "qwen3asr") throw std::runtime_error("Qwen model required");
        auto session = engine->create_session();
        std::vector<std::string> progress;
        auto first = session->accept(audio.data(), 3 * 16000, false,
            [&](const std::string& text) { progress.push_back(text); });
        if (progress.empty() || progress.back() != first.text) throw std::runtime_error("token callback must precede returned draft");
        if (first.final || first.text.empty()) throw std::runtime_error("budgeted partial must yield speech");
        auto next = session->accept(audio.data() + 3 * 16000, 3 * 16000, false);
        if (next.final || next.audio_ms != 6000) throw std::runtime_error("budgeted session must accept more audio");
        bool rejected = false;
        try { session->accept(nullptr, 0, true); }
        catch (const std::runtime_error& e) {
            rejected = std::string(e.what()) == "generation token limit reached";
        }
        if (!rejected) throw std::runtime_error("final token exhaustion must remain an error");
        std::cout << "Qwen partial budget, continued capture and strict final checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
