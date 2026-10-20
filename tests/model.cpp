#include "haolive/haolive.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: haolive-model-tests MODEL.gguf BACKEND\n"; return 2; }
    hl_engine* engine = nullptr;
    hl_session* session = nullptr;
    try {
        auto require = [](bool ok) { if (!ok) throw std::runtime_error(hl_last_error()); };
        auto options = hl_default_options(); options.model_path = argv[1]; options.backend = argv[2];
        options.max_audio_ms = 1000;
        engine = hl_engine_create(&options); require(engine != nullptr);
        session = hl_session_create(engine); require(session != nullptr);
        // Sessions retain model resources across external engine release.
        hl_engine_free(engine); engine = nullptr;
        hl_result result{};
        float nan = std::numeric_limits<float>::quiet_NaN();
        require(hl_session_accept(session, &nan, 1, 0, &result) == -1 && result.text == nullptr);
        require(hl_session_accept(session, nullptr, 0, 2, &result) == -1);
        require(hl_session_accept(session, &nan, std::numeric_limits<size_t>::max(), 0, &result) == -1);
        require(hl_session_accept(session, nullptr, 0, 0, &result) == 0);
        require(result.audio_ms == 0 && !result.final && std::strlen(result.text) == 0);
        require(hl_session_accept(session, nullptr, 0, 1, &result) == 0);
        require(result.final && result.updated && result.audio_ms == 0 && std::strlen(result.text) == 0);
        require(hl_session_accept(session, nullptr, 0, 1, &result) == -1);
        hl_session_free(session); session = nullptr;
        std::cout << "C ABI model lifecycle checks passed: " << argv[2] << '\n';
        return 0;
    } catch (const std::exception& e) {
        hl_session_free(session); hl_engine_free(engine);
        std::cerr << e.what() << '\n'; return 1;
    }
}
