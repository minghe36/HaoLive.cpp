#include "haolive/haolive.h"
#include "haolive/haolive.hpp"
#include <stdexcept>

struct hl_engine { std::shared_ptr<haolive::Engine> value; };
struct hl_vad { std::unique_ptr<haolive::Vad> value; };
struct hl_session {
    std::shared_ptr<haolive::Engine> engine;
    std::unique_ptr<haolive::Session> value;
    std::string text;
};
namespace {
thread_local std::string last_error;
template<class F> bool guarded(F fn) noexcept {
    last_error.clear();
    try { fn(); return true; }
    catch (const std::exception& e) { last_error = e.what(); }
    catch (...) { last_error = "unknown inference error"; }
    return false;
}
}
extern "C" {
hl_options hl_default_options(void) { return {HL_API_VERSION, nullptr, "cpu", 4, 160, 30000, 512}; }
hl_engine* hl_engine_create(const hl_options* options) {
    hl_engine* result = nullptr;
    guarded([&] {
        if (!options || options->api_version != HL_API_VERSION || !options->model_path || !options->backend)
            throw std::invalid_argument("invalid options or API version");
        auto engine = haolive::load({options->model_path, options->backend, options->threads, options->step_ms, options->max_audio_ms, options->max_tokens});
        result = new hl_engine{std::move(engine)};
    });
    return result;
}
void hl_engine_free(hl_engine* engine) { delete engine; }
hl_session* hl_session_create(hl_engine* engine) {
    hl_session* result = nullptr;
    guarded([&] {
        if (!engine) throw std::invalid_argument("null engine");
        auto session = engine->value->create_session();
        result = new hl_session{engine->value, std::move(session), {}};
    });
    return result;
}
void hl_session_free(hl_session* session) { delete session; }
int hl_session_accept(hl_session* session, const float* pcm, size_t count, int final, hl_result* result) {
    if (result) *result = {nullptr, 0, 0, 0};
    return guarded([&] {
        if (!session || !result || (final != 0 && final != 1)) throw std::invalid_argument("invalid session, result or final flag");
        auto r = session->value->accept(pcm, count, final != 0);
        session->text = std::move(r.text);
        *result = {session->text.c_str(), r.final ? 1 : 0, r.updated ? 1 : 0, r.audio_ms};
    }) ? 0 : -1;
}
const char* hl_last_error(void) { return last_error.c_str(); }
hl_vad_options hl_default_vad_options(void) { return {HL_API_VERSION, nullptr, 30000}; }
hl_vad_params hl_default_vad_params(void) { return {0.5f, 100}; }
hl_vad* hl_vad_create(const hl_vad_options* options) {
    hl_vad* result = nullptr;
    guarded([&] {
        if (!options || options->api_version != HL_API_VERSION || !options->model_path)
            throw std::invalid_argument("invalid VAD options or API version");
        auto detector = haolive::load_vad({options->model_path, options->max_audio_ms});
        result = new hl_vad{std::move(detector)};
    });
    return result;
}
void hl_vad_free(hl_vad* vad) { delete vad; }
int hl_vad_detect(hl_vad* vad, const float* pcm, size_t count,
                  const hl_vad_params* params, hl_vad_result* result) {
    if (result) *result = {};
    return guarded([&] {
        if (!vad || !result) throw std::invalid_argument("null VAD or result");
        auto p = params ? haolive::VadParams{params->threshold, params->min_speech_ms} : haolive::VadParams{};
        auto r = vad->value->detect(pcm, count, p);
        *result = {r.is_speech ? 1 : 0, r.sufficient_audio ? 1 : 0, r.max_probability,
                   r.mean_probability, r.speech_ms, r.audio_ms, r.frames};
    }) ? 0 : -1;
}
}
