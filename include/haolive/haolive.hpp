#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace haolive {
inline constexpr int sample_rate = 16000;
inline constexpr int api_version = 1;

struct Options {
    std::string model_path;
    std::string backend = "cpu";
    int threads = 4;
    int step_ms = 160;
    int max_audio_ms = 30000;
    int max_tokens = 512;
};

struct Result {
    std::string text;
    bool final = false;
    bool updated = false;
    int audio_ms = 0;
};

// Every result contains the complete current utterance. Replace the preceding
// partial; only final text is immutable. A finalized session rejects more audio.
using PartialCallback = std::function<void(const std::string&)>;

class Session {
public:
    virtual ~Session() = default;
    virtual Result accept(const float* pcm, size_t count, bool final, const PartialCallback& on_partial = {}) = 0;
};

// Engine owns immutable model weights; sessions own all utterance state.
// Calls on one engine are serialized internally. Engine must outlive sessions.
class Engine {
public:
    virtual ~Engine() = default;
    virtual std::unique_ptr<Session> create_session() = 0;
    virtual std::string architecture() const = 0;
    virtual std::string backend() const = 0;
};

std::unique_ptr<Engine> load(const Options& options);

// Independent CPU VAD; no ASR engine or transcript is needed. Calls are
// stateless windows of float32 mono 16 kHz audio, not incremental appends.
struct VadOptions {
    std::string model_path;
    int max_audio_ms = 30000;
};
struct VadParams {
    float threshold = 0.5f;
    int min_speech_ms = 100;
};
struct VadResult {
    bool is_speech = false;
    bool sufficient_audio = false; // At least one complete 25 ms frame.
    float max_probability = 0;
    float mean_probability = 0;
    int speech_ms = 0; // All above-threshold frames, each representing 10 ms.
    int audio_ms = 0;
    int frames = 0;
};
class Vad {
public:
    virtual ~Vad() = default;
    // Concurrent calls on one detector are safe; no audio history is retained.
    virtual VadResult detect(const float* pcm, size_t count, const VadParams& params = {}) const = 0;
};
std::unique_ptr<Vad> load_vad(const VadOptions& options);
// Compiled backends with a device visible to ggml on this host.
std::vector<std::string> available_backends();
// Strict PCM16LE and WAV PCM16 mono / 16 kHz decoders; no external codecs.
std::vector<float> decode_pcm16(const void* bytes, size_t size);
std::vector<float> decode_wav(const void* bytes, size_t size);
} // namespace haolive
