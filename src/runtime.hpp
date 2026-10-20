#pragma once
#include "haolive/haolive.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace haolive {
struct Device {
    ggml_backend_t primary = nullptr, cpu = nullptr;
    std::string name;
    explicit Device(const Options& options);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

struct Weights {
    gguf_context* meta = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    explicit Weights(const std::string& path);
    ~Weights();
    void upload(const std::string& path, ggml_backend_t backend);
    std::string str(const std::string& key, const std::string& fallback = "") const;
    int integer(const std::string& key, int fallback = 0) const;
    float number(const std::string& key, float fallback) const;
    std::vector<int> integers(const std::string& key, bool allow_zero = false) const;
    std::vector<std::string> strings(const std::string& key) const;
    ggml_tensor* get(const std::string& name) const;
    ggml_tensor* optional(const std::string& name) const;
};
std::vector<float> floats(ggml_tensor* tensor);

struct Graph {
    Device& device;
    ggml_context* ctx;
    ggml_cgraph* graph;
    ggml_backend_sched_t sched = nullptr;
    explicit Graph(Device& device, int nodes = 32768);
    ~Graph();
    ggml_tensor* input(const std::string& name, int64_t a, int64_t b = 1,
                       int64_t c = 1, int64_t d = 1, ggml_type type = GGML_TYPE_F32);
    void output(ggml_tensor* tensor, const std::string& name);
    void allocate();
    void set(const std::string& name, const void* data, size_t bytes);
    void set(const std::string& name, const std::vector<float>& data);
    std::vector<float> read(const std::string& name);
    void compute();
};

struct Features {
    std::vector<float> data;
    int frames = 0;
};
Features kaldi_features(const std::vector<float>& pcm, bool final, int first_frame = 0);
// A session-local cache for append-only PCM and unchanged window/filter weights.
// Keep unnormalized log-mels: a new global maximum must renormalize old frames.
struct WhisperFeatureCache {
    std::vector<float> raw;
    size_t stable_frames = 0, previous_samples = 0;
    int cached_mels = 0;
    Features compute(const std::vector<float>& pcm, const std::vector<float>& window,
                     const std::vector<float>& filters, int mels);
};
Features whisper_features(const std::vector<float>& pcm, const std::vector<float>& window,
                           const std::vector<float>& filters, int mels);
std::string byte_token(const std::string& encoded);
bool qwen_progress_visible(const std::string&, const std::string&);
bool append_qwen_token(std::vector<int32_t>&, int32_t, const std::vector<std::string>&);
std::string clean_asr_text(const std::string& text);

class Model : public Engine {
public:
    Options options;
    Device device;
    Weights weights;
    std::mutex mutex;
    explicit Model(const Options& opts) : options(opts), device(opts), weights(opts.model_path) {}
    std::string backend() const override { return device.name; }
};
std::unique_ptr<Engine> make_xasr(const Options&);
std::unique_ptr<Engine> make_qwen(const Options&);
std::unique_ptr<Engine> make_moonshine(const Options&);
std::unique_ptr<Engine> make_nemotron(const Options&);
std::vector<std::string> parse_moonshine_tokenizer(const uint8_t*, size_t, int vocabulary);
std::vector<std::string> moonshine_vocabulary();
std::string sentencepiece_text(const std::vector<std::string>&, const std::vector<int32_t>&);
Features nemo_features(const std::vector<float>&, const std::vector<float>& window, const std::vector<float>& filters, int mels);
Features firered_features(const float*, size_t count);
VadResult classify_vad(const std::vector<float>& probabilities, size_t samples, const VadParams&);

// Common validation runs before any model or session state is changed.
void validate_audio(const float* pcm, size_t count, size_t previous, int max_ms, bool finished);
} // namespace haolive
