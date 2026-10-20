// FireRed DFSMN equations/GGUF layout informed by the MIT CrispASR reference.
// Independent stateless VAD, with no ASR, GPU or reference runtime dependency.
#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#ifdef __APPLE__
#include <AvailabilityMacros.h>
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 130300
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

namespace haolive {
VadResult classify_vad(const std::vector<float>& probabilities, size_t samples, const VadParams& params) {
    if (!std::isfinite(params.threshold) || params.threshold <= 0 || params.threshold >= 1 ||
        params.min_speech_ms < 1 || params.min_speech_ms > 30000)
        throw std::invalid_argument("VAD threshold must be in (0,1), min_speech_ms in [1,30000]");
    VadResult result;
    result.audio_ms = static_cast<int>(samples / 16);
    result.frames = static_cast<int>(probabilities.size());
    result.sufficient_audio = !probabilities.empty();
    int consecutive = 0, longest = 0;
    double sum = 0;
    for (float probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0 || probability > 1)
            throw std::runtime_error("invalid VAD probability");
        result.max_probability = std::max(result.max_probability, probability);
        sum += probability;
        if (probability >= params.threshold) { ++consecutive; result.speech_ms += 10; }
        else consecutive = 0;
        longest = std::max(longest, consecutive);
    }
    if (!probabilities.empty()) result.mean_probability = static_cast<float>(sum / probabilities.size());
    result.is_speech = longest >= (params.min_speech_ms + 9) / 10;
    return result;
}
namespace {
struct Dense {
    int input = 0, output = 0;
    std::vector<float> weight, bias;
    std::vector<float> apply(const std::vector<float>& x, bool relu) const {
        const int frames = static_cast<int>(x.size() / input);
        std::vector<float> y(static_cast<size_t>(frames) * output);
#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, frames, output, input,
                    1, x.data(), input, weight.data(), input, 0, y.data(), output);
#else
        for (int t = 0; t < frames; ++t)
            for (int o = 0; o < output; ++o) {
                float sum = 0;
                for (int i = 0; i < input; ++i) sum += x[t * input + i] * weight[o * input + i];
                y[t * output + o] = sum;
            }
#endif
        for (int t = 0; t < frames; ++t)
            for (int o = 0; o < output; ++o) {
                auto& v = y[t * output + o];
                if (!bias.empty()) v += bias[o];
                if (relu) v = std::max(0.0f, v);
            }
        return y;
    }
};
struct Memory {
    std::vector<float> back, ahead; // Kernel-major for contiguous channel loops.
    std::vector<float> apply(const std::vector<float>& input) const {
        constexpr int channels = 128, taps = 20;
        int frames = static_cast<int>(input.size() / channels);
        auto out = input;
        for (int t = 0; t < frames; ++t)
            for (int k = 0; k < taps; ++k) {
                int previous = t - (taps - 1 - k), next = t + k + 1;
                for (int c = 0; c < channels; ++c) {
                    if (previous >= 0) out[t * channels + c] += back[k * channels + c] * input[previous * channels + c];
                    if (next < frames) out[t * channels + c] += ahead[k * channels + c] * input[next * channels + c];
                }
            }
        return out;
    }
};
class FireRed final : public Vad {
    int max_audio_ms;
    Dense first, second, dnn, output;
    Memory first_memory;
    struct Block { Dense up, down; Memory memory; };
    std::vector<Block> blocks;
    std::vector<float> mean, deviation;
public:
    explicit FireRed(const VadOptions& options) : max_audio_ms(options.max_audio_ms) {
        Options cpu_options; cpu_options.backend = "cpu";
        Device cpu(cpu_options);
        Weights weights(options.model_path);
        if (weights.str("general.architecture") != "firered-vad") throw std::invalid_argument("VAD requires firered-vad GGUF architecture");
        for (const auto& entry : std::map<std::string, int>{{"R", 8}, {"H", 256}, {"P", 128}, {"N1", 20},
                {"N2", 20}, {"S1", 1}, {"S2", 1}, {"idim", 80}, {"odim", 1}})
            if (weights.integer("firered_vad." + entry.first) != entry.second) throw std::runtime_error("unsupported FireRed VAD geometry");
        weights.upload(options.model_path, cpu.primary);
        auto read = [&](const std::string& name, int a, int b = 1, int c = 1) {
            auto tensor = weights.get(name);
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != a || tensor->ne[1] != b || tensor->ne[2] != c || tensor->ne[3] != 1)
                throw std::runtime_error("invalid FireRed VAD tensor: " + name);
            auto values = floats(tensor);
            for (float v : values) if (!std::isfinite(v)) throw std::runtime_error("nonfinite VAD weights");
            return values;
        };
        auto dense = [&](const std::string& prefix, int input, int out, bool bias) {
            return Dense{input, out, read(prefix + ".weight", input, out), bias ? read(prefix + ".bias", out) : std::vector<float>{}};
        };
        auto memory = [&](const std::string& prefix) {
            auto reorder = [&](const std::string& name) {
                auto source = read(name, 20, 1, 128); std::vector<float> reordered(source.size());
                for (int c = 0; c < 128; ++c) for (int k = 0; k < 20; ++k) reordered[k * 128 + c] = source[c * 20 + k];
                return reordered;
            };
            return Memory{reorder(prefix + ".lookback_filter.weight"), reorder(prefix + ".lookahead_filter.weight")};
        };
        first = dense("dfsmn.fc1.0", 80, 256, true); second = dense("dfsmn.fc2.0", 256, 128, true);
        first_memory = memory("dfsmn.fsmn1");
        for (int i = 0; i < 7; ++i) {
            auto p = "dfsmn.fsmns." + std::to_string(i);
            blocks.push_back({dense(p + ".fc1.0", 128, 256, true), dense(p + ".fc2", 256, 128, false), memory(p + ".fsmn")});
        }
        dnn = dense("dfsmn.dnns.0", 128, 256, true); output = dense("out", 256, 1, true);
        mean = read("cmvn.mean", 80); deviation = read("cmvn.std", 80);
        for (float value : deviation) if (value <= 0) throw std::runtime_error("invalid VAD CMVN standard deviation");
    }
    VadResult detect(const float* pcm, size_t count, const VadParams& params) const override {
        validate_audio(pcm, count, 0, max_audio_ms, false);
        // Validate configuration even for an empty/short window.
        auto empty = classify_vad({}, count, params);
        if (count < 400) return empty;
        auto features = firered_features(pcm, count);
        for (int t = 0; t < features.frames; ++t)
            for (int c = 0; c < 80; ++c) features.data[t * 80 + c] = (features.data[t * 80 + c] - mean[c]) / deviation[c];
        auto state = first_memory.apply(second.apply(first.apply(features.data, true), true));
        for (const auto& block : blocks) {
            auto next = block.memory.apply(block.down.apply(block.up.apply(state, true), false));
            for (size_t i = 0; i < state.size(); ++i) state[i] += next[i];
        }
        auto logits = output.apply(dnn.apply(state, true), false);
        for (auto& v : logits) v = v >= 0 ? 1 / (1 + std::exp(-v)) : std::exp(v) / (1 + std::exp(v));
        return classify_vad(logits, count, params);
    }
};
} // namespace
std::unique_ptr<Vad> load_vad(const VadOptions& options) {
    if (options.model_path.empty() || options.max_audio_ms < 1000 || options.max_audio_ms > 30000)
        throw std::invalid_argument("invalid VAD options");
    return std::make_unique<FireRed>(options);
}
} // namespace haolive
