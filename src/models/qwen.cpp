// Qwen3-ASR GGUF equations/layout informed by the MIT CrispASR reference.
// This implementation deliberately has no CrispASR runtime dependency.
#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace haolive {
namespace {
using T = ggml_tensor*;
std::vector<uint32_t> codepoints(const std::string& value) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < value.size();) {
        auto c = static_cast<unsigned char>(value[i++]);
        int continuation = c < 128 ? 0 : c < 224 ? 1 : c < 240 ? 2 : 3;
        uint32_t cp = c & (continuation == 0 ? 127 : (1 << (6 - continuation)) - 1);
        for (int j = 0; j < continuation; ++j) {
            if (i == value.size()) throw std::runtime_error("invalid vocabulary UTF-8");
            cp = (cp << 6) | (static_cast<unsigned char>(value[i++]) & 63);
        }
        result.push_back(cp);
    }
    return result;
}

class Qwen;
class QSession final : public Session {
    Qwen& model;
    std::vector<float> audio;
    std::vector<int32_t> hypothesis;
    size_t last_decoded = 0;
    WhisperFeatureCache feature_cache;
    bool finished = false;
    std::string current;
public:
    explicit QSession(Qwen& model) : model(model) {}
    Result accept(const float*, size_t, bool, const PartialCallback&) override;
};

class Qwen final : public Model {
public:
    int mels, audio_dim, audio_heads, audio_head, audio_layers, dim, heads, kv_heads, head_dim, layers;
    int chunk_frames, max_positions;
    float theta, epsilon;
    std::vector<float> window, filters;
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int32_t> ids;
    explicit Qwen(const Options& opts) : Model(opts) {
        auto integer = [&](const std::string& key) { int v = weights.integer("qwen3asr." + key); if (v < 1 || v > 65536) throw std::runtime_error("invalid Qwen geometry: " + key); return v; };
        mels = integer("n_mels"); audio_dim = integer("audio.d_model"); audio_heads = integer("audio.n_heads");
        audio_head = integer("audio.head_dim"); audio_layers = integer("audio.n_layers");
        dim = integer("llm.d_model"); heads = integer("llm.n_heads"); kv_heads = integer("llm.n_kv_heads");
        head_dim = integer("llm.head_dim"); layers = integer("llm.n_layers"); max_positions = integer("llm.max_pos");
        theta = weights.number("qwen3asr.llm.rope_theta", 1e6f);
        epsilon = weights.number("qwen3asr.llm.rms_norm_eps", 1e-6f);
        chunk_frames = weights.integer("qwen3asr.n_window", 50) * 2;
        if (weights.integer("qwen3asr.sample_rate") != 16000 || weights.integer("qwen3asr.n_fft") != 400 ||
            weights.integer("qwen3asr.hop_length") != 160 || dim != integer("audio.proj_dim") ||
            audio_dim != audio_heads * audio_head || heads % kv_heads || chunk_frames != 100 ||
            audio_layers > 64 || layers > 80 || weights.str("qwen3asr.variant") == "raon-speech")
            throw std::runtime_error("unsupported Qwen3-ASR geometry");
        vocabulary = weights.strings("tokenizer.ggml.tokens");
        for (size_t i = 0; i < vocabulary.size(); ++i) ids.emplace(vocabulary[i], static_cast<int32_t>(i));
        for (const auto& token : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|audio_start|>", "<|audio_pad|>", "<|audio_end|>", "system", "user", "assistant", "Ċ"}) special(token);
        if (weights.get("token_embd.weight")->ne[0] != dim || weights.get("output.weight")->ne[1] != static_cast<int64_t>(vocabulary.size()))
            throw std::runtime_error("Qwen vocabulary/embedding mismatch");
        weights.upload(opts.model_path, device.primary);
        window = floats(weights.get("audio.mel_window")); filters = floats(weights.get("audio.mel_filters"));
    }
    std::string architecture() const override { return "qwen3asr"; }
    std::unique_ptr<Session> create_session() override { return std::make_unique<QSession>(*this); }
    int32_t special(const std::string& token) const {
        auto it = ids.find(token);
        if (it == ids.end()) throw std::runtime_error("required vocabulary token missing: " + token);
        return it->second;
    }
    T linear(Graph& graph, T x, const std::string& prefix, bool bias = false) {
        auto g = graph.ctx;
        T y = ggml_mul_mat(g, weights.get(prefix + ".weight"), x);
        return bias ? ggml_add(g, y, weights.get(prefix + ".bias")) : y;
    }
    T norm(Graph& graph, T x, const std::string& prefix, bool rms) {
        auto g = graph.ctx;
        x = rms ? ggml_rms_norm(g, x, epsilon) : ggml_norm(g, x, 1e-5f);
        x = ggml_mul(g, x, weights.get(prefix + ".weight"));
        return rms ? x : ggml_add(g, x, weights.get(prefix + ".bias"));
    }
    // Explicit grouped-head expansion ensures adjacent query heads share a KV
    // head (ggml's generic batch broadcasting would use a different order).
    T expand_heads(Graph& graph, T x, int time) {
        auto g = graph.ctx;
        x = ggml_cont(g, x);
        x = ggml_reshape_4d(g, x, head_dim, time, 1, kv_heads);
        x = ggml_repeat_4d(g, x, head_dim, time, heads / kv_heads, kv_heads);
        return ggml_reshape_3d(g, x, head_dim, time, heads);
    }
    T attention(Graph& graph, T q, T k, T v, T mask, int width, int nheads, int queries) {
        auto g = graph.ctx;
        T scores = ggml_mul_mat(g, k, q);
        scores = ggml_soft_max_ext(g, scores, mask, 1.0f / std::sqrt(static_cast<float>(width)), 0);
        T vt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
        T out = ggml_mul_mat(g, vt, scores);
        out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3));
        return ggml_reshape_2d(g, out, width * nheads, queries);
    }
    std::vector<float> encode(const std::vector<float>& audio, WhisperFeatureCache& cache) {
        auto mel = cache.compute(audio, window, filters, mels);
        if (!mel.frames) return {};
        int chunks = (mel.frames + chunk_frames - 1) / chunk_frames;
        const int frames_out = (chunk_frames + 7) / 8;
        int total = frames_out * (chunks - 1) + (mel.frames - chunk_frames * (chunks - 1) + 7) / 8;
        Graph graph(device);
        auto g = graph.ctx;
        T x = graph.input("mel", chunk_frames, mels, 1, chunks);
        for (int i = 1; i <= 3; ++i) {
            auto p = "audio.conv." + std::to_string(i);
            T b = weights.get(p + ".bias");
            x = ggml_conv_2d(g, weights.get(p + ".weight"), x, 2, 2, 1, 1, 1, 1);
            x = ggml_gelu_erf(g, ggml_add(g, x, ggml_cast(g, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32)));
        }
        int64_t freq = x->ne[1], channels = x->ne[2];
        x = ggml_cont(g, ggml_permute(g, x, 2, 0, 1, 3));
        // Flatten batch before quantized projection: no quantized broadcast.
        x = ggml_reshape_2d(g, x, freq * channels, frames_out * chunks);
        x = linear(graph, x, "audio.conv_out");
        if (auto bias = weights.optional("audio.conv_out.bias")) x = ggml_add(g, x, bias);
        x = ggml_reshape_3d(g, x, audio_dim, frames_out, chunks);
        x = ggml_add(g, x, graph.input("position", audio_dim, frames_out));
        x = ggml_reshape_2d(g, ggml_cont(g, x), audio_dim, frames_out * chunks);
        if (total != frames_out * chunks) {
            // Only the last convolution chunk has padding; valid frames form
            // the leading contiguous range after packing.
            x = ggml_cont(g, ggml_view_2d(g, x, audio_dim, total, x->nb[1], 0));
        }
        for (int layer = 0; layer < audio_layers; ++layer) {
            std::string p = "audio.blk." + std::to_string(layer) + ".";
            T h = norm(graph, x, p + "attn_norm", false);
            auto project = [&](const std::string& name) {
                T t = linear(graph, h, p + name, true);
                return ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, t, audio_head, audio_heads, total), 0, 2, 1, 3));
            };
            T a = attention(graph, project("attn_q"), project("attn_k"), project("attn_v"), nullptr, audio_head, audio_heads, total);
            x = ggml_add(g, x, linear(graph, a, p + "attn_out", true));
            h = norm(graph, x, p + "ffn_norm", false);
            h = ggml_gelu_erf(g, linear(graph, h, p + "ffn_up", true));
            x = ggml_add(g, x, linear(graph, h, p + "ffn_down", true));
        }
        x = norm(graph, x, "audio.ln_post", false);
        x = ggml_gelu_erf(g, linear(graph, x, "audio.proj1", true));
        graph.output(linear(graph, x, "audio.proj2", true), "encoded");
        graph.allocate();
        std::vector<float> packed(static_cast<size_t>(chunk_frames) * mels * chunks, 0);
        for (int c = 0; c < chunks; ++c)
            for (int m = 0; m < mels; ++m)
                for (int t = 0; t < std::min(chunk_frames, mel.frames - c * chunk_frames); ++t)
                    packed[t + chunk_frames * (m + mels * c)] = mel.data[m * mel.frames + c * chunk_frames + t];
        graph.set("mel", packed);
        std::vector<float> position(static_cast<size_t>(audio_dim) * frames_out);
        for (int t = 0; t < frames_out; ++t)
            for (int j = 0; j < audio_dim / 2; ++j) {
                float angle = t * std::exp(-std::log(10000.0f) * j / (audio_dim / 2 - 1));
                position[t * audio_dim + j] = std::sin(angle);
                position[t * audio_dim + j + audio_dim / 2] = std::cos(angle);
            }
        graph.set("position", position);
        graph.compute();
        auto out = graph.read("encoded");
        return out;
    }
    void embed(int32_t id, float* destination) {
        auto table = weights.get("token_embd.weight");
        if (id < 0 || id >= table->ne[1]) throw std::runtime_error("invalid token ID");
        std::vector<uint8_t> row(table->nb[1]);
        ggml_backend_tensor_get(table, row.data(), static_cast<size_t>(id) * table->nb[1], row.size());
        if (table->type == GGML_TYPE_F32) std::memcpy(destination, row.data(), dim * sizeof(float));
        else ggml_get_type_traits(table->type)->to_float(row.data(), destination, dim);
    }

    struct Cache {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<T> keys, values;
        ~Cache() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
    };
    std::vector<float> forward(const std::vector<float>& embeddings, int past, Cache& cache) {
        int count = static_cast<int>(embeddings.size() / dim), length = past + count;
        Graph graph(device);
        auto g = graph.ctx;
        T x = graph.input("embeddings", dim, count);
        T positions = graph.input("positions", count, 1, 1, 1, GGML_TYPE_I32);
        T mask = count > 1 ? graph.input("mask", length, count) : nullptr;
        for (int l = 0; l < layers; ++l) {
            auto p = "blk." + std::to_string(l) + ".";
            T h = norm(graph, x, p + "attn_norm", true);
            auto project = [&](const std::string& name, int nheads, bool rotate) {
                T t = ggml_reshape_3d(g, linear(graph, h, p + name), head_dim, nheads, count);
                if (rotate) {
                    t = norm(graph, t, p + name + "_norm", true);
                    t = ggml_rope_ext(g, t, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, max_positions, theta, 1, 0, 1, 32, 1);
                }
                return ggml_cont(g, ggml_permute(g, t, 0, 2, 1, 3));
            };
            T q = project("attn_q", heads, true), k = project("attn_k", kv_heads, true), v = project("attn_v", kv_heads, false);
            auto store = [&](T source, T storage) {
                // SET_ROWS preserves the full cache shape and gives the read
                // an explicit write dependency, including on asynchronous GPUs.
                T written = ggml_set_rows(g, storage, source, positions);
                T full = ggml_view_3d(g, written, head_dim, length, kv_heads, written->nb[1], written->nb[2], 0);
                return expand_heads(graph, full, length);
            };
            k = store(k, cache.keys[l]); v = store(v, cache.values[l]);
            T a = attention(graph, q, k, v, mask, head_dim, heads, count);
            x = ggml_add(g, x, linear(graph, a, p + "attn_output"));
            h = norm(graph, x, p + "ffn_norm", true);
            T ff = ggml_mul(g, ggml_silu(g, linear(graph, h, p + "ffn_gate")), linear(graph, h, p + "ffn_up"));
            x = ggml_add(g, x, linear(graph, ff, p + "ffn_down"));
        }
        x = norm(graph, x, "output_norm", true);
        if (count > 1) x = ggml_view_2d(g, x, dim, 1, x->nb[1], (count - 1) * x->nb[1]);
        graph.output(linear(graph, x, "output"), "logits");
        graph.allocate();
        graph.set("embeddings", embeddings);
        std::vector<int32_t> indices(count);
        for (int i = 0; i < count; ++i) indices[i] = past + i;
        graph.set("positions", indices.data(), indices.size() * sizeof(int32_t));
        if (mask) {
            std::vector<float> values(static_cast<size_t>(length) * count, 0);
            for (int i = 0; i < count; ++i)
                for (int j = past + i + 1; j < length; ++j) values[i * length + j] = -INFINITY;
            graph.set("mask", values);
        }
        graph.compute();
        auto lg = graph.read("logits");
        return lg;
    }
    std::vector<int32_t> generate(const std::vector<float>& audio, const std::vector<int32_t>& prefix, bool final, const PartialCallback& on_partial, WhisperFeatureCache& feature_cache) {
        auto encoded = encode(audio, feature_cache);
        int frames = static_cast<int>(encoded.size() / dim);
        if (!frames) return {};
        std::vector<int32_t> prompt;
        auto add = [&](const char* token) { prompt.push_back(special(token)); };
        add("<|im_start|>"); add("system"); add("Ċ"); add("<|im_end|>"); add("Ċ");
        add("<|im_start|>"); add("user"); add("Ċ"); add("<|audio_start|>");
        size_t audio_start = prompt.size();
        prompt.insert(prompt.end(), static_cast<size_t>(frames), special("<|audio_pad|>"));
        add("<|audio_end|>"); add("<|im_end|>"); add("Ċ"); add("<|im_start|>"); add("assistant"); add("Ċ");
        // Confucius can terminate while generating its own language header.
        // Seed the text boundary directly: bilingual recognition needs no LID
        // model, extra tokenizer, or hardcoded source language.
        add("<asr_text>");
        prompt.insert(prompt.end(), prefix.begin(), prefix.end());
        int capacity = static_cast<int>(prompt.size()) + options.max_tokens;
        if (capacity > max_positions) throw std::length_error("model context limit exceeded");
        Cache cache;
        cache.ctx = ggml_init({ggml_tensor_overhead() * static_cast<size_t>(2 * layers + 4), nullptr, true});
        if (!cache.ctx) throw std::runtime_error("KV metadata allocation failed");
        for (int l = 0; l < layers; ++l) {
            cache.keys.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, capacity, kv_heads));
            cache.values.push_back(ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, capacity, kv_heads));
        }
        cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, device.primary);
        if (!cache.buffer) throw std::runtime_error("KV cache allocation failed");
        std::vector<float> embeddings(prompt.size() * dim);
        for (size_t i = 0; i < prompt.size(); ++i) embed(prompt[i], embeddings.data() + i * dim);
        std::copy(encoded.begin(), encoded.end(), embeddings.begin() + static_cast<ptrdiff_t>(audio_start * dim));
        auto logits = forward(embeddings, 0, cache);
        std::vector<int32_t> result = prefix;
        int past = static_cast<int>(prompt.size());
        // A draft is resumable. Return it promptly instead of letting a
        // hallucinated long continuation block the audio FIFO for 512 tokens.
        const int budget = final ? options.max_tokens : std::min(32, options.max_tokens);
        for (int n = 0; n < budget; ++n) {
            int32_t token = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (token == special("<|im_end|>") || token == special("<|endoftext|>")) return result;
            // The boundary is control state, never an assistant-prefix token.
            // Keeping it makes clean_asr_text hide all subsequent speech.
            if (!append_qwen_token(result, token, vocabulary)) return result;
            if (on_partial) {
                std::string bytes;
                for (auto id : result) if (vocabulary[id].rfind("<|", 0) != 0) bytes += byte_token(vocabulary[id]);
                auto text = clean_asr_text("<asr_text>" + bytes);
                if (!text.empty()) on_partial(text);
            }
            if (n + 1 == budget) {
                if (final) throw std::runtime_error("generation token limit reached");
                return result;
            }
            embeddings.resize(dim); embed(token, embeddings.data());
            logits = forward(embeddings, past++, cache);
        }
        return result;
    }
};

Result QSession::accept(const float* pcm, size_t count, bool final, const PartialCallback& on_partial) {
    std::lock_guard<std::mutex> lock(model.mutex);
    validate_audio(pcm, count, audio.size(), model.options.max_audio_ms, finished);
    if (count) audio.insert(audio.end(), pcm, pcm + count);
    std::string before = current;
    if (audio.size() >= 160 && (final || audio.size() - last_decoded >= static_cast<size_t>(model.options.step_ms) * 16)) {
        // Re-evaluate the recent suffix against all utterance audio. Keep the
        // rest as an assistant prefix, rather than duplicating the reference's
        // turn scheduler or promising append-only partial hypotheses.
        auto prefix = hypothesis;
        if (prefix.size() > 4) prefix.resize(prefix.size() - 4); else prefix.clear();
        hypothesis = model.generate(audio, prefix, final, [&](const std::string& text) {
            // Decoder rollback is internal. Do not flash the shortened prefix
            // while those tokens are being regenerated; final hypotheses can
            // still revise the previous draft.
            if (on_partial && qwen_progress_visible(current, text)) on_partial(text);
        }, feature_cache);
        std::string bytes;
        for (int32_t token : hypothesis) {
            auto piece = model.vocabulary[token];
            if (piece.rfind("<|", 0) != 0) bytes += byte_token(piece);
        }
        current = clean_asr_text("<asr_text>" + bytes);
        last_decoded = audio.size();
    }
    finished = final;
    return {current, final, final || current != before, static_cast<int>(audio.size() / 16)};
}
} // namespace

bool qwen_progress_visible(const std::string& previous, const std::string& candidate) {
    return !candidate.empty() && candidate.size() >= previous.size() && candidate != previous;
}
bool append_qwen_token(std::vector<int32_t>& tokens, int32_t token, const std::vector<std::string>& vocabulary) {
    if (token < 0 || static_cast<size_t>(token) >= vocabulary.size()) throw std::runtime_error("invalid Qwen token ID");
    if (byte_token(vocabulary[token]).find('|') != std::string::npos) return false;
    tokens.push_back(token);
    return true;
}
std::string byte_token(const std::string& encoded) {
    // Invert GPT-2's byte-to-Unicode vocabulary representation.
    static const std::unordered_map<uint32_t, uint8_t> inverse = [] {
        std::unordered_map<uint32_t, uint8_t> result;
        uint32_t extended = 256;
        for (int b = 0; b < 256; ++b) {
            bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            result[direct ? static_cast<uint32_t>(b) : extended++] = static_cast<uint8_t>(b);
        }
        return result;
    }();
    std::string result;
    for (uint32_t cp : codepoints(encoded)) {
        auto found = inverse.find(cp);
        if (found == inverse.end()) throw std::runtime_error("invalid byte vocabulary token");
        result += static_cast<char>(found->second);
    }
    return result;
}
std::string clean_asr_text(const std::string& text) {
    const auto marker = text.find("<asr_text>");
    if (marker == std::string::npos) return {};
    auto result = text.substr(marker + 10);
    if (auto pipe = result.find('|'); pipe != std::string::npos) result.resize(pipe);
    // Token-budgeted drafts can end partway through a UTF-8 character.
    // Keep those bytes in hypothesis, but do not send invalid JSON text.
    if (!result.empty()) {
        size_t start = result.size() - 1;
        while (start > 0 && (static_cast<unsigned char>(result[start]) & 0xc0) == 0x80) --start;
        auto lead = static_cast<unsigned char>(result[start]);
        size_t width = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        if (result.size() - start < width) result.resize(start);
    }
    size_t at;
    while ((at = result.find("<non_speech>")) != std::string::npos) result.erase(at, 12);
    auto begin = result.find_first_not_of(" \r\n\t");
    if (begin == std::string::npos) return {};
    return result.substr(begin, result.find_last_not_of(" \r\n\t") - begin + 1);
}
std::unique_ptr<Engine> make_qwen(const Options& opts) { return std::make_unique<Qwen>(opts); }
} // namespace haolive
