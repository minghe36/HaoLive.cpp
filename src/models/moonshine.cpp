// Moonshine Streaming tensor equations/layout informed by the MIT CrispASR
// reference. Runtime, graph ownership and session policy are independent.
#include "ops.hpp"
#include <algorithm>
#include <numeric>

namespace haolive {
std::vector<std::string> parse_moonshine_tokenizer(const uint8_t* data, size_t size, int vocabulary) {
    if (!data || vocabulary < 1 || vocabulary > 65536) throw std::runtime_error("invalid Moonshine vocabulary");
    size_t offset = 0;
    auto byte = [&]() -> int {
        if (offset == size) throw std::runtime_error("truncated Moonshine tokenizer length");
        return data[offset++];
    };
    std::vector<std::string> result;
    for (int i = 0; i < vocabulary; ++i) {
        int first = byte();
        size_t length = first < 128 ? first : byte() * 128 + first - 128;
        if (length > size - offset) throw std::runtime_error("truncated Moonshine tokenizer piece");
        result.emplace_back(reinterpret_cast<const char*>(data + offset), length);
        offset += length;
    }
    if (offset != size) throw std::runtime_error("Moonshine tokenizer vocabulary mismatch");
    return result;
}
std::vector<std::string> moonshine_vocabulary() {
    static const uint8_t compact[] = {
#include "moonshine_tokens.inc"
    };
    return parse_moonshine_tokenizer(compact, sizeof(compact), 32768);
}
std::string sentencepiece_text(const std::vector<std::string>& vocabulary, const std::vector<int32_t>& tokens) {
    std::string result;
    for (int32_t id : tokens) {
        if (id < 0 || id >= static_cast<int32_t>(vocabulary.size())) throw std::runtime_error("invalid decoded token");
        const auto& piece = vocabulary[id];
        if (!(piece.size() > 2 && piece.front() == '<' && piece.back() == '>')) result += piece;
    }
    size_t at = 0;
    while ((at = result.find("▁", at)) != std::string::npos) { result.replace(at, 3, " "); ++at; }
    auto first = result.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : result.substr(first, result.find_last_not_of(" \t\r\n") - first + 1);
}
namespace {
class Moonshine;
class MoonSession final : public Session {
    Moonshine& model;
    std::vector<float> audio;
    size_t decoded = 0;
    bool finished = false;
    std::string current;
public:
    explicit MoonSession(Moonshine& m) : model(m) {}
    Result accept(const float*, size_t, bool, const PartialCallback&) override;
};
class Moonshine final : public OpsModel {
public:
    int enc_dim, enc_layers, enc_heads, enc_head, dim, layers, nheads, head, intermediate, positions, bos, eos;
    float theta, compression;
    std::vector<int> left, right;
    std::vector<std::string> vocabulary;
    explicit Moonshine(const Options& opts) : OpsModel(opts) {
        auto n = [&](const std::string& key) { int v = weights.integer("moonshine_streaming." + key); if (v < 1 || v > 65536) throw std::runtime_error("invalid Moonshine geometry: " + key); return v; };
        enc_dim = n("encoder.embedding_length"); enc_layers = n("encoder.block_count");
        enc_heads = n("encoder.attention.head_count"); enc_head = n("encoder.attention.head_dim");
        dim = n("decoder.embedding_length"); layers = n("decoder.block_count");
        nheads = n("decoder.attention.head_count"); head = n("decoder.attention.head_dim");
        intermediate = n("decoder.feed_forward_length");
        positions = n("max_position_embeddings"); bos = n("bos_token_id"); eos = n("eos_token_id");
        theta = weights.number("moonshine_streaming.rope.freq_base", 10000);
        if (enc_layers > 64 || layers > 64 || head != 64 || nheads * head != dim ||
            n("encoder.attention.head_count_kv") != enc_heads || n("decoder.attention.head_count_kv") != nheads ||
            n("audio.sample_rate") != 16000 || weights.number("moonshine_streaming.audio.frame_ms", 5) != 5 ||
            weights.number("moonshine_streaming.decoder.partial_rotary_factor", .5f) != .5f)
            throw std::runtime_error("unsupported Moonshine geometry");
        for (int i = 0; i < enc_layers; ++i) {
            auto p = "moonshine_streaming.encoder.layers." + std::to_string(i) + ".";
            left.push_back(weights.integer(p + "window_left")); right.push_back(weights.integer(p + "window_right"));
        }
        int vocab = n("vocab_size");
        // The supplied checkpoint omits vocabulary metadata. Embed its tokenizer
        // to keep deployment to one executable plus the selected GGUF.
        if (gguf_find_key(weights.meta, "tokenizer.ggml.tokens") >= 0)
            vocabulary = weights.strings("tokenizer.ggml.tokens");
        else {
            if (enc_dim != 768 || dim != 640 || enc_layers != 14 || layers != 14 || vocab != 32768)
                throw std::runtime_error("Moonshine variant requires embedded GGUF vocabulary");
            vocabulary = moonshine_vocabulary();
        }
        if (static_cast<int>(vocabulary.size()) != vocab || bos >= vocab || eos >= vocab || weights.get("decoder.embed_tokens.weight")->ne[0] != dim ||
            weights.get("decoder.output.weight")->ne[1] != vocab) throw std::runtime_error("Moonshine vocabulary mismatch");
        weights.upload(opts.model_path, device.primary);
        compression = std::exp(floats(weights.get("encoder.embedder.log_k")).at(0));
    }
    std::string architecture() const override { return "moonshine_streaming"; }
    std::unique_ptr<Session> create_session() override { return std::make_unique<MoonSession>(*this); }
    std::vector<float> encode(const std::vector<float>& audio) {
        int frames = static_cast<int>(audio.size() / 80);
        std::vector<float> normalized(static_cast<size_t>(frames) * 80);
        for (int t = 0; t < frames; ++t) {
            const float* src = audio.data() + t * 80;
            float mean = std::accumulate(src, src + 80, 0.0f) / 80, variance = 0;
            for (int j = 0; j < 80; ++j) variance += (src[j] - mean) * (src[j] - mean);
            float scale = 1 / std::sqrt(variance / 80 + 1e-8f);
            for (int j = 0; j < 80; ++j) normalized[t * 80 + j] = std::asinh(compression * (src[j] - mean) * scale);
        }
        Graph graph(device); auto g = graph.ctx;
        auto x = ggml_silu(g, linear(graph, graph.input("wave", 80, frames), "encoder.embedder.linear"));
        x = ggml_cont(g, ggml_transpose(g, x));
        for (int i = 1; i <= 2; ++i) {
            auto p = "encoder.embedder.conv" + std::to_string(i);
            x = ggml_pad_ext(g, x, 4, 0, 0, 0, 0, 0, 0, 0);
            x = ggml_conv_1d(g, weights.get(p + ".weight"), x, 2, 0, 1);
            auto b = weights.get(p + ".bias");
            x = ggml_add(g, x, ggml_reshape_2d(g, b, 1, b->ne[0]));
            if (i == 1) x = ggml_silu(g, x);
        }
        x = ggml_cont(g, ggml_transpose(g, x));
        int time = static_cast<int>(x->ne[1]);
        if (time > positions) throw std::length_error("Moonshine position limit exceeded");
        for (int l = 0; l < enc_layers; ++l) {
            auto p = "encoder.layers." + std::to_string(l) + ".";
            auto h = norm(graph, x, p + "attn_norm");
            auto project = [&](const char* name) { return heads(graph, linear(graph, h, p + "attn." + name), enc_head, enc_heads); };
            auto a = attend(graph, project("q"), project("k"), project("v"), graph.input("mask" + std::to_string(l), time, time));
            x = ggml_add(g, x, linear(graph, a, p + "attn.o"));
            h = ggml_gelu_erf(g, linear(graph, norm(graph, x, p + "ffn_norm"), p + "ffn.fc1"));
            x = ggml_add(g, x, linear(graph, h, p + "ffn.fc2"));
        }
        graph.output(norm(graph, x, "encoder.output_norm"), "encoded"); graph.allocate();
        graph.set("wave", normalized);
        for (int l = 0; l < enc_layers; ++l) {
            std::vector<float> mask(static_cast<size_t>(time) * time, -INFINITY);
            for (int q = 0; q < time; ++q)
                for (int k = std::max(0, q - left[l]); k <= std::min(time - 1, q + right[l]); ++k) mask[q * time + k] = 0;
            graph.set("mask" + std::to_string(l), mask);
        }
        graph.compute(); return graph.read("encoded");
    }
    std::string generate(const std::vector<float>& audio) {
        auto encoded = encode(audio); int time = static_cast<int>(encoded.size() / enc_dim);
        TensorArena state(layers * 4);
        std::vector<Tensor> keys, values, cross_k, cross_v;
        for (int l = 0; l < layers; ++l) {
            keys.push_back(ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32, head, options.max_tokens, nheads));
            values.push_back(ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32, head, options.max_tokens, nheads));
            cross_k.push_back(ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32, head, time, nheads));
            cross_v.push_back(ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32, head, time, nheads));
        }
        state.allocate(device);
        {
            Graph graph(device); auto g = graph.ctx;
            auto pos = graph.input("pos", time, 1, 1, 1, GGML_TYPE_I32);
            auto x = ggml_add(g, graph.input("enc", enc_dim, time), ggml_get_rows(g, weights.get("decoder.pos_emb.weight"), pos));
            x = linear(graph, x, "decoder.enc_proj");
            for (int l = 0; l < layers; ++l) {
                auto p = "decoder.layers." + std::to_string(l) + ".cross_attn.";
                graph.output(ggml_cpy(g, heads(graph, linear(graph, x, p + "k"), head, nheads), cross_k[l]), "k" + std::to_string(l));
                graph.output(ggml_cpy(g, heads(graph, linear(graph, x, p + "v"), head, nheads), cross_v[l]), "v" + std::to_string(l));
            }
            graph.allocate(); graph.set("enc", encoded);
            std::vector<int32_t> positions_data(time); std::iota(positions_data.begin(), positions_data.end(), 0);
            graph.set("pos", positions_data.data(), positions_data.size() * sizeof(int32_t)); graph.compute();
        }
        std::vector<int32_t> tokens; int32_t token = bos;
        for (int32_t step = 0; step < options.max_tokens; ++step) {
            Graph graph(device); auto g = graph.ctx;
            auto index = graph.input("pos", 1, 1, 1, 1, GGML_TYPE_I32);
            auto x = ggml_get_rows(g, weights.get("decoder.embed_tokens.weight"), graph.input("token", 1, 1, 1, 1, GGML_TYPE_I32));
            for (int l = 0; l < layers; ++l) {
                auto p = "decoder.layers." + std::to_string(l) + ".";
                auto h = norm(graph, x, p + "attn_norm");
                auto project = [&](const char* name, bool rotate) {
                    auto t = ggml_reshape_3d(g, linear(graph, h, p + "attn." + name), head, nheads, 1);
                    if (rotate) t = ggml_rope_ext(g, t, index, nullptr, head / 2, 0, positions, theta, 1, 0, 1, 0, 0);
                    return ggml_cont(g, ggml_permute(g, t, 0, 2, 1, 3));
                };
                auto store = [&](Tensor source, Tensor destination) {
                    auto written = ggml_set_rows(g, destination, source, index);
                    return ggml_view_3d(g, written, head, step + 1, nheads, written->nb[1], written->nb[2], 0);
                };
                auto a = attend(graph, project("q", true), store(project("k", true), keys[l]), store(project("v", false), values[l]));
                x = ggml_add(g, x, linear(graph, a, p + "attn.o"));
                h = heads(graph, linear(graph, norm(graph, x, p + "cross_attn_norm"), p + "cross_attn.q"), head, nheads);
                x = ggml_add(g, x, linear(graph, attend(graph, h, cross_k[l], cross_v[l]), p + "cross_attn.o"));
                h = linear(graph, norm(graph, x, p + "ffn_norm"), p + "ffn.fc1");
                auto value = ggml_view_2d(g, h, intermediate, 1, h->nb[1], 0);
                auto gate = ggml_view_2d(g, h, intermediate, 1, h->nb[1], intermediate * sizeof(float));
                h = ggml_mul(g, value, ggml_silu(g, gate));
                x = ggml_add(g, x, linear(graph, h, p + "ffn.fc2"));
            }
            graph.output(linear(graph, norm(graph, x, "decoder.output_norm"), "decoder.output"), "logits"); graph.allocate();
            graph.set("pos", &step, sizeof(step)); graph.set("token", &token, sizeof(token)); graph.compute();
            auto logits = graph.read("logits"); token = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (token == eos) return sentencepiece_text(vocabulary, tokens);
            tokens.push_back(token);
        }
        throw std::runtime_error("Moonshine generation token limit reached");
    }
};
Result MoonSession::accept(const float* pcm, size_t count, bool final, const PartialCallback&) {
    std::lock_guard<std::mutex> lock(model.mutex);
    validate_audio(pcm, count, audio.size(), model.options.max_audio_ms, finished);
    if (count) audio.insert(audio.end(), pcm, pcm + count);
    auto before = current;
    if (audio.size() >= 80 && (final || audio.size() - decoded >= static_cast<size_t>(model.options.step_ms) * 16)) {
        current = model.generate(audio); decoded = audio.size();
    }
    finished = final;
    return {current, final, final || current != before, static_cast<int>(audio.size() / 16)};
}
} // namespace
std::unique_ptr<Engine> make_moonshine(const Options& opts) { return std::make_unique<Moonshine>(opts); }
} // namespace haolive
