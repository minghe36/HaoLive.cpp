// Zipformer2 GGUF tensor layout and equations were cross-checked against
// CrispASR (MIT, see THIRD_PARTY_NOTICES.md). HaoLive owns the runtime,
// graph lifetime, model/session separation and streaming feature schedule.
#include "runtime.hpp"
#include "xasr_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace haolive {
namespace {
using T = ggml_tensor*;

class Xasr;
class XSession final : public Session {
    Xasr& model;
    std::vector<float> audio, features, decoder;
    std::vector<int> tokens, history;
    std::map<std::string, std::vector<float>> caches;
    size_t received = 0;
    int frame_count = 0, position = 0, processed = 0;
    bool finished = false;
public:
    explicit XSession(Xasr& model);
    Result accept(const float*, size_t, bool, const PartialCallback&) override;
    void decode(const std::vector<float>& encoded);
    void update_decoder();
    std::string text() const;
};

class Xasr final : public Model {
public:
    std::vector<int> dims, layers, ds, heads, qd, vd, pd, kernels;
    int chunk, window, left, pos_dim, context, decoder_dim, join_dim, blank, unk;
    std::vector<std::string> vocabulary, cache_names;
    std::vector<float> dec_embed, dec_conv, dec_weight, dec_bias, join_weight, join_bias;
    std::vector<std::vector<float>> positions;
    std::unique_ptr<Graph> encoder;
    explicit Xasr(const Options& opts) : Model(opts) {
        dims = weights.integers("xasr.dims"); layers = weights.integers("xasr.n_layers");
        ds = weights.integers("xasr.downsample"); heads = weights.integers("xasr.n_heads");
        qd = weights.integers("xasr.query_head_dim"); vd = weights.integers("xasr.value_head_dim");
        pd = weights.integers("xasr.pos_head_dim"); kernels = weights.integers("xasr.conv_kernel");
        if (dims.empty() || dims.size() > 16) throw std::runtime_error("invalid X-ASR stacks");
        for (auto* list : {&layers, &ds, &heads, &qd, &vd, &pd, &kernels})
            if (list->size() != dims.size()) throw std::runtime_error("inconsistent X-ASR metadata");
        auto sizes = weights.integers("xasr.chunk_ms"), contexts = weights.integers("xasr.left_context_frames");
        const auto geometry = xasr_geometry(sizes, contexts, ds);
        chunk = geometry.chunk_frames;
        window = geometry.feature_window;
        left = geometry.left_context;
        for (size_t i = 0; i < dims.size(); ++i)
            if (chunk % ds[i] || left % ds[i] || dims[i] < 4 || kernels[i] % 2 == 0)
                throw std::runtime_error("invalid X-ASR geometry");
        pos_dim = weights.integer("xasr.pos_dim", 48);
        context = weights.integer("xasr.context_size", 2);
        decoder_dim = weights.integer("xasr.decoder_dim", 512);
        join_dim = weights.integer("xasr.joiner_dim", 512);
        blank = weights.integer("xasr.blank_id");
        unk = weights.integer("xasr.unk_id", -1);
        if (weights.integer("xasr.feature_dim", 80) != 80 || context < 1 || context > 16)
            throw std::runtime_error("unsupported X-ASR feature/decoder geometry");
        vocabulary = weights.strings("tokenizer.ggml.tokens");
        if (vocabulary.size() != static_cast<size_t>(weights.integer("xasr.vocab_size")) || blank >= static_cast<int>(vocabulary.size()))
            throw std::runtime_error("invalid X-ASR vocabulary");
        weights.upload(opts.model_path, device.primary);
        dec_embed = floats(weights.get("dec.emb.weight")); dec_conv = floats(weights.get("dec.conv.weight"));
        dec_weight = floats(weights.get("join.decoder_proj.weight")); dec_bias = floats(weights.get("join.decoder_proj.bias"));
        join_weight = floats(weights.get("join.output_linear.weight")); join_bias = floats(weights.get("join.output_linear.bias"));
        build_encoder();
    }
    std::string architecture() const override { return "xasr"; }
    std::unique_ptr<Session> create_session() override {
        std::lock_guard<std::mutex> lock(mutex);
        return std::make_unique<XSession>(*this);
    }

    T weight(const std::string& name) { return weights.get(name); }
    std::vector<float> mixture(const std::string& name) {
        auto values = floats(weight(name));
        float max = *std::max_element(values.begin(), values.end()), sum = 0;
        for (float& v : values) sum += (v = std::exp(v - max));
        for (float& v : values) v /= sum;
        return values;
    }
    T linear(T x, const std::string& name, bool bias = true) {
        auto g = encoder->ctx;
        T y = ggml_mul_mat(g, weight(name + ".weight"), x);
        return bias ? ggml_add(g, y, weight(name + ".bias")) : y;
    }
    T slice(T x, int start, int rows) {
        auto g = encoder->ctx;
        return ggml_cont(g, ggml_view_2d(g, x, rows, x->ne[1], x->nb[1], start * ggml_element_size(x)));
    }
    T tail(T x, int count) {
        auto g = encoder->ctx;
        return ggml_cont(g, ggml_view_2d(g, x, x->ne[0], count, x->nb[1], (x->ne[1] - count) * x->nb[1]));
    }
    T cache(const std::string& name, int64_t a, int64_t b, int64_t c = 1) {
        cache_names.push_back(name);
        return encoder->input(name, a, b, c);
    }
    T cached(T x, const std::string& name, int count) {
        auto g = encoder->ctx;
        T joined = ggml_concat(g, cache(name, x->ne[0], count), x, 1);
        encoder->output(tail(joined, count), "next_" + name);
        return joined;
    }
    T activation(T x, bool right) {
        auto g = encoder->ctx;
        return ggml_sub(g, ggml_softplus(g, ggml_scale_bias(g, x, 1, right ? -1.0f : -4.0f)),
                        ggml_scale_bias(g, x, 0.08f, right ? 0.313261687f : 0.035f));
    }
    T normalize(T x, const std::string& prefix) {
        auto g = encoder->ctx;
        T square = ggml_mean(g, ggml_sqr(g, ggml_sub(g, x, weight(prefix + ".bias"))));
        return ggml_scale(g, ggml_div(g, x, ggml_sqrt(g, square)), std::exp(floats(weight(prefix + ".log_scale"))[0]));
    }
    T blend(T residual, T x, const std::string& name) {
        auto g = encoder->ctx;
        return ggml_add(g, residual, ggml_mul(g, ggml_sub(g, x, residual), weight(name)));
    }
    T downsample(T x, const std::string& name) {
        auto g = encoder->ctx;
        auto coefficients = mixture(name);
        T sum = nullptr;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            T part = ggml_view_2d(g, x, x->ne[0], x->ne[1] / coefficients.size(), x->nb[1] * coefficients.size(), i * x->nb[1]);
            part = ggml_scale(g, ggml_cont(g, part), coefficients[i]);
            sum = sum ? ggml_add(g, sum, part) : part;
        }
        return sum;
    }
    T depthwise(T x, T kernel) {
        auto g = encoder->ctx;
        int64_t width = x->ne[0], time = x->ne[1], k = kernel->ne[0];
        T transposed = ggml_cont(g, ggml_transpose(g, x));
        T y = ggml_ssm_conv(g, ggml_reshape_3d(g, transposed, time, width, 1), ggml_reshape_2d(g, kernel, k, width));
        return ggml_reshape_2d(g, y, width, time - k + 1);
    }
    T convolution(T x, const std::string& p, const std::string& id, int k) {
        auto g = encoder->ctx;
        int dim = static_cast<int>(x->ne[0]), time = static_cast<int>(x->ne[1]), pad = k / 2;
        T h = linear(x, p + "in");
        h = ggml_mul(g, slice(h, 0, dim), ggml_sigmoid(g, slice(h, dim, dim)));
        T causal = ggml_add(g, depthwise(cached(h, id, pad), weight(p + "causal.weight")), weight(p + "causal.bias"));
        T local = ggml_add(g, depthwise(ggml_pad_ext(g, h, 0, 0, pad, pad, 0, 0, 0, 0), weight(p + "chunk.weight")), weight(p + "chunk.bias"));
        T scales = weight(p + "chunk_scale");
        T le = ggml_view_2d(g, scales, k, dim, scales->nb[1], 0);
        T re = ggml_view_2d(g, scales, k, dim, scales->nb[1], scales->nb[2]);
        if (time < k) {
            le = ggml_cont(g, ggml_view_2d(g, le, time, dim, le->nb[1], 0));
            re = ggml_cont(g, ggml_view_2d(g, re, time, dim, re->nb[1], (k - time) * ggml_element_size(re)));
        } else {
            le = ggml_pad_ext(g, ggml_cont(g, le), 0, time - k, 0, 0, 0, 0, 0, 0);
            re = ggml_pad_ext(g, ggml_cont(g, re), time - k, 0, 0, 0, 0, 0, 0, 0);
        }
        T scale = ggml_cont(g, ggml_transpose(g, ggml_scale_bias(g, ggml_add(g, le, re), 1, 1)));
        return linear(activation(ggml_add(g, causal, ggml_mul(g, local, scale)), true), p + "out");
    }
    T layer(T x, const std::string& p, int stack, int index, T pe, T mask) {
        auto g = encoder->ctx;
        const int h = heads[stack], q = qd[stack], v = vd[stack], ph = pd[stack];
        const int seq = chunk / ds[stack], past = left / ds[stack], total = past + seq;
        const int dim = dims[stack], hidden = 3 * dim / 4;
        std::string id = std::to_string(index);
        T residual = x, projected = linear(x, p + "aw.in");
        auto head_layout = [&](T t, int width, int time) {
            return ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, t, width, h, time), 0, 2, 1, 3));
        };
        T query = head_layout(slice(projected, 0, h * q), q, seq);
        T key = head_layout(cached(slice(projected, h * q, h * q), "key" + id, past), q, total);
        T positional = ggml_mul_mat(g, head_layout(linear(pe, p + "aw.pos", false), ph, static_cast<int>(pe->ne[1])),
                                       head_layout(slice(projected, 2 * h * q, h * ph), ph, seq));
        T rel = ggml_view_3d(g, positional, total, seq, h, positional->nb[1] - positional->nb[0], positional->nb[2], (seq - 1) * positional->nb[0]);
        T attention = ggml_soft_max(g, ggml_add(g, ggml_add(g, ggml_mul_mat(g, key, query), ggml_cont(g, rel)), mask));
        auto feedforward = [&](T input, int number) {
            auto f = p + "ff" + std::to_string(number) + ".";
            return linear(activation(linear(input, f + "in"), false), f + "out");
        };
        x = ggml_add(g, x, feedforward(x, 1));
        T nonlin = linear(x, p + "na.in");
        T values = cached(ggml_mul(g, slice(nonlin, hidden, hidden), ggml_tanh(g, slice(nonlin, 0, hidden))), "nonlin" + id, past);
        T first_head = ggml_view_2d(g, attention, total, seq, attention->nb[1], 0);
        T weighted = ggml_mul_mat(g, ggml_cont(g, ggml_transpose(g, values)), first_head);
        x = ggml_add(g, x, linear(ggml_mul(g, weighted, slice(nonlin, hidden * 2, hidden)), p + "na.out"));
        auto self_attention = [&](T input, int number) {
            std::string sa = p + "sa" + std::to_string(number) + ".";
            T val = cached(linear(input, sa + "in"), "value" + id + "_" + std::to_string(number), past);
            val = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, val, v, h, total), 1, 2, 0, 3));
            T out = ggml_mul_mat(g, val, attention);
            out = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)), h * v, seq);
            return linear(out, sa + "out");
        };
        x = ggml_add(g, x, self_attention(x, 1));
        x = ggml_add(g, x, convolution(x, p + "cv1.", "conv" + id + "_1", kernels[stack]));
        x = ggml_add(g, x, feedforward(x, 2));
        x = blend(residual, x, p + "bypass_mid");
        x = ggml_add(g, x, self_attention(x, 2));
        x = ggml_add(g, x, convolution(x, p + "cv2.", "conv" + id + "_2", kernels[stack]));
        x = ggml_add(g, x, feedforward(x, 3));
        return blend(residual, normalize(x, p + "norm"), p + "bypass");
    }
    void build_encoder() {
        encoder = std::make_unique<Graph>(device);
        auto g = encoder->ctx;
        T x = encoder->input("features", 80, window);
        auto conv_bias = [&](const std::string& p) { auto b = weight(p); return ggml_reshape_3d(g, b, 1, 1, b->ne[0]); };
        for (int i = 0; i < 3; ++i) {
            auto p = "emb.conv" + std::to_string(i);
            x = ggml_conv_2d(g, weight(p + ".weight"), x, i == 0 ? 1 : 2, i == 1 ? 2 : 1, i == 0 ? 1 : 0, 0, 1, 1);
            x = activation(ggml_add(g, x, conv_bias(p + ".bias")), true);
        }
        int freq = static_cast<int>(x->ne[0]), channels = static_cast<int>(x->ne[2]);
        if (x->ne[1] != chunk + 3) throw std::runtime_error("X-ASR subsampler shape mismatch");
        T joined = ggml_concat(g, cache("embedding", freq, 3, channels), x, 1);
        encoder->output(ggml_cont(g, ggml_view_3d(g, joined, freq, 3, channels, joined->nb[1], joined->nb[2], chunk * joined->nb[1])), "next_embedding");
        T y = ggml_conv_2d_dw(g, weight("emb.cnx.dw.weight"), joined, 1, 1, 3, 0, 1, 1);
        y = ggml_add(g, y, conv_bias("emb.cnx.dw.bias"));
        y = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, y, 1, 2, 0, 3)), channels, freq * chunk);
        T pw = weight("emb.cnx.pw1.weight");
        y = activation(ggml_add(g, ggml_mul_mat(g, ggml_reshape_2d(g, pw, channels, pw->ne[3]), y), weight("emb.cnx.pw1.bias")), false);
        pw = weight("emb.cnx.pw2.weight");
        y = ggml_add(g, ggml_mul_mat(g, ggml_reshape_2d(g, pw, pw->ne[2], channels), y), weight("emb.cnx.pw2.bias"));
        T residual = ggml_cont(g, ggml_permute(g, ggml_view_3d(g, x, freq, chunk, channels, x->nb[1], x->nb[2], 0), 1, 2, 0, 3));
        y = ggml_add(g, ggml_reshape_3d(g, y, channels, freq, chunk), residual);
        y = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, y, 1, 0, 2, 3)), freq * channels, chunk);
        x = normalize(linear(y, "emb.out"), "emb.norm");
        std::vector<T> outputs;
        int index = 0;
        positions.resize(dims.size());
        for (size_t s = 0; s < dims.size(); ++s) {
            int dim = dims[s], seq = chunk / ds[s], past = left / ds[s];
            x = dim <= x->ne[0] ? slice(x, 0, dim) : ggml_pad_ext(g, x, 0, dim - x->ne[0], 0, 0, 0, 0, 0, 0);
            T original = x;
            std::string p = "z." + std::to_string(s) + ".";
            if (ds[s] > 1) x = downsample(x, p + "ds_bias");
            T pe = encoder->input("pe" + std::to_string(s), pos_dim, past + 2 * seq - 1);
            T mask = encoder->input("mask" + std::to_string(s), past + seq);
            for (int l = 0; l < layers[s]; ++l) x = layer(x, p + std::to_string(l) + ".", static_cast<int>(s), index++, pe, mask);
            if (ds[s] > 1) {
                x = ggml_repeat_4d(g, ggml_reshape_3d(g, x, dim, 1, seq), dim, ds[s], seq, 1);
                x = blend(original, ggml_reshape_2d(g, x, dim, chunk), p + "combiner");
            }
            outputs.push_back(x);
        }
        x = outputs.back();
        for (int i = static_cast<int>(outputs.size()) - 2; i >= 0; --i)
            if (dims[i] > x->ne[0]) x = ggml_concat(g, x, slice(outputs[i], static_cast<int>(x->ne[0]), dims[i] - static_cast<int>(x->ne[0])), 0);
        encoder->output(linear(downsample(x, "z.out_ds_bias"), "join.encoder_proj"), "encoded");
        encoder->allocate();
        for (size_t s = 0; s < dims.size(); ++s) {
            int seq = chunk / ds[s], past = left / ds[s], rows = past + 2 * seq - 1;
            std::vector<float> pe(static_cast<size_t>(rows) * pos_dim);
            float root = std::sqrt(static_cast<float>(pos_dim));
            for (int r = 0; r < rows; ++r) {
                float pos = static_cast<float>(r - (seq + past - 1));
                float compressed = std::copysign(root * std::log1p(std::abs(pos) / root), pos);
                float angle = std::atan(compressed * static_cast<float>(2 * 3.14159265358979323846 / pos_dim));
                for (int f = 0; f < pos_dim / 2; ++f) {
                    pe[r * pos_dim + 2 * f] = std::cos(angle * (f + 1));
                    pe[r * pos_dim + 2 * f + 1] = std::sin(angle * (f + 1));
                }
                pe[(r + 1) * pos_dim - 1] = 1;
            }
            positions[s] = std::move(pe);
        }
    }
};

XSession::XSession(Xasr& m) : model(m) {
    for (const auto& name : m.cache_names) {
        auto t = ggml_graph_get_tensor(m.encoder->graph, name.c_str());
        caches[name].resize(static_cast<size_t>(ggml_nelements(t)), 0);
    }
    history.assign(static_cast<size_t>(m.context), -1); history.back() = m.blank;
    update_decoder();
}
void XSession::update_decoder() {
    const int d = model.decoder_dim, ctx = model.context;
    int group_size = static_cast<int>(model.dec_conv.size() / (static_cast<size_t>(d) * ctx));
    std::vector<float> hidden(d, 0);
    for (int o = 0; o < d; ++o) {
        float sum = 0;
        for (int k = 0; k < ctx; ++k) {
            int id = history[history.size() - ctx + k];
            if (id < 0) continue;
            for (int i = 0; i < group_size; ++i)
                sum += model.dec_conv[(o * group_size + i) * ctx + k] * model.dec_embed[id * d + (o / group_size) * group_size + i];
        }
        hidden[o] = std::max(0.0f, sum);
    }
    decoder = model.dec_bias;
    for (int o = 0; o < model.join_dim; ++o)
        for (int i = 0; i < d; ++i) decoder[o] += model.dec_weight[o * d + i] * hidden[i];
}
void XSession::decode(const std::vector<float>& encoded) {
    int width = model.join_dim;
    std::vector<float> combined(width);
    for (size_t row = 0; row < encoded.size() / width; ++row) {
        for (int i = 0; i < width; ++i) combined[i] = std::tanh(encoded[row * width + i] + decoder[i]);
        int best = 0; float maximum = -INFINITY;
        for (size_t id = 0; id < model.vocabulary.size(); ++id) {
            float score = model.join_bias[id];
            for (int i = 0; i < width; ++i) score += model.join_weight[id * width + i] * combined[i];
            if (score > maximum) { maximum = score; best = static_cast<int>(id); }
        }
        if (best == model.blank || best == model.unk) continue;
        tokens.push_back(best); history.push_back(best); update_decoder();
    }
}
std::string XSession::text() const {
    std::string result;
    for (int id : tokens) {
        auto token = model.vocabulary[id];
        if (token == "<unk>") continue;
        size_t at;
        while ((at = token.find("▁")) != std::string::npos) token.replace(at, 3, " ");
        result += token;
    }
    // Chinese sentence-piece boundaries do not represent word spaces.
    for (size_t i = 1; i + 1 < result.size();) {
        if (result[i] == ' ' && (static_cast<unsigned char>(result[i - 1]) >= 128) && (static_cast<unsigned char>(result[i + 1]) >= 128)) result.erase(i, 1);
        else ++i;
    }
    size_t first = result.find_first_not_of(' ');
    return first == std::string::npos ? "" : result.substr(first);
}
Result XSession::accept(const float* pcm, size_t count, bool final, const PartialCallback&) {
    std::lock_guard<std::mutex> lock(model.mutex);
    validate_audio(pcm, count, received, model.options.max_audio_ms, finished);
    std::string previous = text();
    if (count) audio.insert(audio.end(), pcm, pcm + count);
    received += count;
    if (final && received) audio.resize(audio.size() + (model.window * 10 + 1000) * 16, 0);
    auto fresh = kaldi_features(audio, final, frame_count);
    features.insert(features.end(), fresh.data.begin(), fresh.data.end());
    frame_count += fresh.frames;
    auto& graph = *model.encoder;
    // The window is complete at equality; do not wait for an unused frame.
    while (position + model.window <= frame_count) {
        graph.set("features", features.data() + static_cast<size_t>(position) * 80, static_cast<size_t>(model.window) * 80 * sizeof(float));
        for (const auto& cache : caches) graph.set(cache.first, cache.second);
        for (size_t s = 0; s < model.dims.size(); ++s) {
            // Scheduler scratch may reuse a consumed input's allocation.
            // Constant positional inputs must also be uploaded on every run.
            graph.set("pe" + std::to_string(s), model.positions[s]);
            int past = model.left / model.ds[s], seq = model.chunk / model.ds[s];
            std::vector<float> mask(static_cast<size_t>(past + seq), 0);
            for (int j = 0; j < past; ++j) if (j * model.ds[s] < model.left - processed) mask[j] = -INFINITY;
            graph.set("mask" + std::to_string(s), mask);
        }
        graph.compute();
        for (auto& cache : caches) cache.second = graph.read("next_" + cache.first);
        decode(graph.read("encoded"));
        processed += model.chunk; position += model.chunk * 2;
    }
    finished = final;
    std::string current = text();
    return {current, final, final || current != previous, static_cast<int>(received / 16)};
}
} // namespace
std::unique_ptr<Engine> make_xasr(const Options& opts) { return std::make_unique<Xasr>(opts); }
} // namespace haolive
