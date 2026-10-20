// FastConformer / RNN-T equations and GGUF layout informed by the MIT
// CrispASR reference. This bounded runtime has independent session state.
#include "ops.hpp"
#include <algorithm>

namespace haolive {
namespace {
class Nemotron;
class NemoSession final : public Session {
    Nemotron& model;
    std::vector<float> audio, prediction;
    std::vector<std::vector<float>> attention_cache, convolution_cache;
    std::vector<float> hidden[2], cell[2];
    std::vector<int32_t> tokens;
    int processed = 0, past = 0;
    size_t checked = 0;
    bool finished = false, initialized = false;
public:
    explicit NemoSession(Nemotron&);
    Result accept(const float*, size_t, bool, const PartialCallback&) override;
};
class Nemotron final : public OpsModel {
public:
    int dim, layers, head, nheads, kernel, left, chunk, pred_dim, blank, prompt_count, prompt_id;
    std::vector<float> window, filters;
    std::vector<std::string> vocabulary;
    std::unique_ptr<Graph> encoder, predictor, joint;
    int graph_past = -1, graph_count = -1;
    explicit Nemotron(const Options& opts) : OpsModel(opts) {
        auto n = [&](const std::string& key) { int v = weights.integer("nemotron." + key); if (v < 1 || v > 65536) throw std::runtime_error("invalid Nemotron geometry: " + key); return v; };
        dim = n("d_model"); layers = n("n_layers"); head = n("head_dim"); nheads = n("n_heads");
        kernel = n("conv_kernel"); pred_dim = n("pred_hidden"); blank = n("blank_id"); prompt_count = n("num_prompts");
        auto left_context = weights.integers("nemotron.att_context_left", true);
        auto right_context = weights.integers("nemotron.att_context_right", true);
        if (left_context.empty() || left_context.size() != right_context.size()) throw std::runtime_error("invalid Nemotron context presets");
        left = left_context[0]; chunk = right_context[0] + 1;
        auto languages = weights.strings("nemotron.prompt_langs"); auto ids = weights.integers("nemotron.prompt_ids", true);
        if (languages.size() != ids.size()) throw std::runtime_error("invalid Nemotron prompt table");
        auto automatic = std::find(languages.begin(), languages.end(), "auto");
        if (automatic == languages.end()) throw std::runtime_error("Nemotron automatic language prompt missing");
        prompt_id = ids[static_cast<size_t>(automatic - languages.begin())];
        if (layers > 64 || dim != head * nheads || kernel < 2 || kernel > 65 || left < 1 || left > 512 || chunk > 64 ||
            n("sample_rate") != 16000 || n("n_mels") != 128 || n("n_fft") != 512 || n("win_length") != 400 ||
            n("hop_length") != 160 || n("subsampling_factor") != 8 || n("pred_layers") != 2 ||
            n("subsampling_channels") != 256 || n("prompt_kernel_in") != dim + prompt_count ||
            weights.integer("nemotron.n_tdt_durations") != 0 || prompt_id >= prompt_count ||
            weights.str("nemotron.conv_norm_type") != "layer_norm" || weights.str("nemotron.att_context_style") != "chunked_limited")
            throw std::runtime_error("unsupported Nemotron streaming geometry");
        vocabulary = weights.strings("tokenizer.ggml.tokens");
        if (static_cast<int>(vocabulary.size()) != blank || weights.get("decoder.embed.weight")->ne[0] != pred_dim ||
            weights.get("joint.out.weight")->ne[1] != blank + 1) throw std::runtime_error("Nemotron vocabulary mismatch");
        weights.upload(opts.model_path, device.primary);
        window = floats(weights.get("preprocessor.window")); filters = floats(weights.get("preprocessor.fb"));
        build_predictor(); build_joint();
    }
    std::string architecture() const override { return "nemotron"; }
    std::unique_ptr<Session> create_session() override { return std::make_unique<NemoSession>(*this); }
    std::vector<float> preencode(const std::vector<float>& audio) {
        auto mel = nemo_features(audio, window, filters, 128);
        Graph graph(device); auto g = graph.ctx;
        auto x = graph.input("mel", 128, mel.frames);
        for (int stage : {0, 2, 3, 5, 6}) {
            auto p = "encoder.pre.conv." + std::to_string(stage);
            bool down = stage == 0 || stage == 2 || stage == 5;
            if (down) x = ggml_pad_ext(g, x, 2, 1, 2, 1, 0, 0, 0, 0);
            auto w = weights.get(p + ".weight");
            if (stage == 2 || stage == 5) x = ggml_conv_2d_dw_direct(g, w, x, 2, 2, 0, 0, 1, 1);
            else x = ggml_conv_2d_direct(g, w, x, down ? 2 : 1, down ? 2 : 1, 0, 0, 1, 1);
            auto b = weights.get(p + ".bias");
            x = ggml_add(g, x, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1));
            if (stage == 0 || stage == 3 || stage == 6) x = ggml_relu(g, x);
        }
        int64_t width = x->ne[0] * x->ne[2], time = x->ne[1];
        x = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3)), width, time);
        graph.output(linear(graph, x, "encoder.pre.out"), "frames"); graph.allocate();
        graph.set("mel", mel.data); graph.compute(); return graph.read("frames");
    }
    Tensor tail(Graph& graph, Tensor x, int count) {
        return ggml_cont(graph.ctx, ggml_view_2d(graph.ctx, x, dim, count, x->nb[1], (x->ne[1] - count) * x->nb[1]));
    }
    void build_encoder(int past, int count) {
        if (graph_past == past && graph_count == count) return;
        encoder.reset(); encoder = std::make_unique<Graph>(device); auto& graph = *encoder; auto g = graph.ctx;
        int full_time = past + count;
        auto position = graph.input("position", dim, 2 * full_time - 1);
        auto x = graph.input("frames", dim, count);
        for (int l = 0; l < layers; ++l) {
            auto p = "encoder.layers." + std::to_string(l) + ".";
            auto ff = [&](Tensor input, const std::string& name) {
                auto h = norm(graph, input, p + "norm_" + name);
                h = ggml_silu(g, linear(graph, h, p + name + ".linear1"));
                return ggml_add(g, input, ggml_scale(g, linear(graph, h, p + name + ".linear2"), .5f));
            };
            x = ff(x, "ff1");
            auto full = past ? ggml_concat(g, graph.input("a" + std::to_string(l), dim, past), x, 1) : x;
            graph.output(tail(graph, full, std::min(left, full_time)), "aout" + std::to_string(l));
            auto h = norm(graph, x, p + "norm_attn"), all = norm(graph, full, p + "norm_attn");
            auto q = linear(graph, h, p + "attn.q");
            auto qu = ggml_add(g, q, ggml_reshape_1d(g, weights.get(p + "attn.pos_bias_u"), dim));
            auto qv = ggml_add(g, q, ggml_reshape_1d(g, weights.get(p + "attn.pos_bias_v"), dim));
            qu = heads(graph, qu, head, nheads); qv = heads(graph, qv, head, nheads);
            auto r = heads(graph, linear(graph, position, p + "attn.pos"), head, nheads);
            auto raw = ggml_mul_mat(g, r, qv);
            auto relative = ggml_cont(g, ggml_view_3d(g, raw, full_time, count, nheads, raw->nb[1] - raw->nb[0], raw->nb[2], (count - 1) * raw->nb[0]));
            auto k = heads(graph, linear(graph, all, p + "attn.k"), head, nheads);
            auto v = heads(graph, linear(graph, all, p + "attn.v"), head, nheads);
            x = ggml_add(g, x, linear(graph, attend(graph, qu, k, v, nullptr, relative), p + "attn.out"));
            h = ggml_siglu_swapped(g, linear(graph, norm(graph, x, p + "norm_conv"), p + "conv.pw1"));
            h = ggml_concat(g, graph.input("c" + std::to_string(l), dim, kernel - 1), h, 1);
            // Keep the suffix of old + new, including when a chunk is shorter
            // than the convolution history. No session history is discarded.
            graph.output(tail(graph, h, kernel - 1), "cout" + std::to_string(l));
            h = ggml_reshape_4d(g, ggml_cont(g, ggml_transpose(g, h)), kernel - 1 + count, 1, dim, 1);
            auto dw = ggml_reshape_4d(g, ggml_cast(g, weights.get(p + "conv.dw.weight"), GGML_TYPE_F32), kernel, 1, 1, dim);
            h = ggml_conv_2d_dw_direct(g, dw, h, 1, 1, 0, 0, 1, 1);
            h = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, h, 1, 2, 0, 3)), dim, count);
            h = ggml_silu(g, norm(graph, h, p + "conv.ln"));
            x = ggml_add(g, x, linear(graph, h, p + "conv.pw2"));
            x = norm(graph, ff(x, "ff2"), p + "norm_out");
        }
        auto prompt = graph.input("prompt", prompt_count, count);
        x = ggml_concat(g, x, prompt, 0);
        x = linear(graph, ggml_relu(g, linear(graph, x, "prompt_kernel.0")), "prompt_kernel.2");
        graph.output(linear(graph, x, "joint.enc"), "encoded"); graph.allocate();
        graph_past = past; graph_count = count;
    }
    std::vector<float> encode(const float* frames, int count, int past,
                              std::vector<std::vector<float>>& attn, std::vector<std::vector<float>>& conv) {
        build_encoder(past, count); auto& graph = *encoder;
        int full_time = past + count;
        std::vector<float> position(static_cast<size_t>(dim) * (2 * full_time - 1));
        for (int t = 0; t < 2 * full_time - 1; ++t)
            for (int j = 0; j < dim / 2; ++j) {
                float a = (full_time - 1 - t) * std::exp(-std::log(10000.0f) * 2 * j / dim);
                position[t * dim + 2 * j] = std::sin(a); position[t * dim + 2 * j + 1] = std::cos(a);
            }
        graph.set("position", position); graph.set("frames", frames, static_cast<size_t>(dim) * count * sizeof(float));
        std::vector<float> prompt(static_cast<size_t>(prompt_count) * count, 0);
        for (int t = 0; t < count; ++t) prompt[t * prompt_count + prompt_id] = 1;
        graph.set("prompt", prompt);
        for (int l = 0; l < layers; ++l) {
            if (past) graph.set("a" + std::to_string(l), attn[l]);
            graph.set("c" + std::to_string(l), conv[l]);
        }
        graph.compute(); auto out = graph.read("encoded");
        for (int l = 0; l < layers; ++l) { attn[l] = graph.read("aout" + std::to_string(l)); conv[l] = graph.read("cout" + std::to_string(l)); }
        return out;
    }
    void build_predictor() {
        predictor = std::make_unique<Graph>(device); auto& graph = *predictor; auto g = graph.ctx;
        auto token = graph.input("token", 1, 1, 1, 1, GGML_TYPE_I32);
        auto x = ggml_get_rows(g, weights.get("decoder.embed.weight"), token);
        for (int l = 0; l < 2; ++l) {
            auto p = "decoder.lstm." + std::to_string(l) + ".";
            auto h = graph.input("h" + std::to_string(l), pred_dim), c = graph.input("c" + std::to_string(l), pred_dim);
            auto gates = ggml_add(g, ggml_mul_mat(g, weights.get(p + "w_ih"), x), ggml_mul_mat(g, weights.get(p + "w_hh"), h));
            gates = ggml_add(g, ggml_add(g, gates, weights.get(p + "b_ih")), weights.get(p + "b_hh"));
            auto part = [&](int i) { return ggml_view_1d(g, gates, pred_dim, static_cast<size_t>(i) * pred_dim * sizeof(float)); };
            c = ggml_add(g, ggml_mul(g, ggml_sigmoid(g, part(1)), c), ggml_mul(g, ggml_sigmoid(g, part(0)), ggml_tanh(g, part(2))));
            x = ggml_mul(g, ggml_sigmoid(g, part(3)), ggml_tanh(g, c));
            graph.output(c, "cout" + std::to_string(l)); graph.output(x, "hout" + std::to_string(l));
        }
        graph.output(linear(graph, x, "joint.pred"), "prediction"); graph.allocate();
    }
    std::vector<float> predict(int32_t token, std::vector<float>* hidden, std::vector<float>* cell) {
        auto& graph = *predictor; graph.set("token", &token, sizeof(token));
        for (int l = 0; l < 2; ++l) { graph.set("h" + std::to_string(l), hidden[l]); graph.set("c" + std::to_string(l), cell[l]); }
        graph.compute(); auto out = graph.read("prediction");
        for (int l = 0; l < 2; ++l) { hidden[l] = graph.read("hout" + std::to_string(l)); cell[l] = graph.read("cout" + std::to_string(l)); }
        return out;
    }
    void build_joint() {
        joint = std::make_unique<Graph>(device); auto& graph = *joint; auto g = graph.ctx;
        int joint_dim = static_cast<int>(weights.get("joint.out.weight")->ne[0]);
        auto sum = ggml_add(g, graph.input("encoded", joint_dim), graph.input("prediction", joint_dim));
        graph.output(linear(graph, ggml_relu(g, sum), "joint.out"), "logits"); graph.allocate();
    }
    int32_t choose(const float* frame, const std::vector<float>& prediction) {
        auto& graph = *joint;
        graph.set("encoded", frame, prediction.size() * sizeof(float)); graph.set("prediction", prediction); graph.compute();
        auto logits = graph.read("logits");
        return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
};
NemoSession::NemoSession(Nemotron& m) : model(m), attention_cache(m.layers),
    convolution_cache(m.layers, std::vector<float>(static_cast<size_t>(m.dim) * (m.kernel - 1), 0)) {
    for (int l = 0; l < 2; ++l) { hidden[l].resize(m.pred_dim, 0); cell[l].resize(m.pred_dim, 0); }
}
Result NemoSession::accept(const float* pcm, size_t count, bool final, const PartialCallback&) {
    std::lock_guard<std::mutex> lock(model.mutex);
    validate_audio(pcm, count, audio.size(), model.options.max_audio_ms, finished);
    if (count) audio.insert(audio.end(), pcm, pcm + count);
    auto before = sentencepiece_text(model.vocabulary, tokens);
    size_t threshold = static_cast<size_t>(std::max(model.options.step_ms, model.chunk * 80)) * 16;
    if (!audio.empty() && (final || audio.size() - checked >= threshold)) {
        auto frames = model.preencode(audio); int total = static_cast<int>(frames.size() / model.dim);
        // Hold one encoder chunk while centered STFT's right edge can change.
        int target = final ? total : std::max(0, total - model.chunk) / model.chunk * model.chunk;
        if (!initialized && target > 0) { prediction = model.predict(model.blank, hidden, cell); initialized = true; }
        while (processed < target) {
            int n = std::min(model.chunk, target - processed);
            auto encoded = model.encode(frames.data() + static_cast<size_t>(processed) * model.dim, n, past, attention_cache, convolution_cache);
            for (int t = 0; t < n; ++t) {
                for (int symbol = 0; symbol < 10; ++symbol) {
                    auto id = model.choose(encoded.data() + t * prediction.size(), prediction);
                    if (id == model.blank) break;
                    if (tokens.size() >= static_cast<size_t>(model.options.max_tokens)) throw std::runtime_error("Nemotron generation token limit reached");
                    tokens.push_back(id); prediction = model.predict(id, hidden, cell);
                }
            }
            past = std::min(model.left, past + n); processed += n;
        }
        checked = audio.size();
    }
    finished = final; auto text = sentencepiece_text(model.vocabulary, tokens);
    return {text, final, final || text != before, static_cast<int>(audio.size() / 16)};
}
} // namespace
std::unique_ptr<Engine> make_nemotron(const Options& opts) { return std::make_unique<Nemotron>(opts); }
} // namespace haolive
