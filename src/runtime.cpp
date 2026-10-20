#include "runtime.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace haolive {
namespace {
std::string canonical(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
    if (name == "mtl" || name == "metal") return "metal";
    return name;
}
ggml_backend_dev_t find_device(const std::string& name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        auto reg = ggml_backend_reg_get(i);
        if (canonical(ggml_backend_reg_name(reg)) != name) continue;
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j)
            return ggml_backend_reg_dev_get(reg, j);
    }
    return nullptr;
}
}
std::vector<std::string> available_backends() {
    std::vector<std::string> result;
    for (const auto& name : {"cpu", "metal", "vulkan", "cuda"})
        if (find_device(name)) result.emplace_back(name);
    return result;
}
Device::Device(const Options& opts) : name(opts.backend) {
    if (name != "cpu" && name != "metal" && name != "vulkan" && name != "cuda")
        throw std::invalid_argument("backend must be cpu, metal, vulkan or cuda");
    if (name == "cpu") {
        primary = cpu = ggml_backend_cpu_init();
    } else {
        auto dev = find_device(name);
        if (!dev) throw std::runtime_error("backend unavailable: " + name);
        primary = ggml_backend_dev_init(dev, nullptr);
        if (!primary) throw std::runtime_error("backend initialization failed: " + name);
        cpu = ggml_backend_cpu_init();
    }
    if (!primary || !cpu) {
        if (primary) ggml_backend_free(primary);
        throw std::runtime_error("CPU backend initialization failed");
    }
    ggml_backend_cpu_set_n_threads(cpu, opts.threads);
}
Device::~Device() {
    if (primary != cpu) ggml_backend_free(primary);
    ggml_backend_free(cpu);
}
Weights::Weights(const std::string& path) {
    gguf_init_params params{true, &ctx};
    meta = gguf_init_from_file(path.c_str(), params);
    if (!meta || !ctx) {
        if (meta) gguf_free(meta);
        if (ctx) ggml_free(ctx);
        throw std::runtime_error("cannot read GGUF: " + path);
    }
}
Weights::~Weights() {
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    gguf_free(meta);
}
void Weights::upload(const std::string& path, ggml_backend_t backend) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open model weights");
    const uint64_t size = static_cast<uint64_t>(file.tellg());
    // Validate offsets before allocating GPU memory or copying tensor data.
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        auto tensor = get(gguf_get_tensor_name(meta, i));
        uint64_t offset = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, i);
        if (offset > size || ggml_nbytes(tensor) > size - offset)
            throw std::runtime_error("truncated GGUF tensor: " + std::string(tensor->name));
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) throw std::runtime_error("model allocation failed");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<char> scratch(4 * 1024 * 1024);
    for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) {
        auto tensor = get(gguf_get_tensor_name(meta, i));
        file.seekg(static_cast<std::streamoff>(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, i)));
        for (size_t offset = 0; offset < ggml_nbytes(tensor);) {
            size_t n = std::min(scratch.size(), ggml_nbytes(tensor) - offset);
            if (!file.read(scratch.data(), static_cast<std::streamsize>(n)))
                throw std::runtime_error("model read failed");
            ggml_backend_tensor_set(tensor, scratch.data(), offset, n);
            offset += n;
        }
    }
}
std::string Weights::str(const std::string& key, const std::string& fallback) const {
    auto id = gguf_find_key(meta, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(meta, id) != GGUF_TYPE_STRING) throw std::runtime_error("invalid string: " + key);
    return gguf_get_val_str(meta, id);
}
int Weights::integer(const std::string& key, int fallback) const {
    auto id = gguf_find_key(meta, key.c_str());
    if (id < 0) return fallback;
    int64_t value;
    switch (gguf_get_kv_type(meta, id)) {
    case GGUF_TYPE_UINT32: value = gguf_get_val_u32(meta, id); break;
    case GGUF_TYPE_INT32: value = gguf_get_val_i32(meta, id); break;
    case GGUF_TYPE_INT64: value = gguf_get_val_i64(meta, id); break;
    default: throw std::runtime_error("invalid integer: " + key);
    }
    if (value < 0 || value > std::numeric_limits<int>::max()) throw std::runtime_error("integer out of range: " + key);
    return static_cast<int>(value);
}
float Weights::number(const std::string& key, float fallback) const {
    auto id = gguf_find_key(meta, key.c_str());
    if (id < 0) return fallback;
    if (gguf_get_kv_type(meta, id) != GGUF_TYPE_FLOAT32) throw std::runtime_error("invalid float: " + key);
    return gguf_get_val_f32(meta, id);
}
std::vector<int> Weights::integers(const std::string& key, bool allow_zero) const {
    auto id = gguf_find_key(meta, key.c_str());
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_ARRAY) throw std::runtime_error("missing array: " + key);
    auto type = gguf_get_arr_type(meta, id);
    auto data = gguf_get_arr_data(meta, id);
    std::vector<int> result;
    for (size_t i = 0; i < gguf_get_arr_n(meta, id); ++i) {
        int64_t value = type == GGUF_TYPE_INT32 ? static_cast<const int32_t*>(data)[i] :
                        type == GGUF_TYPE_UINT32 ? static_cast<const uint32_t*>(data)[i] :
                        type == GGUF_TYPE_INT64 ? static_cast<const int64_t*>(data)[i] : -1;
        if (value < (allow_zero ? 0 : 1) || value > 65536) throw std::runtime_error("invalid array value: " + key);
        result.push_back(static_cast<int>(value));
    }
    return result;
}
std::vector<std::string> Weights::strings(const std::string& key) const {
    auto id = gguf_find_key(meta, key.c_str());
    if (id < 0 || gguf_get_kv_type(meta, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, id) != GGUF_TYPE_STRING)
        throw std::runtime_error("missing string array: " + key);
    std::vector<std::string> result;
    for (size_t i = 0; i < gguf_get_arr_n(meta, id); ++i) result.emplace_back(gguf_get_arr_str(meta, id, i));
    return result;
}
ggml_tensor* Weights::optional(const std::string& name) const { return ggml_get_tensor(ctx, name.c_str()); }
ggml_tensor* Weights::get(const std::string& name) const {
    auto tensor = optional(name);
    if (!tensor) throw std::runtime_error("missing tensor: " + name);
    return tensor;
}
std::vector<float> floats(ggml_tensor* tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    else {
        auto convert = ggml_get_type_traits(tensor->type)->to_float;
        if (!convert) throw std::runtime_error("unsupported tensor type");
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        convert(bytes.data(), result.data(), static_cast<int64_t>(result.size()));
    }
    return result;
}
Graph::Graph(Device& dev, int nodes) : device(dev) {
    ctx = ggml_init({ggml_tensor_overhead() * static_cast<size_t>(nodes) + ggml_graph_overhead_custom(nodes, false), nullptr, true});
    if (!ctx) throw std::runtime_error("graph metadata allocation failed");
    graph = ggml_new_graph_custom(ctx, nodes, false);
}
Graph::~Graph() {
    if (sched) ggml_backend_sched_free(sched);
    ggml_free(ctx);
}
ggml_tensor* Graph::input(const std::string& name, int64_t a, int64_t b, int64_t c, int64_t d, ggml_type type) {
    auto t = ggml_new_tensor_4d(ctx, type, a, b, c, d);
    ggml_set_name(t, name.c_str()); ggml_set_input(t); return t;
}
void Graph::output(ggml_tensor* t, const std::string& name) {
    ggml_set_name(t, name.c_str()); ggml_set_output(t); ggml_build_forward_expand(graph, t);
}
void Graph::allocate() {
    ggml_backend_t devices[] = {device.primary, device.cpu};
    sched = ggml_backend_sched_new(devices, nullptr, device.primary == device.cpu ? 1 : 2, ggml_graph_size(graph), false, true);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("graph allocation failed");
}
void Graph::set(const std::string& name, const void* data, size_t bytes) {
    auto t = ggml_graph_get_tensor(graph, name.c_str());
    if (!t || ggml_nbytes(t) != bytes) throw std::runtime_error("graph input shape mismatch: " + name);
    if (bytes) ggml_backend_tensor_set(t, data, 0, bytes);
}
void Graph::set(const std::string& name, const std::vector<float>& data) { set(name, data.data(), data.size() * sizeof(float)); }
std::vector<float> Graph::read(const std::string& name) {
    auto t = ggml_graph_get_tensor(graph, name.c_str());
    if (!t) throw std::runtime_error("missing output: " + name);
    return floats(t);
}
void Graph::compute() {
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("inference failed");
}
void validate_audio(const float* pcm, size_t count, size_t previous, int max_ms, bool finished) {
    if (finished) throw std::logic_error("session already finalized");
    if (count && !pcm) throw std::invalid_argument("null audio");
    size_t limit = static_cast<size_t>(max_ms) * 16;
    if (previous > limit || count > limit - previous) throw std::length_error("utterance audio limit exceeded; finalize and create a new session");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(pcm[i]) || std::abs(pcm[i]) > 1.0f) throw std::invalid_argument("audio must be finite float PCM in [-1,1]");
}
std::unique_ptr<Engine> load(const Options& opts) {
    if (opts.model_path.empty() || opts.threads < 1 || opts.threads > 256 || opts.step_ms < 160 || opts.step_ms > 10000 ||
        opts.max_audio_ms < 1000 || opts.max_audio_ms > 30000 || opts.max_tokens < 1 || opts.max_tokens > 2048)
        throw std::invalid_argument("invalid engine options");
    std::string architecture;
    { Weights probe(opts.model_path); architecture = probe.str("general.architecture"); }
    if (architecture == "xasr") return make_xasr(opts);
    if (architecture == "qwen3asr") return make_qwen(opts);
    if (architecture == "moonshine_streaming") return make_moonshine(opts);
    if (architecture == "nemotron") return make_nemotron(opts);
    throw std::invalid_argument("unsupported GGUF architecture: " + architecture);
}
} // namespace haolive
