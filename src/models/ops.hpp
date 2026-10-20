#pragma once
#include "runtime.hpp"
#include <cmath>

namespace haolive {
using Tensor = ggml_tensor*;
// Small equation helpers; each architecture owns its graphs and session state.
class OpsModel : public Model {
public:
    using Model::Model;
    Tensor linear(Graph& graph, Tensor x, const std::string& prefix) {
        auto y = ggml_mul_mat(graph.ctx, weights.get(prefix + ".weight"), x);
        if (auto bias = weights.optional(prefix + ".bias")) y = ggml_add(graph.ctx, y, bias);
        return y;
    }
    Tensor norm(Graph& graph, Tensor x, const std::string& prefix) {
        auto y = ggml_mul(graph.ctx, ggml_norm(graph.ctx, x, 1e-5f), weights.get(prefix + ".weight"));
        if (auto bias = weights.optional(prefix + ".bias")) y = ggml_add(graph.ctx, y, bias);
        return y;
    }
    Tensor heads(Graph& graph, Tensor x, int width, int nheads) {
        return ggml_cont(graph.ctx, ggml_permute(graph.ctx,
            ggml_reshape_3d(graph.ctx, x, width, nheads, x->ne[1]), 0, 2, 1, 3));
    }
    Tensor attend(Graph& graph, Tensor q, Tensor k, Tensor v, Tensor mask = nullptr, Tensor relative = nullptr) {
        auto g = graph.ctx;
        auto scores = ggml_mul_mat(g, k, q);
        if (relative) scores = ggml_add(g, scores, relative);
        scores = ggml_soft_max_ext(g, scores, mask, 1.0f / std::sqrt(float(q->ne[0])), 0);
        auto vt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
        auto out = ggml_mul_mat(g, vt, scores);
        out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3));
        return ggml_reshape_2d(g, out, q->ne[0] * q->ne[2], q->ne[1]);
    }
};
struct TensorArena {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    explicit TensorArena(int tensors) {
        ctx = ggml_init({ggml_tensor_overhead() * static_cast<size_t>(tensors + 8), nullptr, true});
        if (!ctx) throw std::runtime_error("state metadata allocation failed");
    }
    ~TensorArena() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
    TensorArena(const TensorArena&) = delete;
    TensorArena& operator=(const TensorArena&) = delete;
    void allocate(Device& device) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, device.primary);
        if (!buffer) throw std::runtime_error("state allocation failed");
        ggml_backend_buffer_clear(buffer, 0);
    }
};
} // namespace haolive
