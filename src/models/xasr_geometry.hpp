#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace haolive {
struct XasrGeometry {
    int chunk_frames;
    int feature_window;
    int left_context;
};

inline XasrGeometry xasr_geometry(const std::vector<int>& sizes,
                                  const std::vector<int>& contexts,
                                  const std::vector<int>& downsample) {
    constexpr int chunk_ms = 160;
    auto selected = std::find(sizes.begin(), sizes.end(), chunk_ms);
    if (selected == sizes.end() || contexts.size() != sizes.size())
        throw std::runtime_error("X-ASR requires 160 ms chunk geometry");
    const int chunk = chunk_ms / 20;
    const int left = contexts[static_cast<size_t>(selected - sizes.begin())];
    if (left <= 0 || left > 4096 || downsample.empty())
        throw std::runtime_error("invalid X-ASR context geometry");
    for (int factor : downsample)
        if (factor <= 0 || chunk % factor || left % factor)
            throw std::runtime_error("invalid X-ASR downsample geometry");
    // Two 10 ms feature frames per encoder frame, plus subsampler lookahead.
    return {chunk, 2 * chunk + 13, left};
}
} // namespace haolive
