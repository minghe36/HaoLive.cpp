#pragma once
#include <stdexcept>
#include <utility>
#include <vector>

namespace haolive {
// Called under the connection mutex. Receiving PCM never waits for inference.
class PcmBacklog {
    size_t limit;
    size_t received_samples = 0;
    bool final_requested = false;
    std::vector<float> pending;
public:
    explicit PcmBacklog(size_t limit) : limit(limit) {}
    void append(const std::vector<float>& pcm) {
        if (final_requested) throw std::logic_error("audio after final");
        if (pcm.size() > limit - received_samples) throw std::length_error("utterance audio limit exceeded");
        pending.insert(pending.end(), pcm.begin(), pcm.end());
        received_samples += pcm.size();
    }
    void finish() {
        if (final_requested) throw std::logic_error("duplicate final");
        final_requested = true;
    }
    bool ready() const { return final_requested || !pending.empty(); }
    bool final() const { return final_requested; }
    std::vector<float> take() { return std::exchange(pending, {}); }
};
} // namespace haolive
