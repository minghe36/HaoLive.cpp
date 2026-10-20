#include "runtime.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

namespace haolive {
namespace {
constexpr double pi = 3.14159265358979323846;
using Complex = std::complex<float>;
// Mixed radix FFT supports both the 400-point Whisper window and Kaldi's 512.
void fft(const float* input, int stride, int n, Complex* output) {
    if (n == 1) { output[0] = Complex(*input, 0); return; }
    const int radix = n % 2 == 0 ? 2 : n % 5 == 0 ? 5 : n;
    const int part = n / radix;
    std::vector<Complex> scratch(n);
    for (int r = 0; r < radix; ++r) fft(input + r * stride, stride * radix, part, scratch.data() + r * part);
    for (int k = 0; k < n; ++k) {
        Complex value(0, 0);
        for (int r = 0; r < radix; ++r) {
            const float angle = static_cast<float>(-2 * pi * r * k / n);
            value += scratch[r * part + k % part] * Complex(std::cos(angle), std::sin(angle));
        }
        output[k] = value;
    }
}
std::vector<float> power(const std::vector<float>& frame) {
    std::vector<Complex> spectrum(frame.size());
    fft(frame.data(), 1, static_cast<int>(frame.size()), spectrum.data());
    std::vector<float> result(frame.size() / 2 + 1);
    for (size_t i = 0; i < result.size(); ++i) result[i] = std::norm(spectrum[i]);
    return result;
}
int mirror(int index, int size, bool whisper) {
    while (index < 0 || index >= size)
        index = index < 0 ? -index - (whisper ? 0 : 1) : 2 * size - index - (whisper ? 2 : 1);
    return index;
}
uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint32_t u32(const uint8_t* p) { return uint32_t(u16(p)) | uint32_t(u16(p + 2)) << 16; }
}
std::vector<float> decode_pcm16(const void* data, size_t size) {
    if (size % 2 || (size && !data)) throw std::invalid_argument("PCM16 requires complete little-endian samples");
    auto p = static_cast<const uint8_t*>(data);
    std::vector<float> result(size / 2);
    for (size_t i = 0; i < result.size(); ++i) {
        int value = u16(p + 2 * i);
        if (value >= 32768) value -= 65536;
        result[i] = static_cast<float>(value) / 32768.0f;
    }
    return result;
}
std::vector<float> decode_wav(const void* data, size_t size) {
    auto p = static_cast<const uint8_t*>(data);
    if (!p || size < 12 || std::memcmp(p, "RIFF", 4) || std::memcmp(p + 8, "WAVE", 4))
        throw std::invalid_argument("expected RIFF WAVE PCM16");
    size_t end = size_t(u32(p + 4)) + 8;
    if (end != size) throw std::invalid_argument("WAV RIFF size mismatch");
    bool format = false, audio = false;
    size_t start = 0, length = 0;
    for (size_t pos = 12; pos < end;) {
        if (end - pos < 8) throw std::invalid_argument("truncated WAV chunk header");
        size_t n = u32(p + pos + 4);
        size_t body = pos + 8;
        if (n > end - body) throw std::invalid_argument("truncated WAV chunk");
        if (!std::memcmp(p + pos, "fmt ", 4)) {
            if (format || n < 16 || u16(p + body) != 1 || u16(p + body + 2) != 1 || u32(p + body + 4) != 16000 ||
                u32(p + body + 8) != 32000 || u16(p + body + 12) != 2 || u16(p + body + 14) != 16)
                throw std::invalid_argument("WAV must be 16 kHz mono PCM16");
            format = true;
        } else if (!std::memcmp(p + pos, "data", 4)) {
            if (audio) throw std::invalid_argument("multiple WAV data chunks are unsupported");
            start = body; length = n; audio = true;
        }
        pos = body + n;
        if (n % 2) {
            if (pos >= end) throw std::invalid_argument("missing WAV chunk padding");
            ++pos;
        }
    }
    if (!format || !audio) throw std::invalid_argument("WAV requires fmt and data chunks");
    return decode_pcm16(p + start, length);
}

// FireRed uses snip-edges fbank in int16 scale. Its triangles are linear
// in Hz between HTK mel-spaced centers (different from X-ASR's filterbank).
Features firered_features(const float* pcm, size_t count) {
    Features result;
    if (count < 400) return result;
    result.frames = static_cast<int>((count - 400) / 160 + 1);
    static const std::vector<float> filters = [] {
        std::vector<float> bank(80 * 257, 0);
        float lo = 1127 * std::log1p(20.0f / 700), hi = 1127 * std::log1p(8000.0f / 700);
        float centers[82];
        for (int i = 0; i < 82; ++i) centers[i] = 700 * std::expm1((lo + i * (hi - lo) / 81) / 1127);
        for (int m = 0; m < 80; ++m)
            for (int k = 0; k < 257; ++k) {
                float hz = k * 16000.0f / 512;
                bank[m * 257 + k] = std::max(0.0f, std::min((hz - centers[m]) / (centers[m + 1] - centers[m]),
                    (centers[m + 2] - hz) / (centers[m + 2] - centers[m + 1])));
            }
        return bank;
    }();
    static const std::vector<float> window = [] {
        std::vector<float> w(400);
        for (int j = 0; j < 400; ++j) w[j] = std::pow(.5f - .5f * std::cos(float(2 * pi * j / 399)), .85f);
        return w;
    }();
    result.data.resize(static_cast<size_t>(result.frames) * 80);
    std::vector<float> frame(512, 0);
    for (int t = 0; t < result.frames; ++t) {
        float mean = 0;
        for (int j = 0; j < 400; ++j) mean += (frame[j] = pcm[t * 160 + j] * 32768);
        mean /= 400;
        for (int j = 0; j < 400; ++j) frame[j] -= mean;
        for (int j = 399; j >= 0; --j) frame[j] = (frame[j] - .97f * frame[j ? j - 1 : 0]) * window[j];
        auto spectrum = power(frame);
        for (int m = 0; m < 80; ++m) {
            float energy = 0;
            for (int k = 0; k < 257; ++k) energy += spectrum[k] * filters[m * 257 + k];
            result.data[t * 80 + m] = std::log(std::max(energy, 0x1p-23f));
        }
    }
    return result;
}

// NeMo's centered STFT retains the terminal frame. Preemphasis precedes
// zero padding; the stored 400-sample window is centered in a 512-point FFT.
Features nemo_features(const std::vector<float>& pcm, const std::vector<float>& window,
                       const std::vector<float>& filters, int mels) {
    if (window.size() != 400 || mels < 1 || filters.size() != static_cast<size_t>(257 * mels))
        throw std::invalid_argument("invalid NeMo filterbank");
    if (pcm.empty()) return {};
    Features result;
    result.frames = static_cast<int>(pcm.size() / 160 + 1);
    result.data.resize(static_cast<size_t>(result.frames) * mels);
    std::vector<float> signal(pcm.size());
    signal[0] = pcm[0];
    for (size_t i = 1; i < pcm.size(); ++i) signal[i] = pcm[i] - .97f * pcm[i - 1];
    for (int t = 0; t < result.frames; ++t) {
        std::vector<float> frame(512, 0);
        for (int j = 0; j < 400; ++j) {
            int at = t * 160 - 200 + j;
            if (at >= 0 && at < static_cast<int>(signal.size())) frame[j + 56] = signal[at] * window[j];
        }
        auto spectrum = power(frame);
        for (int m = 0; m < mels; ++m) {
            float energy = 0;
            for (int k = 0; k < 257; ++k) energy += spectrum[k] * filters[m * 257 + k];
            result.data[t * mels + m] = std::log(energy + 0x1p-24f);
        }
    }
    return result;
}

Features kaldi_features(const std::vector<float>& pcm, bool final, int first_frame) {
    Features result;
    const int n = static_cast<int>(pcm.size());
    result.frames = final ? (n + 80) / 160 : n >= 280 ? (n - 280) / 160 + 1 : 0;
    const int total = result.frames;
    if (first_frame < 0 || first_frame > total) throw std::invalid_argument("invalid feature frame offset");
    result.frames -= first_frame;
    if (!result.frames) return result;
    static const std::vector<float> bank = [] {
        std::vector<float> b(80 * 257, 0);
        auto mel = [](float hz) { return 1127.0f * std::log1p(hz / 700.0f); };
        const float lo = mel(20), step = (mel(7600) - lo) / 81;
        for (int m = 0; m < 80; ++m)
            for (int k = 0; k < 256; ++k) {
                float x = (mel(k * 16000.0f / 512) - lo) / step - (m + 1);
                b[m * 257 + k] = std::max(0.0f, 1.0f - std::abs(x));
            }
        return b;
    }();
    std::vector<float> frame(512);
    result.data.resize(static_cast<size_t>(result.frames) * 80);
    for (int frame_index = 0; frame_index < result.frames; ++frame_index) {
        const int t = first_frame + frame_index;
        float mean = 0;
        for (int j = 0; j < 400; ++j) mean += (frame[j] = pcm[mirror(t * 160 - 120 + j, n, false)]);
        mean /= 400;
        for (int j = 0; j < 400; ++j) frame[j] -= mean;
        for (int j = 399; j >= 0; --j) {
            float sample = frame[j] - 0.97f * frame[j ? j - 1 : 0];
            float win = std::pow(0.5f - 0.5f * std::cos(static_cast<float>(2 * pi * j / 399)), 0.85f);
            frame[j] = sample * win;
        }
        std::fill(frame.begin() + 400, frame.end(), 0);
        auto spectrum = power(frame);
        for (int m = 0; m < 80; ++m) {
            float sum = 0;
            for (int k = 0; k < 257; ++k) sum += spectrum[k] * bank[m * 257 + k];
            result.data[frame_index * 80 + m] = std::log(std::max(sum, 1.1920928955078125e-7f));
        }
    }
    return result;
}
Features WhisperFeatureCache::compute(const std::vector<float>& pcm, const std::vector<float>& window,
                                     const std::vector<float>& filters, int mels) {
    if (mels < 1 || window.size() != 400 || filters.size() != static_cast<size_t>(201 * mels))
        throw std::runtime_error("unsupported Whisper feature geometry");
    if (pcm.size() < previous_samples || cached_mels != mels) stable_frames = 0;
    previous_samples = pcm.size(); cached_mels = mels;
    Features result;
    result.frames = static_cast<int>(pcm.size()) / 160;
    raw.resize(static_cast<size_t>(mels) * result.frames);
    if (result.frames == 0) { stable_frames = 0; return result; }
    std::vector<float> frame(400);
    for (int t = static_cast<int>(stable_frames); t < result.frames; ++t) {
        for (int i = 0; i < 400; ++i)
            frame[i] = pcm[mirror(t * 160 + i - 200, static_cast<int>(pcm.size()), true)] * window[i];
        auto spectrum = power(frame);
        for (int m = 0; m < mels; ++m) {
            double sum = 0;
            for (int k = 0; k < 201; ++k) sum += double(spectrum[k]) * filters[k * mels + m];
            raw[static_cast<size_t>(t) * mels + m] = static_cast<float>(std::log10(std::max(sum, 1e-10)));
        }
    }
    // Reflect-padded tail frames change when PCM grows. Very short inputs can
    // also change the left reflection, so retain none until the window fits.
    stable_frames = pcm.size() < 400 ? 0 : std::min(static_cast<size_t>(result.frames), (pcm.size() - 200) / 160 + 1);
    const float maximum = std::max(-10.0f, *std::max_element(raw.begin(), raw.end()));
    result.data.resize(raw.size());
    for (int t = 0; t < result.frames; ++t)
        for (int m = 0; m < mels; ++m)
            result.data[static_cast<size_t>(m) * result.frames + t] =
                (std::max(raw[static_cast<size_t>(t) * mels + m], maximum - 8) + 4) / 4;
    return result;
}
Features whisper_features(const std::vector<float>& pcm, const std::vector<float>& window,
                           const std::vector<float>& filters, int mels) {
    WhisperFeatureCache cache;
    return cache.compute(pcm, window, filters, mels);
}

} // namespace haolive
