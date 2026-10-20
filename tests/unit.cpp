#include "runtime.hpp"
#include "models/xasr_geometry.hpp"
#include "realtime_queue.hpp"
#include "haolive/haolive.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class E = std::exception> void throws(std::function<void()> fn, const char* message) {
    try { fn(); } catch (const E&) { check(true, message); return; }
    check(false, message);
}
void put16(std::string& s, uint16_t n) { s += char(n & 255); s += char(n >> 8); }
void put32(std::string& s, uint32_t n) { put16(s, n & 65535); put16(s, n >> 16); }
std::string wave() {
    std::string body = "WAVEfmt "; put32(body, 16); put16(body, 1); put16(body, 1);
    put32(body, 16000); put32(body, 32000); put16(body, 2); put16(body, 16);
    body += "JUNK"; put32(body, 1); body += 'x'; body += '\0';
    body += "data"; put32(body, 6); put16(body, 0); put16(body, 32767); put16(body, 32768);
    std::string out = "RIFF"; put32(out, static_cast<uint32_t>(body.size())); return out + body;
}
void xasr_geometry_tests() {
    const std::vector<int> sizes{160, 480, 960, 1920}, contexts{96, 256, 256, 256};
    const std::vector<int> downsample{1, 2, 4, 8, 4, 2};
    auto geometry = haolive::xasr_geometry(sizes, contexts, downsample);
    check(geometry.chunk_frames == 8 && geometry.feature_window == 29 &&
          geometry.left_context == 96, "X-ASR 160 ms feature, encoder and cache geometry");
    geometry = haolive::xasr_geometry({480, 160}, {256, 128}, downsample);
    check(geometry.left_context == 128, "context follows matching model metadata, not fixed index");
    throws([] { haolive::xasr_geometry({480}, {256}, {1}); }, "missing 160 ms geometry rejected");
    throws([] { haolive::xasr_geometry({160, 480}, {96}, {1}); }, "mismatched contexts rejected");
    throws([] { haolive::xasr_geometry({160}, {0}, {1}); }, "zero context rejected");
    throws([] { haolive::xasr_geometry({160}, {96}, {0}); }, "zero downsample rejected");
    throws([] { haolive::xasr_geometry({160}, {96}, {16}); }, "incompatible chunk geometry rejected");
    throws([] { haolive::xasr_geometry({160}, {95}, {2}); }, "incompatible context geometry rejected");
}
void audio_tests() {
    const unsigned char pcm[] = {0, 0, 255, 127, 0, 128, 255, 255};
    auto v = haolive::decode_pcm16(pcm, sizeof(pcm));
    check(v.size() == 4 && v[0] == 0 && v[1] == 32767 / 32768.0f && v[2] == -1 && v[3] == -1 / 32768.0f, "PCM16 endian/sign");
    throws<std::invalid_argument>([] { haolive::decode_pcm16("x", 1); }, "odd PCM");
    throws<std::invalid_argument>([] { haolive::decode_pcm16(nullptr, 2); }, "null PCM");
    check(haolive::decode_pcm16(nullptr, 0).empty(), "empty PCM flush");
    auto wav = wave(); auto decoded = haolive::decode_wav(wav.data(), wav.size());
    check(decoded.size() == 3 && decoded[2] == -1, "WAV odd padded metadata");
    for (size_t n = 0; n < wav.size(); ++n)
        throws<std::invalid_argument>([&] { haolive::decode_wav(wav.data(), n); }, "truncated WAV");
    auto bad = wav; bad[22] = 2;
    throws<std::invalid_argument>([&] { haolive::decode_wav(bad.data(), bad.size()); }, "stereo WAV");
    bad = wav; bad[24] = 0;
    throws<std::invalid_argument>([&] { haolive::decode_wav(bad.data(), bad.size()); }, "wrong sample rate");
    bad = wav; bad[20] = 3;
    throws<std::invalid_argument>([&] { haolive::decode_wav(bad.data(), bad.size()); }, "float WAV");
    bad = wav; bad[40] = char(255);
    throws<std::invalid_argument>([&] { haolive::decode_wav(bad.data(), bad.size()); }, "WAV chunk out of bounds");
}
void feature_tests() {
    std::vector<float> silence(1600), window(400, 1), filters(201);
    filters[25] = 1;
    auto mel = haolive::whisper_features(silence, window, filters, 1);
    check(mel.frames == 10 && mel.data.size() == 10, "Whisper drops last STFT frame");
    for (float v : mel.data) check(std::abs(v + 1.5f) < 1e-6f, "Whisper silence floor");
    std::vector<float> sine(1600);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = std::sin(float(2 * 3.14159265358979323846 * 1000 * i / 16000));
    mel = haolive::whisper_features(sine, window, filters, 1);
    float expected = float((std::log10(200.0 * 200.0) + 4) / 4);
    check(std::abs(mel.data[5] - expected) < 1e-4f, "400 point FFT sinusoid energy");
    // Streaming reuse must remain bit-identical even when new loud audio
    // changes global normalization and a formerly reflected tail becomes real.
    std::vector<float> cache_filters(201 * 3);
    for (int k = 0; k < 201; ++k)
        for (int m = 0; m < 3; ++m) cache_filters[k * 3 + m] = static_cast<float>((k + m) % 7) / 100;
    haolive::WhisperFeatureCache feature_cache;
    std::vector<float> growing;
    for (int length : {160, 321, 400, 479, 640, 777, 1600, 2561, 8000}) {
        const size_t old = growing.size();
        growing.resize(length);
        for (size_t i = old; i < growing.size(); ++i)
            growing[i] = std::sin(static_cast<float>(i) * .17f) * (i < 1500 ? .00001f : .9f);
        const auto cached = feature_cache.compute(growing, window, cache_filters, 3);
        const auto full = haolive::whisper_features(growing, window, cache_filters, 3);
        check(cached.frames == full.frames && cached.data == full.data,
              "cached Whisper features preserve reflection and global normalization");
    }
    growing.resize(320);
    check(feature_cache.compute(growing, window, cache_filters, 3).data ==
          haolive::whisper_features(growing, window, cache_filters, 3).data,
          "Whisper feature cache resets on shorter PCM");
    haolive::WhisperFeatureCache independent;
    std::vector<float> zeros(8000);
    check(independent.compute(zeros, window, cache_filters, 3).data ==
          haolive::whisper_features(zeros, window, cache_filters, 3).data,
          "Whisper features never reuse another session's PCM");
    auto all = haolive::kaldi_features(sine, true);
    auto first = haolive::kaldi_features(std::vector<float>(sine.begin(), sine.begin() + 777), false);
    auto rest = haolive::kaldi_features(sine, true, first.frames);
    auto joined = first.data; joined.insert(joined.end(), rest.data.begin(), rest.data.end());
    check(all.frames == first.frames + rest.frames && joined == all.data, "incremental Kaldi feature parity");
    auto quiet = haolive::kaldi_features(silence, true);
    check(quiet.frames == 10, "Kaldi frame count");
    for (float v : quiet.data) check(std::abs(v - std::log(std::numeric_limits<float>::epsilon())) < 1e-5f, "Kaldi silence floor");
    check(haolive::kaldi_features(std::vector<float>(279), false).frames == 0, "defer partial frame");
    check(haolive::kaldi_features(std::vector<float>(280), false).frames == 1, "first frame boundary");
    std::vector<float> nemo_filters(257, 0);
    nemo_filters[0] = 1;
    auto nemo = haolive::nemo_features(silence, window, nemo_filters, 1);
    check(nemo.frames == 11 && nemo.data.size() == 11, "NeMo retains terminal STFT frame");
    for (float v : nemo.data) check(std::abs(v - std::log(0x1p-24f)) < 1e-6f, "NeMo additive log floor");
    // At frame zero an impulse's preemphasis successor is inside the FFT.
    // DC energy = (1 - .97)^2. Padding before preemphasis would differ.
    std::vector<float> impulse(1600, 0); impulse[0] = 1;
    nemo = haolive::nemo_features(impulse, window, nemo_filters, 1);
    check(std::abs(nemo.data[0] - std::log(.03f * .03f + 0x1p-24f)) < 1e-4f, "NeMo preemphasis before zero padding");
    auto prefix = haolive::nemo_features(std::vector<float>(sine.begin(), sine.begin() + 800), window, nemo_filters, 1);
    auto whole = haolive::nemo_features(sine, window, nemo_filters, 1);
    for (int t = 0; t < 4; ++t) check(prefix.data[t] == whole.data[t], "NeMo stable feature prefix");
    check(haolive::nemo_features({}, window, nemo_filters, 1).frames == 0, "NeMo empty flush");
    throws<std::invalid_argument>([&] { haolive::nemo_features(sine, {}, nemo_filters, 1); }, "NeMo invalid window");
    throws<std::invalid_argument>([&] { haolive::nemo_features(sine, window, {}, 1); }, "NeMo invalid filters");
}
void tokenizer_tests() {
    auto parse = [](const std::string& bytes, int n) {
        return haolive::parse_moonshine_tokenizer(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), n);
    };
    std::string encoded(1, '\0'); encoded += char(6); encoded += "▁abc";
    encoded += char(128); encoded += char(1); encoded += std::string(128, 'x');
    auto vocab = parse(encoded, 3);
    check(vocab.size() == 3 && vocab[0].empty() && vocab[1] == "▁abc" && vocab[2].size() == 128, "Moonshine compact tokenizer lengths");
    check(haolive::sentencepiece_text({"<s>", "▁你好", "▁world", "</s>"}, {0, 1, 2, 3}) == "你好 world", "SentencePiece spaces and specials");
    check(haolive::sentencepiece_text({"<", "▁3"}, {0, 1}) == "< 3", "literal less-than preserved");
    throws([&] { haolive::sentencepiece_text(vocab, {-1}); }, "negative decoded token rejected");
    throws([&] { haolive::sentencepiece_text(vocab, {3}); }, "decoded token bounds");
    for (size_t length = 0; length < encoded.size(); ++length) {
        throws([&] { parse(encoded.substr(0, length), 3); }, "truncated compact tokenizer rejected");
    }
    throws([&] { parse(encoded + "x", 3); }, "extra tokenizer entries rejected");
    throws([&] { parse(encoded, 0); }, "invalid vocabulary size");
    auto builtin = haolive::moonshine_vocabulary();
    check(builtin.size() == 32768 && builtin[1] == "<s>" && builtin[2] == "</s>", "embedded Moonshine vocabulary identity");
    throws([&] { haolive::parse_moonshine_tokenizer(nullptr, 0, 3); }, "null vocabulary data");
}
void vad_tests() {
    using haolive::classify_vad;
    auto result = classify_vad({}, 399, {});
    check(!result.is_speech && !result.sufficient_audio && result.audio_ms == 24, "VAD short audio distinguished from silence");
    result = classify_vad(std::vector<float>(10, .5f), 1840, {});
    check(result.is_speech && result.sufficient_audio && result.speech_ms == 100, "VAD threshold and duration inclusive boundary");
    check(result.frames == 10 && result.max_probability == .5f && result.mean_probability == .5f, "VAD probability aggregation");
    check(!classify_vad(std::vector<float>(9, .9f), 1680, {}).is_speech, "VAD short burst rejected");
    check(!classify_vad({.9f, .9f, 0, .9f, .9f}, 1040, {.5f, 30}).is_speech, "VAD needs consecutive speech rather than sum");
    check(!classify_vad(std::vector<float>(10, .9f), 1840, {.5f, 101}).is_speech, "VAD rounds required duration up");
    check(classify_vad({.9f}, 400, {.5f, 10}).is_speech, "VAD configurable brief speech");
    for (float value : {0.0f, 1.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        throws<std::invalid_argument>([&] { classify_vad({}, 0, {value, 100}); }, "invalid VAD threshold even on empty input");
    for (int duration : {0, -1, 30001})
        throws<std::invalid_argument>([&] { classify_vad({}, 0, {.5f, duration}); }, "invalid VAD speech duration");
    throws([&] { classify_vad({std::numeric_limits<float>::quiet_NaN()}, 400, {}); }, "invalid VAD probability");
    throws([&] { haolive::load_vad({}); }, "VAD model path required");
    std::vector<float> silence(1600, 0), dc(1600, .5f);
    auto features = haolive::firered_features(silence.data(), silence.size());
    check(features.frames == 8 && features.data.size() == 640, "FireRed snip edges frame count");
    auto dc_features = haolive::firered_features(dc.data(), dc.size());
    check(features.data == dc_features.data, "FireRed removes DC before preemphasis");
    for (float v : features.data) check(std::abs(v - std::log(0x1p-23f)) < 1e-6f, "FireRed log floor");
    check(haolive::firered_features(silence.data(), 399).frames == 0, "FireRed incomplete frame rejected without OOB");
    check(haolive::firered_features(silence.data(), 400).frames == 1, "FireRed first complete frame");
    check(haolive::firered_features(nullptr, 0).frames == 0, "FireRed empty input");
    std::vector<float> tone(400);
    for (int i = 0; i < 400; ++i) tone[i] = .1f * std::sin(float(2 * 3.141592653589793 * 1000 * i / 16000));
    auto low = haolive::firered_features(tone.data(), tone.size());
    for (auto& v : tone) v *= 2;
    auto high = haolive::firered_features(tone.data(), tone.size());
    check(low.data[28] > 10, "FireRed feature energy uses int16 scale");
    check(std::abs(high.data[28] - low.data[28] - std::log(4.0f)) < 1e-4f, "FireRed log power scales quadratically");
    auto options = hl_default_vad_options(); auto params = hl_default_vad_params();
    check(options.api_version == HL_API_VERSION && options.max_audio_ms == 30000 && params.threshold == .5f && params.min_speech_ms == 100, "VAD C defaults");
    check(!hl_vad_create(nullptr), "VAD null options error contained");
    hl_vad_result c_result{1, 1, 1, 1, 1, 1, 1};
    check(hl_vad_detect(nullptr, nullptr, 0, nullptr, &c_result) == -1 && c_result.frames == 0 && !c_result.is_speech, "VAD C error clears output");
    hl_vad_free(nullptr);
}
void contract_tests() {
    haolive::PcmBacklog backlog(6);
    backlog.append({1, 2});
    check(backlog.take() == std::vector<float>({1, 2}), "socket PCM first batch");
    backlog.append({3}); backlog.append({4, 5});
    check(backlog.take() == std::vector<float>({3, 4, 5}), "socket PCM coalesces while inference runs without loss");
    throws<std::length_error>([&] { backlog.append({6, 7}); }, "socket PCM utterance cap includes consumed audio");
    backlog.append({6}); backlog.finish();
    check(backlog.final() && backlog.ready() && backlog.take() == std::vector<float>{6}, "socket final follows every queued sample");
    throws<std::logic_error>([&] { backlog.append({}); }, "socket audio after final rejected");
    throws<std::logic_error>([&] { backlog.finish(); }, "socket duplicate final rejected");
    float value = 0;
    haolive::validate_audio(nullptr, 0, 0, 1000, false);
    throws<std::logic_error>([&] { haolive::validate_audio(nullptr, 0, 0, 1000, true); }, "closed session");
    throws<std::length_error>([&] { haolive::validate_audio(&value, 1, 16000, 1000, false); }, "audio cap boundary");
    throws<std::length_error>([&] { haolive::validate_audio(&value, std::numeric_limits<size_t>::max(), 0, 1000, false); }, "audio count overflow");
    throws<std::invalid_argument>([&] { haolive::validate_audio(nullptr, 1, 0, 1000, false); }, "null samples");
    value = std::numeric_limits<float>::quiet_NaN();
    throws<std::invalid_argument>([&] { haolive::validate_audio(&value, 1, 0, 1000, false); }, "NaN samples");
    value = 1.1f;
    throws<std::invalid_argument>([&] { haolive::validate_audio(&value, 1, 0, 1000, false); }, "sample range");
    check(haolive::byte_token("ĠhelloĊ") == " hello\n", "GPT byte mapping");
    check(haolive::clean_asr_text("language Chinese<asr_text> 你好 \n") == "你好", "ASR header removed");
    check(haolive::clean_asr_text("language Chinese").empty(), "incomplete header hidden");
    check(haolive::clean_asr_text("language English<asr_text><non_speech>").empty(), "non speech hidden");
    check(haolive::clean_asr_text("<asr_text>hello|unused") == "hello", "R2T2 chunk boundary hidden");
    check(haolive::clean_asr_text(std::string("<asr_text>hello") + char(0xe4) + char(0xbd)) == "hello", "Qwen incomplete UTF8 draft suffix withheld");
    check(haolive::clean_asr_text("<asr_text>你好") == "你好", "Qwen complete multibyte draft preserved");
    check(haolive::clean_asr_text(std::string("<asr_text>") + char(0xe4)).empty(), "Qwen partial first character hidden");
    check(!haolive::qwen_progress_visible("这台相机从", "这"), "Qwen internal rollback does not erase displayed words");
    check(!haolive::qwen_progress_visible("你好", "你好"), "Qwen repeated token progress suppressed");
    check(haolive::qwen_progress_visible("你号", "你好"), "Qwen revised same-length text remains visible");
    check(haolive::qwen_progress_visible("这台相机", "这台相机从"), "Qwen new words stream immediately after rollback catches up");
    std::vector<std::string> qwen_vocab = {"hello", "|", "world", "Ġ|"};
    std::vector<int32_t> qwen_tokens;
    check(haolive::append_qwen_token(qwen_tokens, 0, qwen_vocab), "Qwen speech token retained");
    check(!haolive::append_qwen_token(qwen_tokens, 1, qwen_vocab), "Qwen chunk delimiter ends decode");
    check(qwen_tokens == std::vector<int32_t>{0}, "Qwen delimiter never enters next partial prefix");
    check(haolive::append_qwen_token(qwen_tokens, 2, qwen_vocab), "Qwen next partial extends prior text");
    std::string qwen_bytes;
    for (auto token : qwen_tokens) qwen_bytes += haolive::byte_token(qwen_vocab[token]);
    check(haolive::clean_asr_text("<asr_text>" + qwen_bytes) == "helloworld", "Qwen later speech stays visible beyond chunk boundary");
    check(!haolive::append_qwen_token(qwen_tokens, 3, qwen_vocab), "Qwen combined boundary token excluded");
    throws([&] { haolive::append_qwen_token(qwen_tokens, -1, qwen_vocab); }, "Qwen invalid token rejected");
    auto options = hl_default_options();
    check(options.api_version == 1 && options.step_ms == 160 && haolive::Options{}.step_ms == options.step_ms && options.threads == 4 && std::strcmp(options.backend, "cpu") == 0, "C defaults");
    check(!hl_engine_create(nullptr) && std::strlen(hl_last_error()), "C exception containment");
    check(!hl_session_create(nullptr), "C null engine");
    hl_result result{reinterpret_cast<const char*>(1), 1, 1, 1};
    check(hl_session_accept(nullptr, nullptr, 0, 1, &result) == -1 && result.text == nullptr, "C error output cleared");
    hl_engine_free(nullptr); hl_session_free(nullptr);
    options.model_path = "missing.gguf"; options.api_version = 999;
    check(!hl_engine_create(&options), "C unknown version");
    haolive::Options cpp;
    throws<std::invalid_argument>([&] { haolive::load(cpp); }, "empty model path");
    cpp.model_path = "missing.gguf"; cpp.step_ms = 0;
    throws<std::invalid_argument>([&] { haolive::load(cpp); }, "invalid step rejected before model load");
    cpp.backend = "unsupported";
    throws<std::invalid_argument>([&] { haolive::Device device(cpp); }, "unknown backend rejected");
}
}
int main() {
    try { xasr_geometry_tests(); audio_tests(); feature_tests(); tokenizer_tests(); vad_tests(); contract_tests(); std::cout << checks << " checks passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
