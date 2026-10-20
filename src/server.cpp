#include "haolive/haolive.hpp"
#include "httplib.h"
#include "json.hpp"
#include "licenses.hpp"
#include "version.hpp"
#include "realtime_socket.hpp"
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <cstdlib>
#ifndef _WIN32
#include <unistd.h>
#endif

using Json = nlohmann::json;
namespace {
class ParentWatch {
    std::atomic<bool> done{false};
    std::thread worker;
public:
    explicit ParentWatch(int parent) {
        if (!parent) return;
#ifndef _WIN32
        if (getppid() != parent) throw std::invalid_argument("parent PID must identify the launching process");
        worker = std::thread([this, parent] {
            while (!done.load()) {
                // A killed/crashed host cannot run its normal child cleanup.
                // POSIX reparents us; exit even during model initialization.
                if (getppid() != parent) std::_Exit(0);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
#else
        throw std::invalid_argument("parent PID monitoring is unavailable on this platform");
#endif
    }
    ~ParentWatch() { done.store(true); if (worker.joinable()) worker.join(); }
};
struct HttpError : std::runtime_error {
    int status;
    std::string code;
    HttpError(int s, std::string c, const std::string& message) : std::runtime_error(message), status(s), code(std::move(c)) {}
};
struct Entry {
    std::unique_ptr<haolive::Session> session;
    std::chrono::steady_clock::time_point touched = std::chrono::steady_clock::now();
    uint64_t sequence = 0;
    size_t samples = 0;
};
Json result_json(const haolive::Result& r, const std::string& id, const std::string& model, uint64_t seq) {
    return {{"api_version", 1}, {"session_id", id}, {"model", model}, {"sequence", seq},
            {"type", r.final ? "final" : "partial"}, {"text", r.text}, {"updated", r.updated}, {"audio_ms", r.audio_ms}};
}
void send(httplib::Response& response, const Json& body, int status = 200) {
    response.status = status;
    response.set_content(body.dump(-1, ' ', false, Json::error_handler_t::replace), "application/json; charset=utf-8");
    response.set_header("Cache-Control", "no-store");
}
void error(httplib::Response& response, int status, const std::string& code, const std::string& message) {
    send(response, {{"error", {{"code", code}, {"message", message}}}}, status);
}
std::vector<float> audio(const httplib::Request& request) {
    auto content = request.get_header_value("Content-Type");
    if (content == "audio/wav" || content == "audio/x-wav") return haolive::decode_wav(request.body.data(), request.body.size());
    if (content == "application/octet-stream" || content == "audio/pcm") return haolive::decode_pcm16(request.body.data(), request.body.size());
    throw HttpError(415, "unsupported_audio", "use audio/pcm (PCM16LE mono 16 kHz) or audio/wav");
}
int integer_arg(const std::string& value, int min, int max) {
    size_t used = 0;
    int n;
    try { n = std::stoi(value, &used); }
    catch (const std::out_of_range&) { throw std::invalid_argument("argument out of range: " + value); }
    if (used != value.size() || n < min || n > max) throw std::invalid_argument("argument out of range: " + value);
    return n;
}
void help() {
    std::cout << "HaoLive.cpp 0.1.0\n"
                 "Usage: haolive-server --model MODEL.gguf [--backend cpu|metal|vulkan|cuda]\n"
                 "       [--host 127.0.0.1] [--port 9797] [--threads 4] [--step-ms 160]\n"
                 "       [--max-audio-ms 30000] [--max-tokens 512] [--token SECRET]\n"
                 "       [--vad-model FIRERED.gguf] (also works without --model)\n"
                 "       [--parent-pid PID] (POSIX host lifecycle; --port 0 selects a free port)\n"
                 "       haolive-server --list-backends\n"
                 "       haolive-server --licenses\n"
                 "       haolive-server --version\n";
}
}
int main(int argc, char** argv) {
    try {
        haolive::Options options;
        std::string host = "127.0.0.1", token, vad_path;
        int port = 9797, parent_pid = 0;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { help(); return 0; }
            if (arg == "--version") { std::cout << "haolive-server " << HAOLIVE_VERSION << '\n'; return 0; }
            if (arg == "--licenses") { for (const auto* notice : haolive_licenses) std::cout << notice; return 0; }
            if (arg == "--list-backends") { std::cout << Json(haolive::available_backends()).dump() << '\n'; return 0; }
            if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
            std::string value = argv[++i];
            if (arg == "--model") options.model_path = value;
            else if (arg == "--vad-model") vad_path = value;
            else if (arg == "--backend") options.backend = value;
            else if (arg == "--host") host = value;
            else if (arg == "--token") token = value;
            else if (arg == "--port") port = integer_arg(value, 0, 65535);
            else if (arg == "--parent-pid") parent_pid = integer_arg(value, 1, 2147483647);
            else if (arg == "--threads") options.threads = integer_arg(value, 1, 256);
            else if (arg == "--step-ms") options.step_ms = integer_arg(value, 160, 10000);
            else if (arg == "--max-audio-ms") options.max_audio_ms = integer_arg(value, 1000, 30000);
            else if (arg == "--max-tokens") options.max_tokens = integer_arg(value, 1, 2048);
            else throw std::invalid_argument("unknown argument: " + arg);
        }
        if (options.model_path.empty() && vad_path.empty()) { help(); return 2; }
        ParentWatch parent(parent_pid);
        if (options.model_path.empty() && options.backend != "cpu")
            throw std::invalid_argument("standalone FireRed VAD runs on CPU; --backend configures ASR only");
        auto engine = options.model_path.empty() ? std::unique_ptr<haolive::Engine>{} : haolive::load(options);
        auto vad = vad_path.empty() ? std::unique_ptr<haolive::Vad>{} : haolive::load_vad({vad_path, options.max_audio_ms});
        std::map<std::string, Entry> sessions;
        std::mutex mutex;
        auto realtime = engine ? std::make_unique<haolive::RealtimeSocket>(*engine, mutex, token, options.max_audio_ms) : nullptr;
        std::mt19937_64 random(std::random_device{}());
        httplib::Server server;
        server.set_payload_max_length(1024 * 1024);
        server.set_read_timeout(10, 0);
        server.set_write_timeout(30, 0);
        server.set_keep_alive_max_count(100);
        server.new_task_queue = [] { return new httplib::ThreadPool(4, 32); };
        server.set_pre_routing_handler([&](const httplib::Request& request, httplib::Response& response) {
            if (!token.empty() && request.get_header_value("Authorization") != "Bearer " + token) {
                error(response, 401, "unauthorized", "invalid bearer token");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr exception) {
            try { std::rethrow_exception(exception); }
            catch (const HttpError& e) { error(response, e.status, e.code, e.what()); }
            catch (const std::length_error& e) { error(response, 413, "audio_limit", e.what()); }
            catch (const std::invalid_argument& e) { error(response, 400, "invalid_request", e.what()); }
            catch (const std::logic_error& e) { error(response, 409, "session_closed", e.what()); }
            catch (const std::exception& e) { error(response, 500, "inference_failed", e.what()); }
            catch (...) { error(response, 500, "internal_error", "unknown error"); }
        });
        server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
            if (response.body.empty()) error(response, response.status, "http_error", "HTTP request failed");
        });
        server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
            send(response, {{"status", "ok"}, {"api_version", 1},
                {"websocket_port", realtime ? Json(realtime->port()) : Json(nullptr)},
                {"model", engine ? Json(engine->architecture()) : Json(nullptr)}, {"backend", engine ? engine->backend() : "cpu"},
                {"vad", vad ? Json{{"model", "firered-vad"}, {"backend", "cpu"}} : Json(nullptr)}});
        });
        server.Get("/v1/models", [&](const httplib::Request&, httplib::Response& response) {
            auto models = Json::array();
            if (engine) models.push_back({{"id", engine->architecture()}, {"backend", engine->backend()},
                {"sample_rate", 16000}, {"channels", 1}, {"encoding", "pcm_s16le"},
                {"max_audio_ms", options.max_audio_ms}, {"step_ms", options.step_ms}});
            send(response, {{"data", models}, {"vad", vad ? Json{{"id", "firered-vad"}, {"backend", "cpu"}, {"sample_rate", 16000}} : Json(nullptr)}});
        });
        server.Post("/v1/vad", [&](const httplib::Request& request, httplib::Response& response) {
            if (!vad) throw HttpError(503, "vad_unavailable", "start server with --vad-model");
            haolive::VadParams params;
            if (request.has_param("threshold")) {
                auto value = request.get_param_value("threshold"); size_t used = 0;
                try { params.threshold = std::stof(value, &used); }
                catch (const std::out_of_range&) { throw std::invalid_argument("VAD threshold out of range"); }
                if (used != value.size()) throw std::invalid_argument("invalid VAD threshold");
            }
            if (request.has_param("min_speech_ms")) params.min_speech_ms = integer_arg(request.get_param_value("min_speech_ms"), 1, 30000);
            auto pcm = audio(request);
            auto result = vad->detect(pcm.data(), pcm.size(), params);
            send(response, {{"api_version", 1}, {"model", "firered-vad"}, {"backend", "cpu"},
                {"is_speech", result.is_speech}, {"sufficient_audio", result.sufficient_audio},
                {"max_probability", result.max_probability}, {"mean_probability", result.mean_probability},
                {"speech_ms", result.speech_ms}, {"audio_ms", result.audio_ms}, {"frames", result.frames}});
        });
        auto reap = [&] {
            auto now = std::chrono::steady_clock::now();
            for (auto it = sessions.begin(); it != sessions.end();)
                if (now - it->second.touched > std::chrono::seconds(120)) it = sessions.erase(it); else ++it;
        };
        server.Post("/v1/sessions", [&](const httplib::Request& request, httplib::Response& response) {
            if (!engine) throw HttpError(503, "asr_unavailable", "start server with --model for ASR");
            if (!request.body.empty() && request.body != "{}") throw std::invalid_argument("session creation body must be empty or {}");
            std::lock_guard<std::mutex> lock(mutex);
            reap();
            if (sessions.size() >= 8) throw HttpError(429, "session_limit", "maximum 8 active sessions");
            std::string id;
            do { id = std::to_string(random()) + std::to_string(random()); } while (sessions.count(id));
            Entry entry; entry.session = engine->create_session();
            sessions.emplace(id, std::move(entry));
            send(response, {{"api_version", 1}, {"session_id", id}, {"model", engine->architecture()}, {"sample_rate", 16000}, {"encoding", "pcm_s16le"}}, 201);
        });
        server.Post(R"(/v1/sessions/([0-9]+)/audio)", [&](const httplib::Request& request, httplib::Response& response) {
            bool final = false;
            if (request.has_param("final")) {
                auto flag = request.get_param_value("final");
                if (flag != "true" && flag != "false") throw std::invalid_argument("final must be true or false");
                final = flag == "true";
            }
            auto pcm = audio(request);
            std::string id = request.matches[1];
            if (request.has_param("stream") && request.get_param_value("stream") != "true")
                throw std::invalid_argument("stream must be true");
            if (request.has_param("stream")) {
                response.set_header("Cache-Control", "no-cache");
                response.set_chunked_content_provider("text/event-stream",
                    [&, id, pcm = std::move(pcm), final](size_t, httplib::DataSink& sink) {
                        std::lock_guard<std::mutex> lock(mutex);
                        auto emit = [&](const char* event, const Json& value) {
                            auto data = std::string("event: ") + event + "\ndata: " + value.dump() + "\n\n";
                            if (!sink.write(data.data(), data.size())) throw std::runtime_error("stream disconnected");
                        };
                        try {
                            reap();
                            auto found = sessions.find(id);
                            if (found == sessions.end()) throw std::runtime_error("session missing or expired");
                            auto& entry = found->second;
                            auto seq = ++entry.sequence;
                            auto result = entry.session->accept(pcm.data(), pcm.size(), final,
                                [&](const std::string& text) {
                                    emit("partial", result_json({text, false, true, static_cast<int>((entry.samples + pcm.size()) / 16)}, id, engine->architecture(), seq));
                                });
                            entry.samples += pcm.size();
                            entry.touched = std::chrono::steady_clock::now();
                            emit("result", result_json(result, id, engine->architecture(), seq));
                            if (final) sessions.erase(found);
                        } catch (const std::exception& e) {
                            sessions.erase(id);
                            try { emit("error", {{"error", {{"code", "inference_failed"}, {"message", e.what()}}}}); }
                            catch (...) { return false; }
                        }
                        sink.done();
                        return true;
                    });
                return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            reap();
            auto found = sessions.find(id);
            if (found == sessions.end()) throw HttpError(404, "session_not_found", "session missing or expired");
            auto& entry = found->second;
            haolive::Result result;
            try { result = entry.session->accept(pcm.data(), pcm.size(), final); }
            catch (const std::runtime_error&) { sessions.erase(found); throw; }
            entry.samples += pcm.size();
            entry.touched = std::chrono::steady_clock::now();
            send(response, result_json(result, id, engine->architecture(), ++entry.sequence));
            if (final) sessions.erase(found);
        });
        server.Delete(R"(/v1/sessions/([0-9]+))", [&](const httplib::Request& request, httplib::Response& response) {
            std::lock_guard<std::mutex> lock(mutex);
            reap();
            if (!sessions.erase(request.matches[1])) throw HttpError(404, "session_not_found", "session missing or expired");
            response.status = 204;
        });
        server.Post("/v1/transcribe", [&](const httplib::Request& request, httplib::Response& response) {
            if (!engine) throw HttpError(503, "asr_unavailable", "start server with --model for ASR");
            auto pcm = audio(request);
            // Serialize with streaming calls so a single shared engine has a
            // bounded inference queue and model buffers are never used twice.
            std::lock_guard<std::mutex> lock(mutex);
            auto session = engine->create_session();
            auto result = session->accept(pcm.data(), pcm.size(), true);
            send(response, result_json(result, "", engine->architecture(), 1));
        });
        if (port == 0) {
            port = server.bind_to_any_port(host);
            if (port < 1) throw std::runtime_error("cannot bind server address");
        } else if (!server.bind_to_port(host, port)) throw std::runtime_error("cannot bind server address");
        // Parent processes can request an OS-assigned port without a reserve /
        // release race. No authentication secret is included in the handshake.
        std::cout << Json{{"event", "ready"}, {"api_version", 1}, {"port", port}}.dump() << std::endl;
        std::cerr << "HaoLive " << (engine ? engine->architecture() : "VAD-only")
                  << " backend=" << (engine ? engine->backend() : "cpu")
                  << " vad=" << (vad ? "firered-vad/cpu" : "disabled") << " http://" << host << ':' << port << '\n';
        return server.listen_after_bind() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "HaoLive: " << e.what() << '\n'; return 1;
    }
}
