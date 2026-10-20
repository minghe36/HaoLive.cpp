#pragma once
#include "haolive/haolive.hpp"
#include "realtime_queue.hpp"
#include "ixwebsocket/IXWebSocketServer.h"
#include "json.hpp"
#include <condition_variable>
#include <thread>
#include <chrono>
#include <optional>

namespace haolive {
class RealtimeSocket {
    using Json = nlohmann::json;
    struct Connection {
        Engine& engine;
        std::mutex& inference;
        std::weak_ptr<ix::WebSocket> socket;
        std::mutex mutex;
        std::condition_variable changed;
        PcmBacklog backlog;
        using Clock = std::chrono::steady_clock;
        std::optional<Clock::time_point> pending_since;
        double pending_received_ms = 0, latest_received_ms = 0;
        static double wall_ms() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
        static double elapsed_ms(Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        bool closed = false;
        bool authenticated = false;
        std::string id;
        std::thread worker;
        Connection(Engine& engine, std::mutex& inference, std::weak_ptr<ix::WebSocket> socket, size_t limit)
            : engine(engine), inference(inference), socket(std::move(socket)), backlog(limit) {}
        ~Connection() { stop(); }
        void stop() {
            { std::lock_guard<std::mutex> lock(mutex); closed = true; }
            changed.notify_one();
            if (worker.joinable()) worker.join();
        }
        void send(Json value) {
            auto ws = socket.lock();
            if (!ws || ws->bufferedAmount() > 1024 * 1024 ||
                !ws->sendText(value.dump(-1, ' ', false, Json::error_handler_t::replace)).success)
                throw std::runtime_error("socket output unavailable");
        }
        void error(const std::string& message) {
            try { send({{"event", "error"}, {"session_id", id}, {"error", {{"code", "realtime_failed"}, {"message", message}}}}); }
            catch (...) {}
        }
        void run() {
            try {
                std::unique_ptr<Session> session;
                size_t processed = 0;
                uint64_t sequence = 0;
                while (true) {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return closed || backlog.ready(); });
                    if (closed) return;
                    auto pcm = backlog.take();
                    const auto queued_at = pending_since.value_or(Clock::now());
                    const double received_ms = pending_received_ms;
                    const double latest_ms = latest_received_ms;
                    pending_since.reset();
                    const double queue_ms = elapsed_ms(queued_at);
                    bool final = backlog.final();
                    lock.unlock();
                    double lock_ms = 0, setup_ms = 0;
                    auto inference_started = Clock::now();
                    auto emit = [&](const char* event, const Result& result) {
                        send({{"event", event}, {"session_id", id}, {"sequence", sequence},
                            {"text", result.text}, {"type", result.final ? "final" : "partial"},
                            {"updated", result.updated}, {"audio_ms", result.audio_ms},
                            {"timing", {{"serverReceivedAtMs", received_ms}, {"serverLatestReceivedAtMs", latest_ms}, {"serverEmittedAtMs", wall_ms()},
                                {"queueMs", queue_ms}, {"lockWaitMs", lock_ms}, {"sessionSetupMs", setup_ms},
                                {"inferenceMs", elapsed_ms(inference_started)}}}});
                    };
                    Result result;
                    {
                        const auto lock_started = Clock::now();
                        std::lock_guard<std::mutex> serial(inference);
                        lock_ms = elapsed_ms(lock_started);
                        // Disconnects must not begin a stale queued decode.
                        { std::lock_guard<std::mutex> check(mutex); if (closed) return; }
                        const auto setup_started = Clock::now();
                        if (!session) session = engine.create_session();
                        setup_ms = elapsed_ms(setup_started);
                        inference_started = Clock::now();
                        ++sequence;
                        result = session->accept(pcm.data(), pcm.size(), final, [&](const std::string& text) {
                            emit("partial", {text, false, true, static_cast<int>((processed + pcm.size()) / 16)});
                        });
                    }
                    processed += pcm.size();
                    emit("result", result);
                    if (final) return;
                }
            } catch (const std::exception& e) {
                { std::lock_guard<std::mutex> lock(mutex); closed = true; }
                error(e.what());
            }
        }
    };
    ix::WebSocketServer server;
public:
    RealtimeSocket(Engine& engine, std::mutex& inference, const std::string& token, int max_audio_ms)
        : server(0, "127.0.0.1", 8, 8, 5) {
        server.disablePerMessageDeflate();
        server.setOnConnectionCallback([&, token, max_audio_ms](std::weak_ptr<ix::WebSocket> socket, std::shared_ptr<ix::ConnectionState>) {
            auto connection = std::make_shared<Connection>(engine, inference, socket, static_cast<size_t>(max_audio_ms) * 16);
            if (auto ws = socket.lock()) ws->setOnMessageCallback([connection, token, socket](const ix::WebSocketMessagePtr& message) {
                try {
                    if (message->type == ix::WebSocketMessageType::Open) {
                        const auto& headers = message->openInfo.headers;
                        auto auth = headers.find("Authorization"), id = headers.find("X-Session-Id");
                        if (message->openInfo.uri != "/v1/realtime" ||
                            (!token.empty() && (auth == headers.end() || auth->second != "Bearer " + token)) ||
                            id == headers.end() || id->second.empty() || id->second.size() > 64)
                            throw std::invalid_argument("realtime authentication failed");
                        connection->id = id->second;
                        connection->authenticated = true;
                        connection->send({{"event", "ready"}, {"session_id", connection->id}, {"api_version", 1}});
                        auto* owner = connection.get();
                        connection->worker = std::thread([owner] { owner->run(); });
                    } else if (message->type == ix::WebSocketMessageType::Message) {
                        if (!connection->authenticated) throw std::invalid_argument("realtime authentication required");
                        std::lock_guard<std::mutex> lock(connection->mutex);
                        if (connection->closed) throw std::logic_error("realtime connection closed");
                        if (!connection->pending_since) {
                            connection->pending_since = Connection::Clock::now();
                            connection->pending_received_ms = Connection::wall_ms();
                        }
                        connection->latest_received_ms = Connection::wall_ms();
                        if (message->binary) {
                            if (message->str.size() > 960000) throw std::length_error("PCM size");
                            connection->backlog.append(decode_pcm16(message->str.data(), message->str.size()));
                        } else {
                            if (message->str.size() > 128 || message->str != "{\"event\":\"final\"}")
                                throw std::invalid_argument("expected final control frame");
                            connection->backlog.finish();
                        }
                        connection->changed.notify_one();
                    } else if (message->type == ix::WebSocketMessageType::Close || message->type == ix::WebSocketMessageType::Error) {
                        connection->stop();
                    }
                } catch (const std::exception& e) {
                    connection->error(e.what());
                    connection->stop();
                    if (auto ws = socket.lock()) ws->close(1008, "invalid realtime input");
                }
            });
        });
        auto status = server.listen();
        if (!status.first) throw std::runtime_error(status.second);
        server.start();
    }
    int port() { return server.getPort(); }
    ~RealtimeSocket() { server.stop(); }
};
} // namespace haolive
