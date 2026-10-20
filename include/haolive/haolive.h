#ifndef HAOLIVE_H
#define HAOLIVE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define HL_API_VERSION 1
typedef struct hl_engine hl_engine;
typedef struct hl_session hl_session;
typedef struct hl_vad hl_vad;
typedef struct {
    uint32_t api_version;
    const char* model_path;
    const char* backend;
    int threads;
    int step_ms;
    int max_audio_ms;
    int max_tokens;
} hl_options;
typedef struct {
    const char* text; /* Borrowed until the next call on this session or free. */
    int final;
    int updated;
    int audio_ms;
} hl_result;

hl_options hl_default_options(void);
hl_engine* hl_engine_create(const hl_options* options);
void hl_engine_free(hl_engine* engine);
hl_session* hl_session_create(hl_engine* engine);
/* A session retains its engine, so engine_free may be called first. */
void hl_session_free(hl_session* session);
/* pcm: normalized float32 mono 16 kHz. Returns 0 on success, -1 on error.
 * Input is consumed synchronously. final may be 0 or 1. On inference error,
 * discard the session; validation errors do not consume audio. */
int hl_session_accept(hl_session* session, const float* pcm, size_t count, int final, hl_result* result);
/* Independent FireRed VAD (CPU). No ASR handle is needed. */
typedef struct {
    uint32_t api_version;
    const char* model_path;
    int max_audio_ms;
} hl_vad_options;
typedef struct { float threshold; int min_speech_ms; } hl_vad_params;
typedef struct {
    int is_speech;
    int sufficient_audio;
    float max_probability;
    float mean_probability;
    int speech_ms;
    int audio_ms;
    int frames;
} hl_vad_result;
hl_vad_options hl_default_vad_options(void);
hl_vad_params hl_default_vad_params(void);
hl_vad* hl_vad_create(const hl_vad_options* options);
void hl_vad_free(hl_vad* vad);
/* Stateless window, float32 mono 16 kHz. params=NULL uses defaults.
 * Returns 0 on success, -1 on error; result is cleared on error.
 * Concurrent detect calls are safe. Do not free while a call is running. */
int hl_vad_detect(hl_vad* vad, const float* pcm, size_t count,
                  const hl_vad_params* params, hl_vad_result* result);
/* Error for the last failing call on this thread; borrowed until next call. */
const char* hl_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
