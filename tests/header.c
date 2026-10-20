#include "haolive/haolive.h"
#include <string.h>
int main(void) {
    hl_options options = hl_default_options();
    hl_vad_options vad = hl_default_vad_options();
    hl_vad_params params = hl_default_vad_params();
    hl_vad_result result = {0};
    return options.api_version != HL_API_VERSION || strcmp(options.backend, "cpu") != 0 ||
        vad.api_version != HL_API_VERSION || vad.max_audio_ms != 30000 ||
        params.threshold != 0.5f || hl_vad_detect(NULL, NULL, 0, NULL, &result) != -1;
}
