#include "h3_stages.h"

#include "h3_gpu.h"

#include <string.h>
#include <strings.h>
#include <time.h>

static const char *const stage_names[H3_STAGE_ID_COUNT] = {
    [H3_STAGE_ID_TOKENIZE] = "tokenize",
    [H3_STAGE_ID_ENCODE_CONDITIONING] = "encode_conditioning",
    [H3_STAGE_ID_TEXT_ENCODE] = "text_encode",
    [H3_STAGE_ID_PRECOMPUTE_ADALN] = "precompute_adaln",
    [H3_STAGE_ID_LOAD_TRANSFORMER] = "load_transformer",
    [H3_STAGE_ID_VAE_LOAD] = "vae_load",
    [H3_STAGE_ID_DENOISE] = "denoise",
    [H3_STAGE_ID_AUDIO_DECODE] = "audio_decode",
    [H3_STAGE_ID_VAE_DECODE] = "vae_decode",
    [H3_STAGE_ID_MUX] = "mux",
};

/* Every progress label the generator emits (h3.c's bridges and the DiT
 * loader/sampler), and the stage it belongs to. A label missing here is
 * attributed to whichever stage is running and not reported, so adding a
 * counter without a stage cannot corrupt the stream -- but it will not be
 * seen either; the events test lists every label a render produces. */
static const struct {
    const char *label;
    int stage;
} label_stages[] = {
    {"tokenizer", H3_STAGE_ID_TOKENIZE},
    /* Reference encoders run in sequence inside one stage. */
    {"video VAE encoder", H3_STAGE_ID_ENCODE_CONDITIONING},
    {"Qwen vision", H3_STAGE_ID_ENCODE_CONDITIONING},
    {"audio VAE encoder", H3_STAGE_ID_ENCODE_CONDITIONING},
    {"text encoder", H3_STAGE_ID_TEXT_ENCODE},
    {"refine text", H3_STAGE_ID_TEXT_ENCODE},
    {"precompute AdaLN", H3_STAGE_ID_PRECOMPUTE_ADALN},
    {"load transformer core", H3_STAGE_ID_LOAD_TRANSFORMER},
    /* Loaded up front when denoise previews need a decoder, otherwise
     * after the audio decode; either way the same stage. */
    {"preview VAE load", H3_STAGE_ID_VAE_LOAD},
    {"video VAE load", H3_STAGE_ID_VAE_LOAD},
    /* The enqueue counter runs ahead of the GPU; the sampler's own count
     * follows. Both are the denoise stage, which ends only when the next
     * stage begins -- after the GPU has finished. */
    {"denoise enqueue", H3_STAGE_ID_DENOISE},
    {"denoise", H3_STAGE_ID_DENOISE},
    {"audio VAE", H3_STAGE_ID_AUDIO_DECODE},
    {"video VAE decode", H3_STAGE_ID_VAE_DECODE},
    {"FFmpeg", H3_STAGE_ID_MUX},
};

int h3_stage_for_label(const char *label) {
    if (!label) return -1;
    for (size_t index = 0; index < sizeof(label_stages) / sizeof(*label_stages);
         index++)
        if (!strcasecmp(label_stages[index].label, label))
            return label_stages[index].stage;
    return -1;
}

const char *h3_stage_name(int stage) {
    return stage >= 0 && stage < H3_STAGE_ID_COUNT ? stage_names[stage] : "";
}

static double monotonic_now(void *opaque) {
    (void)opaque;
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double metal_peak_gib(void *opaque) {
    (void)opaque;
    return (double)h3_gpu_process_peak_bytes() / (1024.0 * 1024.0 * 1024.0);
}

static void metal_peak_reset(void *opaque) {
    (void)opaque;
    h3_gpu_process_peak_reset();
}

void h3_stage_tracker_init(h3_stage_tracker *tracker,
                           h3_stage_callback callback, void *opaque) {
    memset(tracker, 0, sizeof(*tracker));
    tracker->callback = callback;
    tracker->opaque = opaque;
    tracker->now = monotonic_now;
    tracker->peak_gib = metal_peak_gib;
    tracker->peak_reset = metal_peak_reset;
    tracker->current = -1;
}

static int deliver(h3_stage_tracker *tracker, const h3_stage_event *event) {
    if (!tracker->callback || tracker->cancelled) return tracker->cancelled;
    if (tracker->callback(event, tracker->opaque)) tracker->cancelled = 1;
    return tracker->cancelled;
}

static int end_current(h3_stage_tracker *tracker) {
    if (tracker->current < 0) return 0;
    h3_stage_event event;
    memset(&event, 0, sizeof(event));
    event.kind = H3_STAGE_END;
    event.stage = h3_stage_name(tracker->current);
    event.elapsed_seconds = tracker->now(tracker->clock_opaque) -
        tracker->started;
    event.peak_memory_gib = tracker->peak_gib ?
        tracker->peak_gib(tracker->clock_opaque) : -1.0;
    tracker->done_mask |= 1u << tracker->current;
    tracker->current = -1;
    return deliver(tracker, &event);
}

int h3_stage_tracker_progress(h3_stage_tracker *tracker, const char *label,
                              int completed, int total) {
    if (!tracker || tracker->cancelled) return tracker ? tracker->cancelled : 0;
    int stage = h3_stage_for_label(label);
    if (stage < 0 || (tracker->done_mask & (1u << stage))) return 0;
    if (stage != tracker->current) {
        if (end_current(tracker)) return 1;
        h3_stage_event event;
        memset(&event, 0, sizeof(event));
        event.kind = H3_STAGE_START;
        event.stage = h3_stage_name(stage);
        event.label = label;
        tracker->current = stage;
        tracker->started = tracker->now(tracker->clock_opaque);
        tracker->last_fraction = -1.0;
        if (tracker->peak_reset) tracker->peak_reset(tracker->clock_opaque);
        if (deliver(tracker, &event)) return 1;
    }
    if (total <= 0) return 0;
    double fraction = (double)completed / (double)total;
    if (fraction < tracker->last_fraction) return 0;
    tracker->last_fraction = fraction;
    h3_stage_event event;
    memset(&event, 0, sizeof(event));
    event.kind = H3_STAGE_PROGRESS;
    event.stage = h3_stage_name(stage);
    event.label = label;
    event.completed = completed;
    event.total = total;
    return deliver(tracker, &event);
}

int h3_stage_tracker_finish(h3_stage_tracker *tracker) {
    return tracker ? end_current(tracker) : 0;
}
