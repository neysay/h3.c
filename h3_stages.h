/* Stage tracking: turns the generator's progress counters into explicit
 * stage boundaries (h3_stage_event, see h3.h).
 *
 * The generator reports progress as (label, completed, total) from many
 * components. Each label belongs to one stage id; the table mapping them
 * lives in h3_stages.c, beside the list of labels the generator emits. The
 * rules, which every consumer can rely on:
 *
 *   - Each stage id starts and ends at most once per render.
 *   - A new stage starting ends the current one, with its measured wall
 *     time and the peak Metal memory in use while it ran.
 *   - Progress within a stage never goes backwards: a stage fed by several
 *     counters in sequence reports only readings at or past the last one.
 *   - A label for a stage that already ended is attributed to the current
 *     stage's time and not reported: a stage is never reopened.
 *   - h3_stage_tracker_finish ends the open stage on success. On failure
 *     the open stage is left open; the render did not complete it. */
#ifndef H3_STAGES_H
#define H3_STAGES_H

#include "h3.h"

#include <stdint.h>

/* The ids of the stages a render passes through, in order. */
typedef enum {
    H3_STAGE_ID_TOKENIZE = 0,
    H3_STAGE_ID_ENCODE_CONDITIONING,
    H3_STAGE_ID_TEXT_ENCODE,
    H3_STAGE_ID_PRECOMPUTE_ADALN,
    H3_STAGE_ID_LOAD_TRANSFORMER,
    H3_STAGE_ID_VAE_LOAD,
    H3_STAGE_ID_DENOISE,
    H3_STAGE_ID_AUDIO_DECODE,
    H3_STAGE_ID_VAE_DECODE,
    H3_STAGE_ID_MUX,
    H3_STAGE_ID_COUNT
} h3_stage_id;

typedef struct {
    h3_stage_callback callback;
    void *opaque;
    /* Injected for tests; production uses the monotonic clock and the
     * process-wide Metal allocation peak. */
    double (*now)(void *clock_opaque);
    double (*peak_gib)(void *clock_opaque);
    void (*peak_reset)(void *clock_opaque);
    void *clock_opaque;

    int current;          /* h3_stage_id, or -1 */
    double started;
    double last_fraction;
    uint32_t done_mask;   /* stages that have ended */
    int cancelled;        /* the callback asked to stop */
} h3_stage_tracker;

/* Production wiring: monotonic clock and process-wide Metal peak. */
void h3_stage_tracker_init(h3_stage_tracker *tracker,
                           h3_stage_callback callback, void *opaque);

/* The stage a progress label belongs to, or -1 for an unknown label. */
int h3_stage_for_label(const char *label);
const char *h3_stage_name(int stage);

/* Feed one progress reading. Returns 0 to continue, nonzero if the
 * callback asked to cancel. */
int h3_stage_tracker_progress(h3_stage_tracker *tracker, const char *label,
                              int completed, int total);

/* End the open stage (the render succeeded). */
int h3_stage_tracker_finish(h3_stage_tracker *tracker);

#endif
