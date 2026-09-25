/* The h3 events protocol: a machine-readable job stream on a file
 * descriptor (--events-fd N). docs/events.md is the normative spec.
 *
 * One JSON object per line, UTF-8, each written with a single write(2)
 * under a mutex so lines from the sampler, preview and main threads never
 * interleave. Unset fields are omitted. The first event is the protocol
 * log; on a normal exit the last is job.end. */
#ifndef H3_EVENTS_H
#define H3_EVENTS_H

#include "h3.h"

#define H3_EVENTS_PROTOCOL 1

/* Event types. */
#define H3_EVENT_LOG "log"
#define H3_EVENT_STAGE_START "stage.start"
#define H3_EVENT_STAGE_PROGRESS "stage.progress"
#define H3_EVENT_STAGE_END "stage.end"
#define H3_EVENT_METRIC "metric"
#define H3_EVENT_ARTIFACT "artifact"
#define H3_EVENT_JOB_END "job.end"

/* Artifact kinds. */
#define H3_ARTIFACT_VIDEO "video"
#define H3_ARTIFACT_PREVIEW "preview"
#define H3_ARTIFACT_FRAMES "frames"
#define H3_ARTIFACT_SETTINGS "settings"

/* Metric keys. */
#define H3_METRIC_PEAK_MEMORY "peak_memory_gb"
#define H3_METRIC_LORA_APPLY "lora_apply_s"

typedef struct h3_events h3_events;

/* Take ownership of an open descriptor and emit the protocol event. The
 * version and commit are stamped into it and into job.end. Returns NULL if
 * `fd` is not open for writing. */
h3_events *h3_events_open(int fd, const char *h3_version,
                          const char *git_commit);
/* Close the descriptor. Safe on NULL. */
void h3_events_close(h3_events *events);

void h3_events_log(h3_events *events, h3_log_level level, const char *text);
/* Stage start/progress/end; an end also reports the stage's peak memory
 * as a metric when known. */
void h3_events_stage(h3_events *events, const h3_stage_event *event);
void h3_events_metric(h3_events *events, const char *stage, const char *key,
                      double value, const char *unit);
void h3_events_artifact(h3_events *events, const char *kind,
                        const char *path);

/* The result: exactly one of these ends every stream. `settings` is the
 * sidecar path, or NULL when none was written. */
void h3_events_complete(h3_events *events, const char *output,
                        const char *settings, const h3_result *result,
                        double elapsed_seconds);
void h3_events_failed(h3_events *events, const char *error,
                      double elapsed_seconds);

/* Callbacks with the library's signatures, for h3_params.on_stage and
 * h3_set_log_callback; `opaque` is the h3_events. */
int h3_events_stage_callback(const h3_stage_event *event, void *opaque);
void h3_events_log_callback(h3_log_level level, const char *message,
                            void *opaque);

#endif
