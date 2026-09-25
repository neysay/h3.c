#include "h3_events.h"

#include "h3_json.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_events {
    int fd;
    int broken; /* the reader went away: stop writing, keep rendering */
    pthread_mutex_t lock;
    char *h3_version;
    char *git_commit;
};

static const char *level_name(h3_log_level level) {
    switch (level) {
        case H3_LOG_DEBUG: return "debug";
        case H3_LOG_INFO: return "info";
        case H3_LOG_WARNING: return "warning";
        case H3_LOG_ERROR: return "error";
    }
    return "info";
}

/* One event line, built in memory then written whole. */
typedef struct {
    FILE *file;
    char *text;
    size_t length;
    int fields;
} line;

static int line_begin(line *out, const char *type) {
    memset(out, 0, sizeof(*out));
    out->file = open_memstream(&out->text, &out->length);
    if (!out->file) return 0;
    fputs("{\"type\":", out->file);
    h3_json_write_string(out->file, type, 1);
    out->fields = 1;
    return 1;
}

static void key(line *out, const char *name) {
    fputc(',', out->file);
    h3_json_write_string(out->file, name, 1);
    fputc(':', out->file);
}

static void field_string(line *out, const char *name, const char *value) {
    if (!value || !*value) return;
    key(out, name);
    h3_json_write_string(out->file, value, 1);
}

static void field_int(line *out, const char *name, long long value) {
    key(out, name);
    fprintf(out->file, "%lld", value);
}

/* Finite values only: JSON has no NaN or infinity. */
static void field_number(line *out, const char *name, double value) {
    key(out, name);
    if (isfinite(value)) fprintf(out->file, "%.6g", value);
    else fputs("null", out->file);
}

static void emit(h3_events *events, line *out) {
    if (!out->file) return;
    fputs("}\n", out->file);
    fclose(out->file);
    out->file = NULL;
    if (events && out->text) {
        pthread_mutex_lock(&events->lock);
        const char *cursor = out->text;
        size_t left = out->length;
        while (!events->broken && left) {
            ssize_t written = write(events->fd, cursor, left);
            if (written < 0) {
                if (errno == EINTR) continue;
                events->broken = 1; /* EPIPE, EBADF: nobody is listening */
                break;
            }
            cursor += written;
            left -= (size_t)written;
        }
        pthread_mutex_unlock(&events->lock);
    }
    free(out->text);
    out->text = NULL;
}

static void meta_build(line *out, const h3_events *events) {
    fprintf(out->file, "\"protocol\":%d,\"h3_version\":", H3_EVENTS_PROTOCOL);
    h3_json_write_string(out->file, events->h3_version, 1);
    fputs(",\"git_commit\":", out->file);
    h3_json_write_string(out->file, events->git_commit, 1);
}

h3_events *h3_events_open(int fd, const char *h3_version,
                          const char *git_commit) {
    int flags = fcntl(fd, F_GETFL);
    if (fd < 0 || flags < 0 || (flags & O_ACCMODE) == O_RDONLY) return NULL;
    h3_events *events = calloc(1, sizeof(*events));
    if (!events) return NULL;
    events->fd = fd;
    events->h3_version = strdup(h3_version ? h3_version : "");
    events->git_commit = strdup(git_commit ? git_commit : "");
    pthread_mutex_init(&events->lock, NULL);
    /* Children (ffmpeg) must not hold the stream open, and a reader that
     * goes away must fail a write, not kill the render. */
    fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
    signal(SIGPIPE, SIG_IGN);
    line out;
    if (line_begin(&out, H3_EVENT_LOG)) {
        field_string(&out, "level", "debug");
        field_string(&out, "source", "h3");
        field_string(&out, "text", "events protocol 1");
        key(&out, "meta");
        fputc('{', out.file);
        meta_build(&out, events);
        fputc('}', out.file);
        emit(events, &out);
    }
    return events;
}

void h3_events_close(h3_events *events) {
    if (!events) return;
    close(events->fd);
    pthread_mutex_destroy(&events->lock);
    free(events->h3_version);
    free(events->git_commit);
    free(events);
}

void h3_events_log(h3_events *events, h3_log_level level, const char *text) {
    line out;
    if (!events || !text || !line_begin(&out, H3_EVENT_LOG)) return;
    field_string(&out, "level", level_name(level));
    field_string(&out, "source", "h3");
    field_string(&out, "text", text);
    emit(events, &out);
}

void h3_events_metric(h3_events *events, const char *stage, const char *name,
                      double value, const char *unit) {
    line out;
    if (!events || !line_begin(&out, H3_EVENT_METRIC)) return;
    field_string(&out, "stage", stage);
    field_string(&out, "key", name);
    field_number(&out, "value", value);
    field_string(&out, "unit", unit);
    emit(events, &out);
}

void h3_events_stage(h3_events *events, const h3_stage_event *event) {
    line out;
    if (!events || !event) return;
    switch (event->kind) {
        case H3_STAGE_START:
            if (!line_begin(&out, H3_EVENT_STAGE_START)) return;
            field_string(&out, "stage", event->stage);
            field_string(&out, "label", event->label);
            emit(events, &out);
            return;
        case H3_STAGE_PROGRESS:
            if (!line_begin(&out, H3_EVENT_STAGE_PROGRESS)) return;
            field_string(&out, "stage", event->stage);
            field_string(&out, "label", event->label);
            field_int(&out, "step", event->completed);
            field_int(&out, "total", event->total);
            emit(events, &out);
            return;
        case H3_STAGE_END:
            if (!line_begin(&out, H3_EVENT_STAGE_END)) return;
            field_string(&out, "stage", event->stage);
            field_number(&out, "elapsed_s", event->elapsed_seconds);
            emit(events, &out);
            if (event->peak_memory_gib >= 0.0)
                h3_events_metric(events, event->stage, H3_METRIC_PEAK_MEMORY,
                                 event->peak_memory_gib, "GiB");
            return;
    }
}

void h3_events_artifact(h3_events *events, const char *kind,
                        const char *path) {
    line out;
    if (!events || !path || !line_begin(&out, H3_EVENT_ARTIFACT)) return;
    field_string(&out, "kind", kind);
    key(&out, "path");
    h3_json_write_path(out.file, path, 1);
    emit(events, &out);
}

void h3_events_complete(h3_events *events, const char *output,
                        const char *settings, const h3_result *result,
                        double elapsed_seconds) {
    line out;
    if (!events || !line_begin(&out, H3_EVENT_JOB_END)) return;
    field_string(&out, "status", "complete");
    field_number(&out, "elapsed_s", elapsed_seconds);
    key(&out, "meta");
    fputc('{', out.file);
    meta_build(&out, events);
    if (output && *output) {
        fputs(",\"output\":", out.file);
        h3_json_write_path(out.file, output, 1);
    }
    if (settings) {
        fputs(",\"settings\":", out.file);
        h3_json_write_path(out.file, settings, 1);
    }
    if (result) {
        fprintf(out.file,
                ",\"width\":%d,\"height\":%d,\"frames\":%d,\"fps\":%d,"
                "\"sample_rate\":%d,\"seed\":\"%" PRIu64 "\"",
                result->width, result->height, result->frames, result->fps,
                result->sample_rate, result->seed);
        fputs(",\"loras\":[", out.file);
        for (size_t index = 0; index < result->lora_count; index++) {
            const h3_lora_report *lora = &result->loras[index];
            fputs(index ? ",{\"path\":" : "{\"path\":", out.file);
            h3_json_write_path(out.file, lora->path, 1);
            fputs(",\"scale\":", out.file);
            h3_json_write_float(out.file, lora->strength);
            fputs(",\"format\":", out.file);
            h3_json_write_string(out.file, lora->format ? lora->format : "", 1);
            fprintf(out.file,
                    ",\"low_rank\":%zu,\"full_deltas\":%zu,\"patched\":%zu,"
                    "\"targets\":%zu,\"alpha_scale\":",
                    lora->low_rank, lora->full_deltas, lora->patched,
                    lora->targets);
            h3_json_write_float(out.file, lora->scale);
            fputc('}', out.file);
        }
        fputc(']', out.file);
    }
    fputc('}', out.file);
    emit(events, &out);
}

void h3_events_failed(h3_events *events, const char *error,
                      double elapsed_seconds) {
    line out;
    if (!events || !line_begin(&out, H3_EVENT_JOB_END)) return;
    field_string(&out, "status", "failed");
    field_string(&out, "error", error && *error ? error : "h3 failed");
    field_number(&out, "elapsed_s", elapsed_seconds);
    emit(events, &out);
}

int h3_events_stage_callback(const h3_stage_event *event, void *opaque) {
    h3_events_stage(opaque, event);
    return 0;
}

void h3_events_log_callback(h3_log_level level, const char *message,
                            void *opaque) {
    h3_events_log(opaque, level, message);
}
