/* The events protocol (docs/events.md) without a model: a scripted render
 * drives the real stage tracker and event writer, and the stream is compared
 * byte for byte with the fixtures in tests/fixtures/events -- so any change to what
 * h3 emits fails here first. Set H3_UPDATE_FIXTURES=1 to rewrite them after
 * an intentional change (and bump H3_EVENTS_PROTOCOL if it is not purely
 * additive).
 *
 * Also checked: escaping (quotes, newlines, non-ASCII, invalid UTF-8),
 * stage start/end pairing, the library log funnel in both modes, and that
 * concurrent writers never interleave lines. */
#include "../h3.h"
#include "../h3_events.h"
#include "../h3_log.h"
#include "../h3_stages.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FIXTURES "tests/fixtures/events/"

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_events.c: %s\n", message);
    exit(1);
}

/* A deterministic clock and memory reading for the tracker. */
typedef struct {
    double now;
    double peak;
} fake_clock;

static double fake_now(void *opaque) {
    fake_clock *clock = opaque;
    clock->now += 0.5;
    return clock->now;
}

static double fake_peak(void *opaque) {
    fake_clock *clock = opaque;
    return clock->peak;
}

static void fake_reset(void *opaque) {
    fake_clock *clock = opaque;
    clock->peak += 1.25;
}

typedef struct {
    h3_stage_tracker tracker;
    fake_clock clock;
    h3_events *events;
    char path[256];
} run;

static void run_open(run *r, const char *name) {
    snprintf(r->path, sizeof(r->path), "/tmp/h3-events-%d-%s.jsonl",
             (int)getpid(), name);
    int fd = open(r->path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) die("cannot create stream file");
    r->events = h3_events_open(fd, "0.1.0-test", "testcommit");
    if (!r->events) die("h3_events_open refused a writable fd");
    memset(&r->clock, 0, sizeof(r->clock));
    h3_stage_tracker_init(&r->tracker, h3_events_stage_callback, r->events);
    r->tracker.now = fake_now;
    r->tracker.peak_gib = fake_peak;
    r->tracker.peak_reset = fake_reset;
    r->tracker.clock_opaque = &r->clock;
}

static void progress(run *r, const char *label, int completed, int total) {
    if (h3_stage_tracker_progress(&r->tracker, label, completed, total))
        die("tracker asked to cancel");
}

static char *slurp(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    if (!text) die("out of memory");
    size_t got = fread(text, 1, (size_t)size, file);
    text[got] = '\0';
    fclose(file);
    if (length) *length = got;
    return text;
}

/* Every stage.start is matched by a stage.end before a complete job.end. */
static void check_pairing(const char *stream, int complete) {
    char open_stages[16][64];
    int open_count = 0;
    const char *line = stream;
    while (*line) {
        const char *end = strchr(line, '\n');
        if (!end) die("stream line without a newline");
        char stage[64] = "";
        const char *at = strstr(line, "\"stage\":\"");
        if (at && at < end) {
            at += 9;
            size_t length = (size_t)(strchr(at, '"') - at);
            if (length >= sizeof(stage)) die("stage id too long");
            memcpy(stage, at, length);
            stage[length] = '\0';
        }
        if (!strncmp(line, "{\"type\":\"stage.start\"", 21)) {
            for (int index = 0; index < open_count; index++)
                if (!strcmp(open_stages[index], stage))
                    die("a stage started twice");
            snprintf(open_stages[open_count++], 64, "%s", stage);
        } else if (!strncmp(line, "{\"type\":\"stage.end\"", 19)) {
            int found = -1;
            for (int index = 0; index < open_count; index++)
                if (!strcmp(open_stages[index], stage)) found = index;
            if (found < 0) die("a stage ended that never started");
            memmove(open_stages[found], open_stages[found + 1],
                    (size_t)(open_count - found - 1) * sizeof(*open_stages));
            open_count--;
        }
        line = end + 1;
    }
    if (complete && open_count) die("a stage was still open at job.end");
}

static void finish(run *r, const char *name, int complete) {
    h3_events_close(r->events);
    size_t length = 0;
    char *actual = slurp(r->path, &length);
    if (!actual) die("cannot read the stream back");
    check_pairing(actual, complete);
    char fixture[256];
    snprintf(fixture, sizeof(fixture), FIXTURES "%s.jsonl", name);
    const char *update = getenv("H3_UPDATE_FIXTURES");
    if (update && *update && strcmp(update, "0")) {
        FILE *file = fopen(fixture, "wb");
        if (!file || fwrite(actual, 1, length, file) != length) die("cannot write fixture");
        fclose(file);
        printf("updated %s\n", fixture);
    } else {
        char *expected = slurp(fixture, NULL);
        if (!expected) die("missing fixture (run with H3_UPDATE_FIXTURES=1)");
        if (strcmp(expected, actual)) {
            fprintf(stderr, "--- expected %s\n%s--- actual %s\n%s", fixture,
                    expected, r->path, actual);
            die("event stream differs from its fixture");
        }
        free(expected);
    }
    free(actual);
    unlink(r->path);
}

/* The label sequence a real 512x512x22 turbo t2va render produces, in
 * order, thinned: the text encoder's second counter ("refine text") and
 * the denoise enqueue running ahead of the sampler are both in it. */
static void script_t2va(run *r) {
    progress(r, "tokenizer", 0, 1);
    progress(r, "tokenizer", 1, 1);
    progress(r, "text encoder", 0, 50);
    progress(r, "text encoder", 25, 50);
    progress(r, "text encoder", 50, 50);
    progress(r, "refine text", 0, 1);   /* would reset: suppressed */
    progress(r, "refine text", 1, 1);
    progress(r, "precompute AdaLN", 1, 50);
    progress(r, "precompute AdaLN", 50, 50);
    progress(r, "load transformer core", 1, 50);
    progress(r, "load transformer core", 50, 50);
    progress(r, "denoise enqueue", 0, 4);
    progress(r, "denoise enqueue", 4, 4);
    progress(r, "denoise", 4, 4);
    progress(r, "audio VAE", 0, 7);
    progress(r, "audio VAE", 7, 7);
    progress(r, "video VAE load", 0, 36);
    progress(r, "video VAE load", 36, 36);
    progress(r, "video VAE decode", 1, 2);
    progress(r, "video VAE decode", 2, 2);
    progress(r, "FFmpeg", 0, 22);
    progress(r, "FFmpeg", 22, 22);
}

static h3_result t2va_result(void) {
    h3_result result;
    memset(&result, 0, sizeof(result));
    result.width = 512;
    result.height = 512;
    result.frames = 22;
    result.fps = 24;
    result.sample_rate = 32000;
    result.seed = UINT64_C(18446744073709551615); /* survives only as text */
    result.video_shift = H3_DEFAULT_VIDEO_SHIFT;
    return result;
}

static void test_t2va(void) {
    run r;
    run_open(&r, "t2va");
    script_t2va(&r);
    if (h3_stage_tracker_finish(&r.tracker)) die("finish cancelled");
    h3_events_artifact(r.events, H3_ARTIFACT_VIDEO, "/nonexistent/out/video.mp4");
    h3_result result = t2va_result();
    h3_events_complete(r.events, "/nonexistent/out/video.mp4", NULL, &result,
                       41.5);
    finish(&r, "t2va", 1);
}

static void test_lora(void) {
    run r;
    run_open(&r, "lora");
    progress(&r, "tokenizer", 1, 1);
    progress(&r, "text encoder", 50, 50);
    progress(&r, "precompute AdaLN", 50, 50);
    progress(&r, "load transformer core", 1, 50);
    h3_events_log(r.events, H3_LOG_INFO,
                  "LoRA /nonexistent/turbo.safetensors (native): 259 low-rank "
                  "+ 0 full deltas, scale 1 x strength 1, patched 259/259 "
                  "tensors");
    h3_events_log(r.events, H3_LOG_INFO, "LoRA apply 13.42 s");
    progress(&r, "load transformer core", 50, 50);
    progress(&r, "denoise", 6, 6);
    progress(&r, "FFmpeg", 22, 22);
    h3_stage_tracker_finish(&r.tracker);
    h3_result result = t2va_result();
    h3_lora_report lora = {"/nonexistent/turbo.safetensors", 0.75f, "native",
                           259, 0, 259, 259, 1.0f};
    result.loras = &lora;
    result.lora_count = 1;
    result.lora_apply_seconds = 13.42;
    h3_events_metric(r.events, "load_transformer", H3_METRIC_LORA_APPLY,
                     result.lora_apply_seconds, "s");
    h3_events_artifact(r.events, H3_ARTIFACT_VIDEO, "/nonexistent/out/video.mp4");
    h3_events_artifact(r.events, H3_ARTIFACT_SETTINGS, "/nonexistent/out/video.json");
    h3_events_complete(r.events, "/nonexistent/out/video.mp4",
                       "/nonexistent/out/video.json", &result, 58.25);
    finish(&r, "lora", 1);
}

static void test_failure(void) {
    run r;
    run_open(&r, "failure");
    progress(&r, "tokenizer", 0, 1);
    progress(&r, "tokenizer", 1, 1);
    progress(&r, "Qwen vision", 1, 27);
    /* The render fails inside a stage: it stays open (the reader closes
     * it), and the failure is the job.end, never a keyword in a log. */
    h3_events_log(r.events, H3_LOG_WARNING,
                  "reference video 1 runs 10.0s; only its first 56 frames "
                  "(2.3s) condition the render");
    h3_events_failed(r.events,
                     "FFmpeg could not decode image /nonexistent/a \"b\".png "
                     "(status 69)", 0.75);
    finish(&r, "failure", 0);
}

static void test_escaping(void) {
    run r;
    run_open(&r, "escaping");
    h3_events_log(r.events, H3_LOG_INFO,
                  "quote \" backslash \\ newline \n tab \t caf\xc3\xa9 "
                  "\xe6\xbc\xa2 bad \xff\xfe end");
    h3_events_artifact(r.events, H3_ARTIFACT_PREVIEW,
                       "/nonexistent/pr\xc3\xa9view \"1\".png");
    h3_events_failed(r.events, "control \x01 char", 0.0);
    finish(&r, "escaping", 0);
}

static void test_tracker_rules(void) {
    /* Unknown labels, reopened stages and backwards counts are not
     * reported; each stage starts and ends once. */
    run r;
    run_open(&r, "rules");
    progress(&r, "tokenizer", 1, 1);
    progress(&r, "not a real counter", 3, 9);
    progress(&r, "Qwen vision", 27, 27);
    progress(&r, "video VAE encoder", 1, 9);    /* backwards in-stage */
    progress(&r, "audio VAE encoder", 9, 9);
    progress(&r, "text encoder", 50, 50);
    progress(&r, "tokenizer", 0, 1);            /* already ended */
    progress(&r, "FFmpeg", 22, 22);
    h3_stage_tracker_finish(&r.tracker);
    h3_stage_tracker_finish(&r.tracker);        /* idempotent */
    h3_result result = t2va_result();
    h3_events_complete(r.events, NULL, NULL, &result, 1.0);
    finish(&r, "rules", 1);
}

typedef struct {
    int level;
    char text[256];
    int calls;
} captured_log;

static void capture_log(h3_log_level level, const char *message, void *opaque) {
    captured_log *log = opaque;
    log->level = (int)level;
    snprintf(log->text, sizeof(log->text), "%s", message);
    log->calls++;
}

static void test_log_funnel(void) {
    /* Default: the exact bytes, on stderr. */
    int pipes[2];
    if (pipe(pipes)) die("pipe");
    fflush(stderr);
    int saved = dup(2);
    dup2(pipes[1], 2);
    h3_log(H3_LOG_WARNING, "h3: warning: %d LoRA target tensors were never "
           "loaded\n", 3);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    close(pipes[1]);
    char buffer[256] = "";
    ssize_t got = read(pipes[0], buffer, sizeof(buffer) - 1);
    close(pipes[0]);
    if (got < 0 || strcmp(buffer,
            "h3: warning: 3 LoRA target tensors were never loaded\n"))
        die("default log output changed");
    /* With a callback: level, prefix stripped, no newline, no stderr. */
    captured_log log;
    memset(&log, 0, sizeof(log));
    h3_set_log_callback(capture_log, &log);
    h3_log(H3_LOG_WARNING, "h3: warning: %d LoRA target tensors were never "
           "loaded\n", 3);
    if (log.calls != 1 || log.level != H3_LOG_WARNING ||
        strcmp(log.text, "3 LoRA target tensors were never loaded"))
        die("log callback did not receive the structured message");
    h3_log(H3_LOG_DEBUG, "h3 profile: %-8s total wall=%.3fs\n", "VAE", 1.5);
    if (log.level != H3_LOG_DEBUG ||
        strcmp(log.text, "h3 profile: VAE      total wall=1.500s"))
        die("profile line lost its text");
    h3_set_log_callback(NULL, NULL);
}

typedef struct {
    h3_events *events;
    int thread;
} writer_arg;

static void *writer(void *opaque) {
    writer_arg *arg = opaque;
    for (int index = 0; index < 500; index++) {
        h3_stage_event event;
        memset(&event, 0, sizeof(event));
        event.kind = H3_STAGE_PROGRESS;
        event.stage = arg->thread % 2 ? "denoise" : "vae_decode";
        event.label = "a label long enough that a torn write would show";
        event.completed = index;
        event.total = 500;
        h3_events_stage(arg->events, &event);
    }
    return NULL;
}

static void test_concurrent_writes(void) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/h3-events-%d-threads.jsonl", (int)getpid());
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    h3_events *events = h3_events_open(fd, "v", "c");
    pthread_t threads[4];
    writer_arg args[4];
    for (int index = 0; index < 4; index++) {
        args[index] = (writer_arg){events, index};
        pthread_create(&threads[index], NULL, writer, &args[index]);
    }
    for (int index = 0; index < 4; index++) pthread_join(threads[index], NULL);
    h3_events_close(events);
    char *text = slurp(path, NULL);
    int lines = 0;
    for (char *line = text; *line;) {
        char *end = strchr(line, '\n');
        if (!end || strncmp(line, "{\"type\":", 8) || end[-1] != '}')
            die("a line was torn by a concurrent write");
        lines++;
        line = end + 1;
    }
    if (lines != 1 + 4 * 500) die("concurrent writes lost lines");
    free(text);
    unlink(path);
}

static void test_bad_descriptor(void) {
    if (h3_events_open(987, "v", "c")) die("accepted a closed descriptor");
    int fd = open("/dev/null", O_RDONLY);
    if (h3_events_open(fd, "v", "c")) die("accepted a read-only descriptor");
    close(fd);
}

static void test_reader_gone(void) {
    /* A closed reader must not kill the render (SIGPIPE) or block it. */
    int pipes[2];
    if (pipe(pipes)) die("pipe");
    h3_events *events = h3_events_open(pipes[1], "v", "c");
    close(pipes[0]);
    for (int index = 0; index < 100; index++)
        h3_events_log(events, H3_LOG_INFO, "nobody is listening");
    h3_events_close(events);
}

int main(void) {
    if (h3_stage_for_label("text encoder") != H3_STAGE_ID_TEXT_ENCODE ||
        strcmp(h3_stage_name(H3_STAGE_ID_MUX), "mux"))
        die("stage table");
    test_t2va();
    test_lora();
    test_failure();
    test_escaping();
    test_tracker_rules();
    test_log_funnel();
    test_concurrent_writes();
    test_bad_descriptor();
    test_reader_gone();
    printf("events: all tests passed\n");
    return 0;
}
