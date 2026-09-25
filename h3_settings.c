#include "h3_settings.h"
#include "h3_build_info.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

char *h3_settings_path_for(const char *output_path) {
    if (!output_path || !*output_path) return NULL;
    const char *slash = strrchr(output_path, '/');
    const char *dot = strrchr(output_path, '.');
    size_t stem = dot && (!slash || dot > slash + 1) ?
        (size_t)(dot - output_path) : strlen(output_path);
    char *path = malloc(stem + sizeof(".json"));
    if (!path) return NULL;
    memcpy(path, output_path, stem);
    memcpy(path + stem, ".json", sizeof(".json"));
    return path;
}

/* ---- writing ------------------------------------------------------------ */

static void write_string(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor; cursor++) {
        switch (*cursor) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (*cursor < 0x20) fprintf(file, "\\u%04x", *cursor);
                else fputc(*cursor, file);
        }
    }
    fputc('"', file);
}

/* Absolute path when the file exists, otherwise the path as given. */
static void write_path(FILE *file, const char *path) {
    if (!path) {
        fputs("null", file);
        return;
    }
    char resolved[PATH_MAX];
    write_string(file, realpath(path, resolved) ? resolved : path);
}

static void write_bool(FILE *file, const char *key, int value) {
    fprintf(file, ",\n  \"%s\": %s", key, value ? "true" : "false");
}

static void write_int(FILE *file, const char *key, int value) {
    fprintf(file, ",\n  \"%s\": %d", key, value);
}

static const char *reference_kind(const h3_reference *reference) {
    switch (reference->kind) {
        case H3_REFERENCE_IMAGE: return "image";
        case H3_REFERENCE_VIDEO:
            return reference->include_embedded_audio ? "video" :
                                                       "silent-video";
        case H3_REFERENCE_VIDEO_AUDIO: return "video-audio";
        case H3_REFERENCE_AUDIO: return "audio";
        default: return "unknown";
    }
}

int h3_settings_write(const char *output_path, const char *model_dir,
                      const char *prompt, const h3_params *params,
                      const h3_result *result, double elapsed_seconds,
                      char *error, size_t error_size) {
    char *path = h3_settings_path_for(output_path);
    if (!path) {
        fail(error, error_size, "no settings path for output %s",
             output_path ? output_path : "(none)");
        return 0;
    }
    char temporary[PATH_MAX];
    snprintf(temporary, sizeof(temporary), "%s.tmp", path);
    FILE *file = fopen(temporary, "w");
    if (!file) {
        fail(error, error_size, "cannot write %s: %s", temporary,
             strerror(errno));
        free(path);
        return 0;
    }
    char created[64] = "";
    time_t now = time(NULL);
    struct tm local;
    if (localtime_r(&now, &local))
        strftime(created, sizeof(created), "%Y-%m-%dT%H:%M:%S%z", &local);
    fprintf(file, "{\n  \"h3_settings\": %d", H3_SETTINGS_VERSION);
    fputs(",\n  \"h3_version\": ", file);
    write_string(file, H3_VERSION);
    fputs(",\n  \"git_commit\": ", file);
    write_string(file, H3_GIT_COMMIT);
    fputs(",\n  \"created\": ", file);
    write_string(file, created);
    fputs(",\n  \"output\": ", file);
    write_path(file, output_path);
    fputs(",\n  \"model-dir\": ", file);
    write_path(file, model_dir);
    fputs(",\n  \"prompt\": ", file);
    write_string(file, prompt);
    /* A u64 seed does not survive a double, so it is written as a string. */
    fprintf(file, ",\n  \"seed\": \"%" PRIu64 "\"", params->seed);
    write_int(file, "width", params->width);
    write_int(file, "height", params->height);
    write_int(file, "render-width", params->render_width);
    write_int(file, "render-height", params->render_height);
    write_int(file, "frames", params->frames);
    write_int(file, "steps", params->steps);
    write_int(file, "reuse", params->denoise_reuse);
    write_int(file, "layers", params->dit_layers);
    write_int(file, "core-reuse", params->core_reuse);
    write_bool(file, "token-reduction", params->token_reduction);
    write_bool(file, "ssd-streaming", params->ssd_streaming);
    write_bool(file, "use-int8-row-fc2", params->use_int8_row_fc2);
    write_bool(file, "use-reference-rope", params->use_reference_rope);
    write_bool(file, "use-slower-bf16-mlp", params->use_slower_bf16_mlp);
    write_bool(file, "use-slower-bf16-qkv", params->use_slower_bf16_qkv);
    write_bool(file, "use-slower-bf16-attention-output",
               params->use_slower_bf16_attention_output);
    write_bool(file, "use-slower-row-major-attention-output",
               params->use_slower_row_major_attention_output);
    write_bool(file, "use-slower-unfused-int8-inputs",
               params->use_slower_unfused_int8_inputs);
    write_bool(file, "use-slower-unfused-qkv-rope",
               params->use_slower_unfused_qkv_rope);
    write_bool(file, "use-slower-scalar-qkv-rms",
               params->use_slower_scalar_qkv_rms);
    write_bool(file, "use-slower-uncached-int8-scales",
               params->use_slower_uncached_int8_scales);
    write_bool(file, "use-slower-dynamic-fc1-k",
               params->use_slower_dynamic_fc1_k);
    write_bool(file, "use-slower-grouped-quantizer",
               params->use_slower_grouped_quantizer);
    fputs(",\n  \"first-frame\": ", file);
    write_path(file, params->first_frame);
    fputs(",\n  \"last-frame\": ", file);
    write_path(file, params->last_frame);
    fprintf(file, ",\n  \"ref-image-size\": \"%s\"",
            params->reference_image_size == H3_REFERENCE_IMAGE_MAX ?
                "max" : "match");
    fprintf(file, ",\n  \"ref-video-size\": \"%s\"",
            params->reference_video_size == H3_REFERENCE_VIDEO_MATCH ?
                "match" : "auto");
    fputs(",\n  \"references\": [", file);
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        fprintf(file, "%s\n    {\"kind\": \"%s\", \"path\": ",
                index ? "," : "", reference_kind(reference));
        write_path(file, reference->path);
        if (reference->kind == H3_REFERENCE_VIDEO_AUDIO) {
            fputs(", \"audio\": ", file);
            write_path(file, reference->audio_path);
        }
        fputc('}', file);
    }
    fputs(params->reference_count ? "\n  ]" : "]", file);
    fputs(",\n  \"loras\": [", file);
    for (size_t index = 0; index < params->lora_count; index++) {
        fprintf(file, "%s\n    {\"path\": ", index ? "," : "");
        write_path(file, params->loras[index].path);
        /* Shortest decimal that reads back to the identical float. */
        float scale = params->loras[index].scale;
        char text[32];
        for (int digits = 6; digits <= 9; digits++) {
            snprintf(text, sizeof(text), "%.*g", digits, (double)scale);
            if (strtof(text, NULL) == scale) break;
        }
        fprintf(file, ", \"scale\": %s}", text);
    }
    fputs(params->lora_count ? "\n  ]" : "]", file);
    if (result) {
        fprintf(file, ",\n  \"result\": {\"width\": %d, \"height\": %d, "
                "\"frames\": %d, \"fps\": %d, \"sample_rate\": %d, "
                "\"seed\": \"%" PRIu64 "\"}",
                result->width, result->height, result->frames, result->fps,
                result->sample_rate, result->seed);
    }
    if (elapsed_seconds >= 0.0)
        fprintf(file, ",\n  \"elapsed_seconds\": %.2f", elapsed_seconds);
    fputs("\n}\n", file);
    int ok = !ferror(file);
    if (fclose(file) != 0) ok = 0;
    if (ok && rename(temporary, path) != 0) ok = 0;
    if (!ok) {
        fail(error, error_size, "cannot write %s: %s", path, strerror(errno));
        remove(temporary);
    }
    free(path);
    return ok;
}

/* ---- reading ------------------------------------------------------------ */

typedef enum {
    JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT
} json_type;

typedef struct json_value {
    json_type type;
    int boolean;
    double number;
    char *text;               /* string value, or number source text */
    struct json_value *items; /* array items / object values */
    char **keys;              /* object keys */
    size_t count;
} json_value;

typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
    int depth;
} json_cursor;

static void json_free(json_value *value) {
    if (!value) return;
    free(value->text);
    for (size_t index = 0; index < value->count; index++) {
        json_free(&value->items[index]);
        if (value->keys) free(value->keys[index]);
    }
    free(value->items);
    free(value->keys);
    memset(value, 0, sizeof(*value));
}

static int json_error(json_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size && !cursor->error[0])
        snprintf(cursor->error, cursor->error_size, "%s", message);
    return 0;
}

static void json_ws(json_cursor *cursor) {
    while (cursor->at < cursor->end && isspace((unsigned char)*cursor->at))
        cursor->at++;
}

static void utf8_append(char *out, size_t *length, uint32_t code) {
    if (code < 0x80) {
        out[(*length)++] = (char)code;
    } else if (code < 0x800) {
        out[(*length)++] = (char)(0xc0 | (code >> 6));
        out[(*length)++] = (char)(0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
        out[(*length)++] = (char)(0xe0 | (code >> 12));
        out[(*length)++] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (code & 0x3f));
    } else {
        out[(*length)++] = (char)(0xf0 | (code >> 18));
        out[(*length)++] = (char)(0x80 | ((code >> 12) & 0x3f));
        out[(*length)++] = (char)(0x80 | ((code >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (code & 0x3f));
    }
}

static int hex4(json_cursor *cursor, uint32_t *code) {
    if (cursor->end - cursor->at < 4) return 0;
    *code = 0;
    for (int index = 0; index < 4; index++) {
        char digit = *cursor->at++;
        *code <<= 4;
        if (digit >= '0' && digit <= '9') *code |= (uint32_t)(digit - '0');
        else if (digit >= 'a' && digit <= 'f')
            *code |= (uint32_t)(digit - 'a' + 10);
        else if (digit >= 'A' && digit <= 'F')
            *code |= (uint32_t)(digit - 'A' + 10);
        else return 0;
    }
    return 1;
}

static char *json_string(json_cursor *cursor) {
    json_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        json_error(cursor, "expected a JSON string");
        return NULL;
    }
    cursor->at++;
    /* Escapes never expand, so the raw span bounds the decoded length. */
    char *out = malloc((size_t)(cursor->end - cursor->at) + 1);
    if (!out) {
        json_error(cursor, "out of memory reading settings");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        char c = *cursor->at++;
        if (c != '\\') {
            out[length++] = c;
            continue;
        }
        if (cursor->at >= cursor->end) break;
        char escape = *cursor->at++;
        uint32_t code;
        switch (escape) {
            case '"': out[length++] = '"'; break;
            case '\\': out[length++] = '\\'; break;
            case '/': out[length++] = '/'; break;
            case 'b': out[length++] = '\b'; break;
            case 'f': out[length++] = '\f'; break;
            case 'n': out[length++] = '\n'; break;
            case 'r': out[length++] = '\r'; break;
            case 't': out[length++] = '\t'; break;
            case 'u':
                if (!hex4(cursor, &code)) goto bad;
                if (code >= 0xd800 && code < 0xdc00) {
                    uint32_t low;
                    if (cursor->end - cursor->at < 6 || cursor->at[0] != '\\' ||
                        cursor->at[1] != 'u') goto bad;
                    cursor->at += 2;
                    if (!hex4(cursor, &low) || low < 0xdc00 || low >= 0xe000)
                        goto bad;
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                }
                utf8_append(out, &length, code);
                break;
            default: goto bad;
        }
    }
    if (cursor->at >= cursor->end) goto bad;
    cursor->at++;
    out[length] = '\0';
    return out;
bad:
    free(out);
    json_error(cursor, "malformed JSON string");
    return NULL;
}

static int json_parse(json_cursor *cursor, json_value *value);

static int json_append(json_value *container, size_t *capacity,
                       json_value *item, char *key, json_cursor *cursor) {
    if (container->count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 8;
        json_value *items = realloc(container->items, next * sizeof(*items));
        if (!items) return json_error(cursor, "out of memory reading settings");
        container->items = items;
        if (container->type == JSON_OBJECT) {
            char **keys = realloc(container->keys, next * sizeof(*keys));
            if (!keys) return json_error(cursor, "out of memory");
            container->keys = keys;
        }
        *capacity = next;
    }
    container->items[container->count] = *item;
    if (container->type == JSON_OBJECT)
        container->keys[container->count] = key;
    container->count++;
    return 1;
}

static int json_compound(json_cursor *cursor, json_value *value, int object) {
    value->type = object ? JSON_OBJECT : JSON_ARRAY;
    cursor->at++;
    if (++cursor->depth > 32) return json_error(cursor, "settings too deep");
    size_t capacity = 0;
    json_ws(cursor);
    if (cursor->at < cursor->end && *cursor->at == (object ? '}' : ']')) {
        cursor->at++;
        cursor->depth--;
        return 1;
    }
    for (;;) {
        char *key = NULL;
        if (object) {
            key = json_string(cursor);
            if (!key) return 0;
            json_ws(cursor);
            if (cursor->at >= cursor->end || *cursor->at++ != ':') {
                free(key);
                return json_error(cursor, "expected ':' in settings");
            }
        }
        json_value item;
        memset(&item, 0, sizeof(item));
        if (!json_parse(cursor, &item) ||
            !json_append(value, &capacity, &item, key, cursor)) {
            json_free(&item);
            free(key);
            return 0;
        }
        json_ws(cursor);
        if (cursor->at >= cursor->end)
            return json_error(cursor, "unterminated settings value");
        char next = *cursor->at++;
        if (next == (object ? '}' : ']')) break;
        if (next != ',') return json_error(cursor, "expected ',' in settings");
    }
    cursor->depth--;
    return 1;
}

static int json_parse(json_cursor *cursor, json_value *value) {
    memset(value, 0, sizeof(*value));
    json_ws(cursor);
    if (cursor->at >= cursor->end) return json_error(cursor, "missing value");
    char c = *cursor->at;
    if (c == '{') return json_compound(cursor, value, 1);
    if (c == '[') return json_compound(cursor, value, 0);
    if (c == '"') {
        value->type = JSON_STRING;
        value->text = json_string(cursor);
        return value->text != NULL;
    }
    static const struct {
        const char *word;
        json_type type;
        int boolean;
    } literals[] = {{"true", JSON_BOOL, 1}, {"false", JSON_BOOL, 0},
                    {"null", JSON_NULL, 0}};
    for (size_t index = 0; index < 3; index++) {
        size_t length = strlen(literals[index].word);
        if ((size_t)(cursor->end - cursor->at) >= length &&
            !strncmp(cursor->at, literals[index].word, length)) {
            cursor->at += length;
            value->type = literals[index].type;
            value->boolean = literals[index].boolean;
            return 1;
        }
    }
    const char *start = cursor->at;
    while (cursor->at < cursor->end &&
           (isdigit((unsigned char)*cursor->at) || strchr("+-.eE", *cursor->at)))
        cursor->at++;
    if (cursor->at == start) return json_error(cursor, "invalid settings value");
    value->type = JSON_NUMBER;
    value->text = strndup(start, (size_t)(cursor->at - start));
    if (!value->text) return json_error(cursor, "out of memory");
    char *end = NULL;
    value->number = strtod(value->text, &end);
    if (!end || *end || !isfinite(value->number))
        return json_error(cursor, "invalid settings number");
    return 1;
}

static char *own(h3_settings *settings, const char *text) {
    if (settings->owned_count == settings->owned_capacity) {
        size_t next = settings->owned_capacity ? settings->owned_capacity * 2 :
                                                 16;
        char **owned = realloc(settings->owned, next * sizeof(*owned));
        if (!owned) return NULL;
        settings->owned = owned;
        settings->owned_capacity = next;
    }
    char *copy = strdup(text);
    if (copy) settings->owned[settings->owned_count++] = copy;
    return copy;
}

static const json_value *member(const json_value *object, const char *key) {
    if (!object || object->type != JSON_OBJECT) return NULL;
    for (size_t index = 0; index < object->count; index++) {
        if (!strcmp(object->keys[index], key)) return &object->items[index];
    }
    return NULL;
}

typedef struct {
    const char *key;
    size_t offset;
} settings_field;

#define INT_FIELD(key, field) {key, offsetof(h3_params, field)}
static const settings_field int_fields[] = {
    INT_FIELD("width", width), INT_FIELD("height", height),
    INT_FIELD("render-width", render_width),
    INT_FIELD("render-height", render_height),
    INT_FIELD("frames", frames), INT_FIELD("steps", steps),
    INT_FIELD("reuse", denoise_reuse), INT_FIELD("layers", dit_layers),
    INT_FIELD("core-reuse", core_reuse),
};
static const settings_field bool_fields[] = {
    INT_FIELD("token-reduction", token_reduction),
    INT_FIELD("ssd-streaming", ssd_streaming),
    INT_FIELD("use-int8-row-fc2", use_int8_row_fc2),
    INT_FIELD("use-reference-rope", use_reference_rope),
    INT_FIELD("use-slower-bf16-mlp", use_slower_bf16_mlp),
    INT_FIELD("use-slower-bf16-qkv", use_slower_bf16_qkv),
    INT_FIELD("use-slower-bf16-attention-output",
              use_slower_bf16_attention_output),
    INT_FIELD("use-slower-row-major-attention-output",
              use_slower_row_major_attention_output),
    INT_FIELD("use-slower-unfused-int8-inputs",
              use_slower_unfused_int8_inputs),
    INT_FIELD("use-slower-unfused-qkv-rope", use_slower_unfused_qkv_rope),
    INT_FIELD("use-slower-scalar-qkv-rms", use_slower_scalar_qkv_rms),
    INT_FIELD("use-slower-uncached-int8-scales",
              use_slower_uncached_int8_scales),
    INT_FIELD("use-slower-dynamic-fc1-k", use_slower_dynamic_fc1_k),
    INT_FIELD("use-slower-grouped-quantizer", use_slower_grouped_quantizer),
};
#undef INT_FIELD

/* Written for provenance only; never applied on load. */
static const char *informational_keys[] = {
    "h3_settings", "h3_version", "git_commit", "created", "output", "result",
    "elapsed_seconds"
};

static int load_references(h3_settings *settings, const json_value *array,
                           char *error, size_t error_size) {
    if (array->type != JSON_ARRAY ||
        array->count > H3_SETTINGS_MAX_REFERENCES) {
        fail(error, error_size, "\"references\" must be an array of at most "
             "%d entries", H3_SETTINGS_MAX_REFERENCES);
        return 0;
    }
    for (size_t index = 0; index < array->count; index++) {
        const json_value *item = &array->items[index];
        const json_value *kind = member(item, "kind");
        const json_value *path = member(item, "path");
        const json_value *audio = member(item, "audio");
        if (!kind || kind->type != JSON_STRING || !path ||
            path->type != JSON_STRING) {
            fail(error, error_size, "reference %zu needs string kind and path",
                 index + 1);
            return 0;
        }
        h3_reference *reference = &settings->references[index];
        memset(reference, 0, sizeof(*reference));
        if (!strcmp(kind->text, "image")) {
            reference->kind = H3_REFERENCE_IMAGE;
        } else if (!strcmp(kind->text, "video")) {
            reference->kind = H3_REFERENCE_VIDEO;
            reference->include_embedded_audio = 1;
        } else if (!strcmp(kind->text, "silent-video")) {
            reference->kind = H3_REFERENCE_VIDEO;
        } else if (!strcmp(kind->text, "video-audio")) {
            reference->kind = H3_REFERENCE_VIDEO_AUDIO;
            if (!audio || audio->type != JSON_STRING) {
                fail(error, error_size, "video-audio reference %zu needs "
                     "\"audio\"", index + 1);
                return 0;
            }
            if (!(reference->audio_path = own(settings, audio->text)))
                goto oom;
        } else if (!strcmp(kind->text, "audio")) {
            reference->kind = H3_REFERENCE_AUDIO;
        } else {
            fail(error, error_size, "reference %zu has unknown kind \"%s\"",
                 index + 1, kind->text);
            return 0;
        }
        if (!(reference->path = own(settings, path->text))) goto oom;
    }
    settings->reference_count = array->count;
    return 1;
oom:
    fail(error, error_size, "out of memory reading settings");
    return 0;
}

static int load_loras(h3_settings *settings, const json_value *array,
                      char *error, size_t error_size) {
    if (array->type != JSON_ARRAY || array->count > H3_MAX_LORAS) {
        fail(error, error_size, "\"loras\" must be an array of at most %d "
             "entries", H3_MAX_LORAS);
        return 0;
    }
    for (size_t index = 0; index < array->count; index++) {
        const json_value *path = member(&array->items[index], "path");
        const json_value *scale = member(&array->items[index], "scale");
        if (!path || path->type != JSON_STRING ||
            (scale && scale->type != JSON_NUMBER)) {
            fail(error, error_size, "LoRA %zu needs a string path and "
                 "numeric scale", index + 1);
            return 0;
        }
        settings->loras[index].path = own(settings, path->text);
        settings->loras[index].scale = scale ? (float)scale->number : 1.0f;
        if (!settings->loras[index].path) {
            fail(error, error_size, "out of memory reading settings");
            return 0;
        }
    }
    settings->lora_count = array->count;
    return 1;
}

static int apply_member(h3_settings *settings, const char *key,
                        const json_value *value, char *error,
                        size_t error_size) {
    h3_params *params = &settings->params;
    for (size_t index = 0;
         index < sizeof(informational_keys) / sizeof(*informational_keys);
         index++) {
        if (!strcmp(key, informational_keys[index])) return 1;
    }
    for (size_t index = 0; index < sizeof(int_fields) / sizeof(*int_fields);
         index++) {
        if (strcmp(key, int_fields[index].key)) continue;
        if (value->type != JSON_NUMBER || value->number < 0 ||
            value->number > INT32_MAX ||
            value->number != floor(value->number)) {
            fail(error, error_size, "\"%s\" must be a non-negative integer",
                 key);
            return 0;
        }
        *(int *)((char *)params + int_fields[index].offset) =
            (int)value->number;
        return 1;
    }
    for (size_t index = 0; index < sizeof(bool_fields) / sizeof(*bool_fields);
         index++) {
        if (strcmp(key, bool_fields[index].key)) continue;
        if (value->type != JSON_BOOL) {
            fail(error, error_size, "\"%s\" must be true or false", key);
            return 0;
        }
        *(int *)((char *)params + bool_fields[index].offset) = value->boolean;
        return 1;
    }
    if (!strcmp(key, "seed")) {
        const char *text = value->type == JSON_STRING ||
                           value->type == JSON_NUMBER ? value->text : NULL;
        char *end = NULL;
        errno = 0;
        unsigned long long seed = text ? strtoull(text, &end, 10) : 0;
        if (!text || errno || !end || *end || *text == '-') {
            fail(error, error_size, "\"seed\" must be an unsigned integer");
            return 0;
        }
        params->seed = (uint64_t)seed;
        return 1;
    }
    if (!strcmp(key, "model-dir") || !strcmp(key, "prompt") ||
        !strcmp(key, "first-frame") || !strcmp(key, "last-frame")) {
        if (value->type == JSON_NULL) return 1;
        if (value->type != JSON_STRING) {
            fail(error, error_size, "\"%s\" must be a string or null", key);
            return 0;
        }
        char *copy = own(settings, value->text);
        if (!copy) {
            fail(error, error_size, "out of memory reading settings");
            return 0;
        }
        if (!strcmp(key, "model-dir")) settings->model_dir = copy;
        else if (!strcmp(key, "prompt")) settings->prompt = copy;
        else if (!strcmp(key, "first-frame")) params->first_frame = copy;
        else params->last_frame = copy;
        return 1;
    }
    if (!strcmp(key, "ref-image-size") || !strcmp(key, "ref-video-size")) {
        int image = !strcmp(key, "ref-image-size");
        const char *text = value->type == JSON_STRING ? value->text : "";
        if (image && !strcmp(text, "match"))
            params->reference_image_size = H3_REFERENCE_IMAGE_MATCH;
        else if (image && !strcmp(text, "max"))
            params->reference_image_size = H3_REFERENCE_IMAGE_MAX;
        else if (!image && !strcmp(text, "auto"))
            params->reference_video_size = H3_REFERENCE_VIDEO_AUTO;
        else if (!image && !strcmp(text, "match"))
            params->reference_video_size = H3_REFERENCE_VIDEO_MATCH;
        else {
            fail(error, error_size, "invalid \"%s\" value", key);
            return 0;
        }
        return 1;
    }
    if (!strcmp(key, "references"))
        return load_references(settings, value, error, error_size);
    if (!strcmp(key, "loras"))
        return load_loras(settings, value, error, error_size);
    fail(error, error_size, "unknown settings key \"%s\"", key);
    return 0;
}

int h3_settings_load(const char *path, h3_settings *settings,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    memset(settings, 0, sizeof(*settings));
    h3_params defaults = H3_PARAMS_DEFAULT;
    settings->params = defaults;
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail(error, error_size, "cannot open settings %s: %s", path,
             strerror(errno));
        return 0;
    }
    char *text = NULL;
    size_t length = 0, capacity = 0;
    for (;;) {
        if (capacity - length < 4096) {
            capacity = capacity ? capacity * 2 : 16384;
            char *grown = capacity <= (1u << 24) ? realloc(text, capacity) :
                                                   NULL;
            if (!grown) {
                free(text);
                fclose(file);
                fail(error, error_size, "settings %s is too large", path);
                return 0;
            }
            text = grown;
        }
        size_t count = fread(text + length, 1, capacity - length, file);
        length += count;
        if (count == 0) break;
    }
    int read_failed = ferror(file);
    fclose(file);
    if (read_failed) {
        free(text);
        fail(error, error_size, "cannot read settings %s", path);
        return 0;
    }
    json_cursor cursor = {text, text + length, error, error_size, 0};
    json_value root;
    int ok = json_parse(&cursor, &root);
    if (ok) {
        json_ws(&cursor);
        if (cursor.at != cursor.end) ok = json_error(&cursor, "trailing data");
    }
    if (ok && root.type != JSON_OBJECT) {
        fail(error, error_size, "settings %s is not a JSON object", path);
        ok = 0;
    }
    const json_value *version = ok ? member(&root, "h3_settings") : NULL;
    if (ok && (!version || version->type != JSON_NUMBER ||
               version->number != H3_SETTINGS_VERSION)) {
        fail(error, error_size, "%s is not an h3 settings v%d file", path,
             H3_SETTINGS_VERSION);
        ok = 0;
    }
    for (size_t index = 0; ok && index < root.count; index++) {
        ok = apply_member(settings, root.keys[index], &root.items[index],
                          error, error_size);
    }
    if (ok) {
        settings->params.references = settings->references;
        settings->params.reference_count = settings->reference_count;
        settings->params.loras = settings->loras;
        settings->params.lora_count = settings->lora_count;
    } else if (error && error_size && error[0] &&
               !strstr(error, path)) {
        char detail[512];
        snprintf(detail, sizeof(detail), "%s", error);
        fail(error, error_size, "%s: %s", path, detail);
    }
    json_free(&root);
    free(text);
    if (!ok) h3_settings_free(settings);
    return ok;
}

void h3_settings_free(h3_settings *settings) {
    if (!settings) return;
    for (size_t index = 0; index < settings->owned_count; index++)
        free(settings->owned[index]);
    free(settings->owned);
    memset(settings, 0, sizeof(*settings));
}

int h3_settings_parse_lora(char *value, h3_lora *lora) {
    lora->path = value;
    lora->scale = 1.0f;
    char *colon = strrchr(value, ':');
    if (colon && colon[1]) {
        char *end = NULL;
        errno = 0;
        float scale = strtof(colon + 1, &end);
        if (!errno && end && !*end && isfinite(scale)) {
            *colon = '\0';
            lora->scale = scale;
        }
    }
    return *lora->path != '\0';
}
