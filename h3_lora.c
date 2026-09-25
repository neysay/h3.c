#define ACCELERATE_NEW_LAPACK
#include "h3_lora.h"
#include "h3_log.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HEAD_DIM 128u
#define MIN_GEMM_RANK 256u
#define DIT_BLOCKS 50u
#define REFINER_BLOCKS 2u

/* How adapter output rows land on checkpoint rows. The released checkpoint
 * stores fused QKV interleaved per head ([h0: q k v, h1: q k v, ...]) and
 * SwiGLU FC1 as [gate; value]. Every trainer except DiffSynth reorders QKV to
 * [q; k; v] before training, and diffusers splits it into to_q/to_k/to_v and
 * stores FC1 as [value; gate]. */
typedef enum {
    ROWS_SAME = 0,
    ROWS_Q,
    ROWS_K,
    ROWS_V,
    ROWS_QKV_GROUPED,
    ROWS_FC1_SWAPPED
} h3_lora_rows;

typedef struct {
    char *target;
    const h3_st_tensor *down;  /* lora_A [rank, in], or the full delta */
    const h3_st_tensor *up;    /* lora_B [out, rank]; NULL for full deltas */
    size_t adapter;
    float scale;
    h3_lora_rows rows;
    int applied;
} h3_lora_entry;

struct h3_lora_set {
    h3_st_header *headers;
    h3_lora_stats *stats;
    size_t adapter_count;
    h3_lora_entry *entries;
    size_t entry_count;
    size_t entry_capacity;
    float *accumulator;       /* reused across tensors; grows to the largest */
    size_t accumulator_elements;
    double apply_seconds;
    double phase_seconds[4];  /* H3_LORA_TIMING: read, widen, gemm, narrow */
    /* F32 GPU scratch reused across tensors: allocating it per tensor
     * zero-fills tens of gigabytes of fresh pages over a full load. */
    h3_gpu_tensor *scratch_sum;
    h3_gpu_tensor *scratch_product;
    size_t scratch_elements;
};

typedef struct {
    char *module;
    const h3_st_tensor *down;
    const h3_st_tensor *up;
    const h3_st_tensor *alpha;
    const h3_st_tensor *diff;
    const h3_st_tensor *diff_b;
} h3_lora_module;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int starts_with(const char *text, const char *prefix) {
    return !strncmp(text, prefix, strlen(prefix));
}

static int ends_with(const char *text, const char *suffix) {
    size_t length = strlen(text), suffix_length = strlen(suffix);
    return length >= suffix_length &&
           !strcmp(text + length - suffix_length, suffix);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((bits >> 16) | 0x40u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (!exponent) {
        if (!mantissa) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float *read_f32(const h3_st_header *header, const h3_st_tensor *tensor,
                       char *error, size_t error_size) {
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t item = h3_dtype_size(tensor->dtype);
    if (tensor->dtype != H3_DTYPE_BF16 && tensor->dtype != H3_DTYPE_F16 &&
        tensor->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "%s: adapter tensor %s has unsupported dtype %s",
             header->path, tensor->name, h3_dtype_name(tensor->dtype));
        return NULL;
    }
    if (!elements || elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "%s: adapter tensor %s is empty or too large",
             header->path, tensor->name);
        return NULL;
    }
    float *values = malloc((size_t)elements * sizeof(float));
    void *raw = tensor->dtype == H3_DTYPE_F32 ? (void *)values :
                malloc((size_t)elements * item);
    if (!values || !raw) {
        free(values);
        if (raw != values) free(raw);
        fail(error, error_size, "out of memory reading adapter tensor %s",
             tensor->name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, raw, (size_t)elements * item,
                         error, error_size)) {
        if (raw != values) free(raw);
        free(values);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_F32) {
        const uint16_t *half = raw;
        for (size_t index = 0; index < (size_t)elements; index++) {
            values[index] = tensor->dtype == H3_DTYPE_BF16 ?
                bf16_to_f32(half[index]) : f16_to_f32(half[index]);
        }
        free(raw);
    }
    return values;
}

/* Parse "<prefix><index>.<rest>" and return rest, or NULL. */
static const char *block_rest(const char *module, const char *prefix,
                              unsigned limit, unsigned *index) {
    if (!starts_with(module, prefix)) return NULL;
    const char *cursor = module + strlen(prefix);
    char *end = NULL;
    unsigned long parsed = strtoul(cursor, &end, 10);
    if (end == cursor || *end != '.' || parsed >= limit) return NULL;
    *index = (unsigned)parsed;
    return end + 1;
}

typedef struct {
    const char *source;
    const char *target;
    h3_lora_rows rows;
} h3_lora_rename;

/* Per-block modules, adapter name -> checkpoint name. */
static const h3_lora_rename diffusers_block[] = {
    {"attn.to_q", "attn.qkv_proj", ROWS_Q},
    {"attn.to_k", "attn.qkv_proj", ROWS_K},
    {"attn.to_v", "attn.qkv_proj", ROWS_V},
    {"attn.to_out.0", "attn.out_proj", ROWS_SAME},
    {"attn.norm_q", "attn.q_norm", ROWS_SAME},
    {"attn.norm_k", "attn.k_norm", ROWS_SAME},
    {"ff.net.0.proj", "mlp.fc1", ROWS_FC1_SWAPPED},
    {"ff.net.2", "mlp.fc2", ROWS_SAME},
    {"adaln_proj.linear", "adaln_proj.linear", ROWS_SAME},
    {"norm1", "norm1", ROWS_SAME},
    {"norm2", "norm2", ROWS_SAME},
};

static const h3_lora_rename native_block[] = {
    {"attn.qkv_proj", "attn.qkv_proj", ROWS_QKV_GROUPED},
    {"attn.out_proj", "attn.out_proj", ROWS_SAME},
    {"attn.q_norm", "attn.q_norm", ROWS_SAME},
    {"attn.k_norm", "attn.k_norm", ROWS_SAME},
    {"mlp.fc1", "mlp.fc1", ROWS_SAME},
    {"mlp.fc2", "mlp.fc2", ROWS_SAME},
    {"adaln_proj.linear", "adaln_proj.linear", ROWS_SAME},
    {"norm1", "norm1", ROWS_SAME},
    {"norm2", "norm2", ROWS_SAME},
};

static const h3_lora_rename diffusers_global[] = {
    {"proj_in", "video_patch_proj", ROWS_SAME},
    {"audio_proj_in", "audio_patch_proj", ROWS_SAME},
    {"context_embedder", "condition_proj", ROWS_SAME},
    {"time_embedder.linear_1", "time_embedder.proj_in", ROWS_SAME},
    {"time_embedder.linear_2", "time_embedder.proj_out", ROWS_SAME},
    {"norm_out.linear", "final_layer.adaln_proj.linear", ROWS_SAME},
    {"norm_out.norm", "final_layer.norm", ROWS_SAME},
    {"proj_out", "final_layer.video_out", ROWS_SAME},
    {"audio_proj_out", "final_layer.audio_out", ROWS_SAME},
    {"token_refiner.final_norm", "token_refiner.final_norm", ROWS_SAME},
};

static const h3_lora_rename native_global[] = {
    {"video_patch_proj", "video_patch_proj", ROWS_SAME},
    {"audio_patch_proj", "audio_patch_proj", ROWS_SAME},
    {"condition_proj", "condition_proj", ROWS_SAME},
    {"time_embedder.proj_in", "time_embedder.proj_in", ROWS_SAME},
    {"time_embedder.proj_out", "time_embedder.proj_out", ROWS_SAME},
    {"final_layer.adaln_proj.linear", "final_layer.adaln_proj.linear",
     ROWS_SAME},
    {"final_layer.norm", "final_layer.norm", ROWS_SAME},
    {"final_layer.video_out", "final_layer.video_out", ROWS_SAME},
    {"final_layer.audio_out", "final_layer.audio_out", ROWS_SAME},
    {"token_refiner.final_norm", "token_refiner.final_norm", ROWS_SAME},
};

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))

static const h3_lora_rename *find_rename(const h3_lora_rename *table,
                                         size_t count, const char *name) {
    for (size_t index = 0; index < count; index++) {
        if (!strcmp(table[index].source, name)) return &table[index];
    }
    return NULL;
}

static int resolve_module(const char *module, int diffusers, int interleaved,
                          char *target, size_t target_size,
                          h3_lora_rows *rows) {
    unsigned index = 0;
    const char *rest;
    const h3_lora_rename *rename;
    const h3_lora_rename *block_table = diffusers ? diffusers_block :
                                                    native_block;
    size_t block_count = diffusers ? COUNT(diffusers_block) :
                                     COUNT(native_block);
    const char *main_prefix = diffusers ? "transformer_blocks." : "blocks.";
    const char *refiner_prefix = diffusers ? "token_refiner.refiner_blocks." :
                                             "token_refiner.blocks.";
    if ((rest = block_rest(module, main_prefix, DIT_BLOCKS, &index))) {
        rename = find_rename(block_table, block_count, rest);
        if (!rename) return 0;
        snprintf(target, target_size, "blocks.%u.%s", index, rename->target);
    } else if ((rest = block_rest(module, refiner_prefix, REFINER_BLOCKS,
                                  &index))) {
        rename = find_rename(block_table, block_count, rest);
        if (!rename || !strcmp(rename->target, "adaln_proj.linear")) return 0;
        snprintf(target, target_size, "token_refiner.blocks.%u.%s", index,
                 rename->target);
    } else {
        rename = diffusers ?
            find_rename(diffusers_global, COUNT(diffusers_global), module) :
            find_rename(native_global, COUNT(native_global), module);
        if (!rename) return 0;
        snprintf(target, target_size, "%s", rename->target);
    }
    *rows = rename->rows;
    /* DiffSynth trains on the raw checkpoint's per-head-interleaved QKV. */
    if (*rows == ROWS_QKV_GROUPED && interleaved) *rows = ROWS_SAME;
    return 1;
}

static int push_entry(h3_lora_set *set, const char *target,
                      const h3_st_tensor *down, const h3_st_tensor *up,
                      size_t adapter, float scale, h3_lora_rows rows) {
    if (set->entry_count == set->entry_capacity) {
        size_t next = set->entry_capacity ? set->entry_capacity * 2 : 256;
        h3_lora_entry *entries = realloc(set->entries,
                                         next * sizeof(*entries));
        if (!entries) return 0;
        set->entries = entries;
        set->entry_capacity = next;
    }
    char *copy = strdup(target);
    if (!copy) return 0;
    h3_lora_entry *entry = &set->entries[set->entry_count++];
    memset(entry, 0, sizeof(*entry));
    entry->target = copy;
    entry->down = down;
    entry->up = up;
    entry->adapter = adapter;
    entry->scale = scale;
    entry->rows = rows;
    return 1;
}

/* Strip trainer wrappers so the remaining key names a module plus a suffix. */
static void normalize_key(const char *key, char *out, size_t out_size) {
    static const char *prefixes[] = {
        "base_model.model.", "model.diffusion_model.", "diffusion_model.",
        "transformer."
    };
    const char *cursor = key;
    for (int changed = 1; changed;) {
        changed = 0;
        for (size_t index = 0; index < COUNT(prefixes); index++) {
            if (starts_with(cursor, prefixes[index])) {
                cursor += strlen(prefixes[index]);
                changed = 1;
            }
        }
    }
    snprintf(out, out_size, "%s", cursor);
    char *infix;
    while ((infix = strstr(out, ".default."))) {
        memmove(infix, infix + 8, strlen(infix + 8) + 1);
    }
}

static h3_lora_module *module_slot(h3_lora_module **modules, size_t *count,
                                   size_t *capacity, const char *name) {
    for (size_t index = 0; index < *count; index++) {
        if (!strcmp((*modules)[index].module, name)) return &(*modules)[index];
    }
    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 128;
        h3_lora_module *grown = realloc(*modules, next * sizeof(*grown));
        if (!grown) return NULL;
        *modules = grown;
        *capacity = next;
    }
    h3_lora_module *slot = &(*modules)[(*count)++];
    memset(slot, 0, sizeof(*slot));
    slot->module = strdup(name);
    if (!slot->module) {
        (*count)--;
        return NULL;
    }
    return slot;
}

static int read_scalar(const h3_st_header *header, const h3_st_tensor *tensor,
                       float *value, char *error, size_t error_size) {
    if (h3_st_tensor_elements(tensor) != 1) {
        fail(error, error_size, "%s: %s is not a scalar", header->path,
             tensor->name);
        return 0;
    }
    float *values = read_f32(header, tensor, error, error_size);
    if (!values) return 0;
    *value = values[0];
    free(values);
    return 1;
}

static int open_adapter(h3_lora_set *set, size_t adapter,
                        const h3_lora *spec, char *error,
                        size_t error_size) {
    h3_st_header *header = &set->headers[adapter];
    h3_lora_stats *stats = &set->stats[adapter];
    if (!h3_st_read_header(spec->path, header, error, error_size)) return 0;
    int diffusers = 0, interleaved = 0;
    for (size_t index = 0; index < header->tensor_count; index++) {
        const char *key = header->tensors[index].name;
        if (starts_with(key, "lora_unet_")) {
            fail(error, error_size,
                 "%s: musubi-tuner flattened lora_unet_ keys are not "
                 "supported; convert the adapter to diffusers or native keys",
                 spec->path);
            return 0;
        }
        if (strstr(key, "transformer_blocks.") ||
            strstr(key, "refiner_blocks.")) diffusers = 1;
        if (strstr(key, ".lora_A.default.") ||
            strstr(key, ".lora_B.default.")) interleaved = 1;
    }
    stats->format = diffusers ? "diffusers" :
                    interleaved ? "native-interleaved" : "native";

    h3_lora_module *modules = NULL;
    size_t module_count = 0, module_capacity = 0;
    int ok = 0;
    for (size_t index = 0; index < header->tensor_count; index++) {
        const h3_st_tensor *tensor = &header->tensors[index];
        char key[512];
        normalize_key(tensor->name, key, sizeof(key));
        static const struct {
            const char *suffix;
            int role;
        } suffixes[] = {
            {".lora_A.weight", 0}, {".lora_down.weight", 0},
            {".lora_B.weight", 1}, {".lora_up.weight", 1},
            {".alpha", 2}, {".diff", 3}, {".diff_b", 4},
        };
        int role = -1;
        size_t suffix_length = 0;
        for (size_t choice = 0; choice < COUNT(suffixes); choice++) {
            if (ends_with(key, suffixes[choice].suffix)) {
                role = suffixes[choice].role;
                suffix_length = strlen(suffixes[choice].suffix);
                break;
            }
        }
        if (role < 0) {
            fail(error, error_size,
                 "%s: unsupported adapter tensor %s (expected lora_A/lora_B, "
                 "lora_down/lora_up, alpha, diff, or diff_b)",
                 spec->path, tensor->name);
            goto done;
        }
        key[strlen(key) - suffix_length] = '\0';
        h3_lora_module *slot = module_slot(&modules, &module_count,
                                           &module_capacity, key);
        if (!slot) {
            fail(error, error_size, "out of memory indexing adapter");
            goto done;
        }
        const h3_st_tensor **field =
            role == 0 ? &slot->down : role == 1 ? &slot->up :
            role == 2 ? &slot->alpha : role == 3 ? &slot->diff :
                        &slot->diff_b;
        if (*field) {
            fail(error, error_size, "%s: duplicate adapter tensor for %s",
                 spec->path, tensor->name);
            goto done;
        }
        *field = tensor;
    }

    float metadata_alpha = 0.0f;
    const char *alpha_text = h3_st_metadata(header, "alpha");
    if (alpha_text) {
        char *end = NULL;
        metadata_alpha = strtof(alpha_text, &end);
        if (end == alpha_text || *end || !isfinite(metadata_alpha) ||
            metadata_alpha <= 0.0f) {
            fail(error, error_size, "%s: invalid alpha metadata \"%s\"",
                 spec->path, alpha_text);
            goto done;
        }
    }

    stats->scale = 0.0f;
    for (size_t index = 0; index < module_count; index++) {
        h3_lora_module *module = &modules[index];
        char target[256];
        h3_lora_rows rows;
        if (!resolve_module(module->module, diffusers, interleaved, target,
                            sizeof(target), &rows)) {
            fail(error, error_size,
                 "%s: %s does not name a MiniMax-H3 DiT module this "
                 "loader can map", spec->path, module->module);
            goto done;
        }
        if (!module->down != !module->up) {
            fail(error, error_size, "%s: %s has only one half of its "
                 "low-rank pair", spec->path, module->module);
            goto done;
        }
        if (module->alpha && !module->down) {
            fail(error, error_size, "%s: %s has alpha but no low-rank pair",
                 spec->path, module->module);
            goto done;
        }
        if (module->down) {
            if (module->down->ndim != 2 || module->up->ndim != 2 ||
                module->down->shape[0] != module->up->shape[1] ||
                !module->down->shape[0]) {
                fail(error, error_size, "%s: %s low-rank shapes disagree",
                     spec->path, module->module);
                goto done;
            }
            float rank = (float)module->down->shape[0];
            float scale = 1.0f;
            if (module->alpha) {
                float alpha;
                if (!read_scalar(header, module->alpha, &alpha, error,
                                 error_size)) goto done;
                scale = alpha / rank;
            } else if (metadata_alpha > 0.0f) {
                scale = metadata_alpha / rank;
            }
            if (stats->scale == 0.0f) stats->scale = scale;
            char name[300];
            snprintf(name, sizeof(name), "%s.weight", target);
            if (!push_entry(set, name, module->down, module->up, adapter,
                            scale * spec->scale, rows)) {
                fail(error, error_size, "out of memory indexing adapter");
                goto done;
            }
            stats->low_rank++;
        }
        if (module->diff) {
            char name[300];
            snprintf(name, sizeof(name), "%s.weight", target);
            if (!push_entry(set, name, module->diff, NULL, adapter,
                            spec->scale, rows)) {
                fail(error, error_size, "out of memory indexing adapter");
                goto done;
            }
            stats->full_deltas++;
        }
        if (module->diff_b) {
            char name[300];
            snprintf(name, sizeof(name), "%s.bias", target);
            if (!push_entry(set, name, module->diff_b, NULL, adapter,
                            spec->scale, rows)) {
                fail(error, error_size, "out of memory indexing adapter");
                goto done;
            }
            stats->full_deltas++;
        }
    }
    if (stats->scale == 0.0f) stats->scale = 1.0f;
    if (!stats->low_rank && !stats->full_deltas) {
        fail(error, error_size, "%s: contains no adapter tensors", spec->path);
        goto done;
    }
    ok = 1;
done:
    for (size_t index = 0; index < module_count; index++)
        free(modules[index].module);
    free(modules);
    return ok;
}

static void count_targets(h3_lora_set *set) {
    for (size_t adapter = 0; adapter < set->adapter_count; adapter++) {
        size_t targets = 0;
        for (size_t index = 0; index < set->entry_count; index++) {
            const h3_lora_entry *entry = &set->entries[index];
            if (entry->adapter != adapter) continue;
            int first = 1;
            for (size_t prior = 0; prior < index; prior++) {
                if (set->entries[prior].adapter == adapter &&
                    !strcmp(set->entries[prior].target, entry->target)) {
                    first = 0;
                    break;
                }
            }
            targets += (size_t)first;
        }
        set->stats[adapter].targets = targets;
    }
}

h3_lora_set *h3_lora_set_open(const h3_lora *adapters, size_t count,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!adapters || !count || count > H3_MAX_LORAS) {
        fail(error, error_size, "between 1 and %d LoRAs are supported",
             H3_MAX_LORAS);
        return NULL;
    }
    h3_lora_set *set = calloc(1, sizeof(*set));
    if (set) {
        set->headers = calloc(count, sizeof(*set->headers));
        set->stats = calloc(count, sizeof(*set->stats));
    }
    if (!set || !set->headers || !set->stats) {
        h3_lora_set_free(set);
        fail(error, error_size, "out of memory opening LoRAs");
        return NULL;
    }
    set->adapter_count = count;
    for (size_t index = 0; index < count; index++) {
        if (!adapters[index].path || !*adapters[index].path ||
            !isfinite(adapters[index].scale)) {
            h3_lora_set_free(set);
            fail(error, error_size, "LoRA %zu has no path or a bad scale",
                 index + 1);
            return NULL;
        }
        if (!open_adapter(set, index, &adapters[index], error, error_size)) {
            h3_lora_set_free(set);
            return NULL;
        }
    }
    count_targets(set);
    return set;
}

void h3_lora_set_release_gpu(h3_lora_set *set) {
    if (!set) return;
    h3_gpu_tensor_free(set->scratch_sum);
    h3_gpu_tensor_free(set->scratch_product);
    set->scratch_sum = set->scratch_product = NULL;
    set->scratch_elements = 0;
}

void h3_lora_set_free(h3_lora_set *set) {
    if (!set) return;
    h3_lora_set_release_gpu(set);
    if (getenv("H3_LORA_TIMING"))
        h3_log(H3_LOG_DEBUG, "h3: LoRA timing stage=%.2fs gpu=%.2fs host-widen=%.2fs "
                "host-narrow=%.2fs\n", set->phase_seconds[0],
                set->phase_seconds[2], set->phase_seconds[1],
                set->phase_seconds[3]);
    for (size_t index = 0; index < set->entry_count; index++)
        free(set->entries[index].target);
    free(set->entries);
    if (set->headers) {
        for (size_t index = 0; index < set->adapter_count; index++)
            h3_st_free_header(&set->headers[index]);
    }
    free(set->headers);
    free(set->stats);
    free(set->accumulator);
    free(set);
}

static size_t source_rows(h3_lora_rows rows, size_t target_rows) {
    return rows == ROWS_Q || rows == ROWS_K || rows == ROWS_V ?
        target_rows / 3 : target_rows;
}

static size_t map_row(h3_lora_rows rows, size_t row, size_t target_rows) {
    switch (rows) {
        case ROWS_Q:
        case ROWS_K:
        case ROWS_V: {
            size_t part = rows == ROWS_Q ? 0 : rows == ROWS_K ? 1 : 2;
            return (row / HEAD_DIM) * 3 * HEAD_DIM + part * HEAD_DIM +
                   row % HEAD_DIM;
        }
        case ROWS_QKV_GROUPED: {
            size_t third = target_rows / 3;
            size_t part = row / third, local = row % third;
            return (local / HEAD_DIM) * 3 * HEAD_DIM + part * HEAD_DIM +
                   local % HEAD_DIM;
        }
        case ROWS_FC1_SWAPPED: {
            size_t half = target_rows / 2;
            return row < half ? row + half : row - half;
        }
        case ROWS_SAME:
        default:
            return row;
    }
}

typedef struct {
    const uint16_t *bf16_in;
    float *f32;
    uint16_t *bf16_out;
    size_t count;
    size_t chunk;
} convert_job;

static void widen_chunk(void *opaque, size_t chunk) {
    convert_job *job = opaque;
    size_t begin = chunk * job->chunk;
    size_t end = begin + job->chunk < job->count ? begin + job->chunk :
                                                   job->count;
    for (size_t index = begin; index < end; index++)
        job->f32[index] = bf16_to_f32(job->bf16_in[index]);
}

static void narrow_chunk(void *opaque, size_t chunk) {
    convert_job *job = opaque;
    size_t begin = chunk * job->chunk;
    size_t end = begin + job->chunk < job->count ? begin + job->chunk :
                                                   job->count;
    for (size_t index = begin; index < end; index++)
        job->bf16_out[index] = f32_to_bf16(job->f32[index]);
}

static void convert_parallel(convert_job *job, int narrow) {
    job->chunk = 1u << 20;
    size_t chunks = (job->count + job->chunk - 1) / job->chunk;
    dispatch_apply_f(chunks,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     job, narrow ? narrow_chunk : widen_chunk);
}

typedef struct {
    const h3_lora_entry *entry;
    const float *up;
    const float *down;
    float *accumulator;
    size_t rows;
    size_t columns;
    size_t rank;
    size_t run;
    size_t source_rows;
} gemm_job;

static void gemm_run(void *opaque, size_t index) {
    const gemm_job *job = opaque;
    size_t begin = index * job->run;
    size_t count = job->source_rows - begin < job->run ?
        job->source_rows - begin : job->run;
    size_t target = map_row(job->entry->rows, begin, job->rows);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)count,
                (int)job->columns, (int)job->rank, job->entry->scale,
                job->up + begin * job->rank, (int)job->rank, job->down,
                (int)job->columns, 1.0f, job->accumulator + target * job->columns,
                (int)job->columns);
}

static int apply_entry(const h3_lora_set *set, const h3_lora_entry *entry,
                       float *accumulator, size_t rows, size_t columns,
                       char *error, size_t error_size) {
    const h3_st_header *header = &set->headers[entry->adapter];
    if ((entry->rows == ROWS_Q || entry->rows == ROWS_K ||
         entry->rows == ROWS_V || entry->rows == ROWS_QKV_GROUPED) &&
        rows % (3 * HEAD_DIM)) {
        fail(error, error_size, "%s: %s rows are not a fused QKV projection",
             header->path, entry->target);
        return 0;
    }
    if (entry->rows == ROWS_FC1_SWAPPED && rows % 2) {
        fail(error, error_size, "%s: %s rows are not a fused SwiGLU input",
             header->path, entry->target);
        return 0;
    }
    size_t expected_rows = source_rows(entry->rows, rows);
    if (!entry->up) {
        if (h3_st_tensor_elements(entry->down) !=
                (uint64_t)expected_rows * columns ||
            entry->down->shape[0] != expected_rows) {
            fail(error, error_size, "%s: %s delta shape does not match "
                 "checkpoint tensor %s", header->path, entry->down->name,
                 entry->target);
            return 0;
        }
        float *delta = read_f32(header, entry->down, error, error_size);
        if (!delta) return 0;
        for (size_t row = 0; row < expected_rows; row++) {
            size_t target = map_row(entry->rows, row, rows);
            vDSP_vsma(delta + row * columns, 1, &entry->scale,
                      accumulator + target * columns, 1,
                      accumulator + target * columns, 1, columns);
        }
        free(delta);
        return 1;
    }
    size_t rank = (size_t)entry->down->shape[0];
    if (entry->down->shape[1] != columns || entry->up->shape[0] !=
            expected_rows || entry->up->shape[1] != rank) {
        fail(error, error_size,
             "%s: %s is [%llu x %llu] @ [%llu x %llu] but checkpoint tensor "
             "%s is %zu x %zu", header->path, entry->down->name,
             (unsigned long long)entry->up->shape[0],
             (unsigned long long)entry->up->shape[1],
             (unsigned long long)entry->down->shape[0],
             (unsigned long long)entry->down->shape[1], entry->target, rows,
             columns);
        return 0;
    }
    float *down = read_f32(header, entry->down, error, error_size);
    float *up = down ? read_f32(header, entry->up, error, error_size) : NULL;
    if (!up) {
        free(down);
        return 0;
    }
    /* Every row mapping keeps 128-row head blocks contiguous, so the product
     * accumulates straight into the checkpoint rows with beta = 1. */
    size_t run = entry->rows == ROWS_SAME ? expected_rows : HEAD_DIM;
    gemm_job job = {entry, up, down, accumulator, rows, columns, rank, run,
                    expected_rows};
    size_t runs = (expected_rows + run - 1) / run;
    if (runs == 1) gemm_run(&job, 0);
    else dispatch_apply_f(runs, dispatch_get_global_queue(
                              QOS_CLASS_USER_INITIATED, 0), &job, gemm_run);
    free(down);
    free(up);
    return 1;
}

static void mark_applied(h3_lora_set *set, size_t index, const char *name) {
    h3_lora_entry *entry = &set->entries[index];
    if (entry->applied) return;
    entry->applied = 1;
    for (size_t prior = 0; prior < index; prior++) {
        if (set->entries[prior].adapter == entry->adapter &&
            set->entries[prior].applied &&
            !strcmp(set->entries[prior].target, name)) return;
    }
    set->stats[entry->adapter].applied++;
}

/* Host path, used for tensors that carry only full-weight deltas. */
static int apply_host(h3_lora_set *set, const char *name, h3_dtype dtype,
                      void *data, size_t rows, size_t columns,
                      char *error, size_t error_size) {
    size_t count = rows * columns;
    float *accumulator = data;
    if (dtype == H3_DTYPE_BF16) {
        if (set->accumulator_elements < count) {
            free(set->accumulator);
            set->accumulator = malloc(count * sizeof(float));
            set->accumulator_elements = set->accumulator ? count : 0;
        }
        accumulator = set->accumulator;
    }
    if (!accumulator) {
        fail(error, error_size, "out of memory applying LoRA to %s", name);
        return 0;
    }
    convert_job job = {data, accumulator, data, count, 0};
    double t0 = now_seconds();
    if (dtype == H3_DTYPE_BF16) convert_parallel(&job, 0);
    set->phase_seconds[1] += now_seconds() - t0;
    t0 = now_seconds();
    int ok = 1;
    for (size_t index = 0; ok && index < set->entry_count; index++) {
        h3_lora_entry *entry = &set->entries[index];
        if (strcmp(entry->target, name)) continue;
        ok = apply_entry(set, entry, accumulator, rows, columns, error,
                         error_size);
        if (ok) mark_applied(set, index, name);
    }
    set->phase_seconds[2] += now_seconds() - t0;
    t0 = now_seconds();
    if (ok && dtype == H3_DTYPE_BF16) convert_parallel(&job, 1);
    set->phase_seconds[3] += now_seconds() - t0;
    return ok;
}

/* Stack every low-rank part as B_cat [rows, R] @ A_cat [R, columns], with B
 * rows already placed at their checkpoint rows and scaled, so one product
 * covers Q/K/V splits and every adapter. Full deltas become one dense term. */
static int build_terms(h3_lora_set *set, const char *name, size_t rows,
                       size_t columns, float **b_cat, float **a_cat_t,
                       size_t *total_rank, float **dense,
                       char *error, size_t error_size) {
    *b_cat = *a_cat_t = *dense = NULL;
    *total_rank = 0;
    int any_dense = 0;
    for (size_t index = 0; index < set->entry_count; index++) {
        const h3_lora_entry *entry = &set->entries[index];
        if (strcmp(entry->target, name)) continue;
        if (entry->up) *total_rank += (size_t)entry->down->shape[0];
        else any_dense = 1;
    }
    /* Zero rank columns add nothing to the product but let a small stacked
     * rank reach the MPS matrix path instead of the scalar fallback kernel,
     * which dominates at AdaLN-sized targets. */
    if (*total_rank < MIN_GEMM_RANK) *total_rank = MIN_GEMM_RANK;
    size_t rank_total = *total_rank;
    *b_cat = calloc(rows * rank_total, sizeof(float));
    *a_cat_t = calloc(columns * rank_total, sizeof(float));
    if (any_dense) *dense = calloc(rows * columns, sizeof(float));
    if (!*b_cat || !*a_cat_t || (any_dense && !*dense)) {
        fail(error, error_size, "out of memory staging LoRA for %s", name);
        return 0;
    }
    size_t offset = 0;
    for (size_t index = 0; index < set->entry_count; index++) {
        h3_lora_entry *entry = &set->entries[index];
        if (strcmp(entry->target, name)) continue;
        const h3_st_header *header = &set->headers[entry->adapter];
        if (!entry->up) {
            if (!apply_entry(set, entry, *dense, rows, columns, error,
                             error_size)) return 0;
            mark_applied(set, index, name);
            continue;
        }
        if ((entry->rows == ROWS_Q || entry->rows == ROWS_K ||
             entry->rows == ROWS_V || entry->rows == ROWS_QKV_GROUPED) &&
            rows % (3 * HEAD_DIM)) {
            fail(error, error_size, "%s: %s rows are not a fused QKV "
                 "projection", header->path, entry->target);
            return 0;
        }
        if (entry->rows == ROWS_FC1_SWAPPED && rows % 2) {
            fail(error, error_size, "%s: %s rows are not a fused SwiGLU input",
                 header->path, entry->target);
            return 0;
        }
        size_t expected_rows = source_rows(entry->rows, rows);
        size_t rank = (size_t)entry->down->shape[0];
        if (entry->down->shape[1] != columns ||
            entry->up->shape[0] != expected_rows ||
            entry->up->shape[1] != rank) {
            fail(error, error_size,
                 "%s: %s is [%llu x %llu] @ [%llu x %llu] but checkpoint "
                 "tensor %s is %zu x %zu", header->path, entry->down->name,
                 (unsigned long long)entry->up->shape[0],
                 (unsigned long long)entry->up->shape[1],
                 (unsigned long long)entry->down->shape[0],
                 (unsigned long long)entry->down->shape[1], entry->target,
                 rows, columns);
            return 0;
        }
        float *down = read_f32(header, entry->down, error, error_size);
        float *up = down ? read_f32(header, entry->up, error, error_size) :
                           NULL;
        if (!up) {
            free(down);
            return 0;
        }
        for (size_t row = 0; row < expected_rows; row++) {
            float *target = *b_cat +
                map_row(entry->rows, row, rows) * rank_total + offset;
            for (size_t k = 0; k < rank; k++)
                target[k] = entry->scale * up[row * rank + k];
        }
        for (size_t k = 0; k < rank; k++) {
            for (size_t column = 0; column < columns; column++)
                (*a_cat_t)[column * rank_total + offset + k] =
                    down[k * columns + column];
        }
        free(down);
        free(up);
        offset += rank;
        mark_applied(set, index, name);
    }
    return 1;
}

static h3_gpu_tensor *patch_gpu(h3_lora_set *set, h3_gpu *gpu,
                                const char *name, h3_dtype dtype,
                                h3_gpu_tensor *loaded, size_t rows,
                                size_t columns, char *error,
                                size_t error_size) {
    float *b_cat = NULL, *a_cat_t = NULL, *dense = NULL;
    size_t rank = 0, count = rows * columns;
    h3_gpu_tensor *b = NULL, *a = NULL, *d = NULL, *product = NULL;
    h3_gpu_tensor *sum = NULL, *result = NULL;
    double t0 = now_seconds();
    int ok = build_terms(set, name, rows, columns, &b_cat, &a_cat_t, &rank,
                         &dense, error, error_size);
    set->phase_seconds[0] += now_seconds() - t0;
    t0 = now_seconds();
    if (ok && (count > UINT32_MAX || rows > UINT32_MAX || rank > UINT32_MAX)) {
        fail(error, error_size, "LoRA target %s is too large", name);
        ok = 0;
    }
    if (ok) {
        b = h3_gpu_tensor_from_f32(gpu, b_cat, rows * rank);
        a = h3_gpu_tensor_from_f32(gpu, a_cat_t, columns * rank);
        d = dense ? h3_gpu_tensor_from_f32(gpu, dense, count) : NULL;
        if (set->scratch_elements < count) {
            h3_lora_set_release_gpu(set);
            set->scratch_sum = h3_gpu_tensor_new_f32(gpu, count);
            set->scratch_product = h3_gpu_tensor_new_f32(gpu, count);
            if (set->scratch_sum && set->scratch_product)
                set->scratch_elements = count;
        }
        product = set->scratch_product;
        sum = set->scratch_sum;
        /* The sum rounds back into the loaded buffer: a fresh result tensor
         * would zero-fill every page of the transformer a second time. */
        result = loaded;
        ok = b && a && (!dense || d) && product && sum &&
             set->scratch_elements >= count;
        if (!ok) fail(error, error_size, "cannot allocate LoRA tensors for "
                      "%s: %s", name, h3_gpu_error(gpu));
    }
    if (ok) {
        ok = h3_gpu_begin(gpu) &&
            (dtype == H3_DTYPE_BF16 ?
                h3_gpu_cast_bf16_to_f32(gpu, sum, loaded, (uint32_t)count) :
                h3_gpu_copy_f32(gpu, sum, 0, loaded, 0, count)) &&
            h3_gpu_linear_f32(gpu, product, b, a, NULL, (uint32_t)rows,
                              (uint32_t)rank, (uint32_t)columns) &&
            h3_gpu_add_scaled_f32(gpu, sum, sum, product, 1.0f, 1.0f,
                                  (uint32_t)count) &&
            (!d || h3_gpu_add_scaled_f32(gpu, sum, sum, d, 1.0f, 1.0f,
                                         (uint32_t)count)) &&
            (dtype == H3_DTYPE_BF16 ?
                h3_gpu_cast_f32_to_bf16(gpu, result, sum, (uint32_t)count) :
                h3_gpu_copy_f32(gpu, result, 0, sum, 0, count)) &&
            h3_gpu_submit(gpu);
        if (!ok) fail(error, error_size, "cannot apply LoRA to %s: %s", name,
                      h3_gpu_error(gpu));
    }
    set->phase_seconds[2] += now_seconds() - t0;
    free(b_cat);
    free(a_cat_t);
    free(dense);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(d);
    return ok ? result : NULL;
}

h3_gpu_tensor *h3_lora_set_patch(h3_lora_set *set, h3_gpu *gpu,
                                 const char *name, h3_dtype dtype,
                                 h3_gpu_tensor *loaded, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    if (!set || !name || !loaded) return loaded;
    int targeted = 0, low_rank = 0;
    for (size_t index = 0; index < set->entry_count; index++) {
        if (strcmp(set->entries[index].target, name)) continue;
        targeted = 1;
        if (set->entries[index].up) low_rank = 1;
    }
    if (!targeted) return loaded;
    if (ndim < 1 || ndim > 2 ||
        (dtype != H3_DTYPE_BF16 && dtype != H3_DTYPE_F32)) {
        fail(error, error_size, "LoRA target %s is not a 1-D/2-D BF16 or F32 "
             "tensor", name);
        h3_gpu_tensor_free(loaded);
        return NULL;
    }
    double started = now_seconds();
    size_t rows = (size_t)shape[0];
    size_t columns = ndim == 2 ? (size_t)shape[1] : 1;
    h3_gpu_tensor *result;
    if (low_rank) {
        result = patch_gpu(set, gpu, name, dtype, loaded, rows, columns,
                           error, error_size);
        if (!result) h3_gpu_tensor_free(loaded);
    } else {
        void *data = h3_gpu_tensor_host_data(loaded);
        result = data && apply_host(set, name, dtype, data, rows, columns,
                                    error, error_size) ? loaded : NULL;
        if (!data) fail(error, error_size, "LoRA target %s is not host "
                        "visible", name);
        if (!result) h3_gpu_tensor_free(loaded);
    }
    set->apply_seconds += now_seconds() - started;
    return result;
}

size_t h3_lora_set_count(const h3_lora_set *set) {
    return set ? set->adapter_count : 0;
}

void h3_lora_set_stats(const h3_lora_set *set, size_t adapter,
                       h3_lora_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    if (set && adapter < set->adapter_count) *stats = set->stats[adapter];
}

double h3_lora_set_apply_seconds(const h3_lora_set *set) {
    return set ? set->apply_seconds : 0.0;
}
