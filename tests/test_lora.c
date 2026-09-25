/* Synthetic end-to-end checks for load-time LoRA patching: tiny checkpoints
 * and adapters in each supported key layout go through the real weight store
 * and GPU patch path, and every value is compared with a direct evaluation of
 * the documented layout rules. */
#include "../h3_gpu.h"
#include "../h3_lora.h"
#include "../h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define HEADS 2u
#define HEAD 128u
#define QKV_ROWS (3u * HEADS * HEAD)  /* 768: per-head [q k v] interleave */
#define FC1_ROWS 512u                 /* [gate; value] */
#define COLUMNS 8u
#define RANK 2u

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_lora.c: %s\n", message);
    exit(1);
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef struct {
    const char *name;
    size_t rows;
    size_t columns;
    const float *values;
} tensor_spec;

/* Write BF16 tensors as a safetensors file, header padded to 8 bytes. */
static void write_safetensors(const char *path, const tensor_spec *tensors,
                              size_t count, const char *metadata) {
    char header[8192];
    size_t used = (size_t)snprintf(header, sizeof(header), "{");
    if (metadata)
        used += (size_t)snprintf(header + used, sizeof(header) - used,
                                 "\"__metadata__\":%s,", metadata);
    size_t offset = 0;
    for (size_t index = 0; index < count; index++) {
        size_t bytes = tensors[index].rows * tensors[index].columns * 2;
        if (tensors[index].columns == 1)
            used += (size_t)snprintf(header + used, sizeof(header) - used,
                "%s\"%s\":{\"dtype\":\"BF16\",\"shape\":[%zu],"
                "\"data_offsets\":[%zu,%zu]}", index ? "," : "",
                tensors[index].name, tensors[index].rows, offset,
                offset + bytes);
        else
            used += (size_t)snprintf(header + used, sizeof(header) - used,
                "%s\"%s\":{\"dtype\":\"BF16\",\"shape\":[%zu,%zu],"
                "\"data_offsets\":[%zu,%zu]}", index ? "," : "",
                tensors[index].name, tensors[index].rows,
                tensors[index].columns, offset, offset + bytes);
        offset += bytes;
    }
    used += (size_t)snprintf(header + used, sizeof(header) - used, "}");
    while (used % 8) header[used++] = ' ';
    FILE *file = fopen(path, "wb");
    if (!file) die(path);
    uint64_t length = used;
    fwrite(&length, 8, 1, file);
    fwrite(header, 1, used, file);
    for (size_t index = 0; index < count; index++) {
        size_t elements = tensors[index].rows * tensors[index].columns;
        for (size_t element = 0; element < elements; element++) {
            uint16_t value = to_bf16(tensors[index].values[element]);
            fwrite(&value, 2, 1, file);
        }
    }
    if (fclose(file) != 0) die(path);
}

static float *filled(size_t count, float scale, float phase) {
    float *values = malloc(count * sizeof(float));
    if (!values) die("out of memory");
    for (size_t index = 0; index < count; index++)
        values[index] = from_bf16(to_bf16(
            scale * sinf(phase + 0.37f * (float)index)));
    return values;
}

/* delta[row, column] = scale * sum_k up[row, k] * down[k, column] */
static void add_product(float *target, size_t target_row, const float *up,
                        size_t up_row, const float *down, float scale) {
    for (size_t column = 0; column < COLUMNS; column++) {
        float sum = 0.0f;
        for (size_t k = 0; k < RANK; k++)
            sum += up[up_row * RANK + k] * down[k * COLUMNS + column];
        target[target_row * COLUMNS + column] += scale * sum;
    }
}

static void check(h3_weight_store *store, h3_gpu *gpu, const char *name,
                  size_t rows, const float *expected, const char *label) {
    char error[512];
    uint64_t shape[2] = {rows, COLUMNS};
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                                error, sizeof(error));
    if (!tensor) die(error);
    const uint16_t *got = h3_gpu_tensor_host_data(tensor);
    for (size_t index = 0; index < rows * COLUMNS; index++) {
        uint16_t want = to_bf16(expected[index]);
        if (abs((int)got[index] - (int)want) > 1) {
            fprintf(stderr, "%s: %s[%zu] got %g want %g\n", label, name,
                    index, (double)from_bf16(got[index]),
                    (double)from_bf16(want));
            die("patched value mismatch");
        }
    }
    h3_gpu_tensor_free(tensor);
}

int main(void) {
    char directory[] = "/tmp/h3-lora-test-XXXXXX";
    if (!mkdtemp(directory)) die("mkdtemp");
    char checkpoint_dir[256], path[320];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/transformer",
             directory);
    mkdir(checkpoint_dir, 0755);

    float *qkv = filled(QKV_ROWS * COLUMNS, 1.0f, 0.1f);
    float *fc1 = filled(FC1_ROWS * COLUMNS, 1.0f, 0.7f);
    tensor_spec base[] = {
        {"blocks.3.attn.qkv_proj.weight", QKV_ROWS, COLUMNS, qkv},
        {"blocks.3.mlp.fc1.weight", FC1_ROWS, COLUMNS, fc1},
    };
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors",
             checkpoint_dir);
    write_safetensors(path, base, 2, NULL);

    /* Native layout: QKV B rows are [q_all; k_all; v_all]; kohya-style
     * per-module alpha scales by alpha / rank. */
    float *a_qkv = filled(RANK * COLUMNS, 0.5f, 1.3f);
    float *b_qkv = filled(QKV_ROWS * RANK, 0.5f, 2.9f);
    float *a_fc1 = filled(RANK * COLUMNS, 0.5f, 0.3f);
    float *b_fc1 = filled(FC1_ROWS * RANK, 0.5f, 4.1f);
    float alpha_value = 1.0f;
    tensor_spec native[] = {
        {"diffusion_model.blocks.3.attn.qkv_proj.lora_down.weight", RANK,
         COLUMNS, a_qkv},
        {"diffusion_model.blocks.3.attn.qkv_proj.lora_up.weight", QKV_ROWS,
         RANK, b_qkv},
        {"diffusion_model.blocks.3.attn.qkv_proj.alpha", 1, 1, &alpha_value},
        {"diffusion_model.blocks.3.mlp.fc1.lora_down.weight", RANK, COLUMNS,
         a_fc1},
        {"diffusion_model.blocks.3.mlp.fc1.lora_up.weight", FC1_ROWS, RANK,
         b_fc1},
    };
    char native_path[320];
    snprintf(native_path, sizeof(native_path), "%s/native.safetensors",
             directory);
    write_safetensors(native_path, native, 5, NULL);

    /* Diffusers layout: split to_q/to_k/to_v and FC1 as [value; gate], with
     * PEFT alpha carried in metadata. */
    float *a_q = filled(RANK * COLUMNS, 0.5f, 5.0f);
    float *b_q = filled(HEADS * HEAD * RANK, 0.5f, 5.5f);
    float *a_v = filled(RANK * COLUMNS, 0.5f, 6.0f);
    float *b_v = filled(HEADS * HEAD * RANK, 0.5f, 6.5f);
    float *a_ff = filled(RANK * COLUMNS, 0.5f, 7.0f);
    float *b_ff = filled(FC1_ROWS * RANK, 0.5f, 7.5f);
    tensor_spec diffusers[] = {
        {"transformer_blocks.3.attn.to_q.lora_A.default.weight", RANK,
         COLUMNS, a_q},
        {"transformer_blocks.3.attn.to_q.lora_B.default.weight", HEADS * HEAD,
         RANK, b_q},
        {"transformer_blocks.3.attn.to_v.lora_A.default.weight", RANK,
         COLUMNS, a_v},
        {"transformer_blocks.3.attn.to_v.lora_B.default.weight", HEADS * HEAD,
         RANK, b_v},
        {"transformer_blocks.3.ff.net.0.proj.lora_A.default.weight", RANK,
         COLUMNS, a_ff},
        {"transformer_blocks.3.ff.net.0.proj.lora_B.default.weight", FC1_ROWS,
         RANK, b_ff},
    };
    char diffusers_path[320];
    snprintf(diffusers_path, sizeof(diffusers_path), "%s/diffusers.safetensors",
             directory);
    write_safetensors(diffusers_path, diffusers, 6, "{\"alpha\":\"4\"}");

    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    /* Both adapters at once, with strengths, into the same two tensors. */
    h3_lora adapters[2] = {{native_path, 0.75f}, {diffusers_path, 1.5f}};
    h3_lora_set *set = h3_lora_set_open(adapters, 2, error, sizeof(error));
    if (!set) die(error);
    h3_lora_stats stats;
    h3_lora_set_stats(set, 0, &stats);
    if (strcmp(stats.format, "native") || stats.low_rank != 2 ||
        fabsf(stats.scale - 0.5f) > 1e-6f) die("native adapter stats");
    h3_lora_set_stats(set, 1, &stats);
    if (strcmp(stats.format, "diffusers") || stats.low_rank != 3 ||
        fabsf(stats.scale - 2.0f) > 1e-6f) die("diffusers adapter stats");

    float *expect_qkv = malloc(QKV_ROWS * COLUMNS * sizeof(float));
    float *expect_fc1 = malloc(FC1_ROWS * COLUMNS * sizeof(float));
    memcpy(expect_qkv, qkv, QKV_ROWS * COLUMNS * sizeof(float));
    memcpy(expect_fc1, fc1, FC1_ROWS * COLUMNS * sizeof(float));
    /* Only the native QKV module carries an alpha tensor (1 / rank 2); its
     * FC1 has none and falls back to 1. The diffusers adapter's metadata
     * alpha 4 / rank 2 applies to every module. */
    float native_scale = 0.5f * 0.75f, native_fc1_scale = 1.0f * 0.75f;
    float diffusers_scale = 2.0f * 1.5f;
    for (size_t part = 0; part < 3; part++) {
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t channel = 0; channel < HEAD; channel++) {
                size_t checkpoint_row = head * 3 * HEAD + part * HEAD + channel;
                size_t grouped_row = part * HEADS * HEAD + head * HEAD +
                                     channel;
                add_product(expect_qkv, checkpoint_row, b_qkv, grouped_row,
                            a_qkv, native_scale);
                size_t split_row = head * HEAD + channel;
                if (part == 0)
                    add_product(expect_qkv, checkpoint_row, b_q, split_row,
                                a_q, diffusers_scale);
                if (part == 2)
                    add_product(expect_qkv, checkpoint_row, b_v, split_row,
                                a_v, diffusers_scale);
            }
        }
    }
    size_t half = FC1_ROWS / 2;
    for (size_t row = 0; row < FC1_ROWS; row++) {
        add_product(expect_fc1, row, b_fc1, row, a_fc1, native_fc1_scale);
        size_t diffusers_row = row < half ? row + half : row - half;
        add_product(expect_fc1, row, b_ff, diffusers_row, a_ff,
                    diffusers_scale);
    }

    h3_weight_store *store = h3_weight_store_open(checkpoint_dir, error,
                                                  sizeof(error));
    if (!store) die(error);
    h3_weight_store_set_loras(store, set);
    check(store, gpu, "blocks.3.attn.qkv_proj.weight", QKV_ROWS, expect_qkv,
          "stacked");
    check(store, gpu, "blocks.3.mlp.fc1.weight", FC1_ROWS, expect_fc1,
          "stacked");
    h3_lora_set_stats(set, 0, &stats);
    if (stats.applied != stats.targets) die("native targets not all applied");
    h3_lora_set_stats(set, 1, &stats);
    if (stats.applied != stats.targets) die("diffusers targets not applied");
    h3_weight_store_set_loras(store, NULL);

    /* Detached: the checkpoint loads unchanged. */
    check(store, gpu, "blocks.3.attn.qkv_proj.weight", QKV_ROWS, qkv,
          "detached");

    /* An adapter naming no H3 module is rejected before any load. */
    tensor_spec unknown[] = {
        {"blocks.3.attn.nonexistent.lora_A.weight", RANK, COLUMNS, a_qkv},
        {"blocks.3.attn.nonexistent.lora_B.weight", QKV_ROWS, RANK, b_qkv},
    };
    char unknown_path[320];
    snprintf(unknown_path, sizeof(unknown_path), "%s/unknown.safetensors",
             directory);
    write_safetensors(unknown_path, unknown, 2, NULL);
    h3_lora bad = {unknown_path, 1.0f};
    h3_lora_set *rejected = h3_lora_set_open(&bad, 1, error, sizeof(error));
    if (rejected || !strstr(error, "does not name")) die("unknown key accepted");

    h3_lora_set_free(set);
    h3_weight_store_free(store);
    h3_gpu_free(gpu);
    unlink(path);
    unlink(native_path);
    unlink(diffusers_path);
    unlink(unknown_path);
    rmdir(checkpoint_dir);
    rmdir(directory);
    free(qkv); free(fc1); free(a_qkv); free(b_qkv); free(a_fc1); free(b_fc1);
    free(a_q); free(b_q); free(a_v); free(b_v); free(a_ff); free(b_ff);
    free(expect_qkv); free(expect_fc1);
    puts("ok: native, diffusers, and stacked LoRAs patch the checkpoint layout");
    return 0;
}
