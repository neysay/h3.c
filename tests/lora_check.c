/* Load named DiT tensors through the weight store, optionally with LoRAs, and
 * dump their host bytes for offline comparison:
 *   h3_lora_check TRANSFORMER_DIR OUT_DIR [--lora PATH[:SCALE]]... NAME... */
#include "../h3_gpu.h"
#include "../h3_lora.h"
#include "../h3_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/lora_check.c: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 4) die("usage: TRANSFORMER_DIR OUT_DIR [--lora P[:S]]... NAME...");
    char error[512];
    h3_lora loras[H3_MAX_LORAS];
    size_t lora_count = 0;
    int first_name = 3;
    while (first_name + 1 < argc && !strcmp(argv[first_name], "--lora")) {
        char *value = argv[first_name + 1];
        char *colon = strrchr(value, ':');
        loras[lora_count].path = value;
        loras[lora_count].scale = 1.0f;
        if (colon) {
            *colon = '\0';
            loras[lora_count].scale = strtof(colon + 1, NULL);
        }
        lora_count++;
        first_name += 2;
    }
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) die(error);
    h3_lora_set *set = NULL;
    if (lora_count) {
        set = h3_lora_set_open(loras, lora_count, error, sizeof(error));
        if (!set) die(error);
        h3_weight_store_set_loras(store, set);
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    for (int index = first_name; index < argc; index++) {
        const char *name = argv[index];
        const h3_st_header *header = NULL;
        const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
        if (!tensor) die(name);
        h3_gpu_tensor *loaded = tensor->dtype == H3_DTYPE_BF16 ?
            h3_weight_load_bf16(store, gpu, name, tensor->ndim, tensor->shape,
                                error, sizeof(error)) :
            h3_weight_load_f32(store, gpu, name, tensor->ndim, tensor->shape,
                               error, sizeof(error));
        if (!loaded) die(error);
        if (!strcmp(argv[2], "-")) {
            h3_gpu_tensor_free(loaded);
            continue;
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.bin", argv[2], name);
        FILE *file = fopen(path, "wb");
        size_t bytes = h3_gpu_tensor_elements(loaded) *
                       h3_dtype_size(tensor->dtype);
        if (!file || fwrite(h3_gpu_tensor_host_data(loaded), 1, bytes, file) !=
                bytes || fclose(file) != 0) die(path);
        h3_gpu_tensor_free(loaded);
    }
    for (size_t adapter = 0; adapter < h3_lora_set_count(set); adapter++) {
        h3_lora_stats stats;
        h3_lora_set_stats(set, adapter, &stats);
        printf("lora %zu: format=%s low_rank=%zu full=%zu targets=%zu "
               "applied=%zu scale=%.6g\n", adapter, stats.format,
               stats.low_rank, stats.full_deltas, stats.targets,
               stats.applied, (double)stats.scale);
    }
    printf("apply %.3f s\n", h3_lora_set_apply_seconds(set));
    h3_weight_store_set_loras(store, NULL);
    h3_lora_set_free(set);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
