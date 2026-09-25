#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

#define H3_MAX_LORAS 8

typedef struct h3_lora_set h3_lora_set;

typedef struct {
    const char *format;   /* "native", "native-interleaved", "diffusers" */
    size_t low_rank;      /* A/B pairs */
    size_t full_deltas;   /* .diff / .diff_b tensors */
    size_t targets;       /* distinct checkpoint tensors touched */
    size_t applied;       /* of those, tensors that were loaded and patched */
    float scale;          /* alpha/rank scale before the user strength */
} h3_lora_stats;

/* Read adapter headers and resolve every entry to a native checkpoint tensor
 * before any weight is loaded. Unknown keys fail the whole set: a partially
 * applied adapter renders plausibly but wrongly. */
h3_lora_set *h3_lora_set_open(const h3_lora *adapters, size_t count,
                              char *error, size_t error_size);
void h3_lora_set_free(h3_lora_set *set);

/* Return `loaded` with every delta targeting `name` added in place. All
 * adapters and low-rank parts are stacked along the rank axis into one GPU
 * product, accumulated in F32 over the checkpoint tensor and rounded once to
 * nearest even. The CPU never touches the checkpoint's copy-on-write pages,
 * which dominated host-side patching. Tensors with only full-weight deltas
 * are patched on the host. Returns NULL on failure, with `loaded` freed.
 * Must be called with no GPU command buffer open. */
h3_gpu_tensor *h3_lora_set_patch(h3_lora_set *set, h3_gpu *gpu,
                                 const char *name, h3_dtype dtype,
                                 h3_gpu_tensor *loaded, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);

/* Free GPU scratch tensors. Call before the GPU context they were made on is
 * destroyed; the weight store does so when the set is detached. */
void h3_lora_set_release_gpu(h3_lora_set *set);

size_t h3_lora_set_count(const h3_lora_set *set);
void h3_lora_set_stats(const h3_lora_set *set, size_t adapter,
                       h3_lora_stats *stats);
double h3_lora_set_apply_seconds(const h3_lora_set *set);

#endif
