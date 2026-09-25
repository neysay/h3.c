#ifndef H3_SETTINGS_H
#define H3_SETTINGS_H

#include "h3.h"
#include "h3_lora.h"

#include <stddef.h>

#define H3_SETTINGS_VERSION 1
#define H3_SETTINGS_MAX_REFERENCES 12

/* A render's complete inputs, loaded from a settings sidecar. Every string in
 * params, references and loras points into storage owned by this struct. */
typedef struct {
    char *model_dir;
    char *prompt;
    h3_params params;
    h3_reference references[H3_SETTINGS_MAX_REFERENCES];
    size_t reference_count;
    h3_lora loras[H3_MAX_LORAS];
    size_t lora_count;
    char **owned;
    size_t owned_count;
    size_t owned_capacity;
} h3_settings;

/* Sidecar path for an output: "clip.mp4" -> "clip.json". Caller frees. */
char *h3_settings_path_for(const char *output_path);

/* Write every input needed to re-render `output_path`, with paths made
 * absolute, to its sidecar. Keys are the CLI's long option names. */
int h3_settings_write(const char *output_path, const char *model_dir,
                      const char *prompt, const h3_params *params,
                      const h3_result *result, double elapsed_seconds,
                      char *error, size_t error_size);

/* Load a sidecar over H3_PARAMS_DEFAULT. Unknown keys are errors so a typo
 * in a hand-edited file cannot silently fall back to a default. */
int h3_settings_load(const char *path, h3_settings *settings,
                     char *error, size_t error_size);
void h3_settings_free(h3_settings *settings);

/* Parse "PATH" or "PATH:SCALE" in place; the suffix is a scale only when it
 * parses fully as a finite number. Returns 0 for an empty path. */
int h3_settings_parse_lora(char *value, h3_lora *lora);

#endif
