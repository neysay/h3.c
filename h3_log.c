#include "h3_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static h3_log_callback h3_log_sink;
static void *h3_log_sink_opaque;

void h3_set_log_callback(h3_log_callback callback, void *opaque) {
    h3_log_sink = callback;
    h3_log_sink_opaque = opaque;
}

static const char *h3_log_strip(const char *text, h3_log_level level) {
    if (!strncmp(text, "h3: ", 4)) text += 4;
    if (level == H3_LOG_WARNING && !strncmp(text, "warning: ", 9)) text += 9;
    return text;
}

void h3_log(h3_log_level level, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    h3_log_callback sink = h3_log_sink;
    if (!sink) {
        vfprintf(stderr, format, arguments);
        va_end(arguments);
        return;
    }
    char stack[512];
    va_list copy;
    va_copy(copy, arguments);
    int length = vsnprintf(stack, sizeof(stack), format, copy);
    va_end(copy);
    char *text = stack;
    if (length >= (int)sizeof(stack)) {
        text = malloc((size_t)length + 1);
        if (text) vsnprintf(text, (size_t)length + 1, format, arguments);
        else text = stack; /* truncated beats silent */
    }
    va_end(arguments);
    if (length < 0) return;
    size_t end = strlen(text);
    while (end && (text[end - 1] == '\n' || text[end - 1] == '\r'))
        text[--end] = '\0';
    if (end) sink(level, h3_log_strip(text, level), h3_log_sink_opaque);
    if (text != stack) free(text);
}
