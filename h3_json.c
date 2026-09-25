#include "h3_json.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Length of the valid UTF-8 sequence at `text`, or 0 if it is not one. */
static size_t utf8_sequence(const unsigned char *text) {
    unsigned char lead = text[0];
    size_t length;
    uint32_t code;
    if (lead < 0x80) return 1;
    if (lead >= 0xc2 && lead <= 0xdf) { length = 2; code = lead & 0x1f; }
    else if (lead >= 0xe0 && lead <= 0xef) { length = 3; code = lead & 0x0f; }
    else if (lead >= 0xf0 && lead <= 0xf4) { length = 4; code = lead & 0x07; }
    else return 0;
    for (size_t index = 1; index < length; index++) {
        if ((text[index] & 0xc0) != 0x80) return 0;
        code = (code << 6) | (text[index] & 0x3f);
    }
    /* Overlong forms, surrogates, and code points past U+10FFFF. */
    if ((length == 3 && code < 0x800) || (length == 4 && code < 0x10000) ||
        (code >= 0xd800 && code <= 0xdfff) || code > 0x10ffff)
        return 0;
    return length;
}

void h3_json_write_string(FILE *file, const char *text, int strict_utf8) {
    fputc('"', file);
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor) {
        switch (*cursor) {
            case '"': fputs("\\\"", file); cursor++; continue;
            case '\\': fputs("\\\\", file); cursor++; continue;
            case '\n': fputs("\\n", file); cursor++; continue;
            case '\r': fputs("\\r", file); cursor++; continue;
            case '\t': fputs("\\t", file); cursor++; continue;
            default: break;
        }
        if (*cursor < 0x20) {
            fprintf(file, "\\u%04x", *cursor);
            cursor++;
            continue;
        }
        if (*cursor < 0x80 || !strict_utf8) {
            fputc(*cursor, file);
            cursor++;
            continue;
        }
        size_t length = utf8_sequence(cursor);
        if (length) {
            fwrite(cursor, 1, length, file);
            cursor += length;
        } else {
            fputs("\\ufffd", file);
            cursor++;
        }
    }
    fputc('"', file);
}

void h3_json_write_path(FILE *file, const char *path, int strict_utf8) {
    if (!path) {
        fputs("null", file);
        return;
    }
    char resolved[PATH_MAX];
    h3_json_write_string(file, realpath(path, resolved) ? resolved : path,
                         strict_utf8);
}

void h3_json_write_float(FILE *file, float value) {
    char text[32];
    for (int digits = 6; digits <= 9; digits++) {
        snprintf(text, sizeof(text), "%.*g", digits, (double)value);
        if (strtof(text, NULL) == value) break;
    }
    fputs(text, file);
}
