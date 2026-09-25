/* The one JSON writer, shared by the settings sidecar and the event stream. */
#ifndef H3_JSON_H
#define H3_JSON_H

#include <stdio.h>

/* A JSON string literal. Control characters are escaped. With strict_utf8,
 * bytes that are not valid UTF-8 become U+FFFD so the output always parses;
 * without it they pass through unchanged (the sidecar keeps paths byte for
 * byte, and its own reader accepts them). */
void h3_json_write_string(FILE *file, const char *text, int strict_utf8);

/* A path as a JSON string: absolute when the file exists, otherwise as
 * given; null for NULL. */
void h3_json_write_path(FILE *file, const char *path, int strict_utf8);

/* The shortest decimal that reads back to the identical float. */
void h3_json_write_float(FILE *file, float value);

#endif
