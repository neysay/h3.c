/* Library diagnostics: one funnel for everything libh3 says.
 *
 * Every library message goes through h3_log. With no callback installed it
 * writes the formatted text to stderr exactly as before, byte for byte, so
 * terminal output is unchanged. With a callback (h3_set_log_callback in h3.h)
 * the message is delivered structured instead: its level, and its text
 * without the "h3: " / "h3: warning: " prefix or the trailing newline. */
#ifndef H3_LOG_H
#define H3_LOG_H

#include "h3.h"

void h3_log(h3_log_level level, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
