# h3 events protocol, version 1

`h3 --events-fd N` reports a one-shot render as a stream of JSON events on an
already-open file descriptor, for programs that supervise h3. It is a plain
job-event stream: h3 does not know or assume who reads it.

This document is normative. `h3_events.h` holds the constants, and
`tests/fixtures/events/*.jsonl` are golden streams produced by
`tests/test_events.c` (part of `make test`), so a change to what h3 emits fails
the test suite before it reaches a reader.

## Activation and framing

- `--events-fd N` writes events to descriptor `N`, which the caller opened for
  writing and passed to h3. h3 marks it close-on-exec, so the FFmpeg processes
  it starts never hold it, and closes it at exit.
- If `N` is not open for writing, h3 exits with code 2 before starting any
  work, and writes no events.
- Other argument errors also exit with code 2 before the stream is opened, so
  they produce no events either; the reason is on stderr, as always.
- The interactive session and `--info` ignore `--events-fd`.
- The format is UTF-8 JSON Lines: one object per line, ended by `\n`, with no
  pretty-printing. Bytes that are not valid UTF-8 (in a path, say) are written
  as U+FFFD, so every line parses.
- Each line is written with one `write(2)` under a lock. Events come from more
  than one thread (the sampler, preview decoding, the main thread), and lines
  never interleave.
- Fields that are unset are left out. A `metric`'s `value` is always present.
- If the reader goes away, h3 stops writing events and finishes the render; it
  is not killed by `SIGPIPE`.

### stderr

Without `--events-fd`, h3's terminal output is unchanged, byte for byte.

With it, h3's own words are events: progress bars, `h3:` messages, warnings and
`--profile` lines are all turned off on stderr, because each is already on the
stream. stderr then carries only output from outside h3's control: errors from
the FFmpeg processes h3 starts, Metal and system messages, and crash output. A
supervisor can treat anything that arrives on stderr as worth showing.

## Event types

| `type` | Required | Optional | Meaning |
|---|---|---|---|
| `log` | `level`, `source` (`"h3"`), `text` | `meta` | A message. `level` is `debug`, `info`, `warning` or `error`. An `error` log is advisory: a render's failure is reported only by `job.end`. |
| `stage.start` | `stage` | `label` | A stage began. `label` is the counter that started it. |
| `stage.progress` | `stage`, `step`, `total` | `label` | A counter within the stage. |
| `stage.end` | `stage`, `elapsed_s` | | A stage finished, with its measured wall time in seconds. |
| `metric` | `key`, `value` | `stage`, `unit` | A numeric reading. |
| `artifact` | `kind`, `path` | | A file was written. `kind` is `video`, `preview`, `frames` (a directory) or `settings`. Paths are absolute when the file exists. |
| `job.end` | `status` (`complete` or `failed`) | `error`, `elapsed_s`, `meta` | The result. Always the last event on a normal exit. |

## Stages

A render passes through these stages, in this order. Stages that do not apply
to a render (conditioning without references, for example) do not appear.

| `stage` | Work |
|---|---|
| `tokenize` | Loading the tokenizer and preparing references |
| `encode_conditioning` | Encoding reference images, video and audio |
| `text_encode` | The Qwen3-VL prompt encoder |
| `precompute_adaln` | Precomputing the DiT's timestep modulation |
| `load_transformer` | Loading the DiT, including applying LoRAs |
| `vae_load` | Loading the video VAE decoder (before `denoise` when previews need it) |
| `denoise` | The sampler |
| `audio_decode` | The audio VAE |
| `vae_decode` | The video VAE decode |
| `mux` | FFmpeg writing the MP4 |

Rules:

- Each stage starts and ends at most once per render.
- A stage ends when the next one starts, or when the render completes; its
  `elapsed_s` is the time between.
- Before a `complete` `job.end`, every started stage has ended. After a
  `failed` one, the stage that was running may be left open: the render did not
  finish it.
- A stage's progress never goes backwards. A stage fed by several counters in
  sequence (the reference encoders; the text encoder's refinement pass; the
  sampler's enqueue counter and its own) reports only readings at or past the
  last one, so `step / total` never decreases. `label` names the counter.
- There are no roll-up stages: no event summarises other events.

## Metrics

Always reported; they do not depend on `--profile`.

| `key` | `unit` | When |
|---|---|---|
| `peak_memory_gb` | `GiB` | After each `stage.end`, with that `stage`: the peak Metal tensor memory in use, across every GPU context, while the stage ran. |
| `lora_apply_s` | `s` | With `stage` `load_transformer`, when LoRAs were applied: the time spent patching the checkpoint. |

## Logs

Messages h3 would print to stderr as `h3: ...` arrive as `log` events, with the
`h3: ` prefix (and `warning: ` for warnings) removed. The LoRA report is one
`info` log per adapter, followed by `LoRA apply <seconds> s`. `--profile` lines
arrive as `debug` logs.

## The result

On success:

```json
{"type":"job.end","status":"complete","elapsed_s":41.5,"meta":{
  "protocol":1,"h3_version":"0.1.0-dev","git_commit":"9acdbdcddc52",
  "output":"/abs/video.mp4","settings":"/abs/video.json",
  "width":512,"height":512,"frames":22,"fps":24,"sample_rate":32000,"seed":"7",
  "loras":[{"path":"/abs/turbo.safetensors","scale":1,"format":"native",
            "low_rank":259,"full_deltas":0,"patched":259,"targets":259,
            "alpha_scale":1}]}}
```

- `seed` is a string: a 64-bit seed does not survive a JSON number.
- `settings` is left out with `--no-settings`, which skips writing the sidecar.
- `loras[].scale` is the strength given to `--lora PATH:S`, as in the sidecar;
  `alpha_scale` is the adapter's own alpha/rank scale.
- `elapsed_s` is measured from argument parsing to the end of the render.

On failure:

```json
{"type":"job.end","status":"failed","error":"<the reason>","elapsed_s":0.5}
```

A process that dies without writing `job.end` (a crash, a Metal abort, a kill)
has failed; its stderr has the cause.

## Versioning

- The first event is always
  `{"type":"log","level":"debug","source":"h3","text":"events protocol 1","meta":{"protocol":1,"h3_version":...,"git_commit":...}}`,
  and `job.end`'s `meta` repeats `protocol`.
- `h3 -d MODEL_DIR --info` prints `events protocol: 1`, so a supervisor can
  check an installed binary without rendering.
- Adding event types, fields, stage ids, metric keys or artifact kinds is
  backward compatible: readers must ignore what they do not know.
- Renaming or removing anything, or changing its meaning, increments the
  protocol number.
