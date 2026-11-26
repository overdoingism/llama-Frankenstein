# Switches

Written with Claude (AI).  Every switch is an environment variable, read once when `llama-server` starts.  **Everything is on
by default** (the default is under each variable) except the parts marked opt-in: to turn a part off, set its variable to
`0`.  `FRANK_ALL=0` turns off every part that changes values or the order of work ([below](#turning-everything-off)).  Set
the variables in the shell that starts the server (`NAME=value` before the command in bash, `$env:NAME = "value"` in
PowerShell).  An on / off switch reads any number but `0` as on, and `0`, an empty value or text as off, unless its row says
otherwise.  The command the README's numbers were measured with, and the server parameters the parts need:
[README](README.md#run-and-switches).  The groups and what their parts do: [STRUCTURE.md](STRUCTURE.md); the lossy parts
and their KLD: [STRUCTURE.md](STRUCTURE.md#lossy-parts).

## Main switches

### J1 attention

| variable | values | what it does |
|---|---|---|
| `FRANK_FA_FP8`<br>(default on, RDNA4) | `1`, `0` | **Lossy.**  Prefill flash attention in fp8, from a `q8_0` cache (head 64 / 128 / 256) or a `q8_0r` cache (head 256): batches of 32 tokens or more over 512 or more KV cells, GQA 1-8, no sinks / ALiBi / softcap.  Where it runs (every GPU RDNA4) it also makes rdna-boosts' `GGML_CUDA_FA_STAGE_MAX_MB` default to 1 ([related](#related-not-from-this-repo)). |
| `FRANK_FA_Q8R`<br>(default on) | `1`, `0` | The decode / verify attention kernel of the `q8_0r` KV cache (`-ctk q8_0r -ctv q8_0r`): head 256, GQA up to 6, 1-31 query rows, RDNA4.  Its split points are fixed, so decoding a token and verifying it in a draft give the same numbers.  `0`: the op runs on f16 copies of K and V instead.  Not bit-identical to the `q8_0` kernels (within the floor).  With `-ctk q8_0r -ctv q8_0r`, a layer this kernel does not take (a head size other than 256, more than 6 query heads per KV head, attention sinks) keeps a `q8_0` cache and logs it, so its attention runs on the `q8_0` kernels instead of on f16 copies. |
| `FRANK_FA_BAND_SPLIT`<br>(default on) | `1`, `0` | With a `q8_0` cache on RDNA4: a draft-verification batch of 9-16 query rows (two slots of a unified KV cache verifying at once, or a long n-gram draft) runs as two halves of at most 8 rows through the decode / verify kernel instead of the generic one; each row gets the result a batch of 8 rows or fewer gives (not bit-identical to the generic kernel).  From `FRANK_FA_BAND_SPLIT_MIN_KV` KV cells (`8192`; `0` = any length).  A `q8_0r` cache does not need it (its kernel takes up to 31 rows). |
| `GGML_CUDA_FA_KQ_BLOCKS_REUSE`<br>(default on here) | `1`, `0` | rdna-boosts' switch (r327; opt-in there): the attention layers of one graph evaluation share one KQ block table (the mask's KV-group bitmap) instead of each building its own.  The same output.  This set turns it on by default.  Measured: no difference on Linux; on Windows about +0.3 % decode with 2 slots and a parked 26K-token slot.  `GGML_CUDA_FA_KQ_BLOCKS_CENSUS=1` logs how often the table was built and reused ([related](#related-not-from-this-repo)). |

### J2 prefill

| variable | values | what it does |
|---|---|---|
| `FRANK_KQ_BLAS`<br>(default `fp8` when every GPU is a gfx1201; otherwise off) | `fp8`, `fp8b`, `f16`, `0` | Prefill matmuls of quantized weights (not MXFP4; dense layers, not MoE experts) through hipBLASLt (K1).  `fp8`: **lossy**, the fastest.  `fp8b`: `fp8` plus the precision recoveries whose share of the fp8 error taken back (against K1 `f16`) is larger than their share of the time `f16` adds: the tail batch (`FRANK_KQ_BLAS_TAIL=own`) and the own f16 GEMM for the batches below K1's threshold (`FRANK_KQ_BLAS_OWN=1`); set either to `0` to leave it out.  `f16`: weights converted to f16, KLD near the floor, slower.  Batches of 384 tokens or more on weights of 20M elements or more, 1024 or more on smaller weights; decode and draft verification never reach it.  Any other value (also `1`) logs a warning and leaves K1 off; an empty value counts as unset.  On other GPUs set it explicitly (the f16 algorithms are then timed at startup and can change between runs). |
| `FRANK_KQ_BLAS_FP8_KEEP`<br>(default `64`) | a number, `0`-`248`; `0` = off | K1 `fp8` only: per input batch, the channels with the largest activations (rounded up to a multiple of 8) get a second fp8 term for their weight and activation rounding errors, in the same GEMM.  About half of the fp8 error above `f16` goes away for about 1% of prefill speed.  Weights with 1024 or more inputs. |
| `FRANK_KQ_BLAS_ADD`<br>(default on) | `1`, `0` | The residual ADD after a matmul K1 or M4 takes goes into the GEMM.  The same output.  Off while `FRANK_MXFP4_ALGOS=timed`, and for K1 `f16` while its algorithms are timed. |
| `FRANK_KQ_BLAS_PRECONV`<br>(default on in a Windows build; otherwise off) | `1`, `0` | Improves reliability on Windows. |
| `FRANK_MXFP4_DIRECT`<br>(default `blaslt`, RDNA4) | `blaslt`, `0` | **Lossy.**  MXFP4 weights only (M4): prefill converts them exactly to fp8 and runs K1's fp8 GEMM.  Batches of 64 tokens or more.  `fp8` and `f16` select older kernels, a little slower; any other value (also `1` or empty) logs a warning and leaves M4 off. |
| `FRANK_PP_TAIL`<br>(default on) | `1`, `0` | The prompt tail: a ubatch of 384 tokens or more (`FRANK_PP_TAIL_MIN`) that ends a sequence with tokens that need logits gives those tokens back to a small ubatch, so the logits a prompt ends with (a reply's first token) come from the decode-sized path, not from K1.  Batches with logits for every token (perplexity) are unchanged. |
| `FRANK_DRAFT_UB`<br>(default `512`) | a number; `0` = no cap | Caps the draft model's ubatch (only ever lowers it).  With `-ub 2048`, `512` saves about 0.7 GiB of VRAM. |
| `FRANK_KQ_BLAS_OWN`<br>(opt-in, default off; on with `FRANK_KQ_BLAS=fp8b`) | `1`, `0` | Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0 prefill matmuls through this repo's own WMMA f16 GEMM, which decodes the weights inside the kernel into the same f16 values as K1 `f16`'s conversion and accumulates in f32 (no f16 copy of the weight per batch).  With K1 `f16` it replaces hipBLASLt and takes batches from 128 tokens (`FRANK_KQ_BLAS_OWN_MIN_T`); with K1 `fp8` it takes only the batches below K1's threshold (128-383 tokens), which otherwise go to MMQ.  Only `1` turns it on (unset with `fp8b`: on). |
| `FRANK_KQ_BLAS_TAIL`<br>(opt-in, default off; `own` with `FRANK_KQ_BLAS=fp8b`) | `own` (or `1`), `mmq`, `0` | **Changes values, toward K1 `f16`.**  K1 `fp8` only: the last 256 tokens (`FRANK_KQ_BLAS_TAIL_N`) of a long prompt run in a batch of their own, below the size K1 takes, so the recurrent state the prompt leaves and the keys and values the first reply token reads are not computed through K1's fp8.  `own`: that batch through the own f16 GEMM (weight types it does not take stay on MMQ); `mmq`: through MMQ.  Long enough: `FRANK_KQ_BLAS_TAIL_MIN_CTX`, `auto` (the default) or a prompt length in tokens; `auto` takes a prompt when the tail costs at most `FRANK_KQ_BLAS_TAIL_BUDGET` percent (`4.5`) of its new tokens' prefill time, from a measurement at load (about 6-8K new tokens on Qwen3.8-27B; 8192 prompt tokens if it could not be measured).  Measured: it takes back 26 % of the fp8 error the generation path sees (KLD against K1 `f16`, 11 segments, UD-Q5_K_M, 8K prompts) for 87.7 ms per prompt. |
| `FRANK_FP8_POW2`<br>(default on) | `1`, `0` | **Changes values.**  The fp8 scales of the prefill activations (K1 `fp8` and M4) and of F8's Q / K / V are powers of two: the largest value a scale is taken from is rounded up to 448·2^k.  With the scale amax / 448, the whole fp8 grid of a batch (K1: every activation of the batch; F8: a KV head, a group of 32 V channels, a query row) moves whenever its largest value moves, so a tiny difference upstream changes the rounding of values far from the one that moved; with a power-of-two scale the grid stays put until the largest value crosses a power of two, and e4m3 keeps its relative precision (the largest value lands in (224, 448] instead of at 448).  The same prefill speed; against a clean reference (K1 `f16`, F8 off, an f16 cache) the KLD at 17 positions where replies part on small differences is slightly lower (mean 0.0074 → 0.0066).  `0`: the scales before J2 v15. |
| `FRANK_KQ_BLAS_OUT16`<br>(opt-in, default off) | `1`, `0` | **Changes values, toward K1 `f16`.**  K1 `fp8` only: the output projections of both attentions (`attn_output`, and Gated DeltaNet's `ssm_out`) run in f16, through the own f16 GEMM where it takes the weight type and K1 `f16`'s hipBLASLt GEMM otherwise.  Their inputs are the attention outputs, rounded to fp8 just before they are added to the residual stream: the place where K1 fp8 amplifies a small difference most.  In f16 that amplification is about halved, and the KLD against an all-f16 run at the same 17 positions falls from 0.0092 to 0.0052 (mean); prefill -5.0 / -4.4 / -3.4 % at 8K / 32K / 110K (Qwen3.8-27B UD-Q4_K_XL, 2 slots); decode does not reach K1. |
| `FRANK_GDN_OWN`<br>(default on) | `1`, `0` | Chunked Gated DeltaNet prefill (RDNA4, 128-dim heads) on this repo's own WMMA kernel instead of rdna-boosts' chunked kernel; values differ within the floor.  `0` (or `FRANK_ALL=0`): rdna-boosts' kernel as released, and the prefill conv writes f32 rows for it. |

### J3 decode / verify

| variable | values | what it does |
|---|---|---|
| `FRANK_MMVK`<br>(default `2`) | `0`-`4` | Q4_K / Q5_K / Q6_K matmuls of 1 to 8 tokens (decode and draft verification) run a kernel that keeps the activations in registers across several weight rows.  `2`: also launches the matmuls that read the same input (gate + up, q + k + v, DeltaNet qkv + z) as one kernel; `1`: one launch each (same numbers, about 1% slower).  `1` and `2` are on only when llama-server verifies drafts (a draft model, MTP or an n-gram type): one-token decode alone is 1-1.6% faster on the stock kernel.  `3` / `4` are the same, always.  RDNA4 only.  The log prints `mmvk on (...)` or `mmvk off (...)`.  Not bit-identical to the stock kernel (float summation order); decode and verification give the same results as each other. |
| `FRANK_MMVK_EXT`<br>(default `1`) | bit mask | `1`: grouped launches may mix weight types (bit-identical to same-type groups).  `2`: IQ4_XS / IQ4_NL / MXFP4 through mmvk too (values change; their stock kernel is already close to the bandwidth limit, so normally leave it). |
| `FRANK_RS_ANCHOR`<br>(default on) | `1`, `0` | qwen35 / qwen35moe with drafts: each slot keeps two recurrent-state planes (after its last batch, and the one that batch started from) and per-token replay factors instead of one full state per draft token; a partly rejected draft is replayed.  Same output; about 1 GiB less VRAM with 2 slots (0.5 GiB with 1).  Off (with a log line) when a recurrent layer is on a device other than the CPU or a CUDA / ROCm GPU. |
| `FRANK_RS_REPLAY_ALL`<br>(default on) | `1`, `0` | With `FRANK_RS_ANCHOR`: after a verify batch that keeps replay factors, the state after the batch is not written; the next batch always starts from the state the batch started from and replays the accepted tokens, also when all were accepted.  Same output; 0.1-0.5% less time per step (less memory traffic). |
| `FRANK_GDN_NT`<br>(default on) | `1`, `0` | The Gated DeltaNet verify kernel reads the recurrent state and writes its per-token snapshots with non-temporal hints, so they do not push the next matmuls' weights out of the cache.  Same values. |
| `FRANK_HEAD_INT2`<br>(default on with one GPU, an RDNA4 one) | `1`, `0` | The DFlash2 draft model's output layer is scored from an int2 copy of its weight; the best 512 of each column (`FRANK_HEAD_INT2_R`) are recomputed through mmvk, bit for bit as the plain path computes them.  When the true top 16 is not among them, only that draft changes, never the verified output.  About 2% less time per step; about 0.4 GiB more VRAM, **outside `--fit`** (if it cannot be allocated, the plain path stays and the log says so).  Acts only where mmvk runs (with `FRANK_MMVK=0` it builds no copy).  `1` forces it with several GPUs. |

### J4 sampling

| variable | values | what it does |
|---|---|---|
| `FRANK_REASONING_GUARD`<br>(default on) | `1`, `0` | When the model samples an end of turn inside its reasoning block, the block's end tag (`</think>`) is emitted instead and generation goes on, so the answer lands in the reply instead of the reasoning.  Nothing before that point changes; it never shortens the reasoning, and a reply cut by `max_tokens` is left as it is.  When it fires the model usually writes its answer again after the reasoning (leave room in `max_tokens`), and the log prints `emitting the end tag`.  Needs a template whose reasoning end tag ends with one token (Qwen's `</think>`); otherwise the log says it is off. |

### J5 speculative decoding

| variable | values | what it does |
|---|---|---|
| `FRANK_DFLASH_ADAPTIVE`<br>(default on) | `1`, `2`, `0` | DFlash2 drafts each sequence's own width (from 3 to `--spec-draft-n-max`) from running acceptance estimates and a fixed cost model; the output does not change.  `2` also logs the width changes and, per request, the verifications per width.  `0`: the fixed width.  Cost constants: [tuning](#tuning-normally-unset). |
| `FRANK_SPEC_ALIGN`<br>(default on) | `1`, `0` | Hybrid / recurrent models only: sequences that verify drafts in the same batch send drafts of one length (DFlash2's width rises to the widest, a longer draft is cut to the shortest), so the verify batch is not split into ubatches.  rdna-boosts' own `LLAMA_SPEC_ALIGN` controls only its `draft-mtp-adaptive`. |
| `FRANK_SPEC_ALIGN_CUT`<br>(default on) | `1`, `0` | The cut to the shortest draft, and the raise of DFlash2's width to a draft another drafter already made, which it relies on.  `0`: unequal drafts are left to the ubatch split, as in the release. |
| `FRANK_SPEC_NGRAM_PROMPT_ONLY`<br>(default on) | `1`, `0` | Only with `ngram-mod`.  Each sequence looks up n-gram drafts only in its current prompt, in its own table with a key check; generated tokens are not added.  With it, `--spec-ngram-mod-n-min` defaults to 1 (any draft the table gives is used).  `0`: upstream's shared table without the check (and its n-min default). |
| `FRANK_SPEC_LENIENCE`<br>(opt-in, default `1` = exact) | a number between 0 and 1 | **Lossy.**  With sampled drafts (llama.cpp's own `--spec-draft-sampling probabilistic`) and a request temperature above 0, a drafted token is accepted with probability min(1, p / (l·q)) instead of min(1, p / q); the rest of the verification is unchanged, so no token becomes more than 1/l times as likely as under the model.  The request field `frank_spec_lenience` sets it per request.  No effect at temperature 0 or with greedy drafts.  What `0.8663` did to speed and quality: [STRUCTURE.md](STRUCTURE.md#lossy-parts). |

### J6 step issue

| variable | values | what it does |
|---|---|---|
| `FRANK_STEP_LOOP`<br>(default on with one GPU, a ROCm one; off with several GPUs) | `1`, `0` | The GPU-resident speculative loop: with a draft model, each step's sampling, draft acceptance (with the reasoning guard, lenience and the Gated DeltaNet verification), the next input and the KQ mask are made on the device, and the steps replay captured graphs.  Per step 5.7-5.9% faster at 8K and 110K on Linux, 6.2-7.7% at 8K-110K on Windows.  Greedy output is the same as on the host path; at temperature > 0 a token can change, as it can between two host-path runs.  With several GPUs it is untested, so it stays off unless you set `1`.  It needs some VRAM outside `--fit` (0.13 GiB with the mapped token embedding, 0.95 GiB without, on Qwen3.8-27B). |
| `FRANK_STEP_LOOP_SEQ2`<br>(default on) | `1`, `0` | With the loop: two sequences are verified in one step. |
| `FRANK_STEP_CHECK`<br>(default on) | `1`, `0` | With the loop: the host checks every step it reads back: the position moved by the tokens emitted, the last token, the recurrent state's parity, rows and cell; for a greedy sequence without a reasoning-guard replacement also that the emitted tokens are the target's and the step ended at the first draft that differs.  A mismatch is logged (`... the loop issues no more steps`), the loop issues no more steps and the host path takes over.  Same output and speed. |
| `FRANK_DEVICE_INPUT_MAPPED`<br>(default on) | `1`, `0` | With the loop: the token embedding stays in pinned host memory that the GPU reads in place, so its lookup runs in the GPU graph without a VRAM copy (0.66 GiB less VRAM on Qwen3.8-27B).  Same bytes.  A tied embedding (a model without an output weight, such as Gemma 4) is not mapped: it stays in VRAM as the output layer, one copy for both. |

### J7 state between requests

| variable | values | what it does |
|---|---|---|
| `FRANK_STATE_ASYNC_BANDS`<br>(default on with one device) | `1`, `0` | A state above rdna-boosts' `LLAMA_STATE_ASYNC_MAX_MIB` (256) is copied through the pinned stage one band at a time instead of tensor by tensor.  Same bytes.  `1` forces it with several devices, where a `-sm tensor` split may not serve the bands. |
| `FRANK_SERVER_CKPT_ASYNC`<br>(default on) | `1`, `0` | Context checkpoints are copied to host memory on a worker thread; every task waits for the copy before it touches the slot. |
| `FRANK_SERVER_CACHE_ASYNC`<br>(default on) | `1`, `0` | When a task takes over a slot while another slot is working, the slot's previous prompt is copied to the prompt cache (pinned entries) on a stream of its own, so the other slots go on decoding instead of waiting for the copy; the new task waits for it.  The data saved is the same; the new prompt may start an iteration later, so the batches it shares, and with K1 `fp8` a word, can change. |
| `FRANK_SERVER_CACHE_PINNED`, `FRANK_SERVER_CKPT_PINNED`<br>(default `100%` each) | `N%`, a number (MiB), `N%,MiB` (the smaller), `0` = off | At startup (on a background thread) the server allocates one pinned host-memory pool for the prompt cache (`N%` of `--cache-ram`) and for the context checkpoints (`N%` of all the checkpoints the context can hold), and saves and restores straight between them and the GPU.  **It takes host RAM**: with the README's command ([Run and switches](README.md#run-and-switches)) on Qwen3.8-27B, 8 GiB for the prompt cache and 5.3 GiB for the checkpoints, capped to the smaller of half the RAM minus 2 GiB and three quarters of the free RAM.  Same text and draft counts with the pools on and off. |
| `FRANK_SERVER_CKPT_ALIGN`<br>(default on) | `1`, `0` | The checkpoint near the end of a prompt (4 + `n_ubatch` tokens before it) is made at the start of the batch that holds it, so a prompt is cut into batches only at the batch size, at user-message breaks and before its last 4 tokens: a request continued from the prompt cache gives the same output as one run from scratch.  `0`: the release's cut (the batches, and so a long prompt's numbers, depend on where the cache ends). |

## Turning everything off

`FRANK_ALL=0` (exactly `0`): every switch above that you leave unset defaults to off, so every part that changes values
or the order of work is off; a switch you set still wins (`FRANK_ALL=0 FRANK_FA_FP8=1` turns on F8 alone).  What stays
on is bit-identical: the fused kernels and A/B switches listed [below](#ab-switches-normally-unset), the state copies in
bands and in the background, and the pinned pools.  It also turns off the shared KQ block table
(`GGML_CUDA_FA_KQ_BLOCKS_REUSE`, as in the release); rdna-boosts' own defaults (`GGML_GLU_INPLACE`,
`GGML_CUDA_GRAPH_MAX_TOKENS`) stay as they are.  `FRANK_FA_Q8R` stays on too, but only acts on a `q8_0r` cache, which
plain rdna-boosts does not have.  On r269, `FRANK_ALL=0` with a `q8_0` cache gave the same 512 decoded tokens as plain
rdna-boosts with the draft model, and a KLD of 0 against it; on r277 and on r295, the same three replies as plain
rdna-boosts (2026-11-14, 2026-11-17).

## Tuning (normally unset)

| variable | default | what it does |
|---|---|---|
| `FRANK_KQ_BLAS_ALGOS` | built-in list on gfx1201 | K1 `f16` only: how hipBLASLt algorithms are picked.  `def.gfx1201v1`: the built-in gfx1201 list (refused on other GPUs, where the default is `tune`).  `K1a2.<...>`: reuse a table the log printed earlier.  `tune`: time the candidates at startup (about 9 s; the choice, and so the output, can change between runs).  `0`: hipBLASLt's own first choice.  The log prints the table in use as `FRANK_KQ_BLAS_ALGOS=K1a2.<...>`. |
| `FRANK_MXFP4_ALGOS` | built-in gfx1201 fp8 list | Every fp8 GEMM (K1 `fp8` and M4).  `heur`: hipBLASLt's first choice.  `timed`: time the candidates on the first full batch (can change between runs; turns the residual ADD into the GEMM off). |
| `FRANK_KQ_BLAS_CHUNK_MB` | `0` (whole) | K1 `f16` only: converts weights in pieces of that many MB.  Saves little VRAM; prefill gets slower and the output changes.  Not recommended. |
| `FRANK_KQ_BLAS_MIN_T` (384; f16: 640, at least 256), `FRANK_KQ_BLAS_OWN_MIN_T` (128), `FRANK_MXFP4_MIN_T` (64), `FRANK_FA_FP8_MIN_T` (32), `FRANK_FA_FP8_MIN_KV` (512), `FRANK_PP_TAIL_MIN` (384), `FRANK_FA_BAND_SPLIT_MIN_KV` (8192) | in brackets | The smallest batch (tokens) or KV length each path takes.  K1's applies to weights of 20M elements or more (smaller ones: at least 1024). |
| `FRANK_KQ_BLAS_OWN_SHAPE` | unset | The own f16 GEMM: `TMxTN/WMxWN`, with `a2` for two activation stages, fixes the block shape to one of the built shapes (e.g. `4x4/4x2a2`: 256 tokens × 128 weight rows in 8 waves); unset: chosen by the batch size (diagnostics). |
| `FRANK_KQ_BLAS_TAIL_N` (256), `_TAIL_MIN_CTX` (`auto`), `_TAIL_BUDGET` (4.5) | in brackets | `FRANK_KQ_BLAS_TAIL`: the tail's length (under 384, so K1 does not take it); the prompt length it starts at; the cost limit of `auto` in percent. |
| `FRANK_FA_FP8_QB` | `4` | F8, head 256: how many 16-token groups of a block share one K / V tile (`2` or `4`; anything else 1).  Bit-identical. |
| `FRANK_FA_Q8R_SPLIT` | `24` | `q8_0r` kernel: KV splits per head.  Another value gives other split points, so other numbers. |
| `FRANK_HEAD_INT2_R` | `512` | Candidates recomputed exactly per column (16-4096). |
| `FRANK_DFLASH_ADAPTIVE_COST` (0.035; 0.018 with `q8_0r`), `_JUMP` (0.08; 0.04), `_JUMP_ROWS` (5; 6) | in brackets | The adaptive width's cost model: the cost of one more verified position as a fraction of a step; the extra cost from `_JUMP_ROWS` rows at 110K, scaled with the context. |
| `FRANK_STEP_KV_HEADROOM` | `1024` | With the loop and a `q8_0r` cache: verify graphs span that many free cells past the used ones, so the loop runs longer before handing back to the host.  `0` = off. |

## A/B switches (normally unset)

Each turns off one bit-identical change (or, for the M4 `fp8` mode, sets an experiment); `0` gives the earlier code path
with the same output, for comparisons: `FRANK_FA_FP8_CONV`, `FRANK_FA_FP8_COMBINE4`, `FRANK_FA_Q8R_SW` (`2` = two
staging warps), `FRANK_KQ_BLAS_SHARE`, `FRANK_KQ_BLAS_DIRECT`, `FRANK_KQ_BLAS_DIRECT_B32`,
`FRANK_MXFP4_SHARE`, `FRANK_MXFP4_AMAX` /
`FRANK_MXFP4_CONV` (`row` = the older kernel), `FRANK_FUSE_CONT_GATE_F32`, `FRANK_GDN_CONV_NORM_INPLACE`,
`FRANK_GDN_BF16_IN`, `FRANK_NORM_PAIR_WAVE`, `FRANK_NORM_ROWS_ROPE_FWHT`, `FRANK_ROPE_FWHT_WAVE`,
`FRANK_DFLASH_CONV_FUSED`, `FRANK_RMS_Q8_1_REGS`, `FRANK_STEP_ONE_STREAM`,
`FRANK_STEP_GRAPH_WIDE`, `FRANK_STEP_SLAB` (on whenever the loop is), `FRANK_SPEC_TRIGGER_STOP` (follows the loop),
`FRANK_SERVER_CKPT_ALIGN_NEXT`.  `FRANK_MXFP4_CFG` and `FRANK_MXFP4_SPLITK` are experiments of M4's `fp8` mode.  Do not
set `FRANK_SLAB_SYNC=0`: it is racy (A/B only).

## Diagnostics (normally unset)

| variable | what it does |
|---|---|
| `FRANK_KQ_BLAS_F16_SET=<list>` (`FRANK_KQ_BLAS_F16_SET_OWN=0`: hipBLASLt only) | For measurements: with K1 `fp8`, the K1 matmuls of the weights in the list run in f16 (the same token thresholds), as `FRANK_KQ_BLAS_OUT16` does for the output projections.  A comma-separated list of groups and name parts: `state` (the keys and values, Gated DeltaNet's q / k / v and gates: what generation reads back from the prompt), `state-noab` (without the gates), `out` (the output projections, the same as `FRANK_KQ_BLAS_OUT16=1`), or a part of a weight's name (`blk.53.`, `ffn_down`).  The log lists the weights it takes. |
| `FRANK_KQ_BLAS_LOG=1` (`2`: with timings) | One line per GEMM shape with its algorithm: K1 and every fp8 GEMM, M4's too. |
| `FRANK_MXFP4_LOG=1` | Per MXFP4 weight on first use: the blocks outside the exact fp8 range (synchronizes). |
| `FRANK_KQ_BLAS_CHECK=1` | Every weight conversion also runs the reference kernel and logs differing bytes (slow). |
| `FRANK_FA_FP8_CONV_CHECK`, `FRANK_FA_FP8_ATTN_CHECK`, `FRANK_FA_FP8_COMBINE_CHECK`, `FRANK_FA_Q8R_COMBINE_CHECK`, `FRANK_FUSE_CONT_GATE_CHECK`, `FRANK_NORM_PAIR_WAVE_CHECK`, `FRANK_NORM_ROWS_ROPE_CHECK`, `FRANK_SSM_GATE_CHECK` | `1`: the fused or faster kernel is compared, bit for bit, with the one it replaces (testing only, slow). |
| `FRANK_GDN_CONV_NORM_DEBUG`, `FRANK_RS_ANCHOR_DEBUG` | `1`: log why a fusion was or was not taken. |
| `FRANK_HEAD_INT2_CHECK=1` (`_CHECK_K`, 16) | Every int2 head call also computes the exact logits and logs, every 64 calls, the columns whose exact top-K was not all among the candidates (with `GGML_CUDA_DISABLE_GRAPHS=1`). |
| `FRANK_STEP_AUDIT=1` | With the loop on, the device-side acceptance is always re-run for greedy host steps and compared (a mismatch is logged).  `1` also has the device write each host step's inputs for the step slab to compare, logs the counts every 64 steps, and works with the loop off. |
| `FRANK_STEP_DEBUG=1` (`2`, `3`: more) | The loop's bound inputs, hand-backs and run counts (WARN level).  Without it, a lasting reason the loop does not run (the drafter, a sampler) is logged once at INFO; `2` logs every reason, also the ones that come and go between steps (the prompt being processed, a pending draft). |

## Old names

Before 2026-11 (llama-cpp-overdoing, patches 0001-0015) the switches were named `GGML_LF_*`, `GGML_CUDA_*` and `LLAMA_*`:
the same name after the prefix became `FRANK_*` (`GGML_LF_KQ_BLAS` → `FRANK_KQ_BLAS`, `GGML_CUDA_MMVK` → `FRANK_MMVK`,
`LLAMA_RS_ANCHOR` → `FRANK_RS_ANCHOR`, `GGML_LF_ALL` → `FRANK_ALL`).  An old name is no longer read; the server warns at
startup when one is set.

On r327 (2026-11-21) three parts moved to rdna-boosts, which now reads its own names: `FRANK_FA_KQ_BLOCKS_REUSE` →
`GGML_CUDA_FA_KQ_BLOCKS_REUSE` (on by default here), `FRANK_FA_KQ_BLOCKS_CENSUS` → `GGML_CUDA_FA_KQ_BLOCKS_CENSUS`,
`FRANK_GRAPH_MAX_TOKENS` → `GGML_CUDA_GRAPH_MAX_TOKENS` (the server warns when an old one is set), and `FRANK_GLU_INPLACE`
→ `GGML_GLU_INPLACE` (no warning).  Removed with what they switched (the server warns when one is set): 6g's
`FRANK_KQ_BLAS_GLU16` and `FRANK_KQ_BLAS_GLU16_ALGOS`, and `FRANK_GDN_NORM_SILU_FP8` / `_DEBUG` (the Gated DeltaNet output
norm and SiLU gate kernel that wrote fixed fp8 activations).

On J2 v10 (2026-11-22) `FRANK_KQ_BLAS_OWN_BMF` and `FRANK_KQ_BLAS_OWN_TN` are no longer read (no warning): the rewritten
own f16 GEMM takes `FRANK_KQ_BLAS_OWN_SHAPE` instead.

On J2 v13 (2026-11-24) `FRANK_KQ_BLAS` takes `fp8b` (above) and `FRANK_KQ_BLAS_OWN` takes Q8_0 weights too.  The norm lock
(K1 fp8's fixed scales of the normalized inputs, called 6d before) is removed with its switches, and the server warns
when one is set: `FRANK_KQ_BLAS_FIXED`, `_FIXED_N`, `_FIXED_MIN_B`, `_FIXED_MIN_T`, `_FIXED_MARGIN`, `_FIXED_KEEP`,
`_FIXED_PRODUCER`, `_FIXED_NOF32`.  Every K1 batch takes its own scales, as `FRANK_KQ_BLAS_FIXED=0` did (why:
[STRUCTURE.md](STRUCTURE.md#lossy-parts)).

On J3 v5.3 (2026-11-26) the Gated DeltaNet output norm fused with its SiLU gate moved to rdna-boosts, which reads its own
name: `FRANK_FUSE_GDN_NORM_GATE_SILU` → `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU` (on by default).  From J7 v2.1 the server
warns when the old one is set.

## Related, not from this repo

| variable | what it does |
|---|---|
| `GGML_CUDA_FA_STAGE_MAX_MB` | rdna-boosts: cap (MiB) of the flash-attention F16 staging copy, which only grows.  Where F8 runs (every GPU RDNA4) and the variable is unset, J1 sets it to 1: the batches F8 does not take (9-31 tokens) then read the cache directly, with the same result and speed, and the copy (about 1.2 GB at 240K context; on Windows it ends up in shared memory) is never made.  An explicit value still wins; `0` means no cap. |
| `GGML_CUDA_FA_KQ_BLOCKS_CENSUS` | rdna-boosts (r327): `1` logs, per GPU context when it is freed, how often the KQ block table was built and reused. |
| `GGML_CUDA_GRAPH_MAX_TOKENS` | rdna-boosts (r326): the most tokens a batch may have and still be captured as a HIP graph (16; `8` = the band before r326), so 9-16-token verify batches (two slots, n-gram drafts) and prefill ubatches are replayed too.  Replaces J6's `FRANK_GRAPH_MAX_TOKENS`; `FRANK_ALL=0` no longer lowers it. |
| `GGML_GLU_INPLACE` | rdna-boosts (r323): a GLU may take its gate's memory when nothing else reads the gate, so an FFN's gate, up and GLU output are not all live at once (about 150 MiB less compute buffer at `-ub 2048` on Qwen3.8-27B); on, `0` off.  Replaces J2's `FRANK_GLU_INPLACE`. |
| `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU` | rdna-boosts (r353): the Gated DeltaNet output norm fused with its SiLU gate (on; `0` keeps the fusion to the sigmoid gate).  Sent from this repo's J3, which carries no copy of its own since v5.3. |
| `LLAMA_KV_ROWS_DISABLE` | rdna-boosts (r359): with `--kv-unified`, `0` turns on its K/V-rows attention (each sequence reads only its own cells), after a probe; unset or any other value keeps the dense mask.  While J6's loop is switched on (`FRANK_STEP_LOOP`, by default with one ROCm GPU, a draft model loaded or not) or its step slab (`FRANK_STEP_SLAB`), this repo keeps it off, since they use the dense mask, and logs it: turn the loop off to use it.  When it is on, J1's `q8_0r` decode / verify kernel, F8 and the 9-16 row halves step aside (a `q8_0r` cache is then read through f16 copies of K and V). |
| `LLAMA_DFLASH_DEV` | rdna-boosts: with a DFlash draft the target model's features stay on the GPU (on by default).  `0` forces the host path. |
| `LLAMA_DEVICE_INPUT` | rdna-boosts: the input layer on the GPU.  Unset, it follows the GPU-resident loop (J6), which needs it. |
| `LLAMA_STATE_ASYNC`, `LLAMA_STATE_ASYNC_MAX_MIB` | rdna-boosts: state copies through a pinned stage (on), and the size above which J7 copies in bands (256). |
| `GGML_CUDA_GRID_STALL_PAD` | rdna-boosts: the empty blocks against the gfx1201 grid stall (on); J3 adds one on mmvk. |
| `HIPBLASLT_TENSILE_LIBPATH` | hipBLASLt's own: where its kernels are.  Not needed when they are where hipBLASLt looks (on Windows, a `gfx1201` folder next to `llama-server.exe`): [BUILD.md](BUILD.md#pitfalls). |
| `HIP_VISIBLE_DEVICES` | HIP's own: which GPUs the program sees.  K1's default needs every visible GPU to be a gfx1201, J3's int2 draft head and J6's loop one GPU ([README](README.md#run-and-switches)). |
| `HSA_USERPTR_FOR_PAGED_MEM` | ROCm's own, Linux.  We set it to `0`: ROCm then allocates pinned host memory (`hipHostMalloc`, the `ROCm_Host` buffers) through the driver instead of as pageable pages registered with the GPU, which the kernel may swap out or move, stopping every GPU queue of the process until the page is back.  See the [README](README.md#run-and-switches).  Not needed on Windows. |

