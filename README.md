# Llama-Frankenstein

Patches for [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) on AMD RDNA4 (gfx1201, Radeon AI PRO
R9700).  They sew techniques from published papers onto llama.cpp and rdna-boosts, a body sewn together from the parts of
others: hence the name.  The optimizations target dense models: on a MoE model the experts' matmuls stay on the
rdna-boosts paths, and only the parts outside them gain.

**Pinned target (recommended).**  What the patches are built and measured on; other versions are not tested.  Full SHAs,
trees and where to get each: [BUILD.md](BUILD.md).

| | version |
|---|---|
| llama.cpp | master `919794500` (build `b12597`) |
| llama-cpp-rdna-boosts | `v18-919794500-r368` |
| ROCm | 10.0.0 (AMD's TheRock packages; the libraries loaded at run time come from it too) |
| HIP compiler | the clang ROCm 10.0.0 ships: AMD clang 23.0.0git (ROCm llvm-project `8f497e0992fb`) |

| code from | license | what it gave |
|---|---|---|
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | MIT | the engine; the WMMA tile helpers (`mma.cuh`) of J2's own Gated DeltaNet kernel and own f16 GEMM |
| [llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) | MIT | the RDNA4 base these patches apply to; its decode-attention band, KQ block table and the table's reuse across layers (J1), the GLU in place (J2), grid-stall padding (J3), graphs of up to 16 tokens (J6) |

The papers each part follows, and thanks: [STRUCTURE.md](STRUCTURE.md).  Patches only; MIT.  Written with Claude (AI).

## Structure

The main pipeline has been rebuilt.  This repo's earlier patches (0001-0015) and a new GPU-resident speculative pipeline
are merged into seven groups, numbered in the order of inference.  Each is one patch, except J2, which is two (J2 v15 is
both files): its part b is applied last, after J7, because it needs J4-J7's server and state code.  The order: J1, J2
part a, J3, J4, J5, J6, J7, J2 part b.

| group | part of the engine | what changes | type |
|---|---|---|---|
| J1 | attention | the full-attention layers' KV cache and every kernel that reads it: fp8 prefill attention, a `q8_0r` cache with a batch-invariant decode / verify kernel (for the layers it takes), 9-16 row verification through the decode kernel | speed/lossy, speed |
| J2 | prefill | part a: large-batch matmuls in fp8 through hipBLASLt (K-quant and MXFP4 weights), fused Gated DeltaNet prefill on an own chunked kernel, the logits a prompt ends with from a decode-sized batch; part b (applied last): an own WMMA f16 GEMM and an f16 batch for a long prompt's last tokens (both opt-in; both on with `FRANK_KQ_BLAS=fp8b`), the f16 conversion's batch amax from the kernels that write its input, fp8 scales as powers of two, the output projections in f16 (opt-in) | speed/lossy, memory, speed, fix |
| J3 | decode / verify | 1-16 token matmuls; Gated DeltaNet draft verification with two state planes instead of one per draft token; fused decode ops; the draft model's output layer scored from an int2 copy, its best candidates recomputed exactly | speed/diff, memory, speed |
| J4 | sampling | an end of turn sampled inside the reasoning becomes the reasoning's end tag | guard |
| J5 | speculative decoding | the draft width from the acceptance, drafts of one length for slots verifying together, n-gram drafts from the prompt, an opt-in lenience for sampled drafts | speed, speed/trade-off, sampling |
| J6 | step issue | the speculative loop on the GPU: sampling, draft acceptance and the next step's input on the device, two sequences in one step, each step checked by the host | speed, guard |
| J7 | state between requests | prompt cache and context checkpoints in pinned memory, saved in the background, cut on the batch grid | speed |

What each group contains, its switches, sources and references, the lossy parts with their KLD, and the change log:
[STRUCTURE.md](STRUCTURE.md).

## Performance

Qwen3.8-27B, official UD-Q5_K_M, R9700, Linux; this repo's defaults and the command under
[Run and switches](#run-and-switches).

| prompt | prefill t/s | decode t/s | KLD, generation (zh / en) | KLD, prompt processing (zh / en) |
|---|---|---|---|---|
| 8K | 2938 | 71.0 | 0.0043 (1.6×) / 0.0026 (3.3×) | 0.038 (4.8×) / 0.0075 (5.3×) |
| 32K | 2742 | 63.5 | 0.0043 (1.7×) / 0.0007 (1.7×) | 0.037 (5.2×) / 0.0027 (3.9×) |

- Speed: a public-domain novel (PG-19 #4047, *The Leavenworth Case*) and a request for a review, 8,163 and 32,709 tokens;
  prefill without the prompt cache, decode 512 greedy tokens with the DFlash2 draft; median of 3 rounds after a warm-up
  (the rounds gave the same tokens; spread under 0.6 %).  Measured 2026-11-13.
- KLD: the mean per token (without the largest one) over 400 teacher-forced tokens after an 8K / 32K prefix of a Chinese and
  an English text, against the same build with every switch off.  That build is plain rdna-boosts: its 512 decoded tokens
  are the same as stock r269's with the draft, and its KLD to stock r269 is 0.  In brackets, the multiple of the floor: the
  same comparison for one bit of K changed in an emulated build, at the same batch width.  Generation: batches of 1, 6 and
  16 tokens (decode, DFlash2 verification, n-gram verification), the one furthest above its floor; here always 1 and 6,
  which give the same bits.  Prompt processing: one 400-token batch, as in tools that score every token of a prompt (a
  reply's first token comes from a decode-sized batch: the prompt tail, J2).  Floor, generation: 0.0028 / 0.0008 (8K),
  0.0026 / 0.0004 (32K); prompt processing: 0.0079 / 0.0014 (8K), 0.0071 / 0.0007 (32K).

### Since the table

The table above was measured on r269 (2026-11-13).  What changed since, newest first; the details are in the
[change log](STRUCTURE.md#change-log).  "The 32K prompt" is the table's 32K row on UD-Q5_K_M (prefill / decode t/s; when the
decode tokens differ, decode t/s also follows the draft acceptance).  "2 slots" is UD-Q4_K_XL with `--parallel 2`.

| date | base | change | output with the defaults | speed |
|---|---|---|---|---|
| 2026&#8209;11&#8209;28 | r368 | J3 v5.4: IQ4_XS / IQ4_NL / MXFP4 weights through mmvk (`FRANK_MMVK_EXT=3`); J6 v11, J7 v3.1: fixes | changes where a model has those weights (UD-Q4_K_XL does); the decode-path checks against a clean reference are within those of the stock kernel ([lossy parts](STRUCTURE.md#lossy-parts)) | per step -0.5 to -0.8 % (2 slots); the 32K prompt 2734 / 62.4 (44.00 ms per step, r359: 44.10) |
| 2026&#8209;11&#8209;26 | r359 | J2 v15: the fp8 scales of K1, M4 and F8 are powers of two | changes; qgate passes (KLD 1.33× / 1.10× the floor, zh / en); with `FRANK_FP8_POW2=0` the same text as before on 22 server requests (2 slots) | the same; the 32K prompt 2736 / 64.4 |
| 2026&#8209;11&#8209;24 | r331 | J2 v13: the norm lock removed (a fix: it clipped later inputs, and replies quoting numbers from tool results got worse, 16 against 24 of 44 passing) | the same as before the lock | prefill gives back the lock's 1.2-2.1 % |
| 2026&#8209;11&#8209;22 | r329, r331 | rdna-boosts on llama.cpp master `919794500`; J2 v10: Gated DeltaNet prefill on an own kernel (`FRANK_GDN_OWN`) | the same tokens on the new base; the own kernel's KLD against an fp32 Gated DeltaNet in the range of rdna-boosts' kernel's | prefill -0.2 to -0.3 %; the 32K prompt 2784 / 62.8 |
| 2026&#8209;11&#8209;21 | r327 | the norm lock (J2, removed in v13); rdna-boosts' KQ block table reuse (J1) | the reuse: the same | the lock: prefill +1.2 to +2.1 % |
| 2026&#8209;11&#8209;14, 11&#8209;17 | r277, r295 | bases only (r295: llama.cpp's 1.0 line) | the same tokens and log-probs as on r269 | the 32K prompt 2738 / 63.0, 2746 / 63.2 |

### History

| date | build | prefill 8K / 32K | decode 8K / 32K | KLD 32K, generation / prompt processing (zh / en) |
|---|---|---|---|---|
| 2026&#8209;11&#8209;01 | [llama-cpp-overdoing](https://github.com/overdoingism/llama-Frankenstein/blob/edef875/README.md): `v16-6050b467c-r195` + K1&nbsp;v7, F8&nbsp;v5, M4&nbsp;v4, mmvk&nbsp;v4 | — / 2510 | — / 59.4 | — |
| 2026&#8209;11&#8209;13 | J1-J7 on `v17-a691b2345-r269` | 2938 / 2742 | 71.0 / 63.5 | 0.0043 / 0.0007; 0.037 / 0.0027 |

The 2026-11-01 row is the last one of this repo's earlier form (then named llama-cpp-overdoing): the same 32K prompt, a
`q8_0` KV cache.

## Reproduction conditions

What follows records how the numbers above were obtained (the pinned sources, the build, the command and the switches) so
that the measurements can be repeated.  It is not a recommendation to run these patches, and no binaries are published
here.

### Build

| what | pinned at |
|---|---|
| llama.cpp | commit `919794500475862f6768c353255469dc661cb734` (master, build `b12597`) |
| llama-cpp-rdna-boosts | release `v18-919794500-r368`, commit `d829586b6d404d2549e980eadeedf004f476f079`, applied with its own `scripts/apply-all.sh`; the same commit is the tag `pin/v18-919794500-r368` in [overdoingism/llama-cpp-rdna-boosts](https://github.com/overdoingism/llama-cpp-rdna-boosts), kept in case the original goes away |
| this repository | the eight patch files in [`patches/`](patches), all of them, in the order J1, J2 part a, J3 to J7, J2 part b |
| ROCm | 10.0.0 with hipBLASLt, the HIP code compiled for gfx1201 |

The method, the principles, a tree hash to check after each stage and the pitfalls, for Linux and Windows:
[BUILD.md](BUILD.md).  It states conditions to meet rather than a script, for a person or an AI agent.  A group alone is
not guaranteed to build: turn parts off with their switches instead.  On another release check with `git apply --check`
first; a release on a new llama.cpp base needs a new set.  The [GitHub Actions workflow](.github/workflows/windows-gfx1201.yml)
can build a Windows package when run by hand (none is published here), with
[`run-server-sample.ps1`](packaging/windows/run-server-sample.ps1): the command below, with the two model paths to fill in.

Tested on gfx1201 only.  On RDNA3 the RDNA4 paths stay off: use rdna-boosts itself.  gfx1200 (RX 9060 XT) takes them but is
untested (build with `GPU_TARGETS=gfx1200`).

### Run and switches

The command behind the [Performance](#performance) numbers (Linux; on Windows the same in PowerShell, without
`HSA_USERPTR_FOR_PAGED_MEM`):

```bash
HSA_USERPTR_FOR_PAGED_MEM=0 ./llama-server -m Qwen3.8-27B-UD-Q5_K_M.gguf -ngl 99 -c 131072 --kv-unified \
  -ub 2048 -b 2048 -fa on -ctk q8_0r -ctv q8_0r \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 --spec-draft-n-max 5 \
  --spec-type draft-dflash \
  --spec-draft-sampling probabilistic
```

| parameter | why |
|---|---|
| `-ub 2048 -b 2048` | J2's prefill matmuls are fastest on 2048-token batches (and take small weights only from 1024 tokens).  The compute buffer grows by about 1.1 MiB per ubatch token: lower it (e.g. 1536) if a long context does not fit. |
| `-fa on -ctk q8_0r -ctv q8_0r` | J1's `q8_0r` cache (the same memory as `q8_0`) and its decode / verify kernel.  The fp8 prefill attention runs on `q8_0r` and `q8_0`, the 9-16 row halves only on `q8_0`.  A layer the `q8_0r` kernel does not take (a head size other than 256, as the DFlash2 draft model's; more than 6 query heads per KV head; attention sinks) keeps `q8_0` and logs it. |
| `-md ... -ngld 99 --spec-draft-n-max 5`, `--spec-type draft-dflash` | The DFlash2 draft model, for Qwen3.8-27B only ([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), Q4_K_M).  The adaptive width, the int2 draft head and the GPU-resident loop act only with it; mmvk's default and the Gated DeltaNet verification act with any drafts.  With any other model leave out the last three lines. |
| `--spec-draft-sampling probabilistic` | llama.cpp's own: sampled drafts, verified exactly by rejection sampling (it matters at temperature > 0).  J5's lenience (lossy, off by default) builds on it. |
| `--spec-type ngram-mod,draft-dflash --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-match 30` | Not in the command above.  J5's n-gram drafts from the prompt act only with an n-gram type, put before DFlash2; with them the n-min defaults to 1. |
| `--poll` | Nothing to set.  Windows builds have no OpenMP, so at llama.cpp's default `--poll 50` the CPU thread pool would spin after the CPU part of every step (4.2-4.4 cores on a Ryzen 7 5700G while generating, 0.85-0.91 at 0, the same speed and text): J6 makes `--poll 0` the default there. |

**Environment.**  `HSA_USERPTR_FOR_PAGED_MEM=0` is ROCm's own variable, Linux only.  By default ROCm allocates pinned host
memory (`hipHostMalloc`, llama.cpp's `ROCm_Host` buffers) as pageable pages registered with the GPU; when the kernel swaps
out or moves one (memory pressure, compaction), the driver stops every GPU queue of the process until the page is back
(5-440 ms at a time, often several in a row).  With `0` the driver allocates them instead and they stay put: under the same
memory pressure, 23 stalls (1.16 s over 19 requests) became none, with the same output and speed.  The cost: those buffers
(about 0.7 GB with every layer on the GPU) stay in RAM, and with large weights in host memory (partial offload without
mmap) they count against the driver's limit.

**Switches.**  Every part has an environment variable, read once at startup.  Everything is on by default, the lossy K1,
F8 and M4 included (they are part of what the numbers above measure), except the parts marked opt-in.
`0` turns a part off; `FRANK_ALL=0` turns off every part that changes values or the order of work, and that build is the
KLD reference above.  Every switch, its values and its default: [SWITCHES.md](SWITCHES.md).

**What to check at run time:**

- The startup log shows `mmvk on (...)`, `FRANK_KQ_BLAS: hipBLASLt found; K1 prefill GEMM in fp8` and
  `K1 warm-up (2048 tokens, ...): fp8 GEMM ready for 8 shapes`, and with the draft model
  `int2 coarse head for output.weight (...)`.  `hipBLASLt has no ... kernels` means hipBLASLt did not find its gfx1201
  kernels and K1 stays off ([BUILD.md](BUILD.md#pitfalls)).
- K1 is on by default only when every GPU the program sees is a gfx1201, and J3's int2 draft head and J6's loop only with
  one GPU: if HIP also sees a CPU's integrated GPU (`found 2 ROCm devices`), hide it with `HIP_VISIBLE_DEVICES`.
- Two things sit outside `--fit`: J3's int2 draft head (about 0.4 GiB with the DFlash2 draft model) and J6's loop
  (0.13 GiB).  On Windows a device allocation that does not fit does not fail: it lands in shared GPU memory and everything
  slows down (the first prefill is far below its usual speed).  Task Manager's shared GPU memory also counts pinned host
  memory, which is RAM, not spilled VRAM: J7's prompt-cache and checkpoint pools (about 13 GiB with this command) and the
  mapped token embedding (0.8 GiB).

## Notice

These patches are offered for research and evaluation.  Every part can be turned off with its switch
([SWITCHES.md](SWITCHES.md)).  Code from other projects is credited under its license in the file headers and in
[STRUCTURE.md](STRUCTURE.md), which also lists the papers each part follows.

## Thanks

To stew675 and the rdna-boosts contributors, to Deadcode and the radiance contributors, and to the llama.cpp / ggml
community.  The authors of the papers we follow are listed in [STRUCTURE.md](STRUCTURE.md).

## Note

Issues and pull requests may not get an answer here.  For problems, try a frontier cloud model first.
