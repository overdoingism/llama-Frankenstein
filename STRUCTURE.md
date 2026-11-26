# Structure

What each group in [`patches/`](patches) contains, its switches (default in brackets), where its ideas and code come
from, the lossy parts with their KLD, and the change log.  The short table is in the [README](README.md#structure).
Written with Claude (AI).

The seven groups follow the order of inference.  Each is one patch, `J<n>-919-<part>-v<version>.patch`, except J2, which
is two: `J2-919-prefill-a-v15.patch` and `J2-919-prefill-b-v15.patch` (J2 v15 is both).  They are applied in number order on
rdna-boosts `v18-919794500-r359`, with J2's part b last, after J7: it needs J4-J7's server and state code.  The files in
order: J1 attention v5, J2 prefill-a v15, J3 decode v5.3, J4 sampling v1, J5 speculative v3.1, J6 step v10.1, J7 state v2.1,
J2 prefill-b v15.  A group alone is not guaranteed to build; every part has a switch.  A version with a decimal (`v4.1`) is
a group adapted to a new base, nothing else; its next change drops the decimal.

Types: `speed` = faster, the same output; `memory` = less VRAM, the same output; `speed/diff` = faster, output not
bit-identical to the stock code but within the run-to-run floor; `speed/lossy` = faster, error above the floor;
`speed/trade-off` = faster on some workloads, slower on others; `sampling` = changes the sampling rule, not the logits;
`guard` = changes behavior only in a failure case; `fix` = corrects a fault.

## Lossy parts

KLD as multiples of the run-to-run floor (zh / en): Qwen3.8-27B official UD-Q5_K_M, 400 teacher-forced tokens after a
48K-token prefix, against the same build with every switch off.  Generation = the largest of the decode / verify widths 1, 6
and 16; prompt processing = one 400-token batch.  Floor: generation 0.0017 / 0.0005, prompt processing 0.0057 / 0.0011.
Speed: the part against the defaults without it (prefill t/s gained; decode time per step saved), 8K / 32K / 110K prompts.
Measured 2026-11-09 on r235 with a 48K prefix (the norm lock, a default from r327 until J2 v13 removed it: below); the defaults on r269 at 8K and 32K:
the README's [Performance](README.md#performance).

| part | group | KLD, generation | KLD, prompt processing | speed |
|---|---|---|---|---|
| K1 (fp8 prefill matmuls) | J2 | 1.42× / 1.72× | 4.01× / 4.35× | prefill +115 / +107 / +83 % |
| F8 (fp8 prefill attention) | J1 | 1.05× / 1.18× | 1.19× / 1.52× | prefill +12 / +29 / +65 % |
| M4 (MXFP4 weights only) | J2 | 0.0021 / 0.0009 (absolute) | 0.0126 / 0.0035 (absolute) | prefill +69 / +66 / +53 % |
| mmvk | J3 | 0.82× / 0.59× | 0 | per step −6.5 / −8.2 / −6.1 % |
| 9-16 row verification | J1 | 0.92× / 1.06× | 0 | copying with a `q8_0` cache: per step −12 % (32K), −29 % (110K) |
| `q8_0r` cache (against `q8_0`) | J1 | 0.84× / 0.53× | 0 | per step −2.3 / −3.4 / −7.3 % |
| **all defaults** | | **1.54× / 1.78×** | **4.54× / 4.72×** | prefill +133 / +137 / +152 %, per step −19 / −11 / −22 % |

The prompt-processing column is K1: the fp8 matmuls compute the scored token's logits directly there.  A reply's first
token does not come from there: the prompt tail (J2) computes the logits a prompt ends with in a decode-sized batch (and a
hybrid model with context checkpoints already ends its prompt with a small batch).  The column applies to tools that score
every token of a prompt (perplexity); later tokens read the prompt through the KV cache and see the generation column.  For
prompt scoring set `FRANK_KQ_BLAS=f16` (K1's matmuls in f16: slower, much smaller error).

**The norm lock (J2, on by default from r327, called 6d at first; removed in J2 v13)** was measured another way: the KLD against K1 `f16` on 22 kinds of content (400
tokens each), as a multiple of the default fp8's (per-batch scales), against the floor of the default fp8 run at another
batch split (`-ub 1024`), after a 3,000-token prefix.  With only the normalized residual streams fixed, every kind's KLD
stayed within that floor, also when the scales were fixed on hexdump, base64 or emoji first; so did top-1 agreement but
in one case (hexdump after fixing on classical Chinese poetry: 4.25 points lower; f16 at another batch split moves
hexdump's by 1.75).  qgate passes (zh 1.23×, en 1.19× its floor).  Fixing every K1 input (the earlier opt-in 6d, retired) made
some kinds of content much worse: on hexdump a KLD 10-50× the per-batch default's and top-1 agreement 4.5-8.75 points
lower.  Speed: prefill +1.6 / +2.1 / +1.2 % at 8K / 32K / 110K against the per-batch default (27B UD-Q4_K_XL, 2 slots),
measured 2026-11-21 on r324; decode does not reach K1.  It was removed in J2 v13 (2026-11-24): the scales came from the
first prompts after the server started, and later inputs outside that range were clipped.  In tool-using conversations
whose system prompt is text, the numbers in later tool results lost precision in the keys and values they wrote, and
replies quoting them got worse (22 tool-using questions run twice at temperature 0: 16 of the 44 replies passed with the
lock, 24 without it).  The KLD above stayed within the floor all the same: it averages over every position, and few of
them must recall an exact value from the context.

**Power-of-two fp8 scales (J2 part b, on by default from v15)** came after the table above.  The scales of K1 fp8's and
M4's activations and of F8's Q / K / V are rounded up to a power of two, so the fp8 grid of a batch no longer moves with
every small change of its largest value ([SWITCHES.md](SWITCHES.md)).  Against a clean reference (K1 `f16`, F8 off, an f16
cache, an exact per-token Gated DeltaNet) at 17 positions where replies to hard tool-using questions part on small
differences, the KLD is slightly lower (mean 0.0074 → 0.0066; 11 of the 17 lower); prefill the same (`pp8192`: 2,898 / 2,896
against 2,894 / 2,895 t/s).  Against the scales before, 400 tokens after a 48K-token
prefix (a Q5_K_M of the same model): KLD 1.33× / 1.10× the floor (zh / en), top-1 agreement 1.0 / 0.25 points lower.

**Output projections in f16 (J2 part b, opt-in from v15, `FRANK_KQ_BLAS_OUT16`)**: with K1 fp8, the attention output and
Gated DeltaNet output projections run in f16.  On the same 17 positions the KLD against an all-f16 run falls from 0.0092
to 0.0052 (mean; the largest from 0.052 to 0.016), and two runs of the Gated DeltaNet that differ by 1e-14 at the op part
half as much; prefill -5.0 / -4.4 / -3.4 % at 8K / 32K / 110K (UD-Q4_K_XL, 2 slots).

**Lenience (J5, sampling, off by default)** changes the rule that accepts sampled drafts, not the logits, so it has no
KLD of the kind above.  Its effect on the sampled distribution can be computed exactly at every verified position: at
`0.8663` and temperature 1.0, 0.005 nats per generated token (1.4 % of the probability moved), about what lowering the
temperature to 0.75-0.8 of its value does; no token the model keeps is ever dropped; greedy decoding is unchanged.  Speed:
+7.3 % on our agentic test set (temperature 1.0, thinking on), with no measurable drop in its pass rate.

## J1 attention

The KV cache of the full-attention layers and every kernel that reads it, for prefill, decode and draft verification.

| part | what | switch | type |
|---|---|---|---|
| F8 | prefill flash attention in fp8 on RDNA4, from a `q8_0` or `q8_0r` cache: batches of 32 tokens or more over 512 KV cells or more; no growing f16 staging copy.  The row sum is counted from the rounded fp8 probabilities (numerator and denominator from the same numbers); barriers order LDS only; WMMAs of two accumulators issued in turn where measured faster | `FRANK_FA_FP8` (on) | speed/lossy |
| `q8_0r` cache | a KV type whose rows hold 256 8-bit codes followed by their 8 scales, written in the same kernel as the KV store, and its own decode / verify kernel: one pass over the KV for the whole GQA group, the split points fixed (they do not move with the KV length or the number of queries), so decode and verification of the same token give the same numbers; every wave loads KV | `-ctk q8_0r -ctv q8_0r`; `FRANK_FA_Q8R` (on) | speed/diff |
| 9-16 row verification | with a `q8_0` cache, a verify batch of 9-16 query rows (two slots verifying at once, a long n-gram draft) runs as two halves through the RDNA4 decode band instead of the generic kernel, from 8192 KV cells.  Each row gets the result a batch of 8 rows or fewer gives | `FRANK_FA_BAND_SPLIT` (on), `_MIN_KV` (8192) | speed/diff |
| shared KQ block table | the attention layers of one graph evaluation share the mask's block table: rdna-boosts' (r327), opt-in there, on by default here | `GGML_CUDA_FA_KQ_BLOCKS_REUSE` (on) | speed |
| `q8_0r` by layer | with `-ctk q8_0r -ctv q8_0r`, only the layers the `q8_0r` kernel takes (head 256, at most 6 query heads per KV head, no attention sinks) get the `q8_0r` cache; the others keep `q8_0` and run on its kernels instead of on f16 copies of K and V, and the log names them.  Qwen3.8-27B is unchanged; Qwen3.6-35B-A3B (8 query heads per KV head): decode +2.6-2.8 %, the same text | — | speed |

rdna-boosts' K/V-rows attention (r359, opt-in: `LLAMA_KV_ROWS_DISABLE=0`) stays off while J6's loop is switched on; when it
is on, the `q8_0r` kernel, F8 and the 9-16 row halves step aside.

- Code: rdna-boosts (MIT): the decode band's group walk and barrier, the KQ block table and its reuse.
- References: J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao, "FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-precision", NeurIPS 2024, arXiv:2407.08608, §3.3, and Colfax Research, "Delivering
  1 PFLOP/s of Performance with FP8 FlashAttention-2", 2024-02-29 (both attention matmuls in fp8, the probabilities rounded to
  fp8 between them: F8).  T. Zadouri, M. Hoehnerbach, J. Shah, T. Liu, V. Thakkar, T. Dao, "FlashAttention-4: Algorithm
  and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451, 2026, §3.1.4, after M. Milakov,
  N. Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867, 2018 (the `q8_0r` kernel rescales its
  output only when a row's running maximum has grown by more than 8 in log2, decided per row so a row never depends on the
  others in its tile).  Y. J. Kim, R. Henry, R. Fahim, H. Hassan Awadalla, "Who Says
  Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production", SustaiNLP 2022, arXiv:2211.10017,
  §3.4.1-3.4.2 (int8 codes turned into f16 by placing them in the mantissa of 1024 and subtracting 1152).  S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman,
  "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", NeurIPS 2024, arXiv:2404.00456 (a Hadamard rotation of the keys
  before the KV cache is quantized; llama.cpp's KV rotation follows it, and the `q8_0r` cache stores and reads the rotated
  rows).  P. Micikevicius et al., "FP8 Formats for Deep Learning", arXiv:2209.05433, 2022 (e4m3).  J. T. Barron,
  D. Gnanapragasam, M. Tschannen, "A Fix for Softmax and Cross-Entropy", arXiv:2610.31642, 2026 (F8's row sum).
  H. He et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference", 2025-09 (fixed splits).  Built on: T. Dao et al., "FlashAttention", arXiv:2205.14135; T. Dao,
  "FlashAttention-2", arXiv:2307.08691; T. Dao, D. Haziza, F. Massa, G. Sizov, "Flash-Decoding for long-context
  inference", 2023-10 (the `q8_0r` kernel's KV splits).

## J2 prefill

The large-batch path apart from attention.  Part a holds everything but the own f16 GEMM, the tail batch, the batch amax of the f16 conversion,
`fp8b`, the algorithm lists by name, the power-of-two scales and the output projections in f16, which are part b.  The GLU in place is rdna-boosts' since r323 (`GGML_GLU_INPLACE`).

| part | what | switch | type |
|---|---|---|---|
| K1 | prefill matmuls of K-quant weights through hipBLASLt in fp8 (or f16), on batches of 384 tokens or more (f16: 640; 1024 on weights under 20M elements); the largest activation channels of each batch get a second fp8 term for their rounding errors in the same GEMM; the residual ADD goes into the GEMM (in f16 too, unless the algorithms are timed) | `FRANK_KQ_BLAS` (`fp8` when every GPU is a gfx1201; `fp8b`: fp8 with the own f16 GEMM and the tail batch on, part b), `_FP8_KEEP` (64), `_ADD` (on) | speed/lossy |
| own f16 GEMM | a WMMA f16 GEMM that decodes Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0 weights inside the kernel (the same f16 values as K1 f16's conversion, f32 accumulation): with K1 f16 it replaces hipBLASLt from 128 tokens; with K1 fp8 it takes the batches below K1's threshold instead of MMQ.  Part b | `FRANK_KQ_BLAS_OWN` (off; on with `fp8b`) | speed |
| batch amax | the f16 conversion of K1 f16 and the own f16 GEMM scales a batch by one power of two from its largest activation: the kernels that write the input (an RMS norm with its weight, a GLU, a gated unary with its multiply) fold their channel maxima into a buffer per stream, and a small kernel takes the largest; without such a producer, one pass of float4 loads over the input; the conversion writes eight values a thread.  The same output.  Part b | — | speed |
| tail batch | the last 256 tokens of a long prompt in a batch of their own through the own f16 GEMM (or MMQ), so the state and the keys and values the reply starts from are not computed in fp8; with K1 fp8 only, prompts long enough by a cost measured at load.  Part b | `FRANK_KQ_BLAS_TAIL` (off, `own` with `fp8b`; `own`, `mmq`), `_N` (256), `_MIN_CTX` (`auto`), `_BUDGET` (4.5) | fix |
| power-of-two scales | the fp8 scales of K1 fp8's and M4's activations and of F8's Q / K / V (F8 is J1's; the change is here) are rounded up to a power of two, so a batch's fp8 grid stays put until its largest value crosses one; e4m3 keeps its relative precision.  Part b | `FRANK_FP8_POW2` (on) | speed/lossy |
| output projections in f16 | with K1 fp8, the attention and Gated DeltaNet output projections (`attn_output`, `ssm_out`) run in f16: where fp8 amplifies a small difference most.  `FRANK_KQ_BLAS_F16_SET` runs any set of weights so, for measurements.  Part b | `FRANK_KQ_BLAS_OUT16` (off) | speed/trade-off |
| algorithm lists by name | K1's and M4's gfx1201 algorithm lists name hipBLASLt's solutions by kernel name, not by index, so another hipBLASLt build finds the same kernels (ROCm 10.2's renumbers every solution); the log says how many of each list's kernels the build has.  Part b | — | fix |
| prompt tail | a batch of 384 tokens or more that ends its prompts with output tokens hands those outputs to a decode-sized batch, so the logits a prompt ends with do not come from K1's prefill matmuls.  Batches with an output for every token (perplexity), embeddings and non-causal models are unchanged | `FRANK_PP_TAIL` (on), `_MIN` (384) | fix |
| M4 | MXFP4 weights converted exactly to fp8 for K1's GEMM, from 64 tokens | `FRANK_MXFP4_DIRECT` (`blaslt`) | speed/lossy |
| draft ubatch | the draft model's ubatch capped at 512: about 0.7 GiB less VRAM with `-ub 2048` | `FRANK_DRAFT_UB` (512) | memory |
| fused prefill | the attention output gate reads Q in place; Q / K norm, rope and the FWHT in one kernel; one-step fp8 conversion of Q8_0 / IQ4_NL weights; Gated DeltaNet: the convolution with the q / k norm, the output norm with the SiLU gate, chunked rows in bf16 for the own kernel | switches in [SWITCHES.md](SWITCHES.md) | speed |
| own Gated DeltaNet kernel | chunked Gated DeltaNet prefill on an own WMMA kernel (128-dim heads): per chunk of 64 tokens and key head, K Kᵀ and Q Kᵀ once for the value heads that share the key head; per 64 value columns the state stays in f32 in the WMMA accumulators, and each 16-token block is solved by forward substitution; bf16 operands, f32 accumulation.  Replaces rdna-boosts' chunked kernel | `FRANK_GDN_OWN` (on; `0` = rdna-boosts' kernel) | speed/diff |

- Code: ggml (MIT): the own Gated DeltaNet kernel and the own f16 GEMM are built on the WMMA tile helpers of `mma.cuh`
  (the f16 GEMM also on its swizzled loads); the f16 GEMM decodes the weights as ggml's `dequantize_row_*` do.
- References: X. Wu, Z. Yao, Y. He, "ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using
  Floating-Point Formats", arXiv:2307.09782, 2023, §3 (4-bit weights cast to fp8 for an fp8 GEMM with fp8 activations: K1,
  M4; with power-of-two block scales, as MXFP4 has, the cast is exact: M4).  Ashkboos et al., QuaRot (J1; the FWHT the fused Q / K kernel
  applies before the KV store).  Micikevicius et al. 2022 (above); T. Dettmers et al., "LLM.int8()", arXiv:2208.07339,
  2022; S. Ashkboos et al.,
  "QUIK", arXiv:2310.09259, 2023; Y. Zhao et al., "Atom", MLSys 2024, arXiv:2310.19102 (outlier channels handled apart at
  higher precision).  K1's channel
  amax folded into the kernel that writes the activation (an RMS norm with its weight, a GLU, a gated unary with its
  multiply), the batch scale then taken from the largest channel amax: H. Xi, H. Cai, L. Zhu, Y. Lu, K. Keutzer, J. Chen,
  S. Han, "COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training", ICLR 2025,
  arXiv:2410.19313, Group Scaling (a per-tensor max reduction in two stages, the per-group stage fused into the previous
  operation); Z. Yao, R. Yazdani Aminabadi, M. Zhang, X. Wu, C. Li, Y. He, "ZeroQuant: Efficient and Affordable
  Post-Training Quantization for Large-Scale Transformers", NeurIPS 2022, arXiv:2206.01861, §4.3 and Appendix D, computes
  the scale from the values still in the producing kernel's registers.  The prefill / decode split:
  A. Panferov et al., "Disaggregated Quantization: Specializing LLM Prefill and Decode", arXiv:2609.26333, 2026 (prefill on
  compute-native low-precision weights, decode on compact weight-only ones).  The residual ADD:
  J. Zeng et al., "Boosting Distributed Training Performance of the Unpadded BERT Model", arXiv:2208.08124, 2022 (an
  addition at a residual taken into the GEMM through β).  Built on: S. Yang, B. Wang, Y. Zhang,
  Y. Shen, Y. Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024,
  arXiv:2406.06484, §3 (the chunkwise form the own Gated DeltaNet kernel computes); S. Yang, J. Kautz, A. Hatamizadeh,
  "Gated Delta Networks", ICLR 2025, arXiv:2412.06464 (the gated chunkwise algorithm); for M4, the OCP Microscaling Formats (MX)
  Specification v1.0, 2023, and B. D. Rouhani et al., "Microscaling Data Formats for Deep Learning", arXiv:2310.10537.
  The batch amax of the f16 conversion: the producers' channel maxima as with K1 fp8 (COAT's Group Scaling, above).
  The f16 scale, a power of two from the amax's exponent: Rouhani et al. (above), Algorithm 1; P. Micikevicius et al.,
  "Mixed Precision Training", ICLR 2018, arXiv:1710.03740, §3.2.  The power-of-two fp8 scales: DeepSeek-AI, "DeepSeek-V3
  Technical Report", arXiv:2412.19437, 2024, §3.3.3 (the scaling factors of fp8 activations as integral powers of 2);
  Rouhani et al. (above), §3 and Algorithm 1 (power-of-two scales shared by groups of 32 values: F8's V channels).  The
  output projections in f16: Llama Team, AI @ Meta, "The Llama 3 Herd of Models", arXiv:2407.21783, 2024, §6.2 (the
  precision chosen by a component's role: fp8 only in the feed-forward layers, the self-attention layers' parameters not
  quantized).

## J3 decode / verify

Batches of 1-16 tokens apart from attention.

| part | what | switch | type |
|---|---|---|---|
| mmvk | Q4_K / Q5_K / Q6_K matmuls of 1-8 tokens keep the activations in registers across several weight rows; the matmuls that read one input go out as one launch.  On when llama-server verifies drafts | `FRANK_MMVK` (2) | speed/diff |
| Gated DeltaNet verification | each slot keeps two recurrent-state planes (after its last batch, and the one that batch started from) and per-token replay factors instead of one full state per draft token; a partly rejected draft is replayed.  About 1 GiB less VRAM with 2 slots on Qwen3.8-27B.  After a verify batch the state after it is not written: the next batch always starts from the state the batch started from and replays the accepted tokens, also when all were accepted (0.1-0.5% less time per step).  The verify kernel reads and writes the state with non-temporal hints | `FRANK_RS_ANCHOR` (on), `FRANK_RS_REPLAY_ALL` (on), `FRANK_GDN_NT` (on) | memory, speed |
| fused decode | the draft model's convolution, the Gated DeltaNet convolution state, the norm before quantization in registers; one empty block against the gfx1201 grid stall on mmvk.  The output norm with the SiLU gate is rdna-boosts' since r353 (sent from here; `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU`) | switches in [SWITCHES.md](SWITCHES.md); `GGML_CUDA_GRID_STALL_PAD` (rdna-boosts') | speed |
| int2 draft head | the DFlash2 draft model's output layer is scored from a resident int2 copy of its weight (asymmetric, 64 values per group); the best 512 of each column are recomputed through mmvk, bit for bit as the plain path computes them, and the rest are set to -inf.  When the true top 16 is not among the 512, only that draft changes, never the verified output.  About 2% less time per step and 0.4 GiB more VRAM (outside `--fit`; if it cannot be allocated the plain path stays).  On with one RDNA4 GPU | `FRANK_HEAD_INT2` (on) | speed |

- Code: rdna-boosts (MIT): the grid-stall padding helper.
- References: A. Gray, C. Gottbrath, R. Olson,
  S. Prasanna, "Deploying Deep Neural Networks with NVIDIA TensorRT", NVIDIA Technical Blog, 2017-04-02 (horizontal layer fusion:
  layers that read one input combined).  J. Wang, D. Paliotta, A. May, A. M. Rush, T. Dao, "The Mamba in the Llama:
  Distilling and Accelerating Hybrid Models", NeurIPS 2024, arXiv:2408.15237, and Y. Wu, Y. Dukler, M. Trager, A. Achille,
  W. Xia, S. Soatto, "Snakes and Ladders: Accelerating SSM Inference with Speculative Decoding", PMLR 262, 2024 (the state
  from before the drafts kept and the accepted tokens replayed, instead of one state per draft token).  Built on: Yang et
  al., Gated Delta Networks (above).  K. Shim, M. Lee, I. Choi, Y. Boo, W. Sung, "SVD-Softmax: Fast Softmax Approximation
  on Large Vocabulary Neural Networks", NeurIPS 2017, and Y. Samaga B L, V. Yerram, C. You, S. Bhojanapalli, S. Kumar,
  P. Jain, P. Netrapalli, "HiRE: High Recall Approximate Top-k Estimation for Efficient LLM Inference", arXiv:2402.09360,
  2024 (a cheap approximation of the output layer picks the candidates, which are then computed exactly).

## J4 sampling

| part | what | switch | type |
|---|---|---|---|
| reasoning guard | an end of turn sampled inside the reasoning becomes the reasoning's end tag (`</think>`), so the model writes its answer instead of leaving it in the reasoning with an empty reply.  Nothing before that point changes; when it fires the model usually writes the answer again (leave room in `max_tokens`) | `FRANK_REASONING_GUARD` (on) | guard |

- Reference: N. Muennighoff et al., "s1: Simple test-time scaling", arXiv:2501.19393, 2025 (budget forcing ends the reasoning
  by appending its end delimiter).

## J5 speculative decoding

Where drafts come from, how wide they are, and how they are accepted.

| part | what | switch | type |
|---|---|---|---|
| adaptive width | DFlash2 drafts each sequence's own width (from 3 to `--spec-draft-n-max`) from running acceptance estimates under a fixed cost model; nothing is timed while running | `FRANK_DFLASH_ADAPTIVE` (on); `_COST`, `_JUMP`, `_JUMP_ROWS` (with a `q8_0r` K cache 0.018 / 0.04 / 6, otherwise 0.035 / 0.08 / 5) | speed |
| one draft length | slots verifying together send drafts of one length (the adaptive width widens, a copying n-gram draft is cut), so a hybrid model's verify batch is not split | `FRANK_SPEC_ALIGN` (on), `FRANK_SPEC_ALIGN_CUT` (on) | speed |
| n-gram from the prompt | with `ngram-mod`, each sequence looks up n-gram drafts only in its current prompt, in a table with a key check; any draft the table gives is used and goes ahead of DFlash2 (ngram-mod's n-min defaults to 1 here instead of the release's 48).  For rewriting whole files | `FRANK_SPEC_NGRAM_PROMPT_ONLY` (on with `ngram-mod`); `--spec-ngram-mod-n-min` still sets the floor | speed/trade-off |
| lenience | sampled drafts accepted with probability min(1, p / (l·q)) instead of min(1, p / q); only at temperature > 0 (see [Lossy parts](#lossy-parts)) | `FRANK_SPEC_LENIENCE` or the request field `frank_spec_lenience` (off; e.g. `0.8663`) | sampling |

- Code: llama.cpp's `ngram-mod` (MIT) is the copy mechanism the prompt-only table builds on.
- References: Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023,
  arXiv:2211.17192 (the lenience follows its relaxed acceptance; §3.5 chooses the draft length from cost and acceptance).
  X. Liu, C. Daniel, L. Hu, W. Kwon, Z. Li, X. Mo, A. Cheung, Z. Deng, I. Stoica, H. Zhang, "Optimizing Speculative Decoding
  for Serving Large Language Models Using Goodput" (SmartSpec; later revised as TurboSpec), arXiv:2406.14066, 2024 (the
  draft length chosen from estimated acceptance and a model of the step's cost, for the most tokens per unit of cost; the
  adaptive width follows it).  Z. He et al., "REST: Retrieval-Based Speculative Decoding", NAACL 2024, arXiv:2311.08252, and Y. Hu et al.,
  "SAM Decoding: Speculative Decoding via Suffix Automaton", arXiv:2411.10666, 2024 (retrieved drafts; SAM chooses between
  retrieved and model drafts by the match length).  H. Qian et al., "BASS: Batched Attention-optimized Speculative
  Sampling", Findings of ACL 2024, arXiv:2404.15778 (one draft length across a batch, so it keeps a regular shape).  Built on: C. Chen et al., "Accelerating Large Language Model Decoding
  with Speculative Sampling", arXiv:2302.01318, 2023; J. Chen, Y. Liang, Z. Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026, arXiv:2602.06036
  (the draft model; DFlash2 is z-lab's later release).

## J6 step issue

How a step goes to the GPU.

| part | what | switch | type |
|---|---|---|---|
| GPU-resident loop | with drafts, a step's sampling, draft acceptance (with J4's guard, J5's lenience and J3's verification), the next input and the KQ mask are made on the device; server requests and two sequences in one step run in it.  The host checks every step it reads back (the position moved, the last token, the recurrent state's cell; for a greedy sequence also that the emitted tokens are the target's); on a mismatch the loop stops and the host path takes over | `FRANK_STEP_LOOP` (on with one GPU, a ROCm one, on Linux and Windows; off with several GPUs), `FRANK_STEP_LOOP_SEQ2` (on), `FRANK_DEVICE_INPUT_MAPPED` (on), `FRANK_STEP_CHECK` (on); rdna-boosts' `LLAMA_DEVICE_INPUT` follows the loop when unset | speed, guard |

The loop gives greedy output identical to the host path; at temperature > 0 the draft model's placement can change a token,
as on the host path.  A tied token embedding (a model without an output weight) is not mapped: it stays in VRAM as the
output layer, one copy for both.  Graphs of 9-16 tokens on the host path are rdna-boosts' since r326
(`GGML_CUDA_GRAPH_MAX_TOKENS`).  The step slab's per-sequence ring of sliding-window cells is made only for the draft
model the loop drives; a target model with sliding-window layers keeps llama.cpp's own layout.  rdna-boosts' K/V-rows
attention (r359, opt-in) stays off while the loop or its slab is switched on: they use the dense mask.

- References: R. Hasan,
  N. Korobov, R. Kobus, H. Huang, D. Mehra, "NVIDIA TensorRT-LLM Now Supports Recurrent Drafting for Optimizing LLM
  Inference", NVIDIA Technical Blog, 2024-12-18 (drafting and validation inside the engine instead of the host
  runtime).  Built on: Alan Gray, "Optimizing
  llama.cpp AI Inference with CUDA Graphs", NVIDIA Technical Blog, 2024-08-07 (llama.cpp's graphs, which the loop's steps
  replay).

## J7 state between requests

| part | what | switch | type |
|---|---|---|---|
| state copy in bands | a state above 256 MiB (the prompt cache saving or restoring a long slot) goes through the pinned stage one band at a time.  One device only | `FRANK_STATE_ASYNC_BANDS` (on with one device) | speed |
| checkpoints in the background | a context checkpoint is saved asynchronously: the save returns once the device copies are queued, and the host copy (about 11 ms for Qwen3.8-27B's 150 MiB recurrent state, plus the draft's) runs on a worker thread beside the decode instead of before the request's first token | `FRANK_SERVER_CKPT_ASYNC` (on) | speed |
| prompt cache in the background | when a task takes over a slot while another slot is working, the slot's previous prompt is copied to the prompt cache on a stream of its own, so the other slots go on decoding (switching slots at 30K: their longest stall 222-298 → 128-208 ms); the new task waits for the copy.  The saved data is the same; the new prompt may start an iteration later | `FRANK_SERVER_CACHE_ASYNC` (on) | speed |
| pinned pools | prompt-cache states and checkpoints live in pinned host memory sized at startup from the tensor shapes, copied straight to and from the device | `FRANK_SERVER_CACHE_PINNED`, `FRANK_SERVER_CKPT_PINNED` | speed |
| checkpoints on the batch grid | checkpoints are cut at batch starts, so a shared prefix is split the same way whatever follows it: a request continued from the cache gives the same output as one run from scratch | `FRANK_SERVER_CKPT_ALIGN` (on) | fix |

- References: R. Pan et al., "Marconi: Prefix Caching for the Era of Hybrid LLMs", MLSys 2025, arXiv:2411.19379 (recurrent
  states kept at chosen prefix positions, at a chunk boundary for chunked models); the SGLang Team, "Towards Deterministic
  Inference in SGLang and Reproducible RL Training", LMSYS blog, 2025-09-22 (updated 2025-09-24) (chunk cuts aligned to a fixed grid, so a sequence is
  split the same way whatever is batched with it).  B. Gao et al., "Cost-Efficient Large Language Model Serving for
  Multi-turn Conversations with CachedAttention", USENIX ATC 2024, arXiv:2403.19708 (cache saved to host memory
  asynchronously, overlapped with computation).

## Thanks

To stew675 and the rdna-boosts contributors; to Deadcode and the radiance contributors; to the llama.cpp / ggml community;
to z-lab for the DFlash2 draft models; and to the authors of the papers above.

## Change log

**2026-11-26 (later): docs.**  The note "do not set `FRANK_MMVK_EXT=3`" is withdrawn.  The wrong token it described was a
measurement artefact: the server reports a token's probabilities only once its UTF-8 character is complete, so comparing
two runs' probability lists entry by entry paired different positions after they parted at a lead byte.  Compared token
by token, those runs part at ordinary near ties (rank 2), as with the defaults; mmvk's grouped launches are bit-identical to
single launches.

**2026-11-26 (later): J7 v2.1.**  The startup warning for old switch names no longer lists
`GGML_CUDA_FUSE_GDN_NORM_GATE_SILU`: since J3 v5.3 it is rdna-boosts' own switch and works, but J7 v2 said it had no
effect.  `FRANK_FUSE_GDN_NORM_GATE_SILU` (J3's name before v5.3) now warns that it is no longer read.  Startup log only:
the other files and the computation are unchanged.

**2026-11-26: J1 v5, J2 v15 (both files), J3 v5.3, J6 v10.1, J7 v2** on rdna-boosts `v18-919794500-r359` (the same
llama.cpp `919794500`); J4 v1 and J5 v3.1 move as they are.  J2 v14 and J6 v10, and the moves to r353 and r358, were not
published here; this entry covers them.  **The default output changes** (the power-of-two scales, J2); everything else is
opt-in, gives the same output, or acts only on models other than Qwen3.8-27B.
- J1 v5: with `-ctk q8_0r`, only the layers the `q8_0r` kernel takes get the `q8_0r` cache (head 256, at most 6 query
  heads per KV head, no attention sinks); the others keep `q8_0` and run on its kernels instead of on f16 copies of K and
  V.  Qwen3.8-27B is unchanged; Qwen3.6-35B-A3B: decode +2.6-2.8 %, the same text.  rdna-boosts r359 makes its K/V-rows
  attention opt-in (`LLAMA_KV_ROWS_DISABLE=0`); the `q8_0r` kernel, F8 and the 9-16 row halves step aside when it is on.
- J2 v14 (part b): K1's and M4's gfx1201 algorithm lists find hipBLASLt's solutions by kernel name instead of by index.
  ROCm 10.2's hipBLASLt renumbers every solution, and with the old lists K1 fell back to hipBLASLt's first choice (prefill
  -11 to -12 %); on ROCm 10.0 the choices and the output are the same.  The log says how many of each list's kernels the
  hipBLASLt build has.
- J2 v15 (part b): the fp8 scales of K1 fp8, M4 and F8 are powers of two (`FRANK_FP8_POW2`, on; `0` = the scales before):
  slightly closer to a clean reference, the same speed ([Lossy parts](#lossy-parts)).  New and off by default: the
  output projections in f16 (`FRANK_KQ_BLAS_OUT16`), and `FRANK_KQ_BLAS_F16_SET` for measurements.
- J3 v5.3: the Gated DeltaNet output norm fused with its SiLU gate is rdna-boosts' since r353 (sent from here); J3's own
  copy is gone.  rdna-boosts' switch is `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU`.
- J6 v10: the step slab's per-sequence ring of sliding-window cells is made only for the draft model the GPU loop drives.
  A target model with sliding-window layers gets llama.cpp's own layout again: before, its second and later slots read
  the wrong cells (a model with sliding-window layers and `--parallel 2`).  Qwen3.8-27B with DFlash2: the same text and draft acceptance.
  J6 v10.1: rdna-boosts' K/V-rows attention stays off while the loop or its slab is switched on (together they stopped
  the server).
- J7 v2: a slot's previous prompt is copied to the prompt cache on a stream of its own while other slots are working
  (`FRANK_SERVER_CACHE_ASYNC`, on), so they go on decoding: switching slots at 30K, the other slot's longest stall went
  from 222-298 to 128-208 ms.
Against the groups before (r353, r358): with `FRANK_FP8_POW2=0`, the same text on 22 server requests (UD-Q4_K_XL, 2 slots);
with the defaults, the time per decode step the same at 8K and 0.6 % less at 110K, and qgate passes for the power-of-two
scales ([Lossy parts](#lossy-parts)).  With every default-off switch on together (`FRANK_KQ_BLAS=fp8b`,
`FRANK_KQ_BLAS_OUT16`, `FRANK_KQ_BLAS_PRECONV`, `FRANK_MMVK_EXT=3`), and with K1's other modes (`f16`, `TAIL=mmq`,
`FRANK_MXFP4_ALGOS=timed`), the same server tests gave no aborts or errors and repeatable output; that set's KLD against
the defaults is 0.0025 / 0.0009 (zh / en, after a 30K-token prefix).

**2026-11-24: J2 v13, a fix** (part a as in v10.1; v11 and v12 were not published here).
- The norm lock is removed with its switches (`FRANK_KQ_BLAS_FIXED` and its tuning; the server warns when one is set):
  every K1 batch takes its own scales and KEEP channels again, as `FRANK_KQ_BLAS_FIXED=0` did.  Why:
  [lossy parts](#lossy-parts).  The default output is the same as v10.1 with `FRANK_KQ_BLAS_FIXED=0` (22 server requests
  on UD-Q4_K_XL with 2 slots); prefill gives back the lock's 1.2-2.1 %.  The rest of this entry is opt-in or gives the
  same output.
- The own f16 GEMM (`FRANK_KQ_BLAS_OWN`, and `fp8b`) takes Q8_0 weights too: eight 32-value blocks a 256-value super-block,
  decoded into the same halfs as K1's Q8_0 conversion.  With K1 `fp8` it replaces MMQ for Q8_0 batches below K1's threshold
  (0.91-1.02x its time); the dense weights of models quantized with Q8_0 throughout now also run in f16 there.
- `FRANK_KQ_BLAS=fp8b`: K1 fp8 with the tail batch (`FRANK_KQ_BLAS_TAIL=own`) and the own f16 GEMM for the batches below
  K1's threshold (`FRANK_KQ_BLAS_OWN=1`) on unless those are set: the recoveries whose share of the fp8 error taken back
  (against K1 `f16`) is larger than their share of the time `f16` adds.  `fp8` and the default are unchanged.
- The batch amax of the f16 conversion (K1 `f16`, the own f16 GEMM) comes from the channel maxima the producing kernels
  fold in, or else from one float4 pass; the conversion writes eight values a thread.  The same output; per 2048-token
  batch 2.1 % less time with the own f16 GEMM, 2.7 % with K1 `f16`.

**2026-11-22: docs.**  J2's references add COAT (Xi et al., ICLR 2025) for K1's channel amax folded into the kernel
that writes the activation, with ZeroQuant's register-side scale.  No patch changes.

**2026-11-22: pinned at rdna-boosts `v17-919794500-r331`.**  No patch changes.  r331 restores block 00's allocator
rules, which rdna-boosts' r328 re-base had dropped: a GLU may take its gate's memory (`GGML_GLU_INPLACE`), view sources
nothing reads are released, and a graph with another layout is not given the previous one's addresses.  It also makes
two block-10 fixes that do nothing on HIP.  Against r329: 0.19 GiB less VRAM, the same tokens and draft acceptance,
prefill and decode within 0.5 % ([README](README.md#performance)).

**2026-11-22: J1 v4.3, J2 v10.1 (both files), J3 v5.2, J5 v3.1, J6 v9.2, J7 v1.3** on rdna-boosts `v17-919794500-r329`
(rdna-boosts' blocks moved to llama.cpp master `919794500`); J4 v1 moves as it is.  Only what the new base needed:
- J1: `q8_0r` rows written by `SET_ROWS` also from a BF16 source (llama.cpp's new source type); warp shuffles through
  ggml's `ggml_cuda_shfl*` helpers, since llama.cpp's HIP build now uses the real `_sync` builtins with their 64-bit mask.
  The same in J2 part a (the own Gated DeltaNet kernel) and J6 (the loop's kernels).
- J3: upstream's op count; mmvk's early return at the new place of upstream's fused-matvec check; the DFlash2 draft's
  conv in `dflash2.cpp`, where llama.cpp now builds DFlash2 apart from its other DFlash variants.
- J5: the adaptive width and its state live in llama.cpp's new DFlash2 class.  J6: the loop looks for the DFlash2 type.
- J7: the prompt cache's pinned pool follows the prompt cache into `server-slot-sched`.
- J2 part b: the same changed lines as v10.
Against r327: the same tokens and log-probs, the same time per step and draft acceptance
([README](README.md#performance)).

**2026-11-22: docs.**  The J7 table no longer says that a cleared slot's checkpoints move into the prompt cache instead
of being copied: llama.cpp has shared them with the prompt cache since its reference-counted checkpoints, and J7 has
not carried that part since then.  No patch changes.

**2026-11-22: J2 v10.**  Both J2 files change; nothing in J2 comes from libr4d or radiance any more, so the radiance
license files and the NOTICE are gone and the patches are MIT.
- Part a: Gated DeltaNet prefill runs on an own chunked kernel (`FRANK_GDN_OWN`, on; `0` = rdna-boosts' chunked kernel as
  released, with f32 rows from the prefill conv).  It replaces J2's earlier changes to rdna-boosts' chunked kernel (the
  scan reading the next chunk's gate early, and its bf16 reading): that file is rdna-boosts' again.  Against J2 v9:
  prefill -0.3 / -0.2 / -0.2 % at 8K / 32K / 110K, the same decode and draft acceptance; KLD in the same range
  ([README](README.md#performance)).
- Part b: the own f16 GEMM is rewritten on ggml's `mma.cuh` tiles, with the same output bit for bit.
  `FRANK_KQ_BLAS_OWN_SHAPE` (diagnostics) replaces `FRANK_KQ_BLAS_OWN_BMF` and `_OWN_TN`.

**2026-11-21: J2 v9.**  Licenses only, no change to the code: the parts of J2 that derive from radiance's libr4d
(part a: the Gated DeltaNet chunked prefill kernels rdna-boosts ported from libr4d; part b: the own f16 GEMM's tile and
fragment order) state radiance's Apache-2.0 license and what was changed in their file headers, and radiance's LICENSE and
NOTICE were in `third_party/radiance/` (removed in J2 v10).

**2026-11-21: J1 v4.2, J2 v8 (two files), J3 v5.1, J6 v9.1, J7 v1.2** on rdna-boosts `v17-4f9dfa0e3-r327` (the same
llama.cpp `4f9dfa0e3`); J4 v1 and J5 v3 move as they are.  J2's new part b is applied last, after J7.
- Moved to rdna-boosts, which now reads its own switches: the KQ block table reuse and its census (J1; r327, opt-in
  there, on by default here: `GGML_CUDA_FA_KQ_BLOCKS_REUSE=0` turns it off; the same output), graphs of up to 16 tokens
  (J6; r326, `GGML_CUDA_GRAPH_MAX_TOKENS`, which `FRANK_ALL=0` no longer lowers to 8) and the GLU in place (J2; r323,
  `GGML_GLU_INPLACE`).  J3 v5.1: the per-evaluation call it shared with J1's reuse is now rdna-boosts'.  J7 v1.2: a warning
  when `FRANK_FA_KQ_BLOCKS_REUSE`, `FRANK_FA_KQ_BLOCKS_CENSUS` or `FRANK_GRAPH_MAX_TOKENS` is set.
- J2 part b, 6d on by default (`FRANK_KQ_BLAS_FIXED`, 2): K1 fp8's scales and KEEP channels are fixed at the first decode
  step only for the normalized residual streams; every other K1 input keeps per-batch scales.  Fixing every input had made
  some kinds of content much worse; this keeps the KLD of all 22 measured kinds within the floor ([Lossy parts](#lossy-parts)).
  Prefill +1.6 / +2.1 / +1.2 % at 8K / 32K / 110K.  The previous mode that fixed every input, and the fp8 writes of the
  GLU and Gated DeltaNet outputs it needed, are gone.
- J2 part b, removed: 6g (`FRANK_KQ_BLAS_GLU16`, 16-bit gate / up products for the GLU; slower once 6d fixes only the
  normalized streams) and the Gated DeltaNet output norm and SiLU gate kernel that wrote fp8 (`FRANK_GDN_NORM_SILU_FP8`).
  Their switches only warn.
- J2 part b, new and off by default: the own WMMA f16 GEMM (`FRANK_KQ_BLAS_OWN`) and the tail batch
  (`FRANK_KQ_BLAS_TAIL`): a long prompt's last 256 tokens through it, which takes back 26 % of the fp8 error the
  generation path sees for 87.7 ms per prompt (8K prompts).

**2026-11-18: J6 v9.**  A model without an output weight uses its token embedding as the output layer.  With the
mapped token embedding (`FRANK_DEVICE_INPUT_MAPPED`, on with the loop) the two were placed apart: a second copy of the
table, read from a file range the load had already released, and the load crashed (Gemma 4).  Such an embedding now stays
in VRAM as the output layer, one copy for both.  Models with an output weight are unchanged (the same output).

**2026-11-17: J3 v5, J6 v8.**  J3: after a verify batch that keeps replay factors, the recurrent state after the
batch is no longer written; the next batch always starts from the state the batch started from and replays the accepted
tokens, also when all were accepted (`FRANK_RS_REPLAY_ALL`, on).  The same output, 0.1-0.5% less time per step.  J6: the
loop's device-written inputs follow the same rule, and the host checks every loop step it reads back; on a mismatch it
logs it, the loop issues no more steps and the host path takes over (`FRANK_STEP_CHECK`, on).  The same output and speed.

**2026-11-17: J1 v4.1, J2 v7.1, J3 v4.1, J6 v7.1** on rdna-boosts `v16-4f9dfa0e3-r295` (llama.cpp `4f9dfa0e3`, the 1.0
line); J4 v1, J5 v3 and J7 v1 move as they are.  Only what the new base needed: J1 includes `<mutex>` itself; J2's fp8
prefill path takes the RMS norm only when its input and output are F32 (upstream's norm can now write BF16); J3 follows
upstream's new `quantize_row_q8_1_cuda` arguments, its always-fused Gated DeltaNet (the anchored verify marks no fused
nodes) and its op count; J6 grows the KV regions with upstream's new count of used cells.  Against r277: the same tokens
and log-probs, the same time per step.

**2026-11-16: J2 v7.**  Improves reliability on Windows.

**2026-11-15: J6 v7.**  With two sequences in one step, the loop read each sequence's recurrent state from the cell its
last host batch read, not the one it wrote.  When `find_slot()` swapped the two sequences' cells (two requests
overlapping, as with `--parallel 2`), each sequence read the other's state and both answers turned to garbage; the loop
now reads the cell written.  `FRANK_STEP_LOOP_SEQ2=0` no longer aborts.  The opt-in step issued ahead
(`FRANK_STEP_AHEAD`, with `FRANK_RS_WHOLE_BATCH`) is removed: one step at a time was as fast or faster.  Otherwise the
same output as J6 v6.

**2026-11-14: J1 v4, J2 v6, J3 v4, J4 v1, J5 v3, J6 v6, J7 v1** (the first release of the groups; the versions count the
changes made before it).  The patches of this repo's earlier form (llama-cpp-overdoing, 0001-0015) and a new GPU-resident
speculative pipeline, merged into seven groups and moved to r277 (llama.cpp `a691b2345`).  Where the old patches went:

| old | new |
|---|---|
| 0001 K1, 0003 M4, 0008 GLU in place | J2 |
| 0002 F8, 0009 9-16 row verification | J1 |
| 0004 reasoning guard | J4 |
| 0005 state bands | J7 |
| 0006 mmvk, 0013 Gated DeltaNet verification | J3 |
| 0007 lenience, 0012 adaptive width, 0014 n-gram from the prompt, 0015 one draft length | J5 |

The old change log: [PATCHES.md at edef875](https://github.com/overdoingism/llama-Frankenstein/blob/edef875/PATCHES.md#change-log).
