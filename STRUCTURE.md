# Structure

What each group in [`patches/`](patches) contains, its switches (default in brackets), where its ideas and code come
from, the lossy parts with their KLD, and the change log.  The short table is in the [README](README.md#structure).
Written with Claude (AI).

The seven groups follow the order of inference, and each is one patch: `J<n>-a69-<part>-v<version>.patch`, applied in
number order on rdna-boosts `v17-a691b2345-r277`.  A group alone is not guaranteed to build; every part has a switch.

Types: `speed` = faster, the same output; `memory` = less VRAM, the same output; `speed/diff` = faster, output not
bit-identical to the stock code but within the run-to-run floor; `speed/lossy` = faster, error above the floor;
`speed/trade-off` = faster on some workloads, slower on others; `sampling` = changes the sampling rule, not the logits;
`guard` = changes behavior only in a failure case; `fix` = corrects a fault.

## Lossy parts

KLD as multiples of the run-to-run floor (zh / en): Qwen3.8-27B official UD-Q5_K_M, 400 teacher-forced tokens after a
48K-token prefix, against the same build with every switch off.  Generation = the largest of the decode / verify widths 1, 6
and 16; prompt processing = one 400-token batch.  Floor: generation 0.0017 / 0.0005, prompt processing 0.0057 / 0.0011.
Speed: the part against the defaults without it (prefill t/s gained; decode time per step saved), 8K / 32K / 110K prompts.
Measured 2026-11-09 on r235 with a 48K prefix; the defaults on r269 at 8K and 32K: the README's
[Performance](README.md#performance).

| part | group | KLD, generation | KLD, prompt processing | speed |
|---|---|---|---|---|
| K1 (fp8 prefill matmuls) | J2 | 1.42× / 1.72× | 4.01× / 4.35× | prefill +115 / +107 / +83 % |
| F8 (fp8 prefill attention) | J1 | 1.05× / 1.18× | 1.19× / 1.52× | prefill +12 / +29 / +65 % |
| fixed KEEP + 16-bit GLU input (off by default) | J2 | within K1's | within K1's | prefill +1 / +7 / +6 % |
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
| shared KQ block table | the attention layers of one graph share the mask's block table | `FRANK_FA_KQ_BLOCKS_REUSE` (on) | speed |

- Code: rdna-boosts (MIT): the decode band's group walk and barrier, the KQ block table.
- References: J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao, "FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-precision", NeurIPS 2024, arXiv:2407.08608, §3.3, and Colfax Research, "Delivering
  1 PFLOP/s of Performance with FP8 FlashAttention-2", 2024 (both attention matmuls in fp8, the probabilities rounded to
  fp8 between them: F8).  T. Zadouri, M. Hoehnerbach, J. Shah, T. Liu, V. Thakkar, T. Dao, "FlashAttention-4: Algorithm
  and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling", arXiv:2603.05451, 2026, §3.1.4, after M. Milakov,
  N. Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867, 2018 (the `q8_0r` kernel rescales its
  output only when a row's running maximum has grown by more than 8 in log2, decided per row so a row never depends on the
  others in its tile).  The Triton tutorial "Fused Attention" (`06-fused-attention.py`, v2.1.0, 2023) (the softmax scale
  and log2 e folded into Q once, for exp2).  vLLM's PagedAttention V2 kernel (v0.2.2, 2023) (each split's normalized
  output kept in 16 bits, its maximum and sum in f32).  Y. J. Kim, R. Henry, R. Fahim, H. Hassan Awadalla, "Who Says
  Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production", SustaiNLP 2022, arXiv:2211.10017,
  §3.4.1-3.4.2 (int8 codes turned into f16 by placing them in the mantissa of 1024 and subtracting 1152).  S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, J. Hensman,
  "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", NeurIPS 2024, arXiv:2404.00456 (a Hadamard rotation of the keys
  before the KV cache is quantized; llama.cpp's KV rotation follows it, and the `q8_0r` cache stores and reads the rotated
  rows).  P. Micikevicius et al., "FP8 Formats for Deep Learning", arXiv:2209.05433, 2022 (e4m3).  J. T. Barron,
  D. Gnanapragasam, M. Tschannen, "A Fix for Softmax and Cross-Entropy", arXiv:2610.31642, 2026 (F8's row sum).
  H. He et al. (Thinking Machines Lab), "Defeating Nondeterminism in LLM Inference", 2025, and the batch-invariance
  documents of vLLM and SGLang (fixed splits).  Built on: T. Dao et al., "FlashAttention", arXiv:2205.14135; T. Dao,
  "FlashAttention-2", arXiv:2307.08691; T. Dao, D. Haziza, F. Massa, G. Sizov, "Flash-Decoding for long-context
  inference", 2023 (the `q8_0r` kernel's KV splits).

## J2 prefill

The large-batch path apart from attention.

| part | what | switch | type |
|---|---|---|---|
| K1 | prefill matmuls of K-quant weights through hipBLASLt in fp8 (or f16), on batches of 384 tokens or more (f16: 640; 1024 on weights under 20M elements); the largest activation channels of each batch get a second fp8 term for their rounding errors in the same GEMM; the residual ADD goes into the GEMM (in f16 too, unless the algorithms are timed) | `FRANK_KQ_BLAS` (`fp8` when every GPU is a gfx1201), `_FP8_KEEP` (64), `_ADD` (on); `FRANK_DRAFT_UB` (512) | speed/lossy |
| fixed KEEP, 16-bit GLU input | K1's outlier channels and fp8 scales are fixed from each input's first batches, so producers write fp8 activations directly; the gate / up GEMM writes 16-bit output for the GLU.  Off by default: faster, but the added error showed in our evaluations | `FRANK_KQ_BLAS_FIXED` (0; 1 = after `_N` (4) batches, 2 = at the first decode step), `FRANK_KQ_BLAS_GLU16` (off; `f16`) | speed/lossy |
| prompt tail | a batch of 384 tokens or more that ends its prompts with output tokens hands those outputs to a decode-sized batch, so the logits a prompt ends with do not come from K1's prefill matmuls.  Batches with an output for every token (perplexity), embeddings and non-causal models are unchanged | `FRANK_PP_TAIL` (on), `_MIN` (384) | fix |
| M4 | MXFP4 weights converted exactly to fp8 for K1's GEMM, from 64 tokens | `FRANK_MXFP4_DIRECT` (`blaslt`) | speed/lossy |
| GLU in place | the graph allocator may give a GLU its gate's memory when nothing else reads the gate: 150 MiB less compute buffer at `-ub 2048` on Qwen3.8-27B | `FRANK_GLU_INPLACE` (on) | memory |
| fused prefill | the attention output gate reads Q in place; Q / K norm, rope and the FWHT in one kernel; one-step fp8 conversion of Q8_0 / IQ4_NL weights; Gated DeltaNet: the convolution with the q / k norm, the output norm with the SiLU gate, chunked rows in bf16, the scan reading the next chunk's gate early | switches in [SWITCHES.md](SWITCHES.md) | speed |

- Code: rdna-boosts (MIT): the chunked Gated DeltaNet kernel the scan is adapted from.
- References: X. Wu, Z. Yao, Y. He, "ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using
  Floating-Point Formats", arXiv:2307.09782, 2023, §3 (4-bit weights cast to fp8 for an fp8 GEMM with fp8 activations: K1,
  M4; with power-of-two block scales, as MXFP4 has, the cast is exact: M4).  Ashkboos et al., QuaRot (J1; the FWHT the fused Q / K kernel
  applies before the KV store).  Micikevicius et al. 2022 (above); T. Dettmers et al., "LLM.int8()", arXiv:2208.07339,
  2022; S. Ashkboos et al.,
  "QUIK", arXiv:2310.09259, 2023; Y. Zhao et al., "Atom", MLSys 2024, arXiv:2310.19102 (outlier channels handled apart at
  higher precision; with fixed KEEP, chosen once from calibration).  Fixed KEEP: R. Krishnamoorthi, "Quantizing deep
  convolutional networks for efficient inference: A whitepaper", arXiv:1806.08342, 2018 (activation ranges from
  calibration); NVIDIA Transformer Engine's FP8 delayed scaling (scales from earlier batches); Z. Yao, R. Yazdani Aminabadi,
  M. Zhang, X. Wu, C. Li, Y. He, "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale
  Transformers", NeurIPS 2022, arXiv:2206.01861, §4.3 and Appendix D, and Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao,
  C. Gan, S. Han, "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving", MLSys 2025,
  arXiv:2405.04532, §5.1 (the activation quantization fused into the kernel that produces the activation).  The prefill / decode split:
  A. Panferov et al., "Disaggregated Quantization: Specializing LLM Prefill and Decode", arXiv:2609.26333, 2026 (prefill on
  compute-native low-precision weights, decode on compact weight-only ones).  The residual ADD: J. J. Dongarra, J. Du Croz,
  S. Hammarling, I. S. Duff, "A Set of Level 3 Basic Linear Algebra Subprograms", ACM TOMS 16(1), 1990 (the GEMM's βC term),
  and J. Zeng et al., "Boosting Distributed Training Performance of the Unpadded BERT Model", arXiv:2208.08124, 2022 (an
  addition at a residual taken into the GEMM through β).  16-bit GLU input: P. Micikevicius et al., "Mixed Precision
  Training", ICLR 2018, arXiv:1710.03740 (fp32 accumulation, 16-bit output).  GLU in place: T. Chen, B. Xu, C. Zhang,
  C. Guestrin, "Training Deep Nets with Sublinear Memory Cost", arXiv:1604.06174, 2016 (an operation writes into its input's
  memory when it is that input's last reader, the rule of ggml's graph allocator).  Built on: S. Yang, B. Wang, Y. Zhang,
  Y. Shen, Y. Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024,
  arXiv:2406.06484, §3 (the chunkwise form the Gated DeltaNet scan computes); S. Yang, J. Kautz, A. Hatamizadeh, "Gated
  Delta Networks", ICLR 2025, arXiv:2412.06464; for M4, the OCP Microscaling Formats (MX)
  Specification v1.0, 2023, and B. D. Rouhani et al., "Microscaling Data Formats for Deep Learning", arXiv:2310.10537.

## J3 decode / verify

Batches of 1-16 tokens apart from attention.

| part | what | switch | type |
|---|---|---|---|
| mmvk | Q4_K / Q5_K / Q6_K matmuls of 1-8 tokens keep the activations in registers across several weight rows; the matmuls that read one input go out as one launch.  On when llama-server verifies drafts | `FRANK_MMVK` (2) | speed/diff |
| Gated DeltaNet verification | each slot keeps two recurrent-state planes (after its last batch, and the one that batch started from) and per-token replay factors instead of one full state per draft token; a partly rejected draft is replayed.  About 1 GiB less VRAM with 2 slots on Qwen3.8-27B.  The verify kernel reads and writes the state with non-temporal hints | `FRANK_RS_ANCHOR` (on), `FRANK_GDN_NT` (on) | memory, speed |
| fused decode | the draft model's convolution, the Gated DeltaNet convolution state, the norm before quantization in registers, the output norm with the SiLU gate; one empty block against the gfx1201 grid stall on mmvk | switches in [SWITCHES.md](SWITCHES.md); `GGML_CUDA_GRID_STALL_PAD` (rdna-boosts') | speed |
| int2 draft head | the DFlash2 draft model's output layer is scored from a resident int2 copy of its weight (asymmetric, 64 values per group); the best 512 of each column are recomputed through mmvk, bit for bit as the plain path computes them, and the rest are set to -inf.  When the true top 16 is not among the 512, only that draft changes, never the verified output.  About 2% less time per step and 0.4 GiB more VRAM (outside `--fit`; if it cannot be allocated the plain path stays).  On with one RDNA4 GPU | `FRANK_HEAD_INT2` (on) | speed |

- Code: rdna-boosts (MIT): the grid-stall padding helper.
- References: A. Kerr, D. Merrill, J. Demouth, J. Tran, "CUTLASS: Fast Linear Algebra in CUDA C++", NVIDIA Technical Blog,
  2017 (mmvk's register tile: values held in registers reused across rows and columns); A. Gray, C. Gottbrath, R. Olson,
  S. Prasanna, "Deploying Deep Neural Networks with NVIDIA TensorRT", NVIDIA Technical Blog, 2017 (horizontal layer fusion:
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
  with Speculative Sampling", arXiv:2302.01318, 2023; A. Saxena, "Prompt Lookup Decoding", GitHub, 2023 (llama.cpp's n-gram
  drafts); J. Chen, Y. Liang, Z. Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026, arXiv:2602.06036
  (the draft model; DFlash2 is z-lab's later release).

## J6 step issue

How a step goes to the GPU.

| part | what | switch | type |
|---|---|---|---|
| GPU-resident loop | with drafts, a step's sampling, draft acceptance (with J4's guard, J5's lenience and J3's verification), the next input and the KQ mask are made on the device; server requests and two sequences in one step run in it.  On this llama.cpp base the host checks each step before the next is queued (as fast at 8K, 1.6 % faster per step at 32K and 110K than queuing one ahead) | `FRANK_STEP_LOOP` (on with one GPU, a ROCm one, on Linux and Windows; off with several GPUs), `FRANK_STEP_LOOP_SEQ2` (on), `FRANK_DEVICE_INPUT_MAPPED` (on), `FRANK_STEP_AHEAD` (off; on queues one step ahead, with the recurrent state's whole-batch rollback `FRANK_RS_WHOLE_BATCH`); rdna-boosts' `LLAMA_DEVICE_INPUT` follows the loop when unset | speed |
| graphs for wider verification | the host path also records verify batches of 9-16 tokens as HIP graphs | `FRANK_GRAPH_MAX_TOKENS` (16) | speed |

The loop gives greedy output identical to the host path; at temperature > 0 the draft model's placement can change a token,
as on the host path.

- References: K. Zhu, Y. Gao, Y. Zhao et al., "NanoFlow: Towards Optimal Large Language Model Serving Throughput",
  OSDI 2025, arXiv:2408.12757, §4.2.1 (the next batch built and issued before the current one ends, its results checked a
  step later: `FRANK_STEP_AHEAD`).  The vLLM Gaudi plugin's documentation, "Bucketing mechanism" (lengths padded so a
  recorded graph is replayed; the loop's verify graphs span `FRANK_STEP_KV_HEADROOM` free cells for the same reason).
  R. Hasan, N. Korobov, R. Kobus, H. Huang, D. Mehra, "NVIDIA TensorRT-LLM Now Supports Recurrent Drafting for
  Optimizing LLM Inference", NVIDIA Technical Blog, 2024 (drafting and validation inside the engine instead of the host
  runtime); the SGLang Team, "SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured
  Outputs", LMSYS blog, 2024 (`FRANK_STEP_AHEAD`: the scheduler one batch ahead).  Built on: Alan Gray, "Optimizing
  llama.cpp AI Inference with CUDA Graphs", NVIDIA Technical Blog, 2024 (llama.cpp's graphs, which the wider verify batches
  join).

## J7 state between requests

| part | what | switch | type |
|---|---|---|---|
| state copy in bands | a state above 256 MiB (the prompt cache saving or restoring a long slot) goes through the pinned stage one band at a time.  One device only | `FRANK_STATE_ASYNC_BANDS` (on with one device) | speed |
| checkpoints in the background | context checkpoints are saved asynchronously; when a slot is cleared or changes hands they move into the prompt cache instead of being copied | `FRANK_SERVER_CKPT_ASYNC` (on) | speed |
| pinned pools | prompt-cache states and checkpoints live in pinned host memory sized at startup from the tensor shapes, copied straight to and from the device | `FRANK_SERVER_CACHE_PINNED`, `FRANK_SERVER_CKPT_PINNED` | speed |
| checkpoints on the batch grid | checkpoints are cut at batch starts, so a shared prefix is split the same way whatever follows it: a request continued from the cache gives the same output as one run from scratch | `FRANK_SERVER_CKPT_ALIGN` (on) | fix |

- References: R. Pan et al., "Marconi: Prefix Caching for the Era of Hybrid LLMs", MLSys 2025, arXiv:2411.19379 (recurrent
  states kept at chosen prefix positions, at a chunk boundary for chunked models); the SGLang Team, "Towards Deterministic
  Inference in SGLang and Reproducible RL Training", LMSYS blog, 2025 (chunk cuts aligned to a fixed grid, so a sequence is
  split the same way whatever is batched with it).  B. Gao et al., "Cost-Efficient Large Language Model Serving for
  Multi-turn Conversations with CachedAttention", USENIX ATC 2024, arXiv:2403.19708 (cache saved to host memory
  asynchronously, overlapped with computation).  LMCache documentation, "CPU RAM" storage backend (caches in reserved
  page-locked host memory).  M. Harris, "How to Optimize Data Transfers in CUDA C/C++", NVIDIA Technical Blog, 2012
  (transfers staged through pinned memory, small ones batched into one).

## Thanks

To stew675 and the rdna-boosts contributors; to Deadcode and the radiance contributors; to the llama.cpp / ggml community;
to z-lab for the DFlash2 draft models; and to the authors of the papers above.

## Change log

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
