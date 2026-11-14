llama-Frankenstein for Windows, AMD gfx1201 (Radeon AI PRO R9700, RX 9070 / 9070 XT)
====================================================================================

llama.cpp + llama-cpp-rdna-boosts + the patches of https://github.com/overdoingism/llama-Frankenstein, built for gfx1201
only, with the ROCm 10.0.0 runtime DLLs and the hipBLASLt kernels for gfx1201 included.  Other GPUs: use rdna-boosts
directly (this package's speed-ups are RDNA4-only).  An RX 9060 XT (gfx1200) is RDNA4 too but untested here: build it
yourself (BUILD.md in the repository; an AI assistant can do it for you).

Run
---

Everything is on by default; each part can be turned off with its switch (SWITCHES.md).  Speculative decoding needs a draft
model and server parameters; run-server-sample.ps1 has them.  Open it (or a copy of it) in a text editor, fill in the two paths at
the top (the model, and the DFlash2 draft model), then in PowerShell, in this folder:

    .\run-server-sample.ps1

- The draft model is for Qwen3.8-27B only: https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M).  For any other
  model set the draft path to "" (no speculative decoding).
- The script runs llama-server with a 131072-token context, 1 slot, at http://127.0.0.1:8080.  Other llama-server
  arguments can follow it and win over its own, e.g.  .\run-server-sample.ps1 --ctx-size 65536 --port 8100
  .\run-server-sample.ps1 -DryRun prints the command without starting.
- If Windows refuses to run the script (downloaded files are blocked): Unblock-File .\run-server-sample.ps1, or
  powershell -ExecutionPolicy Bypass -File .\run-server-sample.ps1
- If HIP also sees the CPU's integrated GPU (the log says "found 2 ROCm devices"), run $env:HIP_VISIBLE_DEVICES = "0"
  (the R9700's index) first, or the fp8 prefill (K1) stays off.
- With K1 on, the log shows "K1 warm-up (2048 tokens, ...): fp8 GEMM ready".
- VRAM: on Windows an allocation that does not fit lands in shared GPU memory, and everything slows down (the first
  prefill is then far below its usual speed).  J3's int2 draft head (about 0.4 GiB with the draft model) is outside
  --fit: if that happens, run $env:FRANK_HEAD_INT2 = "0" first, or lower --ctx-size.
- J5's lenience (lossy) is off by default (SWITCHES.md).

Without the script:

    .\llama-server.exe -m <model.gguf> -ngl 99 -c 131072 -ub 2048 -b 2048 -fa on -ctk q8_0r -ctv q8_0r --kv-unified `
      -md <Qwen3.8-27B-DFlash2-Q4_K_M.gguf> -ngld 99 --spec-draft-n-max 5 `
      --spec-type draft-dflash --spec-draft-sampling probabilistic

All switches: SWITCHES.md in the repository.

These patches are offered for research and evaluation.  Every patch can be turned off with its switch.

Check the download
------------------

Compare the SHA-256 of the zip with the .sha256 file (PowerShell: Get-FileHash <zip>), and with the GitHub CLI:
gh attestation verify <zip> --repo overdoingism/llama-Frankenstein (it was built by the repository's GitHub Actions
workflow from the public sources).

Licenses: LICENSES.txt (llama.cpp, rdna-boosts, this repository: MIT; the ROCm components: MIT, BSD, Apache 2.0 with LLVM
exceptions).  No warranty.  Written with Claude (AI).

