# Build

Written with Claude (AI).  How the build behind the [README](README.md#reproduction-conditions)'s numbers was made, as
conditions to meet rather than a script: what the source tree must be, the versions, the principles, a tree hash to check
after each stage, and the pitfalls we ran into on Linux and Windows.  It is meant to be enough for a person or an AI agent
to repeat the build on either system.

## The result

A HIP build of llama.cpp for gfx1201 whose source tree is llama.cpp at the pinned commit, with the rdna-boosts release
applied by its own script, then this repository's eight patch files.  `llama-server` is all the measurements need
(`llama-bench` for speed alone).

## Pinned versions

| what | version |
|---|---|
| llama.cpp | commit `919794500475862f6768c353255469dc661cb734` (master, the build tag `b12597`, 238 commits after `v1.0.0`; also in rdna-boosts' `release.json`, field `base`) |
| llama-cpp-rdna-boosts | release tag `v17-919794500-r329` = commit `d5bb66e66adbf9eba05eb58115436546e2733d8d`.  If the tag or the repository is gone: the tag `pin/v17-919794500-r329` in [overdoingism/llama-cpp-rdna-boosts](https://github.com/overdoingism/llama-cpp-rdna-boosts), the same commit |
| this repository | all eight files in `patches/`, in this order: `J1-*`, `J2-*-prefill-a-*`, `J3-*`, `J4-*`, `J5-*`, `J6-*`, `J7-*`, `J2-*-prefill-b-*` (J2 v10 is the two J2 files; part b needs J4-J7's server and state code, so it goes last) |
| ROCm | 10.0.0 with hipBLASLt.  Linux: AMD's TheRock Python packages `rocm[libraries,devel,device-gfx1201]==10.0.0`, whose `_rocm_sdk_devel` folder is the ROCm root.  Windows: TheRock's `therock-dist-windows-gfx120X-all-10.0.0` (URL and SHA-256 in the [CI workflow](.github/workflows/windows-gfx1201.yml)).  HIP compiler: that release's clang (`AMD clang version 23.0.0git`, llvm-project `8f497e0992fb`) |
| compilers, build tools | Linux: GCC 13.3 for C / C++, CMake 3.28, make.  Windows: the SDK's clang for C, C++ and HIP, Visual Studio 2022 (not 2026) for the libraries and tools, CMake 4.3, Ninja, Git for Windows |
| GPU | gfx1201 (Radeon AI PRO R9700).  gfx1200 takes the same paths but is untested |

## Method

1. A clean llama.cpp checkout at the pinned commit.
2. The rdna-boosts release on top of it.  rdna-boosts is a patch series, not a fork of llama.cpp: its
   `scripts/apply-all.sh <llama.cpp folder>`, run from the release checkout, commits one block after another with
   `git am` on a new branch (`rdna-boosts`), with the series author's identity and dates, so every machine gets the same
   commits; when `jq` is installed it also compares the result with the tree in its `release.json`.
3. This repository's eight patch files with `git apply`, in the order J1, J2 part a, J3, J4, J5, J6, J7, J2 part b, each
   checked with `git apply --check` right before it is applied: each patch builds on the ones before it, so checking them
   all up front fails.  J2 part b goes last, after J7: J2 v10 is the two J2 files together, and part b needs J4-J7's server
   and state code.  They are plain diffs; nothing more is committed.
4. A HIP build for gfx1201 with the ROCm 10.0.0 compiler and libraries, configured after step 3.
5. The checks below.

## Principles

- **Pin by commit, check by tree.**  A tag can move or disappear; a tree hash says whether the files are exactly the ones
  that were measured.  Check the tree after each stage ([checkpoints](#checkpoints)); on a mismatch stop there and fix the
  base instead of building.
- **Never merge by hand.**  A patch that does not apply means a wrong base or converted line endings.  Do not use fuzz,
  `--3way`, `--reject` or edits to the patches, and do not leave a patch out: a group alone is not guaranteed to build.
  Parts are turned off at run time with their switches, never by building without them.
- **One ROCm.**  The HIP compiler, the CMake packages it finds and the libraries loaded at run time all come from the same
  ROCm 10.0.0.  A second ROCm on the machine (a distribution's `/opt/rocm-*`, an older SDK in `HIP_PATH`) gets picked up
  silently when nothing says otherwise.
- **The measured configuration is the default one.**  Set the build options below and leave everything else at the
  llama.cpp / rdna-boosts defaults.  The switches are environment variables read when the server starts, not build
  options.
- **Configure after applying.**  The patches add source files, which the build lists only when it is configured.

## Checkpoints

| after | command | tree |
|---|---|---|
| the llama.cpp checkout | `git rev-parse HEAD^{tree}` | `916941663346eb86004bed77f74e90fcf9c43126` (`release.json` field `base_tree`) |
| `apply-all.sh` | `git rev-parse HEAD^{tree}` | `63195b0ceb8dd9dea206a6a92b9bcd4a52e494e6` (`release.json` field `tree`); the branch tip is commit `bbd08a3004dc482c151a5f060ec2d53bcc030ebc` (field `tip`) |
| J1 to J7 (J2 part a only) | `git add -A`, then `git write-tree` | `127799f35e73c0dfaa13d0f72d959cd2938232ff` |
| all eight files (J2 part b last) | `git add -A`, then `git write-tree` | `8a78791ac57d765c98aa6deefec01cba55b2a0c1` |

`git apply` leaves the new files untracked, so the last two checks stage everything first (that changes no file).  A build
folder named `build*` at the top is ignored by llama.cpp's `.gitignore`; check before configuring if it lives elsewhere
inside the tree.

## Build options

| option | value | why |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | |
| `GGML_HIP` | `ON` | |
| `GPU_TARGETS` (we also set its older name `AMDGPU_TARGETS`) | `gfx1201` | the GPU the measurements ran on |
| `GGML_HIP_NO_VMM` | `ON` (the default) | with `OFF` a long context overflows into shared GPU memory on Windows and prefill drops to about a quarter |
| `FRANK_HIPBLASLT` | `ON` (the default) | links hipBLASLt for the fp8 prefill matmuls (J2's K1 and M4).  The configure log must say `FRANK_HIPBLASLT: hipBLASLt found, FRANK_KQ_BLAS available`; `not found, FRANK_KQ_BLAS compiled out` means the build lacks them |
| `CMAKE_HIP_COMPILER` | ROCm 10.0.0's `lib/llvm/bin/clang` (`clang.exe` on Windows) | otherwise CMake may find another ROCm's compiler |
| `CMAKE_PREFIX_PATH` | the ROCm root | the HIP, hipBLAS and hipBLASLt CMake packages of the same ROCm |
| `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` | Windows only: the SDK's `clang.exe`, `clang++.exe` | the MSVC ABI build the SDK expects |
| `GGML_HIP_RCCL` | Linux: either (we built with `1`); Windows: leave it out | it matters only with several GPUs, and the Windows SDK may not have RCCL (configure then fails) |
| `LLAMA_USE_PREBUILT_UI` | `OFF` (we built with it; optional) | otherwise configure tries to download the web UI |

The CI workflow adds `GGML_BACKEND_DL`, `GGML_NATIVE=OFF` and a C flag for a portable package; a local build does not need
them.

## Pitfalls

All systems:

- `v17-919794500-r329` is an annotated tag: `git rev-parse v17-919794500-r329` prints the tag object (`101611a7...`), not
  the commit.  Compare `git rev-parse v17-919794500-r329^{commit}` with the commit above.  The `pin/` tag in the fork
  points at the commit itself.
- `apply-all.sh` needs a clean llama.cpp tree and no existing `rdna-boosts` branch (it sets the committer identity
  itself).  Without `jq` it skips its tree check without saying so: check the tree yourself.  If it reports that it fell
  back to `git am -3`, the base was not the pinned commit.  Do not pass `--include-optional` or `--include-beta`: the
  result is then not the measured tree.  It prints one `new blank line at EOF` warning while applying block 10; harmless,
  the tree check covers it.
- `git apply` of J6 warns about one new blank line at the end of a file (`ggml/src/ggml-cuda/spec-step.cu`).  Harmless:
  the patch is applied as it is, and the tree check above includes that line.
- Configure logs `UI: no assets available - building without an embedded UI` with `LLAMA_USE_PREBUILT_UI=OFF`, and may
  log `UI: download dist.tar.gz... failed` without it.  Harmless: the build works offline, and `llama-server
  --no-webui` does not need the web UI.
- llama.cpp uses `ccache` by itself when it is installed (`GGML_CCACHE=OFF` turns that off); harmless.
- The HIP sources take long to compile and a lot of memory per job: all the default targets (679 files) took about 17
  minutes with 12 jobs on a 12-core Ryzen 9 3900X.  Lower the job count if the machine starts to swap.

Linux:

- With the Python-package ROCm, `libggml-hip.so` records no path to its libraries: when a system ROCm is in the loader's
  path (`/etc/ld.so.conf.d`), `libamdhip64` and `libhipblaslt` come from there without any message.  Run with
  `LD_LIBRARY_PATH=<build>/bin:<rocm root>/lib` and check with `ldd <build>/bin/libggml-hip.so` (or, while it runs,
  `/proc/<pid>/maps`) that every ROCm library is the 10.0.0 one.
- hipBLASLt loads its kernels from the `hipblaslt/library/gfx1201` folder next to `libhipblaslt.so`: nothing to copy as
  long as the 10.0.0 `libhipblaslt.so` is the one loaded.  Otherwise the log says `hipBLASLt has no ... kernels` and K1
  stays off; `HIPBLASLT_TENSILE_LIBPATH` can point at the folder.
- At run time set `HSA_USERPTR_FOR_PAGED_MEM=0` ([README](README.md#run-and-switches)).

Windows:

- `git config --global core.autocrlf false` before cloning: with converted line endings the patches do not apply.
- Build in a Developer PowerShell for VS 2022 (it provides `rc.exe`, `mt.exe` and the Windows SDK libraries).  Not the
  2026 toolset (MSVC 14.51): every HIP file failed to compile with it.  The linker embeds an application manifest in every
  `.exe` and calls `mt.exe` from `PATH`: without it the DLLs build and the `.exe` files fail (`unable to find mt.exe in
  PATH`); put the Windows SDK's `bin\<version>\x64` folder in `PATH` and build again.
- Run `apply-all.sh` with Git for Windows' `bash`.  It usually has no `jq`, so check the trees yourself.
- Copy the SDK's `hipblaslt\library\gfx1201` folder (in the tarball under `bin\`; with the Python packages it may sit in
  a neighbouring package folder: look for a `gfx1201` folder whose parents are `library` and `hipblaslt`) into the folder
  of `llama-server.exe` (`build\bin`): hipBLASLt looks for it there.  Without it the log says `hipBLASLt has no ...
  kernels` and K1 stays off.  Elsewhere it needs `HIPBLASLT_TENSILE_LIBPATH`.
- Which ROCm DLLs load depends on the folder of the `.exe` and on `PATH`; a system-wide `HIP_PATH` may name another SDK.
  Put the 10.0.0 SDK's `bin` first in `PATH`, or copy its runtime DLLs next to the `.exe` as
  [`package.ps1`](packaging/windows/package.ps1) does.

## Checking the result

- The tree hashes above, and the configure line of `FRANK_HIPBLASLT`.
- The ROCm libraries the binary loads (Linux: `ldd`; Windows: the DLLs found next to the `.exe` or in `PATH`) are the
  10.0.0 ones.
- With a model, the startup log lines listed under [Run and switches](README.md#run-and-switches).
- The KLD reference of the README is the same binary with `FRANK_ALL=0`.  With a `q8_0` cache it gives the same greedy
  tokens as plain rdna-boosts (the tree after `apply-all.sh`, built the same way).
