# RELOM & DCT: Residual Order-of-Magnitude Quantization, Horner-Bucket Matmul, and Dual-Clutch MoE Tiering

Measured on 2× Intel Arc Pro B60 (24 GB, PCIe Gen5 x8 each), 2× Predator GM7 1 TB NVMe (`C:` CPU-direct Gen4 x4 + `M:` PCH Gen4 x4), 64 GB DDR5-6000 (4× 16 GB), and a remote NVIDIA RTX 3070 (8 GB) over a direct 10 GbE link (9.12 Gbps measured; RPC rows marked 2.5 GbE were measured before the link was upgraded), running Windows 11.

---

## Honest Prior-Art Audit: What Is Known vs. What Is Genuinely New Here

To avoid over-claiming or reinventing textbook terms, here is the exact line between prior art and what is new in this repository:

### What Already Exists in the Literature (Prior Art)
1. **Plane-Split Lossless Float Compression (`ZipNN` 2024, `DFloat11` 2025):**
   - Separating `bf16` into its 1-bit sign + 7-bit mantissa byte and 8-bit exponent byte works because the mantissa is near-uniform (~6.97 bits of entropy) while the exponent has low entropy (~2.6–3.0 bits).
   - `ZipNN` and `DFloat11` entropy-code (Huffman/ANS) the exponent plane. Their limitation: variable-length streams require a decompression pass before or during inference.
2. **Horner's Scheme / Bucketed Radix Accumulation (Classical Numerical Analysis):**
   - Factoring $\sum_j m_j 2^{e_j}$ by grouping terms with identical exponents $e \in \{e_{\max} \dots e_{\min}\}$ and shifting the accumulator by 1 bit per bucket ($\text{acc} = 2 \cdot \text{acc} + \sum_{j \in B_e} m_j x_j$) is the radix-2 Horner scheme (used in exact floating-point accumulators such as Kulisch/binned accumulators).
3. **Cache Replacement with Frequency + Recency (`LFU-Aging`, `ARC`, `W-TinyLFU`):**
   - Combining decayed frequency with a recency window and keeping ghost/metadata scores for evicted items is known in database/cache literature (`W-TinyLFU`, `ARC`).

### What Is Genuinely New & Worth Sharing in This Work
1. **`GGML_TYPE_Q12H` (`QL`) Fixed-Stride Random-Access Block with Shift-Offset Mode (`v2`):**
   - Unlike `DFloat11` or `ZipNN`, `Q12H` is a fixed-stride 256-weight block (`412 B`, `12.875 bpw = 0.805x bf16`) where 98.1–98.5% of blocks use a LUT-less **shift-offset mode** (`exp = emin + nibble`), reconstructing IEEE-754 bits directly inside the fused GEMV (`mul_mat_vec`) and `mul_mat_id` shaders with **zero decompression pass**, running **faster than `bf16`** on both Intel Arc B60 (`2.1x–2.5x` prefill, `+5–10%` decode), NVIDIA RTX 3070 (`+16%` decode on 7B), and AVX2 CPU.
2. **`RELOM` (Residual Order-of-Magnitude Horizon) for Per-Layer & Intra-Block Precision Allocation:**
   - Instead of choosing per-layer bit-widths by heuristic sensitivity sweeps, `RELOM` measures the physical **exponent shift** $\Delta E = E(x_{\text{residual}}) - E(\Delta_{\text{layer}})$ at the `bf16(x + Δ)` residual addition.
   - In `Qwen2.5-Coder-14B` and `Qwen3-Coder-30B-A3B`:
     - In **Layers `0..4` and `45..47`**, $\Delta E \in [-3, 0]$ ($\Delta$ is larger than or equal to the residual stream; all 7 mantissa bits survive).
     - In **Middle Layers `5..44`**, $\Delta E = +3.0\text{ bits}$ ($x_{\text{residual}}$ is $8\times$ larger than $\Delta$, so the hardware `bf16` adder shifts $\Delta$'s mantissa right by 3 bits and discards the bottom 3 bits anyway).
   - **Measured Result (`Qwen3-Coder-30B-A3B` on 2× Arc Pro B60):**
     - Keeping Layers `0..4`, `45..47`, `token_embd`, `output`, and all 48 `attn_*` projections in lossless `QL` (`Q12H`) and putting middle expert layers (`5..44`) in `Q8_0` shrinks the model from `45.94 GiB` to **`33.63 GiB` (`16.8 GiB/card`, 0 CPU spill)**, achieving **identical KL divergence (`0.00148` mean, `0.000085` median, `98.86%` top-1 agreement) to full `Q12H`**, at **`1,154.5 tok/s` prefill and `60.87 tok/s` decode**.
     - With `Q4_K_M` in the middle expert layers (`23.16 GiB`, `11.6 GiB/card`), mean KLD drops **2.5×** (`0.0166 -> 0.0067`) and maximum KLD spikes drop **9.8×** (`12.10 -> 1.24`) compared to uniform `UD-Q4_K_XL`.
     - **Lossless `QL` Backbone Only (`17.83 GiB`, `+1.38 GiB` over uniform `UD-Q4_K_XL`):** Upgrading *only* the 194 non-expert Backbone tensors (`token_embd`, `output.weight`, and all 48 `attn_*` layers) to `QL` while keeping 100% of all 48 MoE expert layers in `Q4_K` cuts mean KLD by **2.13×** (`0.01657 -> 0.00778`) and worst-case KLD spikes by **7.7×** (`12.10 -> 1.57`), running at **`1,462.6 tok/s` prefill and `77.49 tok/s` decode**.
   - **Surgical HTTP `Range` `BF16 -> QL` Backbone Transplant for `80B+` / `120B+` Models (`surgical_ql_80b.py`):**
     - Standard GGUF conversions of giant MoEs (`Qwen3-Coder-Next 80B-A3B`, `gpt-oss-120b`) quantize the Attention/Backbone to `Q8_0`, which corrupts **81.8%** of `BF16` weights (`max 15,090 ULPs` off) because linear integer grids destroy the mantissa of smaller weights sharing a 32-weight block with a larger weight.
     - Using HTTP `Range` requests against HuggingFace `.safetensors` shards, `surgical_ql_80b.py` skips 155 GB of MoE experts on the remote CDN, downloads *only* the 230 `BF16` Backbone tensors (`1.40B` weights, `2.8 GB`) of `Qwen/Qwen3-Coder-Next`, quantizes them into **lossless `12-bit QL` (`Q12H`)**, and splices them into `Qwen3-Coder-Next-QL-Backbone-Q4_K_XL.gguf` (`46.91 GiB`, holding **`45.7 tok/s` @ `d=0..4K`, `44.9 tok/s` @ `d=16K`, and `41.7 tok/s` @ `d=32K`** on 2× B60 + DDR5).
     - Likewise, `surgical_ql_120b.py` pulls only the 146 `BF16` Backbone tensors (`3.94 GiB`) of `openai/gpt-oss-120b`, converting the 36 `4096`-wide `attn_output` matrices to `QL` (`Q12H`) and keeping the 110 `2880`-wide (`11×256 + 64`) matrices in exact `BF16`, splicing `gpt-oss-120b-QL-Backbone-MXFP4.gguf` (`60.71 GiB`, holding **`428.1 tok/s` prefill / `34.5 tok/s` decode** at `-ncmoe 12 -ts 24/12`, and **`32.4 -> 27.0 tok/s` all the way to `d=32,768`** at `-ncmoe 14 -ts 25/11`).
   - **On-Die NPU + iGPU "Track Switchman" (`npu_router_switchman.py`):**
     - Offloads all 48 layers of MoE Router Gate matrices (`ffn_gate_inp.weight`) onto the idle on-die **Intel AI Boost NPU** and **Arrow Lake iGPU** via OpenVINO (`12.59 µs/layer` on NPU, `7.58 µs/layer` on iGPU for `Qwen3-Coder-30B-A3B` `6,144` experts; `42.89 µs/layer` on NPU, `29.93 µs/layer` on iGPU, and **`35,336 layer-routes/sec` combined** for `Qwen3-Coder-Next 80B-A3B` `24,576` experts), freeing the discrete B60 GPUs and AVX2 CPU threads from lookahead router scoring.
3. **Exponent-Sorted Horner-Bucket + 2D `(Exponent, Mantissa)` Bin-Table Matmul (`horner_relom.py`):**
   - Sorting weights within a block/row by exponent descending (`d = emax - exp` from `0` to `15`) solves three GPU problems at once:
     1. Replaces the 128-byte per-weight nibble array (`4.0 bpw`) with a 16-byte bucket run-length header (`0.5 bpw`).
     2. Eliminates all 256 per-weight exponent shifts via Horner's scheme across the 16 buckets (`100.00%` bit-exact `bf16` match vs `fp32` reference).
     3. Allows **changing the significand operand width and instruction (`INT8 -> INT5/4 -> INT3/2 -> Early Exit`) only ONCE at each `RELOM` zone boundary**, eliminating warp divergence inside the inner loop.
   - **2D `(Exponent, Mantissa)` Binning (`885` unique buckets across `5,120` weights = `5.78×` fewer multiplies, `0` index bits):** Because `QL` has only `16 × 128 = 2,048` possible `(exp_offset, mantissa)` values per row, binning input activations into a `4 KiB` shared-memory table before multiplying cuts multiplications by **5.78×** on `Qwen2.5-Coder-14B` (`5,120 -> 885` muls/row) with `100.00%` exact `bf16` output.
4. **Wallace & Gromit "Train-Track" Size-Invariance Law (`25B -> 120B` Sweep + Standalone 8 GB RTX 3070 Proof):**
   - Keeping the tiny Attention + Router backbone (`~1.8–3.0 GiB`) pinned in GPU VRAM while laying down MoE expert layers from DDR5-6000 (`-ncmoe N -lm none`) proves that **decode speed is governed by active bytes per token and track bandwidth, not total model size**:
     - **100% of MoE Experts on DDR5 Track (`-ncmoe all`, using only `1.8–3.0 GiB` total GPU VRAM):**
       - `Gemma-4-26B-A4B` (`25.2B`, `15.83 GiB`): **`30.14 tok/s`** (`657.5 tok/s` prefill)
       - `Qwen3-Coder-30B-A3B` (`30.5B`, `16.45 GiB`): **`27.85 tok/s`** (`543.4 tok/s` prefill)
       - `Qwen3.6-35B-A3B` (`34.7B`, `20.81 GiB`): **`32.83 tok/s`** (`569.3 tok/s` prefill)
       - `Qwen3-Coder-Next 80B-A3B` (`79.7B`, `46.20 GiB`): **`25.96 tok/s`** (`320.7 tok/s` prefill) — and **`9.22 tok/s` on the integrated Intel iGPU alone (`0` discrete GPUs)**!
       - `gpt-oss-120b` (`116.8B`, `59.02 GiB`): **`18.74 tok/s`** (`144.0 tok/s` prefill) at `-ncmoe 36`, and **`40.27 tok/s` (`524.3 tok/s` prefill)** at `-ncmoe 9 -ts 22/14` (or **`36.18 tok/s` @ `d=0`, `35.98 tok/s` @ `d=4K`, `33.76 tok/s` @ `d=16K`, `31.56 tok/s` @ `d=32K`** at deep-safe `-ncmoe 12 -ts 24/12 -fa 1`).
       - `DeepSeek-V4-Flash-0731` (**`284.3B`**, `80.76 GiB`, `43 layers`, `256 experts`, `6 active`): **`171.76 tok/s` prefill and `13.72 tok/s` decode** at `-ncmoe 24 -ts 31/12 -lm none` (`19` MoE layers + `8.20 GiB` Backbone in B60 VRAM, `24` MoE layers = `40.5 GiB` in DDR5-6000 RAM, `0` SSD paging!).
       - `GLM-5.3-Flash` (**`320.8B`**, `94.84 GiB`, `46 layers`, `288 experts`, `8 active = 18B active`): Striped across both NVMe drives (`Shard 2` `46.47 GiB` on CPU-direct NVMe `C:` + `Shards 1, 3, 4` `48.37 GiB` on PCH NVMe `M:` via native Dual-NVMe split-shard loader in `llama-model-loader.cpp`), running at **`32.71 tok/s` prefill and `9.45 tok/s` decode** (`-ncmoe 28 -ts 31/15 -lm mmap`).
     - **Standalone 8 GB NVIDIA RTX 3070 + 32 GB DDR4-2667 (`Qwen3-Coder-30B-A3B`, `16.45 GiB`):**
       - Runs a `30.5B` model on a single 8 GB card at **`829.7 tok/s` prefill and `26.55–28.36 tok/s` decode** (`-ncmoe 32..28`), tapering cleanly to **`17.60 tok/s` (`677.6 tok/s` prefill)** with 100% of experts in DDR4 (`-ncmoe 48`, `1.80 GiB` VRAM).
5. **Dual-Clutch Transmission (DCT) + Asymmetric Dual-Channel NVMe (`C: + M:`) + Reddit & Reloaded Score for MoE:**
   - **Asymmetric 2:1 NVMe Striping:** Striping expert records 2:1 across CPU-direct NVMe `C:` (`6.32 GiB/s`) and Chipset NVMe `M:` (`3.23–3.73 GiB/s`) reaches **`9.23–9.61 GiB/s` (`10.3 GB/s`)**.
   - **Reddit Score + `Reloaded Score` (Speculative Unloading of "Abstract Tourists"):** Preserves per-expert frequency/recency across evictions and tracks `reloaded_score` (regret reloads within $\le 32$ steps) while unloading one-shot "abstract tourist" experts first, cutting SSD misses by **12–24%** and eliminating thousands of regret reloads on 3,206-token traces.
   - **DCT GPU+CPU Split Overlap (`ggml-backend.cpp`):** Fires the GPU's resident expert subgraph (`out_hit`) asynchronously *before* starting the CPU's missed-expert subgraph (`out_miss`), and overlaps multi-layer lookahead NVMe reads (`3.22×` end-to-end NVMe tier speedup in `dct_nvme_race.py`).
6. **Distributed RPC Head Fix & On-the-Fly `Q12H` Wire Compression (`ggml-rpc.cpp` + `llama-graph.cpp`):**
   - Eliminated two unused `has_mixed` embedding input branches on pure token batches in `llama-graph.cpp` (`-95%` incoming wire bytes to an RPC head stage: `45.2 KiB -> 2.3 KiB/token`).
   - Added live `Q12H` wire compression for `F32` activation tensors in `ggml-rpc.cpp` (`SET_TENSOR` and `GET_TENSOR`, `2.49×` smaller on the wire: `14.00 KiB -> 5.63 KiB`). **This is lossy and opt-in (`GGML_RPC_WIRE_Q12H=1`, off by default, RPC protocol `8.0.2`):** activations are cut from `F32` to 16 bits and outliers beyond a block's 4 exception slots are snapped. Measured on the 30B `QL+Q8` across `RTX 3070 + 2× B60`: wire on against wire off = mean KLD `0.000971`, same top token `99.098%` (off against off repeats exactly: KLD `0`, `100%`); against `BF16`, off = `0.001632` / `98.870%`, on = `0.001707` / `98.729%`. It made no tok/s difference on 10 GbE.
   - **Whole-file round-trip check (`ql_roundtrip.py`):** every `Q12H` tensor of both surgical backbones was dequantized and compared bit for bit with the Hugging Face `BF16` source: 80B `230` tensors / `1,402,470,400` weights and 120B `36` tensors / `424,673,280` weights, **zero differ**.
   - Runs `Qwen3-Coder-Next 80B-A3B` (`46.2 GiB`) and `Qwen3-Coder-30B-A3B Q12H` (`45.94 GiB`) **100% in VRAM (`-ncmoe 0`)** across `ROG RTX 3070 (Head) + 2× Arc Pro B60 (Body)` over 2.5 GbE at **`1,055.7 tok/s` prefill / `50.3 tok/s` decode** (80B) and **`538.8 tok/s` prefill / `41.6 tok/s` decode** (30B Q12H). Over the direct 10 GbE link: `1,033.5` / `50.3` (80B), `534.9` / `42.7` (30B Q12H), `1,206.8` / `51.2` (30B `QL+Q8`); the link is not the bottleneck.

---

## Contents

- `horner_relom.py` / `horner_relom.output.txt` — Exponent-Sorted Horner-Bucket + `RELOM` Zone-Stepped matrix multiply proof on `Qwen2.5-Coder-14B`.
- `ramoe_dct.py` / `dct_nvme_race.py` / `dct_nvme_race.output.txt` — Dual-Channel NVMe (2:1 `C: + M:`) + Reddit Score + DCT lookahead engine and benchmark on `Qwen3-30B-A3B` routing trace.
- `surgical_ql_80b.py` / `surgical_ql_120b.py` — Surgical HTTP `Range` `BF16 -> QL (Q12H) / BF16` Backbone transplant scripts for `Qwen3-Coder-Next 80B-A3B` and `gpt-oss-120b`.
- `npu_router_switchman.py` — OpenVINO MoE Router Gate offload across all 48 layers onto the Intel AI Boost NPU and Arrow Lake iGPU.
- `patches/` — The 6 commits on top of `llama.cpp` (`q12h-moe` branch):
  - `0001` — `Q12H_SLOT_EMPTY = 0xFFFF` sentinel fix, chunked Vulkan dequant dispatch, `BF16` overflow fallback.
  - `0002` — AVX2 `Q12H` CPU dot product (`+63%` decode), direct IEEE-754 bit-shift `Q12H` Vulkan shader, Reddit-score MoE cache with profile prewarm, and DCT GPU+CPU split overlap.
  - `0003` — Skip unused `has_mixed` embedding branches on pure token batches (`-95%` RPC head wire traffic).
  - `0004` — Live `Q12H` wire compression for `F32` activations in `ggml-rpc.cpp` (`2.49×` smaller wire payload).
  - `0005` — Prioritize top-4 largest outliers for `Q12H` 16-bit exception slots and snap sub-`RELOM` tail outliers.
  - `0006` — Native Dual-NVMe split-shard loader (`llama-model-loader.cpp`) for striping multi-shard `95+ GiB` GGUFs across `C:` and `M:` without cross-volume symlinks.

---

## Credits & Engineering Process

**Architecture, Physical Intuition & Experimental Direction:** `@wcarsond` (Carson) — conceived the `QL` (`Q12H`) lossless exponent-window float layout, the `RELOM` residual-horizon law ($\Delta E$ bit-survival across layer depth), the Exponent-Sorted Horner / 2D `(Exponent, Mantissa)` bin-table matmul, and the Dual-Clutch Transmission (`DCT`) + Reddit/Reloaded-Score MoE "Wallace & Gromit Train-Track" architecture, driving the hypothesis generation and black-box A/B verification across hardware tiers.

**Implementation, Kernel Optimization & Benchmarking:** Built in pair-programming collaboration with GitHub Copilot (`llama.cpp` AVX2 / Vulkan GLSL / RPC C++ kernels, Python Win32 unbuffered Dual-NVMe `OVERLAPPED` engines, and HTTP `Range` surgical `BF16 -> QL` GGUF splicers).

