# Kernel Design Agents Flow Prompt — Overlap the Fused SiTU+Requant Epilogue with the MMA

You are working in a task implementation workspace. The fused FC1 kernel
(`m_grouped_fp8_fp4_gemm_nt_contiguous_unpadded_situ_quant`) fuses the SiTU-GLU v2 activation + FP8
requant into the GEMM1 epilogue. It is correct (bit-exact) but still carries a residual latency
**vs the un-fused FC1 GEMM**. The single, direct goal: drive that residual to zero by **overlapping
the epilogue with the MMA**, not by removing work. This doc is a re-exploration brief — it records
what has been tried, what landed, what was ruled out, and where to look next.

## Baseline & current gap (the number to close)

Baseline = `bench_situ_tp8_dgunpadded_v_b8.csv` (the pre-fusion / un-fused FC1: plain GEMM, kineto
self-time). The current fused FC1 (lever #1 landed) vs that baseline, kineto `fc1_us`:

| bs | v_b8 plain FC1 | fused FC1 (now) | Δ |
|---|---|---|---|
| 4096 | 176.9 | 217.5 | +41µs (+23%) |
| 8192 | 242.8 | 278.1 | +35µs (+15%) |
| 16384 | 405.4 | 514.5 | +109µs (+27%) |
| 32768 | 678.0 | 774.7 | +97µs (+14%) |
| 65536 | 1324.7 | 1435.0 | +110µs (+8%) |
| 131072 | 2539.6 | 2804.1 | **+265µs (+10%)** |

**FC2 is NOT the problem — rule it out.** FC2 (`fc2_us`) matches v_b8 within ±6% (noise) at every bs;
it is the down-projection (n=H=7168, k=I=256) that writes `M×7168` bf16 — ~56× the fused FC1 output —
so it is inherently store-bound and ≈ FC1 in absolute time. That is shape-intrinsic, present in v_b8
too, and unaffected by the fusion. Do not chase FC2.

So the residual to close is **FC1: +8–27% (≈ +40–265µs, growing with bs)** — the part of the
activation+requant that is not yet hidden behind the MMA.

## Task Contract

- Task name: `overlap the fused SiTU+FP8-requant FC1 epilogue with the MMA (close FC1 vs v_b8)`
- Objective: `drive fused FC1 (..._unpadded_situ_quant) kineto latency down to the v_b8 plain-FC1 baseline. Perfect overlap => fused FC1 == plain FC1 (the activation/requant becomes free).`
- Correctness requirements: `Bit-exact vs per_token_cast_to_fp8(situglu_v2(d1_il), use_ue8m0=True, gran_k=32): dequant diff == 0.0, UE8M0 scales 100% identical. tests/test_fp8_fp4_unpad.py::{test_situ_fused,test_block_m_grid} pass. bench_situ.py --backend dg_unpadded (correctness sweep) cosine vs DG ref >= 0.999. Works for partial tiles, tiny M, cluster_n in {1,2}.`
- Performance target: `close the FC1 vs v_b8 gap above (currently +8–27%). Realistic range (bs<=4096) is already ~aligned; the residual is concentrated at bs>=8192 and grows in absolute µs with bs. Stretch: <=0% everywhere.`
- Allowed approaches: `epilogue + pipeline config ONLY -- sm100_store_cd_swap_ab.cuh (kHasActivation/kQuant store branch), transform.cuh (EpilogueSiTU*), sm100_fp8_fp4_gemm_1d1d.cuh (epilogue/TMEM staging), and the store-wg / num-stages / block_m knobs in csrc/jit_kernels/heuristics/sm100.hpp (gate to desc.heavy_epilogue so other GemmTypes are untouched). Optionally integrate the situ op into the embedded autotuner. Do NOT change the GEMM math, the weight interleave, the SFA layout, or the (d_fp8, scale) GEMM2 contract.`
- Validation: `cd DeepGEMM && PYTHONPATH=$PWD:$PWD/tests python3 -c "import sys;sys.path.insert(0,'tests');import test_fp8_fp4_unpad as t;t.test_block_m_grid();t.test_situ_fused()"`
- Evaluation: `plain-vs-fused FC1 micro-bench (below), AND cd flashinfer && DG_NO_AUTOTUNE=1 PYTHONPATH=<DeepGEMM> python3 bench_situ.py --backend dg_unpadded --bench --bs 4096,8192,16384,32768,65536 -> compare fc1_us vs bench_situ_tp8_dgunpadded_v_b8.csv.`
- Promotion: `(1) bit-exact retained; (2) fused FC1 strictly closer to v_b8 at bs>=8192 with no regression at bs<=4096 and no FC2/plain-GEMM regression; (3) tiny/partial M + cluster_n in {1,2} still correct.`

## Current code state (what is actually in the tree)

- **LANDED — lever #1 (store-warpgroup parallelism).** `GemmDesc.heavy_epilogue` (set in the unpadded
  launcher when `epilogue_type` is a SiTU activation) makes `get_num_store_wg` return
  `min(6, chunks_per_tile)` for the fused epilogue regardless of K (plain GEMMs stay at their short-K
  rule). This spreads the per-tile activation+requant over up to 6 store warpgroups. It cut the
  original catastrophic overhead (+455%/+552% in the first naive store) down to today's +8–27%.
  This is the ONLY perf change currently in the kernel for the fused path.
- `kNumEpilogueStages = 2` (hardcoded). The store path is `sm100_store_cd_swap_ab.cuh`; the quant
  branch uses 4 threads per (token, 32-ch group), a 4-lane amax butterfly, and an 8-byte (uint2)
  coalesced fp8 store.

## What has been tried and RULED OUT (do not repeat without new evidence)

- **Lever #2 — cheaper per-element math (faster tanh). REJECTED.** Replacing libdevice `tanhf` with
  the `__expf` identity `tanh(x)=2·sigmoid(2x)-1` is ~4% SLOWER on SM100: `tanhf` lowers to a single
  hardware `MUFU.TANH`, while the `__expf` form needs more MUFU ops (3×EX2+3×RCP vs 2×TANH+1×EX2).
  Accuracy stayed ~bit-exact, but it is not faster. The transcendentals are already optimal; they are
  not the lever. (The `fast_math` flag was implemented then reverted.)
- **Lever #3 — deeper TMEM accumulator pipeline (`kNumEpilogueStages` 2→3+ via smaller block_m).
  TRIED, REVERTED, no win on this shape.** Implemented `kNumEpilogueStages` as a static function of
  block_m (smaller block_m -> more stages, host barrier/TMEM budgets kept consistent), bit-exact, no
  default regression. But the block_m sweep showed the only kineto win — **bs16384 block_m=192,
  -12% FC1** — comes from a **2-stage** tile (better geometry), NOT the extra stages: the 3-stage
  configs (block_m=128/160) did *not* beat the 2-stage block_m=192, and forcing a smaller block_m
  **regresses bs>=32768** (+8–12%). And block_m=192 was already a legal 2-stage config before the
  change — so deeper pipelining did not enable the win. Conclusion: on this shape (n=512, k=7168,
  BLOCK_M~240 already near the 512-col TMEM limit) tile geometry dominates, deeper pipeline does not
  pay. Reverted to `kNumEpilogueStages=2`.

## Overlap model & the masking-rate metric (how to quantify "hidden")

Warp roles (`sm100_fp8_fp4_gemm_1d1d.cuh`): warp0 TMA load, warp1 MMA issue, warp2 UTCCP SF
transpose, warps `[kNumNonEpilogueThreads/32, …)` = epilogue (store) warpgroups. The TMEM accumulator
is 2-stage; `tmem_full`/`tmem_empty` barriers hand each tile MMA→epilogue→MMA. `tmem_empty` is arrived
right after the TMEM→smem STSM read, so the activation runs from smem and does not hold TMEM — but the
SAME warps do the read and the activation, so a slow activation delays the *next* tile's read and
throttles the MMA.

Masking rate (two-stream overlap): with T_mma = plain-FC1 time (= v_b8), T_epi = the added epilogue's
own busy time (from ncu pipe-active deltas fused−plain, max of the parallel pipes ALU/FMA/XU/LSU),
`masking = (T_mma + T_epi − D_fused) / min(T_mma, T_epi)` (1 = perfectly hidden). Because
T_epi (~150µs at bs16384) ≪ T_mma (~900µs), the epilogue is *theoretically* 100% hideable — the
residual is overlap inefficiency (dependency/latency), confirmed by ncu (no pipe saturated: TC ~46%,
ALU ~33%, DRAM ~56% at the earlier cap-4 state). The simplest black-box proxy is just
`exposed = fused_fc1 − v_b8_fc1` (the un-hidden µs in the table above).

## Candidate directions for the NEXT round (ranked)

1. **Re-profile at the CURRENT cap=6 state (do this FIRST).** The earlier ncu (TC 60.6%→45.7%) was at
   cap=4. Re-run `ncu --section SpeedOfLight --section WarpStateStats --section SchedulerStats
   --section ComputeWorkloadAnalysis` on the fused FC1 at bs=16384 and 65536 and find the NEW dominant
   stall after lever #1 (barrier/`tmem` wait? MIO/LSU throttle? scoreboard?). The right next lever
   depends on which stall now dominates — don't guess.
2. **Per-shape block_m (the one concrete known win).** block_m=192 gives −12% kineto FC1 at bs16384;
   default block_m elsewhere. The clean way is to **integrate `..._situ_quant` into the embedded
   autotuner** (it is currently heuristic-only; the plain unpadded op IS autotuned) so block_m /
   cluster_n / store_wg are searched per m-bucket and the best is picked automatically — likely the
   highest-leverage practical step, and it needs no risky kernel surgery.
3. **Decouple the TMEM-drain warps from the activation warps (untried, deepest fix).** Dedicate a
   subset of epilogue warps to ONLY the STSM TMEM→smem read (so `tmem_empty` frees at the MMA's rate,
   never blocked by the activation), and the rest to activation+requant+store from a deeper smem ring
   buffer. This breaks the read↔activation coupling that throttles the MMA.
4. **Shorten the epilogue dependency chain / software-pipeline it.** The per-(token,group) chain
   (smem read → tanh/exp → 4-lane amax shuffle → ÷sf → e4m3 → store) is latency-bound; overlap
   activation(chunk i) [XU/FMA] with store(chunk i−1) [LSU], and hoist the amax/reciprocal so fewer
   dependent ops sit on the critical path.

Non-goals: do not drop the requant or change granularity; do not touch the weight interleave or the
(d_fp8, scale) contract. Wins must come from overlap/throughput. Guard: any restructure must keep
`tmem_empty` arriving after the TMEM read and before the activation, or the MMA re-couples to it.

## Measurement harness (epilogue isolation)

Plain-vs-fused FC1 micro-bench (CUDA-event median, L2-flushed, same interleaved weights, same M): time
`m_grouped_fp8_fp4_gemm_nt_contiguous_unpadded` (plain, bf16 [M,2I]) vs `..._situ_quant` (fused, fp8
[M,I] + scale) at bs ∈ {256,1024,4096,16384,65536} (M = bs*8 over E=384); report `fused/plain − 1`.
`override_block_m` / `override_cluster_n` are exposed on the op for the block_m sweep (and via
`DG_SITU_BLOCK_M` in bench_situ.py's fused path). Cross-check the kineto `fc1_us` from
`bench_situ.py --bench` against `bench_situ_tp8_dgunpadded_v_b8.csv`. NOTE: `DG_NO_AUTOTUNE=1` for a
deterministic heuristic read; the bench autotunes the *plain* GEMM2 but not the situ FC1 today (see #2).

## Workflow

1. Read the kernel + epilogue files; re-derive the per-tile MMA vs epilogue cost ratio.
2. Reproduce the v_b8 gap with the micro-bench + bench; capture a fresh `ncu` stall breakdown (#1).
3. Write a plan draft to `docs/draft.md` (don't edit code first).
4. Implement one candidate at a time; validate bit-exactness, then evaluate the v_b8 gap.
5. Record candidate results + `ncu` evidence; keep changes scoped to the epilogue/pipeline.

## Plan Draft Requirements

`docs/draft.md` should include: the v_b8 baseline + current FC1 gap per bs and how bit-exactness is
validated; the fresh ncu stall breakdown at cap=6 and the per-tile MMA-vs-epilogue model; candidate
directions ranked (above), with what's already ruled out (#2 fast-tanh, #3 deeper-pipeline); the first
concrete step (re-profile, then per-shape block_m / autotuner integration); the exact validation +
evaluation commands; and the evidence to promote/reject (gap-vs-v_b8 + scales-100% + no FC2/plain
regression). Do not start implementation until the draft exists.
