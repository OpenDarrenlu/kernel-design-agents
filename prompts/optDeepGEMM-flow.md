# Kernel Design Agents Flow Prompt — Optimize the w4a8 SM100 Unpadded GEMM at Large Batch

You are working in a task implementation workspace. Target: the **w4a8 (fp8 activation × fp4 weight)
SM100 m-grouped *unpadded* 1D1D GEMM** — the shared launcher
`sm100_m_grouped_fp8_fp4_gemm_contiguous_unpadded_1d1d`
(`DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:257-351`) and the device kernel it
generates (`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`). This launcher backs both
MoE projections in the bench: the plain GEMM and the fused-SiTU FC1 ride the same device kernel.

**The goal is to find headroom at large batch (per-call M = bs·K, bs ≥ 2048) and realize it.** This is
a *big-picture, bottleneck-first* brief — deliberately NOT another epilogue-overlap pass (that angle
is exhausted, see below). Do not pre-commit to a lever. **Profile first, identify the dominant
limiter, then choose the optimization. Step 1 is measurement, not a code change.**

## Hard constraint (read before proposing anything)

**Do NOT restructure the operator.** Specifically, keep the **persistent scheduler** and the **warp
specialization** intact: warp0 = TMA load, warp1 = UMMA (tcgen05) issue, warp2 = UTCCP scale-factor
transpose, the epilogue store warpgroups, and the 2-stage TMEM accumulator handoff
(`tmem_full`/`tmem_empty`). Wins must come from *within* this structure — tiling/stage/SF/scheduling
knobs, load or SF pipeline efficiency, wave/tail quantization, swizzle/TMA/multicast efficiency,
small device-side micro-optimizations — not from a new kernel shape or a different parallelization
scheme. Also off-limits: changing the GEMM math, the fp4 weight interleave, the SFA/SFB scale layout,
or the (d_fp8, scale) GEMM2 contract.

## Scope: the two real shapes (per-GPU, tp=8)

MoE config E=384, H=7168, I=256, K=8. Both GEMMs go through the unpadded launcher; `emp = bs·8/384`:

| GEMM | role | M | N | K | per-expert tokens (emp) at bs 2048 / 16384 / 131072 |
|---|---|---|---|---|---|
| FC1 | up/gate (n=2I) | bs·8 | 512 | 7168 | 43 / 341 / 2731 |
| FC2 | down (n=H) | bs·8 | 7168 | 256 | 43 / 341 / 2731 |

FC1 is N-skinny + K-deep (many K-blocks ⇒ much SF traffic + a long MMA main loop). FC2 is N-wide +
K-shallow and writes M×7168 bf16 (≈56× FC1's fp8 output) so it is plausibly store/DRAM-bound. **Both
are in scope — let the profile say which carries the headroom at bs ≥ 2048;** don't assume.

## Already landed / autotuned (do NOT redo)

Recent commits (`git log`) built the autotuner + the unpadded path + epilogue parallelism:
- `block_m`, `load_k_mult` (bigger TMA K-loads per stage), and `num_store_wg` (epilogue store
  warpgroups) are **already searched by the embedded FlashInfer autotuner** per m-bucket
  (`deep_gemm/autotune/_mgc.py`: `DGMGCURunner` for the plain op, `DGMGCUSituRunner` for the fused
  FC1). `cluster_n`/`num_stages` are heuristic. So "sweep block_m" / "tune store warpgroups" is
  done — a new lever must be *outside* this existing search, or a smarter heuristic, not a re-sweep.
- The heavy fused epilogue runs up to 6 store warpgroups (`get_num_store_wg` gated on
  `desc.heavy_epilogue`).

## Dead ends from the prior task (do NOT re-tread without new evidence)

The sibling brief `mmaOverlapEpilogu-flow.md` chased "overlap the fused-FC1 epilogue with the MMA."
Outcome, with evidence:
- **Decoupling TMEM-drain warps from activation warps (a flat per-chunk CD ring) is SLOWER** — tie-to-
  −14% vs the coupled 6-warpgroup store across bs 2048–131072; ncu showed it *lowers* SM throughput.
  Built, validated bit-exact, measured, then **deleted**. The fused FC1 is **activation-latency-bound**
  (ncu: TC ≈ 51%, top stall = the activation MUFU/FFMA dependency chain), not MMA-throttle-bound, so
  the coupled epilogue already maximizes activation parallelism. Don't re-open epilogue↔MMA overlap.
- Faster-tanh and deeper `kNumEpilogueStages` (2→3) were also tried and reverted (no win on this
  shape; 512-col TMEM ceiling forces small block_m).
- **Conclusion that motivates THIS task:** the *epilogue* is understood and tapped out. The open
  question is the **GEMM core** — MMA main loop, A/B/SF load pipeline, and tile scheduling — at large
  batch. That is what to profile now.

## Task Contract

- Task name: `find + close the dominant large-batch (bs>=2048) bottleneck in the w4a8 SM100 unpadded 1D1D GEMM`
- Objective: `profile the unpadded fp8xfp4 GEMM (FC1 and FC2 shapes) at bs in {2048,8192,32768,131072}, identify the single dominant limiter at large batch, and realize a measurable latency win WITHOUT touching the persistent / warp-specialized structure.`
- Correctness requirements: `bit-exact / within-tolerance vs the existing references. tests/test_fp8_fp4_unpad.py (block_m grid + situ-fused + moe-4modes) pass; bench_situ.py --backend dg_unpadded correctness sweep stays within current tolerance; partial tiles, tiny M, cluster_n in {1,2} still correct.`
- Performance target: `a reproducible per-GEMM (kineto fc1_us / fc2_us) speedup at bs>=2048 vs the current committed baseline (bench_situ_tp8_dgunpadded.csv), with no regression at bs<2048 and no regression on the plain GEMM / FC2 if FC1 is the target (and vice versa). Quantify against the ncu roofline (e.g. "+X% tensor-core active" or "-Y% DRAM stall") so the win is mechanistic, not noise.`
- Allowed approaches: `within the persistent + warp-spec structure only -- e.g. SF (scale-factor) load/transpose pipeline efficiency, A/B TMA load + multicast/cluster efficiency, pipeline-stage depth vs smem budget, wave/tail quantization & tile scheduling across unpadded per-expert segments (scheduler/gemm.cuh), swizzle / L2 / TMA-descriptor tuning, smarter autotuner heuristics, and small device micro-opts on the hot path. Knobs live in csrc/jit_kernels/heuristics/sm100.hpp + config.hpp; the device kernel + scheduler are in deep_gemm/include/deep_gemm/{impls,scheduler}/.`
- Validation: `cd DeepGEMM && DG_NO_AUTOTUNE=1 PYTHONPATH=$PWD python3 tests/test_fp8_fp4_unpad.py  (expects PASS on block_m grid + situ-fused + moe-4modes).`
- Evaluation: `cd flashinfer && PYTHONPATH=<DeepGEMM> python3 bench_situ.py --backend dg_unpadded --bs sweep --bench --iters 3 --warmup 1 --tp 8 --ep 1 --E 384 --H 7168 --I 2048 --K 8 --csv <out>.csv  -> compare fc1_us/fc2_us/total_us at bs>=2048 vs bench_situ_tp8_dgunpadded.csv. PLUS the ncu breakdown from Step 1 to show the limiter moved.`
- Promotion: `(1) correctness retained; (2) a bs>=2048 fc1_us or fc2_us improvement that reproduces across runs (the 3-iter bench is noisy at small bs -- confirm large-bs wins with a warm >=20-iter micro-bench or ncu), traceable to the profiled bottleneck; (3) no regression elsewhere; (4) persistent + warp-spec structure untouched.`

## Step 1 (MANDATORY, do this before any code) — find the bottleneck

Capture a fresh ncu breakdown of the **GEMM core** (not just the epilogue) at large batch. Run the
fused FC1 and the plain FC2 separately, at bs = 16384 and 131072, warm (skip the first launches):

```
ncu -f --set @roofline \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --section SchedulerStats --section WarpStateStats --section LaunchStats --section Occupancy \
    --kernel-name regex:"fp8_fp4_gemm" --launch-skip 10 --launch-count 1 \
    -o gemm_<shape>_<bs> --clock-control base \
    python3 <single-shape driver>
```

Then **classify the dominant limiter** and let it pick the lever — examples of what the data could say:
- **Tensor-core-bound & near peak** → little GEMM headroom; the win (if any) is elsewhere (FC2 store,
  or wave/tail).
- **Tensor core NOT saturated + top stall = memory/`long scoreboard`** → A/B/SF load pipeline is the
  limiter ⇒ look at stage depth vs smem, `load_k_mult`, multicast/cluster, TMA/swizzle, SF traffic.
- **Top stall = barrier / `tmem`/MMA-issue wait with low DRAM** → latency/dependency in the main loop
  or SF transpose (warp2 UTCCP) not keeping up ⇒ SF pipeline.
- **Low achieved occupancy / many idle SMs in the last wave** (likely at the unpadded per-expert
  boundaries with imbalanced emp) → wave/tail quantization ⇒ tile scheduling / block_m-vs-emp.
- **DRAM-bound** (expected for FC2's M×7168 bf16 store) → confirm it's intrinsic before chasing it.

Write the breakdown (per shape, per bs: SOL TC%/DRAM%, achieved vs peak, top-3 stalls, occupancy,
last-wave utilization) into `docs/draft.md` and choose the lever from THAT, not from this list.

## Candidate directions (profile-driven — ranked by *a priori* likelihood, not pre-decided)

Use these only as hypotheses to confirm/reject against Step 1:
1. **SF (scale-factor) pipeline.** w4a8 needs per-block fp4-weight scales loaded + UTCCP-transposed
   (warp2) every K-block; at FC1's K=7168 that's many SF ops. If warp2/SF is on the critical path,
   improving SF load batching/reuse or transpose throughput helps — and it's w4a8-specific (a generic
   fp8 GEMM wouldn't have it).
2. **Wave / tail quantization at the unpadded per-expert boundaries.** At large bs with imbalanced
   routing, per-expert tiles don't fill the last wave evenly → idle SMs. A scheduling-side fix
   (tile ordering / block_m chosen vs the emp distribution; cf. the FlashInfer device-side
   CTA→expert schedule idea) can raise last-wave utilization without touching warp roles. Check
   `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`.
3. **Load-pipeline depth & multicast efficiency.** Confirm `num_stages` is smem-optimal at large
   block_m and that cluster_n=2 multicast is actually halving A/B traffic; a better stage/cluster
   heuristic (not just the existing block_m/load_k_mult search) could lift a memory-stalled main loop.
4. **FC2 store path** (only if Step 1 shows FC2 is the bigger opportunity and NOT purely intrinsic
   DRAM): store coalescing / swizzle / TMA-store staging for the M×7168 bf16 write.

Non-goals: no persistent/warp-spec restructure; no GEMM-math, weight-interleave, or SF-layout change;
no re-running the existing block_m/load_k_mult/num_store_wg autotune as if it were new.

## Measurement harness

- Correctness: `tests/test_fp8_fp4_unpad.py` (block_m grid, situ-fused, moe-4modes autotune/cache).
- End-to-end / kineto: `flashinfer/bench_situ.py --backend dg_unpadded --bench` → `fc1_us`, `fc2_us`,
  `total_us` per bs; baseline = `bench_situ_tp8_dgunpadded.csv` (current committed),
  `bench_situ_tp8_dgunpadded_v_b8.csv` (pre-fusion un-fused reference). The `--iters 3 --warmup 1`
  bench is noisy at small bs — for a clean large-bs read use a warm ≥20-iter CUDA-event micro-bench
  on the single op (`override_block_m` / `override_num_store_wg` / `override_load_k_mult` are exposed
  to pin a config; `DG_NO_AUTOTUNE=1` for a deterministic heuristic read), or ncu.
- Profiling: ncu per Step 1 + the repo's profiling convention in `CLAUDE.md`/`AGENTS.md`.

## Workflow

1. Read the unpadded launcher + device kernel + scheduler + the SF path; re-derive, per shape, the
   large-batch per-tile cost (MMA cycles vs A/B/SF load bytes vs store bytes).
2. **Step 1 profile first** — capture the ncu breakdown at bs 16384 & 131072 for FC1 and FC2; classify
   the dominant limiter. No code edits yet.
3. Write `docs/draft.md` (baseline numbers + the profile + the chosen bottleneck + ranked candidates).
4. Implement ONE candidate at a time, inside the persistent/warp-spec constraint; validate
   correctness, then evaluate the large-bs latency + re-profile to confirm the limiter moved.
5. Record candidate results + ncu evidence; keep changes scoped and the structure intact.

## Plan Draft Requirements

`docs/draft.md` must include: the baseline `fc1_us`/`fc2_us` at bs ≥ 2048 and how correctness is
validated; the **Step 1 ncu bottleneck breakdown** (per shape/bs: SOL, top stalls, occupancy,
last-wave util) and the per-tile cost model; the chosen dominant limiter with the evidence for it;
candidate levers ranked *from the profile* (with the prior dead ends — epilogue overlap / decouple /
fast-tanh / deeper-pipeline — explicitly excluded); the first concrete step; the exact validation +
evaluation commands; and the promotion evidence (large-bs win that reproduces + limiter moved + no
regression + structure untouched). **Do not start implementation until the draft exists and names the
bottleneck.**
