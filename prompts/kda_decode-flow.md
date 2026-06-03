# Kernel Design Agents Basic Flow Prompt

You are working in a task implementation workspace. Your job is to produce the best correct implementation for the task described below.

## Task Contract

- Task name: `optimize run_batched_tma & run_kda_decode_v2`
- Objective: `optimize run_batched_tma & run_kda_decode_v2 kernel latency`
- Correctness requirements: `fused_conv_recurrent_kda_decode` output (`o` and `ht`) must match the Triton reference backend within `assert_close` tolerance of 0.01 (RMSE-style). Must support shapes B∈[1,256], H/HV∈[2,32], K=V=128, KERNEL_WIDTH=4, with `use_onorm=True`, `use_lower_bound=True`, `use_beta_sigmoid_in_kernel=True`, `transpose_state_layout=True`. Output must be free of NaN/Inf for all valid inputs.
- Performance or quality target: On the benchmark shape matrix (B∈[1,32] powers of 2, H∈{2,12}, K=V=128), TMAversion latency must remain below the Triton baseline across all shapes, and achieve ≥ 1.30× speedup on the critical large-batch shapes (B=32/64/128, H=12). Target B=32～128 H=2 latency optimized to above 1.4x（其他不能劣化，当前版本优化数据参考 @prof_v5.log）.
- Allowed implementation approaches: Modify `fla/ops/kda/backends/cutedslkda.py` only — specifically `tma_pipeline_kernel` (CuTeDSL kernel body) and `run_batched_tma` (host launch wrapper). Not allowed: changes to PyTorch dispatch layer, Triton kernels, or `naive.py`.
- Validation command: `pytest x4/tests/test_cutedslkda_recurrent.py::test_cutedslkda_conv_decode_dispatch_tma -v` & `pytest x4/tests/test_cutedslkda_recurrent.py::test_cutedslkda_conv_decode_dispatch_v2 -v`
- Evaluation command: `pytest x4/tests/test_cutedslkda_recurrent.py::test_cutedslkda_conv_decode_cudagraph_benchmark -v -s`
- Promotion criteria: (1) `test_cutedslkda_conv_decode_dispatch_tma` & `test_cutedslkda_conv_decode_dispatch_v2` passes on all parameterized shapes with `assert_close("o_dispatch_tma", ref_o, tri_o, 0.01)` & `assert_close("o_dispatch_v2", ref_o, tri_o, 0.01)`. (2) CUDAGraph benchmark shows no regression vs Triton on any shape, and ≥ 1.40× speedup on B=32/64/128 H=2. (3) No new NaN/Inf failures in `test_kda_decode.py` or `test_kda_naive.py`.


## Workflow

1. Read the repository structure, existing implementation, tests, and task documentation.
2. Identify the baseline behavior and the validation path.
3. Research only the references needed for this task.
4. Write an implementation-plan draft to `docs/draft.md`.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time.
7. Run validation after each meaningful candidate.
8. Record candidate results, parent relationships, and evidence in the workspace.
9. Keep the final change scoped to the task contract.

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- The current baseline and how it is validated.
- The main risks and unknowns.
- Candidate implementation directions ranked by expected value and risk.
- The first concrete implementation steps.
- The exact validation and evaluation commands to run.
- The evidence required to promote, revise, or reject a candidate.

Do not start implementation until the draft exists.
