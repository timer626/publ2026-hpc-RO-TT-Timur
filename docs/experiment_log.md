# Experiment Log

Project: TT KV-cache compression for per-example HotpotQA RAG.

Repository state:

- Remote: `https://github.com/aidagroup/publ2026-hpc-RO-TT-Timur.git`
- Branch: `tt-rag-kv`
- Upstream: `origin/tt-rag-kv`
- Initial local review date: 2026-10-28

## 2026-10-28 Repository Review

The `tt-rag-kv` branch is an independent project line rather than a normal
feature branch on top of `origin/main`: `git diff origin/main...origin/tt-rag-kv`
has no merge base. The branch replaces the older notebooks/docs-oriented tree
with a runnable HotpotQA RAG prototype.

Implemented pieces:

- Data/model setup pipeline for `Qwen/Qwen3-8B`, `Qwen/Qwen3-Embedding-0.6B`,
  and `hotpotqa/hotpot_qa`.
- Document construction from HotpotQA contexts and document embeddings saved to
  `datasets/hotpot_qa_documents`.
- Per-example RAG baseline in `experiments/baseline.py`.
- TT-compressed KV-cache baseline in `experiments/baseline_tt.py`.
- Transformers Cache API adapter in `experiments/tt_cache.py`.
- JSONL predictions, JSON summaries, and plots for quality, generation, and
  TT-cache metrics.

Important methodology note: retrieval is restricted to documents indexed under
the current HotpotQA `example_id`. These experiments test generation and
TT-KV-cache behavior under a controlled RAG prompt, not open-corpus retrieval.

Local checks completed:

- `git status --short --branch` is clean on `tt-rag-kv...origin/tt-rag-kv`.
- Python syntax check passed with
  `PYTHONPYCACHEPREFIX=/tmp/codex-pycache python3 -m py_compile config.py main.py metrics.py logging_config.py setup/*.py experiments/*.py`.

Not checked yet:

- `uv sync` dependency installation.
- Actual compatibility of `transformers==5.19.0` Cache API with
  `TTCompressedCache`.
- Dataset/model download.
- GPU smoke runs.

## Compute Choice

MLCore KB note: the available KB resource for this session contains the answer
contract and seamless authorization overview, but does not document comparative
hardware characteristics for `aidahpc` vs `aidahpc2`. CLI observations below are
therefore empirical, from `mlc job ls`, not KB documentation.

Observed active/recent flavors on 2026-10-28:

- `aidahpc`: `h100-2x`, `a100-2x`
- `aidahpc2`: `v100-2x`

Preferred compute for this project: `aidahpc`, ideally H100 first and A100 as a
fallback. Reason: Qwen3-8B, long prompts, and repeated dense TT-cache
reconstruction are memory- and bandwidth-sensitive. `aidahpc2`/V100 is suitable
for small smoke tests or fallback queue capacity, but is not the first choice
for full sweeps.

## Planned Runs

### Stage 0: Environment and API Smoke Test

Goal: prove the code runs before spending GPU time on sweeps.

Configuration:

- Dataset split: reduce to `validation[:10]` if needed for setup speed.
- `baseline --limit 3 --max-new-tokens 16 --top-k 5 --seed 42`
- `baseline_tt --limit 3 --max-new-tokens 16 --top-k 5 --seed 42 --tt-epsilon 0.01 --tt-block-size 16`

Record:

- dependency install status;
- model/dataset download status;
- whether `TTCompressedCache` is accepted by `model.generate`;
- first `summary.json` paths;
- any exceptions and exact stack traces.

### Stage 1: Baseline vs TT Baseline

Goal: compare quality and performance at the default TT setting.

Configuration:

- `limit`: 10, then 50, then 300 if stable;
- `top-k`: 5;
- `max-new-tokens`: 64;
- `seed`: 42;
- TT: `epsilon=0.01`, `block_size=16`.

Primary metrics:

- answer EM/F1;
- retrieval recall/precision;
- mean generation seconds;
- generated tokens/sec;
- peak CUDA memory;
- TT compression ratio;
- TT reconstruction error;
- max TT rank;
- TT decomposition and reconstruction time.

### Stage 2: TT Epsilon Sweep

Goal: estimate quality/compression/error tradeoff.

Configuration:

- `tt-epsilon`: `0.1`, `0.03`, `0.01`, `0.003`, `0.001`;
- `tt-block-size`: 16;
- use the largest stable `limit` from Stage 1.

### Stage 3: TT Block Size Sweep

Goal: estimate dense-tail and recompression tradeoff.

Configuration:

- `tt-block-size`: `4`, `8`, `16`, `32`, `64`;
- use selected `tt-epsilon` from Stage 2.

### Stage 4: Prompt and Generation Length Scaling

Goal: test whether TT storage benefits grow with longer cache.

Configuration:

- `top-k`: `1`, `3`, `5`, `10`;
- `max-new-tokens`: `32`, `64`, `128`, optionally `256`;
- selected TT settings from previous stages.

## Run Records

Add one section per completed run with:

- date/time;
- cluster/project/flavor;
- git commit;
- command;
- output directory;
- status;
- summary metrics;
- notes and follow-up.

## Earlier Compute Reservation

Reserved for this TT RAG KV-cache task on 2026-10-28:

- `aidahpc/hpc0n1`: `h100-2x`, region `ix-m5-sm12`, MLCore job
  `hpc0n1-5qvtti`, later ended.
- `aidahpc/hpc0n2`: `a100-2x`, region `ix-m4-sm4`, MLCore job
  `hpc0n2-9cgm5t`, later ended.

These entries were current at the initial review and are historical.

## 2026-10-28 GPU Runs

The earlier `hpc0n1-5qvtti` and `hpc0n2-9cgm5t` reservations ended. For this
round, `hpc0n1-7wbvdt` and `hpc0n3-wpxhzf` were started in `aidahpc`. Both
expose two H100 80 GB GPUs; MLCore changed the stored `hpc0n3` V100 flavor to
H100 on start. Initial runs used the source at commit `e152dae`; later runs
also include the local BLAS-thread and int8-core changes. Both used the frozen
environment: Python 3.13.16, PyTorch 2.14.1+cu132, Transformers 5.19.0, and
ttpy 2.0.0rc3. Models and data live under `/tmp/tt-rag-kv`; small result
artifacts live under `/qwarium/home/p.a.belozerova/tt-rag-kv-results` and
were copied to `experiments/results/hpc0n1` and `experiments/results/hpc0n3`
locally.

Preparation completed on both nodes. `hpc0n1` prepared `validation[:300]`
(2971 documents). `hpc0n3` first prepared `validation[:10]` (100 documents),
then prepared `validation[:300]` in separate paths (2971 documents), both with
`batch_size=16`. A synthetic two-layer Qwen3 `model.generate` test accepted
both plain and int8 `TTCompressedCache` on CUDA.

Comparable 10-example runs: `top_k=5`, `max_new_tokens=64`, `seed=42`.

| Node | Run | Epsilon | Block | Cores | EM | F1 | Gen s | Ratio | Error |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| hpc0n1 | `baseline_20261028T164801852507Z` | — | — | — | 0.600 | 0.786 | 0.510 | — | — |
| hpc0n3 | `baseline_tt_20261028T164925304393Z` | 0.01 | 16 | float | 0.600 | 0.786 | 20.947 | 0.271 | 0.0052 |
| hpc0n3 | `baseline_tt_20261028T170822587223Z` | 0.01 | 16 | float | 0.600 | 0.786 | 10.651 | 0.271 | 0.0052 |
| hpc0n3 | `baseline_tt_20261028T171138806135Z` | 0.03 | 16 | float | 0.600 | 0.786 | 10.702 | 0.292 | 0.0164 |
| hpc0n1 | `baseline_tt_20261028T165207081483Z` | 0.1 | 16 | float | 0.600 | 0.786 | 39.215 | 0.355 | 0.0564 |
| hpc0n3 | `baseline_tt_20261028T171429381697Z` | 0.1 | 8 | float | 0.600 | 0.786 | 9.549 | 0.382 | 0.0545 |
| hpc0n3 | `baseline_tt_20261028T171429917373Z` | 0.1 | 4 | float | 0.500 | 0.642 | 11.457 | 0.423 | 0.0399 |
| hpc0n3 | `baseline_tt_20261028T171149720979Z` | 0.2 | 16 | float | 0.400 | 0.688 | 10.447 | 0.447 | 0.1136 |
| hpc0n1 | `baseline_tt_20261028T170038451424Z` | 0.3 | 16 | float | 0.300 | 0.516 | 45.245 | 0.559 | 0.1760 |
| hpc0n3 | `baseline_tt_20261028T165852310293Z` | 0.5 | 16 | float | 0.000 | 0.004 | 44.526 | 1.125 | 0.3714 |
| hpc0n3 | `baseline_tt_20261028T171926319941Z` | 0.01 | 16 | int8 | 0.600 | 0.742 | 10.575 | 1.077 | 0.0686 |
| hpc0n3 | `baseline_tt_20261028T172219520511Z` | 0.01 | 8 | int8 | 0.600 | 0.786 | 10.579 | 1.329 | 0.0550 |
| hpc0n3 | `baseline_tt_20261028T172219606077Z` | 0.1 | 8 | int8 | 0.600 | 0.786 | 12.099 | 1.522 | 0.0782 |

The TT runs starting at 16:49, 16:52, 16:58, and 17:00 UTC used the original
BLAS thread defaults; later runs used one BLAS thread by explicit environment
or by the updated `baseline_tt.py`. Several later runs were concurrent on
different GPUs, so compare their latency with that limitation in mind.

The cache ratio is dense payload bytes divided by stored payload bytes. A value
below 1 means the TT payload is larger. Peak CUDA memory is not a measure of
the stored cache alone because attention receives reconstructed dense K/V.

The 300-example baseline on `hpc0n1`, run
`baseline_20261028T164926208761Z`, reached EM 0.4767, F1 0.6313,
Recall@5 0.8700, mean generation time 0.344 s, and peak CUDA memory 15.68 GiB.
On the first 50 examples, baseline EM was 0.500 and F1 was 0.651. Plain TT
with `epsilon=0.01`, block 16, and one BLAS thread reached EM 0.520, F1 0.682,
mean generation time 14.78 s, and cache ratio 0.271 in run
`baseline_tt_20261028T170926290940Z` on `hpc0n1`. A separate 50-example
baseline run on `hpc0n3`, `baseline_20261028T172728449851Z`, matched EM 0.500
and F1 0.651 with mean generation time 0.408 s.

The 50-example int8 comparison used the same first 50 questions, retrieval
settings, and sampling seed. The `hpc0n3` baseline above is the direct
comparison for the two block-8 variants.

| Node | Run | Epsilon | Block | EM | F1 | Gen s | Ratio | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hpc0n1 | `baseline_tt_20261028T172327923949Z` | 0.01 | 16 | 0.460 | 0.589 | 16.090 | 1.077 | 0.0686 |
| hpc0n3 | `baseline_tt_20261028T172823233518Z` | 0.01 | 8 | 0.480 | 0.612 | 11.910 | 1.328 | 0.0550 |
| hpc0n3 | `baseline_tt_20261028T172741484492Z` | 0.1 | 8 | 0.480 | 0.631 | 12.607 | 1.521 | 0.0782 |

All three rows use int8 TT cores. The best observed 50-example tradeoff is
`epsilon=0.1`, block 8, int8: payload compression with a 0.020 absolute F1
drop from baseline. Peak CUDA memory remained 15.45 GiB because the cache is
reconstructed as dense K/V for attention. Generation was about 31 times
slower than the matching baseline.

One-example exploratory TT runs on `hpc0n3` with the same retrieval/generation
settings as above:

| Run | Epsilon | Block | F1 | Generation s | Cache ratio | Mean reconstruction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_tt_20261028T164810863905Z` | 0.1 | 16 | 1.0 | 22.91 | 0.355 | 0.05660 |
| `baseline_tt_20261028T165351800692Z` | 0.1 | 64 | 1.0 | 66.99 | 0.434 | 0.05495 |
| `baseline_tt_20261028T165559539802Z` | 0.5 | 16 | 0.0 | 37.63 | 1.112 | 0.37254 |

Single-example quality values are diagnostic only.

For a full 300-example comparison, both runners now support `--offset` and
`--seed-per-example`. The seed for dataset index `i` is `42 + i`, allowing
nonoverlapping TT shards to reproduce a single run's sampling. The seeded
300-example baseline, `baseline_20261028T174222723131Z`, reached EM 0.4900,
F1 0.6388, and Recall@5 0.8700. Four 75-example TT shards using the selected
int8/block-8/epsilon-0.1 setting completed on the four H100 GPUs.

The first aggregation against a single baseline prepared on `hpc0n1` found
20/300 different retrieved document lists in TT shards prepared on `hpc0n3`.
This came from small embedding differences under different preparation batch
sizes (`60` versus `16`). To remove this confound, the baseline was rerun in
the same four 75-example ranges on the same nodes as the TT shards. The final
aggregation validated **zero** retrieval mismatches and complete, nonoverlapping
coverage of indices 0–299. All eight runs used per-example seeds, `top_k=5`,
and `max_new_tokens=64`. Exact run IDs are in `full_300_comparison.json`.

| Metric | Matched baseline | TT int8, epsilon 0.1, block 8 |
| --- | ---: | ---: |
| Answer EM | 0.4933 | 0.4333 |
| Answer F1 | 0.6413 | 0.5741 |
| Mean generation seconds | 0.3821 | 14.6675 |
| Peak CUDA memory GiB | 15.696 | 15.446 |
| Persistent cache payload ratio | — | 1.521 |
| Mean relative reconstruction error | — | 0.0781 |

The paired TT-minus-baseline difference is -0.0600 EM (95% paired bootstrap
interval [-0.1033, -0.0167]) and -0.0672 F1 (interval [-0.1043, -0.0297]).
These intervals resample examples under one sampling seed policy; additional
generation seeds were not tested. TT was about 38.4 times slower in
generation. The small peak CUDA-memory difference does not indicate a durable
GPU cache-memory saving: TT cores are stored on CPU and dense K/V are
reconstructed on GPU for attention.

Current conclusion: optional int8 quantization makes the TT cache payload
smaller, but this implementation has a measurable quality cost and very high
latency. The next implementation step is to change the compression/reconstruction
scheme before treating it as a practical KV-cache optimization. Longer prompt
and generation sweeps remain to be run after that change.

All results were copied locally before `hpc0n1` and `hpc0n3` were stopped.
Their `/tmp` model and environment files are ephemeral; the small per-run
artifacts are also in the Qwarium persistent home directories.

## 2026-10-28 Follow-up implementation

The 300-example profiling data attributes about 10.08 seconds per example to
TT decomposition and 3.05 seconds to reconstruction. Prompts average 729
tokens, while answers average 7.3 tokens, so prompt prefill dominates. The
cache now accepts a separate `prefill_block_size`; generated-token blocks keep
their original size. An optional `int8_row` mode stores a float32 scale for
each left-rank slice of each TT core to test whether lower quantization error
improves answer quality. The shard aggregator checks the prefill size as part
of TT run compatibility. These are experimental settings; no quality or speed
improvement has been established for them yet.

The next comparison uses `epsilon=0.1`, generated-token block 8, prefill
blocks 128 and 256, and `int8` versus `int8_row` on the first 10 questions.
Both Qwarium environments were recreated, including the pinned Qwen3-8B and
embedding models. A two-layer synthetic Qwen3 CUDA generation test accepted
both quantization modes with separate prefill and generation block sizes.

The following runs use the first 10 questions, `top_k=5`, 64 maximum new
tokens, and seed 42. The first four used `validation[:10]`; the last two used
`validation[:300]`. Retrieved document lists matched across all six runs and
also matched the earlier 10-question baseline. These are exploratory quality
measurements on a small sample.

| Node | Run suffix | Prefill block | Cores | EM | F1 | Gen s | Ratio | Error |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| hpc0n3 | `201929680481Z` | 128 | int8 | 0.500 | 0.686 | 7.426 | 1.877 | 0.0928 |
| hpc0n3 | `201929517476Z` | 128 | int8_row | 0.600 | 0.786 | 7.643 | 1.867 | 0.0560 |
| hpc0n1 | `201952617346Z` | 256 | int8 | 0.400 | 0.582 | 7.423 | 1.961 | 0.0930 |
| hpc0n1 | `201952899454Z` | 256 | int8_row | 0.600 | 0.711 | 7.262 | 1.955 | 0.0555 |
| hpc0n1 | `202323037012Z` | 64 | int8_row | 0.700 | 0.842 | 8.575 | 1.719 | 0.0575 |
| hpc0n1 | `202323390884Z` | 128 | int8_row | 0.600 | 0.711 | 7.595 | 1.867 | 0.0560 |

`int8_row` reduced mean relative reconstruction error compared with `int8`
at the same prefill size. The repeated 128/int8_row runs had different F1
despite identical retrieved document lists, so quality claims require the
larger matched evaluation now in progress.

The first 50-question pair used the same `validation[:300]` preparation on
`hpc0n3` and seed 42 without per-example reseeding. All 50 retrieval lists
matched. The baseline run `baseline_20261028T202312111376Z` reached EM 0.500,
F1 0.6515, and mean generation time 0.392 s. The 128/int8_row TT run
`baseline_tt_20261028T202311591140Z` reached EM 0.540, F1 0.6904,
mean generation time 8.864 s, cache ratio 1.868, and mean relative
reconstruction error 0.0559. The quality difference is exploratory; TT is
still 22.6 times slower on this pair. A 300-question matched comparison with
per-example seeds is in progress.

The second 50-question pair used the same protocol on `hpc0n1`. Baseline
`baseline_20261028T202555167463Z` reached EM 0.500, F1 0.6515, and
0.398 s mean generation time. TT with prefill block 64 and `int8_row`, run
`baseline_tt_20261028T202554756883Z`, reached EM 0.580, F1 0.7311,
9.645 s mean generation time, cache ratio 1.720, and error 0.0574. All 50
retrieval lists matched. This variant is slower and stores more data than
prefill block 128; its higher 50-question F1 needs a larger confirmation.

### Matched 300-question follow-up: prefill 128

Four 75-question TT shards with `epsilon=0.1`, generated block 8, prefill
block 128, and `int8_row` completed on the four H100 GPUs. Four baseline
shards were prepared and run on the same nodes with the same dataset, retrieval,
generation settings, and per-example seeds. The aggregator verified complete,
nonoverlapping indices 0–299 and **zero retrieval mismatches**. Exact run IDs
and paired intervals are in `full_300_int8_row_prefill128_comparison.json`.

| Metric | Matched baseline | TT prefill 128, int8_row |
| --- | ---: | ---: |
| Answer EM | 0.4900 | 0.4767 |
| Answer F1 | 0.6410 | 0.6218 |
| Mean generation seconds | 0.3523 | 8.6981 |
| Peak CUDA memory GiB | 15.696 | 15.446 |
| Persistent cache payload ratio | — | 1.871 |
| Mean relative reconstruction error | — | 0.0560 |

The paired TT-minus-baseline difference is -0.0133 EM (95% paired bootstrap
interval [-0.0433, 0.0167]) and -0.0193 F1 (interval [-0.0446, 0.0046]).
The intervals include zero, so this run does not establish a quality difference
at the chosen seed policy. TT remains 24.7 times slower than baseline in
generation; its stored payload is smaller, while peak GPU memory remains near
the dense-cache baseline because attention receives reconstructed K/V.

The 50-question prefill-64 result warranted the same 300-question evaluation
with the four baseline shards above. All four TT shards completed; the
aggregator verified complete, nonoverlapping indices and **zero retrieval
mismatches**. Exact run IDs and paired intervals are in
`full_300_int8_row_prefill64_comparison.json`.

| Metric | Matched baseline | TT prefill 64, int8_row |
| --- | ---: | ---: |
| Answer EM | 0.4900 | 0.4800 |
| Answer F1 | 0.6410 | 0.6267 |
| Mean generation seconds | 0.3523 | 9.6517 |
| Peak CUDA memory GiB | 15.696 | 15.446 |
| Persistent cache payload ratio | — | 1.722 |
| Mean relative reconstruction error | — | 0.0573 |

The paired TT-minus-baseline difference is -0.0100 EM (95% paired bootstrap
interval [-0.0367, 0.0167]) and -0.0143 F1 (interval [-0.0378, 0.0079]).
The intervals include zero. Generation remains 27.4 times slower than
baseline. In a direct paired comparison of both TT runs, prefill 64 minus
prefill 128 is +0.0050 F1 (95% paired bootstrap interval [-0.0095, 0.0205])
with identical retrieval lists. This small quality difference is unresolved
while prefill 128 is faster and compresses the persistent payload more.

Compared descriptively with the earlier 300-question int8/block-8 TT run,
prefill 128 plus `int8_row` raised F1 from 0.5741 to 0.6218 and reduced mean
generation time from 14.6675 to 8.6981 s. Retrieval lists across those two
TT experiments matched on 287 of 300 questions. Their matched baselines were
close (F1 0.6413 versus 0.6410). This is a useful implementation improvement,
but the cache still needs a design that avoids repeated CPU TT-SVD and full
dense reconstruction before it can be considered a practical latency or GPU
memory optimization.

## TODO: follow-up after the matched 300-question comparisons

- [ ] **1. Repeat quality measurements across generation seeds.** Run the
  matched baseline and the selected TT setting (`epsilon=0.1`, generated block
  8, prefill block 128, `int8_row`) on the same 300 questions with base seeds
  101 and 202, in addition to the completed seed 42. Use `--seed-per-example`
  and the same four 75-question ranges; verify identical retrieved documents
  within every pair. Report EM/F1 and TT-minus-baseline differences for each
  seed, then summarize variability across the three seeds. Resample question
  indices jointly across seeds for a pooled paired interval, so repeated
  answers to one question are not treated as independent examples. Save raw
  predictions, summaries, and comparison files in Git.
- [ ] **2. Redesign the cache path around the measured bottlenecks.** Profile
  prefill TT-SVD, block reconstruction, transfer, and attention separately.
  Prototype a path that does not reconstruct the entire dense history at every
  decoding step; measure peak GPU memory and tokens per second against the
  current implementation and dense baseline. Decide whether the research goal
  favors compressed CPU storage with chunked attention or a TT-aware attention
  kernel before implementing the larger change.
- [ ] **3. Test where compression can amortize.** After the cache-path change,
  run a small matched pilot with longer prompts (about 2k, 4k, and 8k tokens)
  and longer continuations (64, 256, and 512 tokens, within model limits).
  Record end-to-end latency, decode throughput, persistent payload, peak GPU
  memory, and answer quality. Expand only where the pilot shows a useful
  latency or memory tradeoff.
