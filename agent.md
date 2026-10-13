# agent.md: GPU Scheduling Data Plane and Simulator ("gsdp")

**Audience:** a coding agent building this repository, and the humans reviewing its work.

**Tags used throughout**
- `[V]` verified from a source fetched while writing this file (Slurm docs, the survey excerpt, the Alibaba v2020 README).
- `[M]` recalled from memory. Verify before relying on it, and record the result in section 17.
- `[A]` a design decision or modeling assumption. Change it when you have data.

## Quick start for the coding agent

1. Read sections 0 (rules), 3 (architecture), 4 (data model) and 14 (milestones) first. Skim the rest as needed.
2. Create the repository layout from section 13 and complete milestone M0 before anything else.
3. Write `config/assumptions.yaml` before writing any simulator code (section 8.5).
4. Do not skip an acceptance test in section 14. If one cannot pass, write the reason in `QUESTIONS.md` and stop at that milestone.
5. Any `[M]` item you depend on must be checked against the real system or its documentation, and the outcome recorded in section 17.

---

## 0. Rules of engagement

1. **No fabricated measurements.** Every metric value carries a `quality` tag: `measured`, `derived`, `estimated`, `simulated` or `unavailable`. Never emit a silent zero for a missing value.
2. **Origin labels.** Every output file and table carries `data_origin` in {`slurm_live`, `alibaba_replay`, `synthetic_sim`}. Simulated results are never presented as real results.
3. **Slurm is read-only by default.** Anything that changes cluster state (`sbatch`, `scancel`, `scontrol update/hold/release`) requires `allow_enforce: true` in config and the `--enforce` flag. The default mode is shadow (section 5.3).
4. **Interface parity.** The simulator and the Slurm adapter emit identical canonical events. Policies, the feature builder and the ledger must not be able to tell which source they are reading.
5. **Point-in-time correctness.** A feature computed for decision time `t` may only use information with timestamp `<= t`. Outcomes (end time, final utilization, runtime) live in separate label tables. Tests enforce this (section 9).
6. **Explicit units in column names** (`mem_gib`, `duration_s`, `req_cpu_cores`). Convert at ingestion, never downstream.
7. **No magic numbers.** Every simulator parameter lives in `config/assumptions.yaml` with a status (`assumed`, `calibrated`, `verified`) and a source.
8. **Probe, don't assume.** Metrics (DCGM fields, NVML calls) and Slurm fields are discovered by a capability probe. Missing capabilities degrade to `unavailable`, never to a guess.
9. **Determinism.** Same config plus same seed gives byte-identical outputs.
10. **Privacy.** Hash user and account names at ingestion with a salt kept in local config. Never write raw names to logs or exports.
11. **Docs win.** If a document and this file disagree, trust the document and update section 17.
12. **Ask, don't guess.** If an unresolved item in section 16 blocks a decision, record it in `QUESTIONS.md` and continue with the default stated here.

---

## 1. Context in plain words

The wider project is a quantum-inspired, interference-aware GPU scheduler for multi-tenant clusters that use NVIDIA Multi-Instance GPU (MIG) partitioning. Three pieces already exist as separate work:

- `formulation.md`: the QUBO/Ising formulation of the assignment problem (the optimizer policy in M8 comes from here).
- `isolation_sim.py`: a working cross-tenant isolation check (the rules in section 10 come from here).
- `profiler_plan.md` and `extraction_plan.md`: the profiler design and the mapping of the 22 required scheduler-model fields to sources.

A teammate is training a scheduling model on the Alibaba trace. That trace is training data only. To schedule in real time, every input must come from live sources. This repository is the **data plane** that produces those inputs, feeds them to a scheduler, and, until a real cluster is available, simulates the same data.

**Non-goals:** this repo does not implement the quantum optimizer's mathematics beyond calling it as a policy, does not train the teammate's model, and does not modify Slurm internals (no C plugins in v1).

## 2. Goal and scope

Deliver one system with two interchangeable data sources:

| Source | Used for | Status |
|---|---|---|
| Slurm adapter plus node agents | The real system | Built in M5 and M6, validated in M9 |
| Discrete-event simulator | Development, testing, policy comparison, counterfactuals | Built in M1 and M2 |
| Alibaba v2020 replay | Training data and realism checks for the simulator | Built in M3 |

It must provide, for every scheduling decision:
1. The 22 required fields (section 4), correct as of the decision time.
2. Richer profiler telemetry (section 6).
3. A hard isolation check (section 10).
4. A stable policy interface so any scheduler (heuristic, ILP, QUBO, learned) plugs in unchanged (section 11).

---

## 3. Architecture

```
REAL PATH                                      FALLBACK PATH
Slurm (slurmctld / slurmdbd)                   Simulator (discrete-event)
  | sacct / scontrol (adapter)                   | emits the same canonical events
Node agents (profiler) on GPU nodes              | plus synthetic telemetry
  | telemetry samples                            |
  +----------------------+-----------------------+
                         v
   Canonical event log (append-only; Parquet + DuckDB)
   tables: jobs, nodes, telemetry, decisions
                         v
   Cluster-state ledger (reserved vs measured, as-of queries)
                         v
   Feature builder (point-in-time) ---> Predictor (teammate's model, optional)
                         v
   Isolation engine (hard constraints)
                         v
   Policy (FIFO ... backfill ... ILP ... QUBO ... variational)
                         v
   Executor: SimExecutor | SlurmExecutor (shadow -> advisory -> enforce)
                         v
   Decision log ---> Evaluation harness (metrics, CIs, reports)
```

| Component | Responsibility |
|---|---|
| Source (slurm / sim / alibaba) | Produce canonical events: job lifecycle, node inventory, telemetry |
| Event log | Single source of truth. Supports record and replay of any live session |
| Ledger | Maintains reserved resources per node from events. Answers "what was free at time t" |
| Feature builder | Builds model and policy inputs using only information available at `t` |
| Predictor | Optional plug-in returning predicted runtime and demand with uncertainty |
| Isolation engine | Rejects placements that put different tenants on non-isolated hardware |
| Policy | Maps (pending jobs, cluster view) to placements |
| Executor | Applies placements (simulated, or via Slurm in enforce mode) |
| Evaluation | Computes metrics with confidence intervals, compares policies and seeds |

**Feed interface** (what the scheduler and the teammate's model consume):

```python
class Feed(Protocol):                     # illustrative sketch, not tested
    def cluster_view(self, as_of: Timestamp) -> ClusterView: ...
    def pending_jobs(self, as_of: Timestamp) -> list[JobView]: ...
    def candidate_features(self, job_id: str, as_of: Timestamp) -> FeatureFrame: ...
```

Expose it first as a Python API, then as an optional local HTTP service (`/state`, `/features?job_id=`). Latency budget for one decision epoch: 50 ms for a few hundred candidates `[A]`.

Two loops run independently: an asynchronous **collection loop** (polls Slurm, receives node telemetry, appends to the log) and an event-triggered **decision loop** (on job arrival, job end, or a periodic tick).

---

## 4. Canonical data model

### 4.1 Conventions
- Timestamps: UTC epoch seconds (float). Convert Slurm local times at ingestion `[A]`.
- Memory: GiB (2^30 bytes). Slurm reports MiB `[V]`; Alibaba says "GB" without defining it, so treat GB as GiB and record that assumption `[A]`.
- GPU request: `req_gpu_frac` (1.0 is one full GPU, fractions allowed because the Alibaba trace uses percent shares `[V]`), plus `req_gpu_count` (integer ceiling) and `req_mig_profile` (nullable string).
- CPU: cores (float). Alibaba `plan_cpu` is percent of a core (600 is six cores) `[V]`, so divide by 100.
- Canonical row granularity: one allocation (Slurm job, or Alibaba instance), with a `job_nodes` child table for multi-node jobs.

### 4.2 Tables

**`jobs`** (one row per allocation; fields 1 to 10)

| Column | Field # | Notes |
|---|---|---|
| `job_id`, `instance_id` | 1 | Alibaba: `inst_id` is the job id, `worker_name` the instance `[V]`. Slurm: `JobID` with array (`A_B`), heterogeneous (`A+B`) and step forms handled `[V]` |
| `submit_ts` | 2 | Slurm `Submit` `[V]`; Alibaba job table `start_time` `[V]` |
| `start_ts` | 3 | Slurm `Start` `[V]`; Alibaba task or instance `start_time` `[V]` |
| `end_ts` | 4 | Nullable while running |
| `status` | 5 | Enum: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, NODE_FAIL, OOM, OTHER. Map Alibaba Waiting, Running, Terminated, Failed to PENDING, RUNNING, COMPLETED, FAILED `[V]` |
| `node_id` | 6 | In `job_nodes` for multi-node allocations |
| `req_gpu_frac`, `req_gpu_count`, `req_mig_profile` | 7 | |
| `req_cpu_cores` | 8 | |
| `req_mem_gib` | 9 | |
| `req_gpu_type` | 10 | Nullable (no requirement) |
| `tenant_id` (hashed), `partition`, `qos`, `priority`, `timelimit_s` | n/a | Slurm `Account`, `User`, `QOS`, `Partition`, `Timelimit` `[V]` |
| `data_origin` | n/a | See rule 2 |

**`nodes`** (fields 11 to 15; slowly changing, with `valid_from`/`valid_to` so MIG reconfiguration is recorded)

`node_id` (11), `gpu_model` (12), `gpu_count` (13), `cpu_cores` (14), `mem_gib` (15), `mig_enabled`, `mig_layout` (per GPU, list of profile names), `sharing_mode` per device in {`exclusive`, `mig`, `time_slice`, `shard`, `mps`}.

**`telemetry`** (long format; fields 16 to 18 plus extras)

`ts`, `node_id`, `device_id`, `mig_instance_id` (nullable), `job_id` (nullable), `metric`, `value`, `unit`, `quality`, `source`.
Required metrics: `gpu_util_pct` (16), `gpu_mem_used_gib` (17), `cpu_util_cores` (18). Extras from `profiler_plan.md`: `sm_active`, `tensor_active`, `fp16_pipe`, `fp32_pipe`, `dram_active`, `mem_bw_gbps` (derived), `pcie_tx/rx`, `nvlink_tx/rx`, `power_w`, `temp_c`, `sm_clock_mhz`, `throttle_reasons`, `host_mem_gib`, `net_tx/rx`.

**`job_usage_summary`** (derived)
Lifetime-to-date average and max per metric, plus windowed statistics. The lifetime-average view exists for parity with the Alibaba sensor table, which stores lifetime averages and not time series `[V]`.

**`cluster_snapshots`** (fields 19 to 22; written at every job start and every decision epoch)

`ts`, `node_id`, `avail_gpu_frac` (19), `avail_cpu_cores` (20), `avail_mem_gib` (21), `running_jobs` (22), plus `measured_free_*` (capacity minus profiler-measured usage) so reserved and actual can be compared.

**`features`** (one row per job and candidate node; section 9) and **`labels`** (outcomes, separate table). **`decisions`**: `decision_id`, `ts`, `policy`, `job_id`, `chosen_node/device/profile`, `candidates`, `scores`, `isolation_checks`, `mode` (shadow, advisory, enforce).

---

## 5. Slurm adapter

### 5.1 Source map for the 22 fields

| Fields | Source | Status |
|---|---|---|
| 1 to 6 | `sacct -P` with `--format=JobID,User,Account,QOS,Partition,Submit,Eligible,Start,End,State,NodeList` | Field names `[V]`. Flags `-X` (allocations only) and `-S/-E` (time window) `[M]` |
| 7 to 9 | `ReqTRES` (GPU count appears as `gres/gpu`), `ReqCPUS`, `ReqMem` | Field names `[V]`. `ReqMem` suffix semantics `[M]`: the docs I fetched do not explain them, so parse defensively |
| 10 | GPU type from the `ReqTRES` or `--gres` string, or constraints | `[M]` |
| 11 to 15 | `scontrol show node` (`-o` or `--json`) | `RealMemory`, `AllocMem`, `FreeMem` documented in MiB `[V]`. `CPUTot`, `CPUAlloc`, `CfgTRES`, `AllocTRES`, `Gres` were not in the man-page text I fetched `[M]`: **probe them on your cluster** |
| 16, 17 | `sacct` `TRESUsageIn*` for `gres/gpuutil` and `gres/gpumem` | Needs `AccountingStorageTRES=gres/gpu`, NVIDIA with `AutoDetect=nvml`. **Not available for MIG devices** `[V]` |
| 18 | `AveCPU`, `MaxRSS`, `AveRSS` | Field names `[V]` |
| 19 to 21 | Computed by the ledger (section 7); cross-check against `scontrol show node` allocation fields | `[A]` |
| 22 | Count of running allocations per node from the ledger; cross-check with `squeue -w <node> -t R` | `[M]` |

### 5.2 Slurm facts that constrain the design `[V]`
- **MIG support** exists since Slurm 21.08. MIG instances appear as individual GPUs with cgroup isolation. Partitioning must be done beforehand, and **dynamic partitioning is not supported**. A GRES type is a substring of the MIG profile string, for example `gpu:a100_3g.20gb:2`.
  Consequence: "dynamic MIG allocation" needs an out-of-band reconfiguration workflow (drain the node, reconfigure MIG, update Slurm's view, resume). Do not design this alone; it needs the cluster admin. Model its cost in the simulator (section 8.8).
- **GPU accounting does not cover MIG.** NVML reports no utilization for MIG devices, so Slurm records no `gpuutil` or `gpumem` for them. Per-instance telemetry must come from the node agent (DCGM), which is a core reason the profiler exists.
- **Shard** lets jobs share a GPU without fencing processes from each other. **MPS** allows only one user's server at a time, so different users are time-sliced. A GPU is allocated as either `gpu` or `shard`/`mps`, not both. These are the non-isolated modes the isolation engine must police.
- `--gpus*` options require the `select/cons_tres` plugin.
- `scontrol` supports `--json`, `--yaml` and `-o` (one line per record). `sacct -P` gives pipe-delimited output.
- Default timestamp format is `YYYY-MM-DDTHH:MM:SS`. Set `SLURM_TIME_FORMAT` explicitly in the collector.
- `AutoDetect=nvidia` (Slurm 24.11) does not detect MIG devices.

### 5.3 Modes
| Mode | Behavior |
|---|---|
| `shadow` (default) | Slurm schedules as usual. The system collects data and logs what its policy would have decided. Evaluate by re-simulating the logged history under the new policy (section 8), never by assuming Slurm would have behaved the same |
| `advisory` | Writes recommendations for humans or a wrapper script |
| `enforce` | Applies decisions. Option A (preferred): the system is the submission front end and calls `sbatch --nodelist/--gres/--constraint` `[M]`. Option B: submit with `--hold`, then `scontrol release` or `scontrol update` on pending jobs `[M]`. A Slurm scheduler plugin is out of scope for v1 |

### 5.4 Collection cadence `[A]`
- Lifecycle: incremental `sacct` poll every 30 to 60 s using a cursor (`--starttime` = last seen), dedupe on job id plus state.
- Node state: `scontrol show node` every 10 to 30 s. Do not poll more often than needed, and ask the admin about slurmctld load.
- If the admin agrees, prefer Prolog/Epilog hooks for exact start and end events `[M]`.

### 5.5 Capability probe (`gsdp slurm probe`)
Writes `capabilities.json` recording: Slurm version, whether accounting (slurmdbd) answers `sacct`, `AccountingStorageTRES`, `SelectType`, `GresTypes`, the keys actually present in `scontrol show node`, whether MIG nodes exist, and which GPU accounting TRES are collected. Everything downstream reads this file and degrades gracefully.

### 5.6 Parsing rules
- Keep the raw line in a `raw` column so records can be reprocessed.
- `sacct` returns job rows and step rows. Use allocations only unless step data is needed.
- Handle `Unknown`, `None` and empty times; state strings with suffixes (for example a cancelled state with the user id appended) `[M]`; TRES strings such as `cpu=4,mem=16G,gres/gpu=2` `[M]`.
- Memory units differ between commands. Convert at the parser boundary with unit tests.

### 5.7 Fake Slurm shims (test without a cluster)
`tests/shims/bin/{sacct,scontrol,squeue,sinfo}` print output in the real formats, generated from simulator state. Add the shim directory to `PATH` in tests. Real recordings go in `tests/golden/real_slurm/` as soon as access exists. Until then, label the adapter **"unvalidated against real Slurm"** in all reports. Optionally use a community Docker-based Slurm cluster for integration tests `[M]`.

---

## 6. Node agent (the profiler)

Follow `profiler_plan.md`. Summary of what this repo must implement:

- **Backends:** `mock`, `nvml`, `dcgm`, `host` (cgroup, `psutil`, NIC counters). A common interface returns `{field: (value, unit, source, quality)}`.
- **Fast and slow groups:** cheap metrics sampled often, expensive ones rarely `[A]`.
- **Per-job attribution on Slurm:** map each PID to a job by reading `/proc/<pid>/cgroup` and extracting the job id from Slurm's cgroup path `[M]`; fall back to `SLURM_JOB_ID` in the process environment. In exclusive and MIG modes, device or instance metrics equal job metrics. In shared modes (time-slice, shard, MPS), split device metrics by per-process shares and tag them `estimated`.
- **MIG:** per-instance metrics through DCGM, keyed by MIG UUID `[M]`. Emit a "topology changed" event on reconfiguration.
- **Output:** windowed statistics (mean, max, p95, std, slope), a phase label (compute-bound, memory-bound, IO-bound, idle), and the lifetime-to-date averages for parity.
- **Transport:** local spool file, then push to the collector over HTTP or a shared filesystem `[A]`. Authenticate and encrypt before leaving a lab network.
- **Overhead budget:** under 1.5 percent throughput loss at the default cadence, measured as in `profiler_plan.md` section 6.

**How profiler data is used** (important; avoids a design error): a job being placed has no measurements yet, so its own live utilization cannot be an input at decision time. The profiler feeds the scheduler in three legitimate ways:
1. **Node load:** current utilization of jobs already running on each candidate node.
2. **Job history:** measured profiles of earlier runs of repeated jobs (Alibaba tags repeats with the `group` field `[V]`).
3. **Training labels:** measured outcomes the predictor learns to estimate from request features.

---

## 7. Cluster-state ledger

Event-sourced. For instance `i` placed on node `m` at time `t_i`, with `d_j` the requested demand (GPU, CPU or memory) of instance `j`:

```
used_before_i = sum(d_j  for j on m with start_j <= t_i)
              - sum(d_j  for j on m with end_j   <= t_i)
              - d_i
available_i   = capacity_m - used_before_i
running_i     = count(start_j <= t_i) - count(end_j <= t_i) - 1
```

Implement with per-machine sorted arrays and cumulative sums (`searchsorted`), as sketched in `extraction_plan.md`.

Rules:
- Missing end times mean "still running".
- Ties at identical timestamps need a documented tie-break (default: count as concurrent).
- "Available" means capacity minus **reserved** requests. Also store `measured_free` from telemetry, because the gap between reserved and actually used is a central research quantity.
- In live mode, reconcile against `scontrol show node` allocation fields on every poll and log discrepancies.

Invariants (tested in section 15): available is never below zero without a logged oversubscription flag; per-node reserved never exceeds capacity unless oversubscription is configured; replaying the log reproduces identical snapshots.

---

## 8. Simulator (the fallback "simulated tracing")

### 8.1 Purpose and honest limits
The simulator exists to (a) develop and test everything without a cluster, (b) compare policies under controlled conditions, and (c) re-simulate logged Slurm history for counterfactual evaluation. It is **not** evidence about real hardware until its assumed parameters are calibrated (M9). Every simulated output is labeled `synthetic_sim`.

### 8.2 What the survey says, and what we implement

Source: the MDPI survey "Algorithmic Techniques for GPU Scheduling" (Algorithms 18(7):385). **The excerpt I could read stopped after section 2.3 and was summarized by an automated reader, so verify every item below against the full paper before citing it.** Section numbers are the excerpt's.

| Survey item `[V]` (excerpt) | What we implement |
|---|---|
| Simulators are event-driven, synthetic or trace-driven (2.3) | Discrete-event engine (8.3), synthetic generator (8.5), trace replay (8.6) |
| Poisson arrivals are the classical assumption but do not capture real behavior (2.2) | Poisson as a validation baseline, plus diurnal and bursty arrival modes (8.5) |
| Durations are heavy-tailed. Cited P90/mean ratios: about 3 (Google Borg 2019) and about 3.5 (Alibaba 2021) (1.5) | Mixture-based duration model calibrated to target ratios (8.5). Verify the ratios in the full text |
| Job types range from multi-day training to millisecond inference (1.3) | Job classes: training, inference, interactive (8.5) |
| GPU demand: fractional, single, multi-GPU, CPU-only (1.3) | Demand distribution including fractional and CPU-only requests (8.5) |
| MIG multiplexes only at coarse integer granularity (2.1) | Discrete MIG profiles from `config/mig_profiles.yaml` (8.8) |
| Cross-job interference causes prediction error and tail volatility; the survey gives no explicit interference model (2.1, 2.2) | **Our own** contention model, flagged `[A]` (8.8) |
| Mixed precision is reported as hardware metrics only; the survey does not model it in scheduling (1.2) | **Our own** precision model, flagged `[A]` (8.7) |
| Heterogeneous GPU generations (2.1) | Heterogeneous cluster generator (8.4) |
| Power: SM and HBM modeled separately; DVFS (2.1) | Simple power and thermal model, `[A]` (8.8) |
| Preemption is often impractical; checkpoint-restart has overhead (2.1) | No preemption by default; optional checkpoint-restart cost parameter |
| Prediction error shrinks scheduler benefits; use auditing and fallbacks (2.2) | Noisy runtime-estimate model and conservative fallbacks (8.5, section 11) |
| Metrics: completion time, waiting time `W_j = C_j - r_j - p_j`, makespan, throughput, slowdown, max slowdown, SLO attainment, Jain's index, utilization over time, tail latency (p95), energy (2.1, 2.3) | Evaluation harness (section 12) |
| Baselines: FIFO, SRPT, first-fit/best-fit, backfilling, DJF, default Kubernetes and Slurm (2.2, 2.3) | Baseline policies P0 to P5 (section 11) |

### 8.3 Engine
- Heap-based event queue; events: `ARRIVE`, `START`, `FINISH`, `TICK` (telemetry sampling), `RECONFIG_BEGIN/END`, `NODE_FAIL` (optional).
- Order by `(time, event_priority, sequence)` for determinism.
- Time in seconds. Progress model: each running job has remaining `work`; its current `rate` (0 to 1) changes whenever co-location or device state changes, so finish times are recomputed on every such change.
- One RNG per component, seeded from the run seed.

### 8.4 Cluster generator
Config-driven node types: GPU model, GPUs per node, CPU cores, RAM, MIG enabled, MIG layout per GPU, sharing mode per device. Provide presets: `small` (4 nodes), `hetero` (mixed GPU generations), `mig_heavy`. A100 and H100 profile tables live in `config/mig_profiles.yaml` and **must be checked against NVIDIA's MIG user guide, including which profile combinations are legal** `[M]`.

### 8.5 Synthetic workload generator
Every default below is a **placeholder** `[A]`. The preferred approach is to fit the generator to the Alibaba v2020 marginals (arrival process, `plan_gpu`, `plan_cpu`, `plan_mem`, runtime) and validate with distribution tests.

| Component | Modes |
|---|---|
| Arrivals | (a) homogeneous Poisson (validation baseline); (b) non-homogeneous Poisson with a diurnal rate profile, generated by thinning; (c) bursty compound Poisson (sweeps arrive as groups with heavy-tailed group size) |
| Durations | **Mixture**: a mass of very short jobs plus a heavy-tailed long component, calibrated to a target P90/mean. See the warning below |
| Job classes | training (long, multi-GPU), inference (short, small), interactive (idle-heavy). Class mix configurable |
| GPU demand | fractional, 1, 2, 4, 8 GPUs, and CPU-only, with configurable probabilities (replace with the trace histogram when fitted) |
| CPU and memory | Correlated with GPU demand plus noise |
| Precision | Per job in {FP32, FP16/BF16, INT8} with configurable shares (8.7) |
| Tenants | Zipf-distributed activity; each tenant has a default job mix |
| Estimates | `timelimit` and the predictor input are noisy: `estimate = true_runtime * LogNormal(0, sigma)`. Calibrate `sigma` from Slurm `Timelimit` versus `Elapsed` once real accounting exists |
| Declared vs true demand | Declared request = true peak demand times an over-ask factor (log-normal). Calibrate from Alibaba `plan_gpu` versus sensor `gpu_wrk_util` `[V]` |
| Failures | Per-job failure probability and failure-time distribution |

**Calibration warning (my own calculation, checked numerically while writing this file):** the P90/mean ratio of a single log-normal peaks at about 2.27 (at sigma near 1.28), of a single Weibull at about 2.68 (shape near 0.56), and an exponential gives 2.30. The survey's reported ratios (about 3 to 3.5) therefore cannot be reached with one log-normal or Weibull component. A mixture with a large mass of near-zero jobs can: for example, 75 to 80 percent near-zero jobs plus an exponential tail gives about 3.5 to 3.7. Implement the mixture, keep a test that recomputes these achievable ratios (section 15), and verify the survey's numbers in the full paper before using them as targets.

### 8.6 Trace replay (Alibaba v2020)
Adapter that reads the seven tables `[V]` (`pai_job_table`, `pai_task_table`, `pai_instance_table`, `pai_sensor_table`, `pai_group_tag_table`, `pai_machine_spec`, `pai_machine_metric`) and emits canonical `jobs`, `nodes` and lifetime-average telemetry. Follow `extraction_plan.md` for joins, unit conversion and the derived availability fields. Limits to state in every report: lifetime averages only, no rejected candidates, fake dates, no MIG.

### 8.7 Demand, telemetry and precision model
- Each simulated job has a **hidden true demand vector** `d = (sm, dram, pcie, mem_gib)` as fractions of a full device, plus a phase process (Markov chain over compute-bound, memory-bound, IO-bound, idle) that modulates demand over time `[A]`.
- The simulated profiler samples the true demand with noise (AR(1) plus measurement noise), optional missing samples and counter glitches, so the profiler pipeline is tested against messy input `[A]`.
- Precision changes work and memory: speedup and footprint factors per precision in `assumptions.yaml`. Placeholder values only (for example FP32 = 1.0, FP16 = 2.0, INT8 = 3.0) and **must be calibrated with microbenchmarks on the target GPU** `[A]`.

### 8.8 Sharing, interference, MIG, power `[A]`
**Rate model.** For job `j` on a device or slice with capacities `cap_r` (fraction of a full GPU per resource `r`) shared with other jobs whose demands sum to `S_r`:

```
rate_j = min(1, min over r used by j of  cap_r / S_r)
```

- Exclusive job on a MIG slice: `S_r = d_j,r`, so a job demanding more than its slice is slowed proportionally.
- Time-sliced, shard or MPS device: `S_r` sums all co-located jobs, plus an optional time-slice overhead factor per extra tenant.
- **MIG instances do not interfere with each other** on SM and memory bandwidth. Allow a small configurable coupling through shared power and PCIe.
- **Reconfiguration:** only on an idle GPU; costs a configurable delay; the node is unavailable meanwhile. This reflects that Slurm cannot repartition dynamically `[V]`.
- **Power and thermal:** `P = P_idle + (P_max - P_idle) * (w_sm*sm + w_dram*dram)`; temperature follows a first-order lag toward `T_amb + P*R_th`; above a threshold, the clock scale (and `rate`) drops.

These models are deliberately simple and replaceable. M9 replaces assumed parameters with measured ones.

### 8.9 Slurm-like baseline behavior
Policy P0 emulates priority ordering plus backfill. It is an approximation, not a reimplementation of Slurm. When real accounting is available, compare simulated and real wait-time distributions and record the gap.

### 8.10 Outputs
Canonical Parquet tables identical to the live path, plus Slurm-format text through the shims (section 5.7), plus `run_manifest.json` (config hash, seed, git commit, `data_origin`).

### 8.11 Validating the simulator itself
1. **Queueing check:** single-GPU jobs, Poisson arrivals, exponential service, `c` GPUs, FIFO. Mean wait must fall inside the 95 percent confidence interval of the analytic M/M/c value (Erlang C: `Wq = P_wait / (c*mu - lambda)`).
2. **Conservation:** resources are never negative or double-booked; every started job finishes or is accounted for.
3. **Determinism:** two runs with the same seed give identical hashes.
4. **Distribution tests:** synthetic marginals match their fitted targets (KS tests).
5. **Sensitivity:** vary each `assumed` parameter and report which ones change policy rankings. Those get calibrated first.

### 8.12 Never claim
- Simulated utilization or slowdown gains as hardware results.
- That the interference or power models are accurate before calibration.
- That the Slurm shim matches real Slurm output before real recordings exist.

---

## 9. Feature builder and leakage rules

**Candidate rows:** one per (job, feasible candidate node or device or MIG profile), filtered by GPU type requirement and capacity.

| Group | Examples |
|---|---|
| Job request | `req_gpu_frac`, `req_cpu_cores`, `req_mem_gib`, `req_gpu_type`, precision hint, class, timelimit |
| Candidate hardware | GPU model, GPU count, capacities, MIG layout, sharing mode |
| Node state (as of `t`) | `avail_*`, `running_jobs`, node-load from telemetry (measured utilization of residents), residents' tenants |
| Co-location context | Resident fingerprints (SM, DRAM, tensor shares), predicted slowdown if co-located |
| History (repeat jobs) | Lifetime-average usage of earlier runs of the same job group or user script |
| Time | Queue length, hour of day |

**Labels** (separate table): wait time, runtime, final status, mean utilization, slowdown versus solo when known.

**Leakage tests (mandatory):**
- A mutation test injects a future value into the telemetry and the build must fail or the value must not appear.
- A job's own post-start utilization must never appear in its own placement features.
- Features built in streaming mode equal features rebuilt from the full log with an as-of cutoff (differential test).

Use DuckDB `ASOF JOIN` or an equivalent as-of join `[M]`.

**Predictor plug-in (the teammate's model).** Interface: `predict(features) -> (pred_runtime_s, pred_gpu_util_pct, pred_gpu_mem_gib, uncertainty)`. Start with a file-based exchange (Parquet in and out), then in-process, then HTTP. A `NullPredictor` returns the user estimate so everything runs without the model.

---

## 10. Isolation engine integration

Port the logic of `isolation_sim.py` into `gsdp/isolation`:

| Device mode | Rule |
|---|---|
| `mig` instance | Hardware-isolated. One allocation at a time, so cross-tenant sharing is structurally impossible |
| `exclusive` | Same as above |
| `time_slice`, `shard`, `mps` | **Not isolated** (shard does not fence processes `[V]`; MPS time-slices between users `[V]`). Forbid two different tenants on the same device |

- **Tenant key:** default Slurm `Account` (hashed). Make it configurable (`User`, `QOS`) `[A]`.
- The isolation engine runs after the policy proposes a placement and before the executor acts. Violations are blocked, logged with the reason, and counted as a metric. Policies receive a clean "rejected" signal and must propose again.
- Provide a soft-penalty form of the same rule so the QUBO policy can include it as `H_isolation` in its cost function while the hard check remains as the final guard.
- Property test: for every policy, no executed placement ever puts two tenants on a non-isolated device.

---

## 11. Scheduler interface and policy roadmap

```python
class Policy(Protocol):                      # illustrative sketch, not tested
    def decide(self, now: Timestamp, pending: list[JobView],
               cluster: ClusterView, predictor: Predictor) -> list[Placement]: ...
```

Decisions are event-triggered (arrival, finish) plus a periodic tick. Optimizers use a rolling horizon and a **time box**: return the best solution found within the budget, never block the loop.

| ID | Policy | Notes |
|---|---|---|
| P0 | Slurm-like (priority plus backfill) | Baseline (8.9) |
| P1 | FIFO | `[V]` survey baseline |
| P2 | SJF / SRPT using noisy estimates | `[V]` |
| P3 | First-fit, best-fit bin packing | `[V]` |
| P4 | EASY backfilling | `[V]` |
| P5 | LAS-style (least attained service) | `[V]` Tiresias idea; works without size estimates |
| P6 | Greedy interference-aware | Uses profiler fingerprints and the contention model |
| P7 | Exact ILP / CP-SAT baseline | OR-Tools `[M]`. Gives ground truth on small instances |
| P8 | QUBO plus simulated annealing | From `formulation.md`; PyQUBO and `dwave-neal` or OpenJij `[M]` |
| P9 | Variational (QAOA-style) | Stretch; classical simulation only |
| Stretch | DRF, Gavel-style heterogeneity, Optimus-style throughput probing, Pollux-style adaptation | Names from the survey `[V]` |

The survey also names AlloX, HEFT, Themis, DeepRM and Decima as further approaches `[V]`; add only if there is a research reason.

---

## 12. Evaluation harness

**Metrics (survey definitions `[V]`):** completion time `C_j`; waiting time `W_j = C_j - r_j - p_j`; makespan `max_j C_j`; throughput; slowdown (response time over processing time); max slowdown; SLO attainment; Jain's fairness index `(sum x)^2 / (n * sum x^2)`; utilization over time; p95 tail latency; energy.

**Project metrics:** interference slowdown versus solo; isolation violations blocked; reserved-versus-measured gap; MIG reconfiguration count and cost; power and thermal indicators; profiler overhead; policy decision latency.

**Protocol**
- Scenario matrix: cluster preset, workload mode, load level (utilization targets from low to near saturation), policy.
- At least 20 seeds per cell; report means with bootstrap 95 percent confidence intervals `[A]`.
- Always include P0 to P4 as baselines and the ILP baseline on small instances.
- Report heavy-tail behavior (p95 and p99), not only means.
- Every table states `data_origin` and which parameters were `assumed` versus `calibrated`.

---

## 13. Repository layout and stack

```
gsdp/
  agent.md  QUESTIONS.md  pyproject.toml  Makefile
  config/
    assumptions.yaml  mig_profiles.yaml  slurm.yaml
    cluster_{small,hetero,mig_heavy}.yaml  workload_*.yaml
  src/gsdp/
    schema/        pydantic models, Parquet schemas, units
    store/         event log, DuckDB and Parquet IO
    sources/
      base.py      Source protocol
      slurm/       probe.py sacct.py scontrol.py parse.py executor.py
      alibaba/     loader.py adapter.py
      sim/         engine.py cluster.py workload.py demand.py
                   interference.py power.py slurm_shim.py
    agent/         backends/{mock,nvml,dcgm,host}.py attribution.py
                   window.py phase.py spool.py
    ledger/        state.py
    features/      builder.py asof.py
    isolation/     engine.py rules.py
    policies/      base.py fifo.py sjf.py fit.py backfill.py las.py
                   interference_greedy.py ilp_cpsat.py qubo_sa.py variational.py
    predictors/    base.py null.py teammate.py
    eval/          metrics.py scenarios.py stats.py report.py
    cli.py
  tests/           unit/ property/ golden/ leakage/ shims/
  docs/
```

**Stack `[A]`:** Python 3.11 or newer; pydantic v2; numpy; pandas with pyarrow; DuckDB; PyYAML; Typer; pytest and Hypothesis; ruff; mypy. Optional: OR-Tools, PyQUBO, `dwave-neal` or OpenJij, `nvidia-ml-py`, `psutil`, DCGM Python bindings `[M]`. Write the event engine yourself (a heap) for determinism and speed.

**CLI** (illustrative):
```
gsdp sim run    --cluster small --workload synth_diurnal --policy fifo --seed 1
gsdp sim sweep  --config sweep.yaml
gsdp alibaba load --path <dir>   |  gsdp alibaba replay ...
gsdp slurm probe | collect --once | shadow
gsdp agent run  --backend mock|nvml|dcgm
gsdp features build --source sim --run <id>
gsdp eval report --runs <ids>
```

---

## 14. Milestones and acceptance tests

| M | Deliverable | Acceptance |
|---|---|---|
| M0 | Scaffolding, schemas, units, config loader, CI, `assumptions.yaml`, `QUESTIONS.md` | `make test` passes; schema round-trip tests; auto-generated data dictionary lists all 22 fields with sources |
| M1 | Simulator core, ledger, simple FIFO on Poisson/exponential workload | M/M/c check within CI; conservation and determinism tests pass |
| M2 | Realistic workload generator, demand and telemetry model, baselines P1 to P4, metrics | Distribution tests pass; heavy-tail calibration test passes; metrics match hand-computed cases |
| M3 | Alibaba v2020 adapter, replay, derived availability and concurrency | Quality report per `extraction_plan.md`; derived concurrency correlates with `machine_num_worker`; negative-availability rate reported |
| M4 | Feature builder, leakage tests, lifetime-average and windowed views | Mutation test catches injected leakage; streaming equals rebuilt features |
| M5 | Node agent: mock, then host, then NVML; attribution; windowing; overhead harness | Calibration microbenchmarks from `profiler_plan.md` pass on the dev GPU; overhead report |
| M6 | Slurm adapter: probe, parsers, shims, shadow mode | Differential test: simulator, then shim, then adapter yields the same canonical tables as the direct simulator output (excluding declared lossy fields); parser golden tests |
| M7 | Isolation engine wired in | Property test: no policy ever executes a cross-tenant placement on a non-isolated device; violations logged |
| M8 | Interference-aware greedy, CP-SAT baseline, QUBO plus SA, (stretch) variational | On small instances CP-SAT is never worse than heuristics; time boxes respected; comparison tables with CIs |
| M9 | Real-hardware validation and calibration (A100/A30 with MIG; a short rented session is acceptable); Slurm sandbox trial with the admin | Assumed parameters replaced by calibrated ones, with data; overhead under 1.5 percent; adapter validated against real recordings |
| M10 | Reports and paper-ready artifacts | Reproducible from manifests; every number traceable to a run |

Dependencies: M3 and M4 can proceed in parallel with M5 after M2. M6 needs M1 and M2. M8 needs M4 and M7.

---

## 15. Testing strategy

- **Unit tests** for parsers, unit conversions, ledger arithmetic, metric formulas.
- **Property tests (Hypothesis):** event ordering; no negative resources; reserved never exceeds capacity; no overlapping MIG assignments; replaying the log reproduces snapshots; isolation invariant.
- **Golden tests:** fixed seed runs compared to stored hashes; Slurm output samples compared to parsed tables.
- **Differential tests:** simulator versus shim plus adapter (M6); streaming versus rebuilt features (M4).
- **Statistical tests:** M/M/c agreement; KS tests on workload marginals; a numeric test of achievable P90/mean ratios per distribution family (confirms section 8.5's warning).
- **Mutation tests** for leakage (section 9).
- **Performance budget `[A]`:** simulate 100k jobs on 200 nodes in a few minutes on a laptop; report the actual figure.

---

## 16. Open questions (put answers in `QUESTIONS.md`)

1. Slurm version; is `slurmdbd` accounting enabled; `AccountingStorageTRES`; `SelectType` (`cons_tres`?); `GresTypes`.
2. Does the cluster use MIG, shard, MPS or exclusive GPUs? Which GPU models, and how many nodes?
3. Access level: read-only, or may we submit and update jobs? Is enforce mode ever allowed?
4. What defines a tenant: Account, User or QOS?
5. How does the teammate's model consume features (file, in-process, HTTP), and what does it predict?
6. Which Alibaba trace and version does the teammate use? Is the training row an instance, a task or a job?
7. Are multi-node jobs in scope for v1?
8. Is there a MIG reconfiguration procedure the admin supports, and what is its downtime?
9. Where will the decision service run (login node, dedicated host)? Data retention and privacy rules?
10. When will real GPU hardware with MIG be available for M9?

Defaults used until answered: Account as tenant, single-node jobs, shadow mode, file-based predictor exchange, no MIG reconfiguration.

---

## 17. Provenance: what was verified and what was not

**Verified while writing this file `[V]`**
- Slurm `sacct`: field names for submit, eligible, start, end, state, node list, requested and allocated TRES, requested CPUs and memory, usage fields; `-P` output; job id formats; default time format.
- Slurm `scontrol`: `--json`, `--yaml`, `-o`; node memory fields in MiB; partial `show node` and `show job` fields.
- Slurm GRES: MIG since 21.08, static partitioning only, type naming; shard and MPS behavior; GPU accounting TRES and the MIG exclusion; `cons_tres` requirement for `--gpus*`.
- Alibaba v2020 README: the seven tables, their columns, units and caveats.
- Survey excerpt: taxonomy, metrics, workload statements. **The excerpt was cut off after section 2.3 and summarized automatically; verify against the full paper.**

**From memory `[M]` (verify before depending)**
- DCGM field names and availability; NVML call names.
- `scontrol show node` keys `CPUTot`, `CPUAlloc`, `CfgTRES`, `AllocTRES`, `Gres`; `sacct -X`, `-S`, `-E`; `ReqMem` suffixes; state-string suffixes; TRES string format.
- `sbatch --nodelist`, `--hold`, `scontrol release/update`, Prolog/Epilog hooks.
- Slurm cgroup path layout and PID-to-job mapping; `SLURM_JOB_ID` in process environments.
- MIG legal profile combinations and per-model profile tables.
- Existence of community Docker Slurm clusters; DuckDB `ASOF JOIN`; OR-Tools, PyQUBO, OpenJij APIs.

**Assumptions `[A]`**
- Every simulator parameter: arrival modes, duration mixtures, demand distributions, precision factors, interference, power and thermal models, estimate noise, failure rates.
- Polling cadences, latency budgets, seed counts, performance targets.
- Canonical units (GB treated as GiB for Alibaba), UTC conversion, tenant key.

---

## 18. Glossary

- **Tenant:** a customer or team whose jobs must not interfere with or see another tenant's work.
- **MIG (Multi-Instance GPU):** hardware partitioning of one GPU into isolated slices, each with dedicated compute and memory.
- **Time-slicing, shard, MPS:** ways to share one GPU between jobs without hardware isolation.
- **Ledger:** the running record of what each node has reserved, used to answer "what was free at time t".
- **Shadow mode:** the system decides but does not act, so its decisions can be compared safely.
- **Point-in-time / leakage:** using information that would not have existed yet at decision time. It inflates training results and breaks in real time.
- **Interference:** slowdown a job suffers because another job shares hardware.
- **QUBO:** the binary quadratic cost form that quantum and quantum-inspired solvers accept.
