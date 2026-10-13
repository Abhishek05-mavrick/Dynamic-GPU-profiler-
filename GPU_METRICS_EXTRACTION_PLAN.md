# Real-Time GPU Metrics Extraction Plan

## Executive summary

The current project is a timing-based interference profiler. It already detects CUDA
device properties and infers SM saturation, memory-bandwidth pressure, cache
sensitivity, and compute-vs-memory behavior from kernel timings. It does **not**
yet have a general telemetry sampler.

The requested metrics are feasible, but not all through one API or at the same
fidelity:

- **NVML** is the right first layer for device identity, power, temperature,
  coarse GPU utilization, memory used, memory utilization where supported, and
  PCIe counters.
- **DCGM field telemetry** is the preferred production layer on A100/H100/H200
  class systems. It exposes more engine and fabric fields, but availability is
  driver, GPU, MIG, and DCGM-version dependent.
- **CUPTI/Nsight Compute** is needed for kernel-level SM, Tensor, DRAM, and
  achieved-memory-bandwidth counters. These counters are intrusive, may require
  elevated profiling permissions, and should not be sampled continuously in a
  latency-sensitive service.
- **Host OS telemetry** (for example `psutil`) is needed for CPU utilization,
  host memory, and network RX/TX. These are not GPU metrics and should be
  collected in the same timestamped sample, not queried from NVML.

The RTX 2050 is a valid development target for the sampler. It is not a valid
target for assuming that every datacenter counter exists: it has no NVLink and
may not expose separate Tensor/DRAM engine counters through the installed
driver. Missing fields must be represented explicitly as `unsupported` or
`unavailable`, never as zero.

## Repository context

The current implementation is split into:

1. `dynamic_profiler/gpu_config.py`: PyTorch-based device detection and static
   hardware configuration.
2. `dynamic_profiler/profiler.py`: one-shot kernel fingerprint orchestration.
3. `dynamic_profiler/fingerprint.py`: JSON-serializable kernel interference
   scores.
4. `util-gpu/gpu-util-interference`: CUDA/NVML experiments and Nsight guidance.

The existing NVML experiment intentionally demonstrates that `nvidia-smi`/NVML
GPU utilization can report close to 100% for a single active block even when
the GPU still has spare capacity. Therefore the new telemetry must label
`gpu_utilization` as **device busy time**, not as SM occupancy or compute
capacity consumed. The existing timing-based inference should remain the source
of the profiler's deeper interference scores.

## Metric feasibility and source

| Requested metric | First source | RTX 2050 expectation | A100/H100/H200 expectation | Notes |
|---|---|---:|---:|---|
| GPU utilization | NVML `utilization.gpu` | Available | Available | Coarse engine busy percentage over the driver's sampling window |
| SM/Compute utilization | DCGM field if exposed; otherwise CUPTI/Nsight Compute | Often unavailable as a direct live counter | Usually available through DCGM or profiling counters | Do not equate this with NVML GPU utilization |
| Tensor utilization | DCGM tensor field or CUPTI tensor instructions | May be unavailable or zero for workloads that do not use Tensor Cores | Better support, but field/counter varies by architecture and MIG | Report capability and sampling source |
| GPU memory used | NVML memory used | Available | Available | Include total and free if useful |
| GPU memory utilization | NVML memory percentage where supported; otherwise `used/total` | Derived fallback is reliable | Derived fallback is reliable | Name the derived value distinctly |
| DRAM utilization | DCGM memory-pipeline field or CUPTI/Nsight | Usually not guaranteed | Typically available on supported datacenter stacks | This is a traffic/busy counter, not capacity allocated |
| GPU memory bandwidth | CUPTI/Nsight bytes or DCGM bandwidth field | Prefer profiler-derived `bytes / interval`; exact hardware counter may be unavailable | DCGM/CUPTI preferred | Keep units in bytes/s and document read/write semantics |
| PCIe TX/RX | NVML PCIe TX/RX throughput counters | Available on supported driver, may be coarse/zero when idle | Available | Sample over time and calculate rate if API returns cumulative bytes |
| NVLink TX/RX | NVML/DCGM NVLink link counters | Unsupported: RTX 2050 has no NVLink | Available only when links exist and are enabled | Return per-link values and aggregate; unsupported is not zero |
| CPU utilization | Host OS (`psutil` or Windows performance counters) | Available | Available | Process and system scope should be separate |
| Host memory utilization | Host OS (`psutil.virtual_memory`) | Available | Available | Keep used, available, and percentage |
| Network RX/TX | Host OS interface counters (`psutil.net_io_counters`) | Available | Available | Aggregate and optionally expose per-interface counters |

## Recommended architecture

### 1. Stable sample schema

Add a telemetry module with a typed sample object rather than extending
`KernelFingerprint` directly. A sample should contain:

- `timestamp_ns`, `device_index`, `gpu_name`, driver version
- `interval_ms` and `source` per metric (`nvml`, `dcgm`, `cupti`,
  `derived`, `host_os`)
- numeric values in stable units: percentages in `[0, 100]`, memory in bytes,
  bandwidth/rates in bytes per second
- `status` per field: `ok`, `unsupported`, `unavailable`, or `permission_denied`
- optional `error` text for a failed field

This avoids conflating “zero traffic” with “counter does not exist” and makes
the JSON output portable between RTX and datacenter systems.

### 2. Tiered collectors

Implement collectors behind one interface:

1. **Host collector**: CPU, host memory, network RX/TX.
2. **NVML collector**: identity, temperature/power, GPU busy, memory, PCIe,
   and NVLink when present.
3. **DCGM collector (optional)**: richer live GPU fields and datacenter
   deployment support.
4. **Kernel counter collector (optional)**: CUPTI or Nsight Compute for
   short profiling windows, not every polling interval.

The sampler should continue when one collector is missing. It should emit a
structured field status and a startup capability report.

### 3. Sampling behavior

- Default interval: **500 ms or 1 s** for a low-overhead dashboard.
- Use the same monotonic timestamp for all collectors.
- Initialize NVML once, reuse handles, and shut it down on clean exit.
- Compute rates from cumulative counters as
  `(new_value - old_value) / elapsed_seconds`; handle counter resets.
- Keep a bounded in-memory ring buffer and offer JSON-lines/CSV export.
- Use a separate short profiling mode for CUPTI/Nsight counters.
- Do not run `nvidia-smi` as the primary polling mechanism; process startup and
  text parsing add overhead and lose structured error/capability information.

### 4. Interaction with the existing profiler

Keep the existing fingerprint fields and semantics unchanged. Add telemetry
only as:

- a standalone `sample_telemetry()`/streaming API for real-time observation;
- optional telemetry snapshots around `build_fingerprint()` experiments;
- a later correlation layer that compares latency/interference scores with
  observed device and host samples.

The timing experiments remain necessary because sampled device utilization cannot
reliably identify scheduler, cache, warp-pipeline, or co-location contention.

## Phased implementation plan

### Phase 0 — capability probe

- Detect GPU, driver, CUDA runtime, MIG mode, NVML availability, and linked
  NVLink count.
- Enumerate which requested fields are supported on the current machine.
- Add a diagnostic command that prints the capability matrix.
- Acceptance: on this RTX 2050, GPU busy, memory, PCIe, CPU, host memory, and
  network fields are either sampled or explicitly marked unavailable; NVLink is
  explicitly unsupported.

### Phase 1 — low-overhead real-time sampler

- Add Python dependencies only if needed: an NVML binding and `psutil`.
- Implement the typed schema, NVML collector, host collector, rate handling,
  and JSON-lines output.
- Add tests using fake collectors for counter deltas, reset handling, missing
  fields, and unit conversion.
- Acceptance: stable 1 Hz samples for at least five minutes with no leaked
  NVML handles and no crash when an optional field is absent.

### Phase 2 — deeper GPU counters

- Add an optional DCGM backend for supported datacenter hosts.
- Add a short-window CUPTI/Nsight backend for SM, Tensor, DRAM, and bandwidth
  counters where permissions and architecture support them.
- Record counter scope and collection overhead in every profile.
- Acceptance: compare counters with known compute-heavy, Tensor Core, and
  memory-copy workloads; do not claim a metric is validated until the value
  changes predictably across those workloads.

### Phase 3 — profiler correlation and portability

- Capture before/during/after samples around fingerprint experiments.
- Add a correlation report showing timing-inferred pressure beside observed
  telemetry, without replacing either signal.
- Test on RTX 2050, one Ampere datacenter GPU (A100), and one Hopper GPU
  (H100/H200) when hardware access exists.
- If “A1000” means the NVIDIA RTX A1000 workstation GPU rather than A100, treat
  it as a separate workstation target; it should not be grouped with A100.

## Validation workloads

Use small deterministic workloads:

1. Idle process: all activity counters should approach zero.
2. Large CUDA copy/elementwise kernel: memory traffic and bandwidth should
   increase while Tensor activity remains absent.
3. FP32 matrix multiplication: GPU/SM activity should increase without
   requiring Tensor activity.
4. Tensor Core matrix multiplication: Tensor activity should increase on
   architectures and toolchains that expose the counter.
5. Host CPU loop and memory allocator: host CPU/memory fields should respond.
6. Network transfer: host RX/TX deltas should respond.

For each workload, compare timestamps, kernel duration, expected direction of
change, and field status. Exact percentages should not be treated as portable
across driver versions or GPU architectures.

## Risks and decisions

- **Counter availability varies** by driver, architecture, MIG mode, permissions,
  and whether another profiler has reserved the performance-counter interface.
- **Sampling is not kernel attribution**: NVML/DCGM device samples can include
  other users and kernels. CUPTI/Nsight is needed for attribution.
- **Polling has overhead**: keep the normal sampler lightweight and make deep
  counters opt-in.
- **Windows vs Linux differs**: the local RTX 2050 is Windows; deployment on
  H200/A100 systems is commonly Linux. Capability probing and OS-specific host
  collectors are required.
- **Virtualized GPUs may hide fields**: status must preserve unavailable versus
  zero.

## Immediate next steps

1. Implement the schema and capability probe.
2. Add NVML and host collectors with 1 Hz JSON-lines output.
3. Validate on the local RTX 2050 using the six workloads above.
4. Add DCGM only after the first layer is stable and a datacenter test host is
   available.
5. Add CUPTI/Nsight counters as a separate profiling mode, then correlate them
   with the existing timing fingerprints.

This sequence delivers useful real-time telemetry quickly while preserving the
project's central insight: top-level GPU utilization alone is not a sufficient
measure of interference or available compute capacity.
