"""Low-overhead, capability-aware telemetry for GPU schedulers.

The sampler deliberately separates values from availability. A fractional GPU VM
may expose memory and busy time while hiding engine counters or NVLink fields.
Missing counters are therefore reported as ``unsupported`` or ``unavailable``
instead of being converted to zero.

Optional dependencies:
    nvidia-ml-py  (import name: pynvml)
    psutil
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Iterator, Optional


class MetricStatus(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class Metric:
    value: Optional[float] = None
    unit: str = ""
    source: str = ""
    status: MetricStatus = MetricStatus.UNAVAILABLE
    error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "status": self.status.value,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class TelemetrySample:
    timestamp_ns: int
    interval_ms: float
    gpu_name: str = ""
    device_index: int = 0
    metrics: Dict[str, Metric] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "interval_ms": self.interval_ms,
            "gpu_name": self.gpu_name,
            "device_index": self.device_index,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
        }


@dataclass(frozen=True)
class Capability:
    name: str
    source: str
    supported: bool
    note: str = ""


GPU_SCHEDULER_PROFILES = {
    "H200": {
        "family": "Hopper datacenter",
        "expected_memory_bytes": 141 * 1024**3,
        "peak_bandwidth_bytes_s": 4_800 * 1024**3,
        "expected_nvlink": True,
        "deep_counters": "dcgm/cupti",
    },
    "H100": {
        "family": "Hopper datacenter",
        "expected_memory_bytes": 80 * 1024**3,
        "peak_bandwidth_bytes_s": 3_350 * 1024**3,
        "expected_nvlink": True,
        "deep_counters": "dcgm/cupti",
    },
    "A100": {
        "family": "Ampere datacenter",
        "expected_memory_bytes": 40 * 1024**3,
        "peak_bandwidth_bytes_s": 2_039 * 1024**3,
        "expected_nvlink": True,
        "deep_counters": "dcgm/cupti",
    },
    "RTX": {
        "family": "GeForce/workstation fallback",
        "expected_memory_bytes": 0,
        "peak_bandwidth_bytes_s": 0,
        "expected_nvlink": False,
        "deep_counters": "unavailable unless explicitly exposed",
    },
}


def _metric(value: float, unit: str, source: str) -> Metric:
    return Metric(value=float(value), unit=unit, source=source, status=MetricStatus.OK)


def _unsupported(source: str, note: str) -> Metric:
    return Metric(source=source, status=MetricStatus.UNSUPPORTED, error=note)


def _unavailable(source: str, error: Exception) -> Metric:
    status = MetricStatus.PERMISSION_DENIED if "permission" in str(error).lower() else MetricStatus.UNAVAILABLE
    return Metric(source=source, status=status, error=str(error))


class TelemetrySampler:
    """Collect scheduler telemetry without assuming privileged GPU access.

    NVML is initialized lazily and once per sampler. The sampler does not invoke
    ``nvidia-smi`` and continues when either NVML or psutil is absent.
    """

    def __init__(self, device_index: int = 0, enable_host: bool = True):
        self.device_index = device_index
        self.enable_host = enable_host
        self._nvml = None
        self._handle = None
        self._gpu_name = ""
        self._previous: Dict[str, tuple[int, float]] = {}
        self._load_nvml()

    def _load_nvml(self) -> None:
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._gpu_name = _decode(pynvml.nvmlDeviceGetName(self._handle))
        except pynvml.NVMLError:
            self._nvml = None

    @property
    def gpu_name(self) -> str:
        return self._gpu_name or "unknown"

    def hardware_profile(self) -> dict:
        """Return scheduler defaults without treating them as live measurements."""
        name = self.gpu_name.upper()
        for model in ("H200", "H100", "A100"):
            if model in name:
                return {"model": model, **GPU_SCHEDULER_PROFILES[model]}
        return {"model": "generic", **GPU_SCHEDULER_PROFILES["RTX"]}

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass
            finally:
                self._nvml = None
                self._handle = None

    def __enter__(self) -> "TelemetrySampler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def capabilities(self) -> Dict[str, Capability]:
        """Return the conservative capabilities of the current VM/device."""
        capabilities = {
            "gpu_utilization": Capability("gpu_utilization", "nvml", self._nvml is not None),
            "gpu_memory_used": Capability("gpu_memory_used", "nvml", self._nvml is not None),
            "gpu_memory_utilization": Capability("gpu_memory_utilization", "derived", self._nvml is not None),
            "pcie_tx": Capability("pcie_tx", "nvml", self._nvml is not None),
            "pcie_rx": Capability("pcie_rx", "nvml", self._nvml is not None),
            "nvlink_tx": Capability("nvlink_tx", "nvml", False, "Requires exposed NVLink links"),
            "nvlink_rx": Capability("nvlink_rx", "nvml", False, "Requires exposed NVLink links"),
            "sm_compute_utilization": Capability(
                "sm_compute_utilization", "dcgm/cupti", False,
                "Requires an exposed DCGM or CUPTI engine counter",
            ),
            "tensor_utilization": Capability(
                "tensor_utilization", "dcgm/cupti", False,
                "Requires an exposed DCGM or CUPTI Tensor Core counter",
            ),
            "dram_utilization": Capability(
                "dram_utilization", "dcgm/cupti", False,
                "Requires an exposed DCGM or CUPTI DRAM counter",
            ),
            "gpu_memory_bandwidth": Capability(
                "gpu_memory_bandwidth", "dcgm/cupti/derived", False,
                "Use DCGM/CUPTI or workload byte counts",
            ),
        }
        if "H200" in self.gpu_name.upper():
            capabilities["nvlink_tx"] = Capability(
                "nvlink_tx", "nvml/dcgm", True,
                "Only if the fractional VM exposes NVLink counters",
            )
            capabilities["nvlink_rx"] = Capability(
                "nvlink_rx", "nvml/dcgm", True,
                "Only if the fractional VM exposes NVLink counters",
            )
        return capabilities

    def sample(self, interval_ms: float = 0.0) -> TelemetrySample:
        """Collect one timestamped sample.

        ``interval_ms`` is supplied by the caller when sampling in a loop. It is
        metadata only; NVML's utilization values already represent a driver
        sampling window.
        """
        timestamp_ns = time.monotonic_ns()
        metrics: Dict[str, Metric] = {}
        if self._nvml is None:
            metrics.update(self._nvml_unavailable_metrics())
        else:
            metrics.update(self._sample_nvml())
        if self.enable_host:
            metrics.update(self._sample_host())
        metrics.update(self._deep_counter_fallbacks())
        return TelemetrySample(
            timestamp_ns=timestamp_ns,
            interval_ms=float(interval_ms),
            gpu_name=self.gpu_name,
            device_index=self.device_index,
            metrics=metrics,
        )

    def stream(self, interval_seconds: float = 1.0) -> Iterator[TelemetrySample]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        previous_ns = time.monotonic_ns()
        while True:
            time.sleep(interval_seconds)
            now_ns = time.monotonic_ns()
            yield self.sample((now_ns - previous_ns) / 1_000_000)
            previous_ns = now_ns

    def _sample_nvml(self) -> Dict[str, Metric]:
        assert self._nvml is not None and self._handle is not None
        nvml = self._nvml
        handle = self._handle
        result: Dict[str, Metric] = {}
        result.update(self._read_utilization(nvml, handle))
        result.update(self._read_memory(nvml, handle))
        result.update(self._read_pcie(nvml, handle))
        result.update(self._read_nvlink(nvml, handle))
        return result

    def _read_utilization(self, nvml: Any, handle: Any) -> Dict[str, Metric]:
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            return {"gpu_utilization": _metric(utilization.gpu, "%", "nvml")}
        except nvml.NVMLError as error:
            return {"gpu_utilization": _unavailable("nvml", error)}

    def _read_memory(self, nvml: Any, handle: Any) -> Dict[str, Metric]:
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            used = _metric(memory.used, "bytes", "nvml")
            utilization = _metric((memory.used / memory.total) * 100, "%", "derived")
            return {
                "gpu_memory_used": used,
                "gpu_memory_total": _metric(memory.total, "bytes", "nvml"),
                "gpu_memory_utilization": utilization,
            }
        except nvml.NVMLError as error:
            unavailable = _unavailable("nvml", error)
            return {
                "gpu_memory_used": unavailable,
                "gpu_memory_total": unavailable,
                "gpu_memory_utilization": unavailable,
            }

    def _read_pcie(self, nvml: Any, handle: Any) -> Dict[str, Metric]:
        result: Dict[str, Metric] = {}
        for name, counter in (("pcie_tx", nvml.NVML_PCIE_UTIL_TX_BYTES), ("pcie_rx", nvml.NVML_PCIE_UTIL_RX_BYTES)):
            try:
                # NVML returns KB/s for this API. Normalize to bytes/s.
                result[name] = _metric(nvml.nvmlDeviceGetPcieThroughput(handle, counter) * 1024, "bytes/s", "nvml")
            except nvml.NVMLError as error:
                result[name] = _unavailable("nvml", error)
        return result

    def _read_nvlink(self, nvml: Any, handle: Any) -> Dict[str, Metric]:
        if not hasattr(nvml, "nvmlDeviceGetNvLinkState") or not hasattr(
            nvml, "nvmlDeviceGetNvLinkUtilizationCounter"
        ):
            unsupported = _unsupported("nvml", "NVLink API is not exposed by this binding")
            return {"nvlink_tx": unsupported, "nvlink_rx": unsupported}
        tx = rx = 0.0
        found = False
        try:
            link_count = nvml.NVML_NVLINK_MAX_LINKS
            for link in range(link_count):
                if not nvml.nvmlDeviceGetNvLinkState(handle, link):
                    continue
                found = True
                counter = nvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, link, 0)
                tx_value = getattr(counter, "tx", getattr(counter, "tx_bytes", None))
                rx_value = getattr(counter, "rx", getattr(counter, "rx_bytes", None))
                if tx_value is None or rx_value is None:
                    unsupported = _unsupported(
                        "nvml", "NVLink counter shape is not understood by this binding"
                    )
                    return {"nvlink_tx": unsupported, "nvlink_rx": unsupported}
                tx += float(tx_value)
                rx += float(rx_value)
            if not found:
                unsupported = _unsupported("nvml", "No active NVLink links exposed by the VM")
                return {"nvlink_tx": unsupported, "nvlink_rx": unsupported}
            return {
                "nvlink_tx": _metric(tx, "bytes", "nvml"),
                "nvlink_rx": _metric(rx, "bytes", "nvml"),
            }
        except nvml.NVMLError as error:
            unavailable = _unavailable("nvml", error)
            return {"nvlink_tx": unavailable, "nvlink_rx": unavailable}

    def _sample_host(self) -> Dict[str, Metric]:
        try:
            import psutil
        except ImportError as error:
            return {
                name: _unavailable("host_os", error)
                for name in ("cpu_utilization", "host_memory_utilization", "network_rx", "network_tx")
            }
        result = {
            "cpu_utilization": _metric(psutil.cpu_percent(interval=None), "%", "host_os"),
        }
        try:
            memory = psutil.virtual_memory()
            result["host_memory_utilization"] = _metric(memory.percent, "%", "host_os")
        except psutil.Error as error:
            result["host_memory_utilization"] = _unavailable("host_os", error)
        try:
            counters = psutil.net_io_counters()
            result["network_rx"] = self._rate("network_rx", counters.bytes_recv, "bytes/s", "host_os")
            result["network_tx"] = self._rate("network_tx", counters.bytes_sent, "bytes/s", "host_os")
        except psutil.Error as error:
            result["network_rx"] = _unavailable("host_os", error)
            result["network_tx"] = _unavailable("host_os", error)
        return result

    def _rate(self, name: str, value: int, unit: str, source: str) -> Metric:
        now_ns = time.monotonic_ns()
        previous = self._previous.get(name)
        self._previous[name] = (value, now_ns)
        if previous is None:
            return Metric(source=source, unit=unit, status=MetricStatus.UNAVAILABLE, error="rate requires two samples")
        old_value, old_ns = previous
        elapsed = (now_ns - old_ns) / 1_000_000_000
        if elapsed <= 0 or value < old_value:
            return Metric(source=source, unit=unit, status=MetricStatus.UNAVAILABLE, error="counter reset or invalid interval")
        return _metric((value - old_value) / elapsed, unit, source)

    @staticmethod
    def _nvml_unavailable_metrics() -> Dict[str, Metric]:
        metrics = {}
        for name in ("gpu_utilization", "gpu_memory_used", "gpu_memory_total",
                     "gpu_memory_utilization", "pcie_tx", "pcie_rx"):
            metrics[name] = _unavailable("nvml", ImportError("nvidia-ml-py is not installed or NVML is not accessible"))
        metrics["nvlink_tx"] = _unsupported("nvml", "No NVML handle; NVLink is not observable")
        metrics["nvlink_rx"] = _unsupported("nvml", "No NVML handle; NVLink is not observable")
        return metrics

    @staticmethod
    def _deep_counter_fallbacks() -> Dict[str, Metric]:
        note = "Requires DCGM/CUPTI counters; not available from the unprivileged sampler"
        return {
            "sm_compute_utilization": _unsupported("dcgm/cupti", note),
            "tensor_utilization": _unsupported("dcgm/cupti", note),
            "dram_utilization": _unsupported("dcgm/cupti", note),
            "gpu_memory_bandwidth": _unsupported("dcgm/cupti", note),
        }


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
