"""
GPU Configuration — Auto-detect GPU properties at runtime.

Supports any NVIDIA GPU. Tested on RTX 2050 (CC 8.6) and H200 (CC 9.0).
Uses torch.cuda.get_device_properties() so no special permissions needed.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


# Known GPU bandwidth and cache specs not exposed by cudaDeviceProp.
# Keyed by (architecture, chip_variant) or GPU name substring.
_KNOWN_GPU_SPECS = {
    # RTX 2050 Laptop (GA107, Ampere, CC 8.6)
    "RTX 2050": {
        "peak_bandwidth_gbps": 128.0,
        "l2_cache_bytes": 2 * 1024 * 1024,        # 2 MB
        "l1_cache_per_sm_bytes": 128 * 1024,       # 128 KB (unified L1 + shared)
    },
    # RTX 3090 (GA102, Ampere, CC 8.6)
    "RTX 3090": {
        "peak_bandwidth_gbps": 936.0,
        "l2_cache_bytes": 6 * 1024 * 1024,         # 6 MB
        "l1_cache_per_sm_bytes": 128 * 1024,
    },
    # RTX 4090 (AD102, Ada Lovelace, CC 8.9)
    "RTX 4090": {
        "peak_bandwidth_gbps": 1008.0,
        "l2_cache_bytes": 72 * 1024 * 1024,        # 72 MB
        "l1_cache_per_sm_bytes": 128 * 1024,
    },
    # A100 (GA100, Ampere, CC 8.0)
    "A100": {
        "peak_bandwidth_gbps": 2039.0,
        "l2_cache_bytes": 40 * 1024 * 1024,        # 40 MB
        "l1_cache_per_sm_bytes": 192 * 1024,
    },
    # H100 (GH100, Hopper, CC 9.0)
    "H100": {
        "peak_bandwidth_gbps": 3350.0,
        "l2_cache_bytes": 50 * 1024 * 1024,        # 50 MB
        "l1_cache_per_sm_bytes": 256 * 1024,
    },
    # H200 (GH100, Hopper, CC 9.0)
    "H200": {
        "peak_bandwidth_gbps": 4800.0,
        "l2_cache_bytes": 50 * 1024 * 1024,        # 50 MB
        "l1_cache_per_sm_bytes": 256 * 1024,
    },
}


def _match_gpu_specs(gpu_name: str) -> dict:
    """Match a GPU name to known specs. Returns best match or conservative defaults."""
    gpu_name_upper = gpu_name.upper()
    for key, specs in _KNOWN_GPU_SPECS.items():
        if key.upper() in gpu_name_upper:
            return specs
    # Conservative defaults for unknown GPUs
    return {
        "peak_bandwidth_gbps": 100.0,
        "l2_cache_bytes": 2 * 1024 * 1024,
        "l1_cache_per_sm_bytes": 128 * 1024,
    }


@dataclass
class GPUConfig:
    """
    GPU configuration auto-detected from the current device.
    All fields are populated automatically — no hardcoding needed.
    """
    # Identity
    name: str = ""
    compute_capability: tuple = (0, 0)
    device_index: int = 0

    # Compute resources
    num_sms: int = 0
    max_threads_per_sm: int = 0
    max_threads_per_block: int = 0
    warp_size: int = 32
    max_warps_per_sm: int = 0
    num_warp_schedulers_per_sm: int = 4  # 4 for all modern NVIDIA GPUs

    # Memory
    total_memory_bytes: int = 0
    peak_bandwidth_gbps: float = 0.0

    # Cache
    l2_cache_bytes: int = 0
    l1_cache_per_sm_bytes: int = 0
    shared_mem_per_sm_bytes: int = 0
    shared_mem_per_block_bytes: int = 0

    # Registers
    registers_per_sm: int = 0
    registers_per_block: int = 0

    # Concurrent execution
    supports_concurrent_kernels: bool = False
    max_blocks_per_sm: int = 0

    @classmethod
    def detect(cls, device_index: int = 0) -> "GPUConfig":
        """
        Auto-detect GPU configuration from the current CUDA device.
        Works on any NVIDIA GPU — no root or special permissions needed.

        Args:
            device_index: CUDA device index (default 0)

        Returns:
            GPUConfig with all fields populated
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install PyTorch with CUDA support:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        props = torch.cuda.get_device_properties(device_index)
        gpu_name = props.name

        # Look up specs not available via cudaDeviceProp
        known = _match_gpu_specs(gpu_name)

        # Safe attribute lookup with architecture-aware fallbacks
        l2_bytes = getattr(props, "L2_cache_size", 0)
        if not l2_bytes:
            l2_bytes = known.get("l2_cache_bytes", 2 * 1024 * 1024)

        config = cls(
            name=gpu_name,
            compute_capability=(props.major, props.minor),
            device_index=device_index,
            # Compute
            num_sms=props.multi_processor_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
            warp_size=getattr(props, "warp_size", 32),
            max_warps_per_sm=props.max_threads_per_multi_processor // getattr(props, "warp_size", 32),
            # Memory
            total_memory_bytes=props.total_memory,
            peak_bandwidth_gbps=known.get("peak_bandwidth_gbps", 128.0),
            # Cache
            l2_cache_bytes=l2_bytes,
            l1_cache_per_sm_bytes=known.get("l1_cache_per_sm_bytes", 128 * 1024),
            shared_mem_per_sm_bytes=getattr(props, "max_shared_memory_size", 100 * 1024),
            shared_mem_per_block_bytes=getattr(props, "shared_memory_per_block", 48 * 1024),
            # Registers
            registers_per_sm=getattr(props, "regs_per_multiprocessor", 65536),
            registers_per_block=getattr(props, "regs_per_block", 65536),
            # Concurrent
            supports_concurrent_kernels=True,  # All modern GPUs support this
        )

        # Max blocks per SM (hardware limit, typically 16 or 32)
        # Not directly in props, use architecture-based defaults
        if config.compute_capability[0] >= 8:
            config.max_blocks_per_sm = 32  # Ampere, Ada, Hopper
        elif config.compute_capability[0] >= 7:
            config.max_blocks_per_sm = 32  # Volta, Turing
        else:
            config.max_blocks_per_sm = 32  # Conservative default

        return config

    def safe_test_array_bytes(self, fraction: float = 0.25) -> int:
        """Return a safe array size for testing (fraction of total memory)."""
        return int(self.total_memory_bytes * fraction)

    def half_sm_threads(self) -> int:
        """Threads per block to fill half an SM (for co-location experiments)."""
        return self.max_threads_per_sm // 2

    def __str__(self) -> str:
        mem_gb = self.total_memory_bytes / (1024 ** 3)
        l2_mb = self.l2_cache_bytes / (1024 ** 2)
        l1_kb = self.l1_cache_per_sm_bytes / 1024
        return (
            f"=== GPU Configuration ===\n"
            f"  GPU:                  {self.name}\n"
            f"  Compute Capability:   {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"  SMs:                  {self.num_sms}\n"
            f"  Max threads/SM:       {self.max_threads_per_sm}\n"
            f"  Max warps/SM:         {self.max_warps_per_sm}\n"
            f"  Max threads/block:    {self.max_threads_per_block}\n"
            f"  Total Memory:         {mem_gb:.1f} GB\n"
            f"  Peak Bandwidth:       {self.peak_bandwidth_gbps:.0f} GB/s\n"
            f"  L2 Cache:             {l2_mb:.1f} MB\n"
            f"  L1 Cache/SM:          {l1_kb:.0f} KB\n"
            f"  Registers/SM:         {self.registers_per_sm}\n"
            f"  Max blocks/SM:        {self.max_blocks_per_sm}\n"
            f"========================="
        )
