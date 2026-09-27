"""
engine/system_specs.py
======================
Comprehensive host system hardware specification and telemetry querying.
Accurately reports real physical hardware specs (CPU brand, physical cores,
logical threads, total/available RAM, disk storage, and GPU acceleration)
separate from internal engine worker thread allocation.
"""

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field

import psutil


@dataclass
class HardwareSpecs:
    """Encapsulates host machine hardware specs and compute capabilities."""

    cpu_name: str
    physical_cores: int
    logical_threads: int
    ram_total_gb: float
    ram_available_gb: float
    ram_percent_used: float
    disk_total_gb: float
    disk_free_gb: float
    disk_percent_used: float
    gpus: list[str] = field(default_factory=list)
    cuda_available: bool = False
    cuda_device_name: str | None = None
    cuda_vram_gb: float | None = None
    recommended_threads: int = 4


def query_cpu_name() -> str:
    """Retrieves accurate CPU brand string across platforms (e.g. 'Intel(R) Core(TM) 5 120U')."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and val.strip():
                return val.strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            if out.strip():
                return out.strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown Processor"


def query_installed_gpus() -> list[str]:
    """Discovers installed display adapters (integrated and discrete)."""
    gpus: list[str] = []
    if sys.platform == "win32":
        try:
            import winreg

            base_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_path)
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    i += 1
                    if subkey_name.isdigit():
                        sub = winreg.OpenKey(key, subkey_name)
                        try:
                            desc, _ = winreg.QueryValueEx(sub, "DriverDesc")
                            if desc and desc.strip() and desc.strip() not in gpus:
                                gpus.append(desc.strip())
                        except Exception:
                            pass
                except OSError:
                    break
        except Exception:
            pass

    return gpus or ["Integrated Graphics"]


def query_cuda_acceleration() -> tuple[bool, str | None, float | None]:
    """Queries NVIDIA CUDA GPU availability and dedicated VRAM via PyTorch/CTranslate2."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            try:
                import torch

                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
                    return True, name, round(total_vram, 2)
            except Exception:
                return True, "NVIDIA CUDA Device", None
    except Exception:
        pass
    return False, None, None


def calculate_optimal_cpu_threads(
    total_ram_gb: float | None = None,
    logical_threads: int | None = None,
    avail_ram_gb: float | None = None,
) -> int:
    """
    Dynamically computes the optimal CTranslate2 CPU thread allocation based on host specs.
    Balances inference parallelism against Intel MKL thread-local memory buffer overhead,
    prioritizing real-time available RAM headroom to protect against mkl_malloc exhaustion.
    """
    if logical_threads is None:
        logical_threads = psutil.cpu_count(logical=True) or 4

    if total_ram_gb is None:
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024.0**3)
        except Exception:
            total_ram_gb = 8.0

    if avail_ram_gb is None:
        try:
            avail_ram_gb = psutil.virtual_memory().available / (1024.0**3)
        except Exception:
            avail_ram_gb = None

    # Base hardware tier capping based on total installed RAM
    if total_ram_gb <= 8.5:
        max_safe = 4
    elif total_ram_gb <= 16.5:
        max_safe = 6
    elif total_ram_gb <= 32.5:
        max_safe = 8
    else:
        max_safe = 12

    # Dynamic headroom scaling based on actual available RAM pool:
    # MADLAD-400 3B INT8 requires ~2.8 GB base weights. Each OpenMP worker thread allocates
    # ~150-200 MB for Intel MKL scratchpads. When memory headroom is constricted,
    # scale threads down to prevent allocation failure.
    if avail_ram_gb is not None:
        if avail_ram_gb <= 2.2:
            max_safe = min(max_safe, 2)
        elif avail_ram_gb <= 3.5:
            max_safe = min(max_safe, 3)
        elif avail_ram_gb <= 5.0:
            max_safe = min(max_safe, 4)

    return max(1, min(max_safe, logical_threads))


def get_hardware_specs(target_path: str = ".") -> HardwareSpecs:
    """Queries and returns the full machine hardware specifications."""
    cpu_name = query_cpu_name()
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_threads = psutil.cpu_count(logical=True) or physical_cores

    vm = psutil.virtual_memory()
    ram_total_gb = vm.total / (1024.0**3)
    ram_available_gb = vm.available / (1024.0**3)
    ram_percent_used = vm.percent

    target_dir = os.path.abspath(target_path)
    if not os.path.exists(target_dir):
        target_dir = os.path.dirname(target_dir) or "."
    disk = shutil.disk_usage(target_dir)
    disk_total_gb = disk.total / (1024.0**3)
    disk_free_gb = disk.free / (1024.0**3)
    disk_percent_used = round(100.0 * (1.0 - (disk.free / disk.total)), 1) if disk.total else 0.0

    gpus = query_installed_gpus()
    has_cuda, cuda_name, cuda_vram = query_cuda_acceleration()
    recommended_threads = calculate_optimal_cpu_threads(ram_total_gb, logical_threads)

    return HardwareSpecs(
        cpu_name=cpu_name,
        physical_cores=physical_cores,
        logical_threads=logical_threads,
        ram_total_gb=round(ram_total_gb, 2),
        ram_available_gb=round(ram_available_gb, 2),
        ram_percent_used=round(ram_percent_used, 1),
        disk_total_gb=round(disk_total_gb, 2),
        disk_free_gb=round(disk_free_gb, 2),
        disk_percent_used=disk_percent_used,
        gpus=gpus,
        cuda_available=has_cuda,
        cuda_device_name=cuda_name,
        cuda_vram_gb=cuda_vram,
        recommended_threads=recommended_threads,
    )
