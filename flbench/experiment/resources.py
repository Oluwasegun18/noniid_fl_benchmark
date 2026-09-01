from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import psutil
import torch


def payload_num_bytes(payload: Any) -> int:
    """Return the in-memory tensor payload size in bytes.

    The function recursively handles tensors, state dictionaries, lists,
    tuples, and scalar metadata. It intentionally measures tensor payloads,
    not Python serialization overhead.
    """
    if payload is None:
        return 0
    if isinstance(payload, torch.Tensor):
        return int(payload.numel() * payload.element_size())
    if isinstance(payload, dict):
        return sum(payload_num_bytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(payload_num_bytes(value) for value in payload)
    if isinstance(payload, (bool, int, float)):
        return 8
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    return 0


def model_num_bytes(model: torch.nn.Module) -> int:
    return payload_num_bytes(model.state_dict())


def state_dict_num_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return payload_num_bytes(state_dict)


@dataclass
class EnergyReading:
    cpu_energy_measured_j: Optional[float]
    gpu_energy_measured_j: Optional[float]
    compute_energy_modeled_j: Optional[float]
    # Backward-compatible hybrid total. Do not use this field as the primary
    # ranking metric because it can mix measured and modeled components.
    compute_energy_reported_j: Optional[float]
    # Explicitly separated energy quantities for analysis.
    gpu_energy_primary_j: Optional[float]
    total_energy_hybrid_j: Optional[float]
    cpu_utilization_fraction: Optional[float]
    elapsed_sec: float
    energy_source: str
    energy_is_estimated: bool
    energy_role: str
    energy_valid_for_ranking: bool
    energy_invalid_reason: Optional[str]
    gpu_counter_scope: str
    gpu_uuid: Optional[str]
    concurrent_gpu_processes_detected: bool

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        values = asdict(self)
        if not prefix:
            return values
        return {f"{prefix}{key}": value for key, value in values.items()}


@dataclass
class _RaplDomain:
    energy_path: Path
    max_range_path: Optional[Path]

    def read_joules(self) -> Optional[float]:
        try:
            return float(self.energy_path.read_text().strip()) / 1_000_000.0
        except (OSError, ValueError):
            return None

    def max_range_joules(self) -> Optional[float]:
        if self.max_range_path is None:
            return None
        try:
            return float(self.max_range_path.read_text().strip()) / 1_000_000.0
        except (OSError, ValueError):
            return None


class _NvmlPowerSampler:
    """Integrate NVML power samples when cumulative GPU energy is unavailable."""

    def __init__(self, pynvml_module, handle, interval_sec: float) -> None:
        self.pynvml = pynvml_module
        self.handle = handle
        self.interval_sec = max(float(interval_sec), 0.02)
        self.samples: list[tuple[float, float]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                watts = (
                    float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle))
                    / 1000.0
                )
                self.samples.append((time.perf_counter(), watts))
            except Exception:
                pass
            self.stop_event.wait(self.interval_sec)

    def start(self) -> None:
        self.samples = []
        self.stop_event.clear()
        self._sample_once()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample_once(self) -> None:
        try:
            watts = (
                float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle))
                / 1000.0
            )
            self.samples.append((time.perf_counter(), watts))
        except Exception:
            pass

    def stop(self) -> Optional[float]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_sec * 4))
        self._sample_once()

        if len(self.samples) < 2:
            return None

        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples[:-1], self.samples[1:]):
            dt = max(0.0, t1 - t0)
            energy_j += 0.5 * (p0 + p1) * dt
        return energy_j


class EnergyTracker:
    """Cross-platform computation-energy tracker.

    Measurement priority:
      1. Intel/AMD RAPL-compatible Linux energy counters for CPU/package.
      2. NVML cumulative energy counter for supported NVIDIA GPUs.
      3. NVML sampled-power integration if cumulative energy is unavailable.
      4. Configured CPU power model as a portable fallback, including Windows.

    Modeled energy is clearly marked as estimated and should not be mixed with
    hardware measurements without reporting the source.
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        *,
        phase: str = "compute",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.phase = phase

        energy_cfg = cfg.get("resources", {}).get("energy", {})
        self.energy_role = str(energy_cfg.get("role", "descriptive")).lower()
        self.require_valid_for_ranking = bool(
            energy_cfg.get("require_valid_for_ranking", False)
        )
        self.check_gpu_isolation = bool(
            energy_cfg.get("check_gpu_isolation", True)
        )
        legacy_mode = cfg.get("resources", {}).get("energy_measurement", "auto")
        self.mode = str(energy_cfg.get("mode", legacy_mode)).lower()
        self.allow_modeled_fallback = bool(
            energy_cfg.get("allow_modeled_fallback", True)
        )

        cpu_cfg = energy_cfg.get("cpu", {})
        self.cpu_power_model = str(
            cpu_cfg.get("power_model", "process_utilization")
        ).lower()
        self.cpu_active_power_w = float(
            cpu_cfg.get("active_power_watts", 25.0)
        )
        self.cpu_idle_power_w = float(cpu_cfg.get("idle_power_watts", 5.0))
        self.include_idle_power = bool(
            cpu_cfg.get("include_idle_power", False)
        )

        gpu_cfg = energy_cfg.get("gpu", {})
        self.gpu_device_index = int(gpu_cfg.get("device_index", 0))
        self.gpu_sample_interval_sec = float(
            gpu_cfg.get("sample_interval_sec", 0.1)
        )

        self._process = psutil.Process(os.getpid())
        self._start_wall: Optional[float] = None
        self._start_process_cpu_sec: Optional[float] = None

        self._rapl_domains = self._discover_rapl_domains()
        self._rapl_start: list[Optional[float]] = []

        self._pynvml = None
        self._gpu_handle = None
        self._gpu_start_j: Optional[float] = None
        self._gpu_counter_supported = False
        self._gpu_power_sampler: Optional[_NvmlPowerSampler] = None
        self._gpu_counter_scope = "unavailable"
        self._gpu_uuid: Optional[str] = None
        self._foreign_gpu_pids: set[int] = set()
        self._isolation_stop = threading.Event()
        self._isolation_thread: Optional[threading.Thread] = None
        self._initialize_nvml()

    @staticmethod
    def _discover_rapl_domains() -> list[_RaplDomain]:
        if platform.system().lower() != "linux":
            return []

        root = Path("/sys/class/powercap")
        if not root.exists():
            return []

        domains: list[_RaplDomain] = []
        # Use top-level package domains only to avoid double counting nested
        # core/DRAM domains. Common names are intel-rapl:0, intel-rapl:1, etc.
        for directory in sorted(root.glob("*-rapl:*")):
            if ":" in directory.name.split("rapl:")[-1]:
                continue
            energy_path = directory / "energy_uj"
            if not energy_path.exists():
                continue
            max_path = directory / "max_energy_range_uj"
            domains.append(
                _RaplDomain(
                    energy_path=energy_path,
                    max_range_path=max_path if max_path.exists() else None,
                )
            )
        return domains

    def _visible_cuda_token(self) -> Optional[str]:
        """Return the CUDA_VISIBLE_DEVICES token for the logical device.

        SLURM commonly remaps physical GPUs to logical CUDA index 0. Using
        NVML index 0 directly can therefore target the wrong physical device.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            return None
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        logical_index = self.device.index if self.device.index is not None else 0
        if 0 <= logical_index < len(tokens):
            return tokens[logical_index]
        return None

    def _initialize_nvml(self) -> None:
        if self.device.type != "cuda":
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            token = self._visible_cuda_token()
            handle = None

            if token and (token.startswith("GPU-") or token.startswith("MIG-")):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(token)
                except TypeError:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(token.encode())
                self._gpu_counter_scope = (
                    "mig_device" if token.startswith("MIG-") else "physical_gpu"
                )
            elif token and token.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(token))
                self._gpu_counter_scope = "physical_gpu"
            else:
                index = (
                    self.device.index
                    if self.device.index is not None
                    else self.gpu_device_index
                )
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self._gpu_counter_scope = "physical_gpu"

            self._pynvml = pynvml
            self._gpu_handle = handle
            try:
                raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(raw_uuid, bytes):
                    raw_uuid = raw_uuid.decode()
                self._gpu_uuid = str(raw_uuid)
                if self._gpu_uuid.startswith("MIG-"):
                    self._gpu_counter_scope = "mig_device"
            except Exception:
                self._gpu_uuid = token

            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                self._gpu_counter_supported = True
            except Exception:
                self._gpu_counter_supported = False
                self._gpu_power_sampler = _NvmlPowerSampler(
                    pynvml,
                    handle,
                    self.gpu_sample_interval_sec,
                )
        except Exception:
            self._pynvml = None
            self._gpu_handle = None
            self._gpu_counter_scope = "unavailable"

    def _foreign_compute_pids(self) -> set[int]:
        if (
            not self.check_gpu_isolation
            or self._pynvml is None
            or self._gpu_handle is None
        ):
            return set()

        processes = None
        for function_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            function = getattr(self._pynvml, function_name, None)
            if function is None:
                continue
            try:
                processes = function(self._gpu_handle)
                break
            except Exception:
                continue
        if processes is None:
            return set()

        own_pids = {os.getpid()}
        try:
            own_pids.update(child.pid for child in self._process.children(recursive=True))
        except Exception:
            pass
        return {
            int(proc.pid)
            for proc in processes
            if getattr(proc, "pid", None) is not None
            and int(proc.pid) not in own_pids
        }

    def _start_isolation_monitor(self) -> None:
        if (
            self.energy_role != "ranking"
            or not self.check_gpu_isolation
            or self._gpu_handle is None
        ):
            return

        def monitor() -> None:
            while not self._isolation_stop.is_set():
                self._foreign_gpu_pids.update(self._foreign_compute_pids())
                self._isolation_stop.wait(1.0)

        self._isolation_stop.clear()
        self._isolation_thread = threading.Thread(target=monitor, daemon=True)
        self._isolation_thread.start()

    def _stop_isolation_monitor(self) -> None:
        self._isolation_stop.set()
        if self._isolation_thread is not None:
            self._isolation_thread.join(timeout=2.0)
            self._isolation_thread = None
        self._foreign_gpu_pids.update(self._foreign_compute_pids())

    def _sync_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                torch.cuda.synchronize()

    def start(self) -> None:
        self._sync_device()
        self._start_wall = time.perf_counter()

        cpu_times = self._process.cpu_times()
        self._start_process_cpu_sec = float(cpu_times.user + cpu_times.system)

        self._rapl_start = [domain.read_joules() for domain in self._rapl_domains]

        if self._gpu_handle is not None:
            self._foreign_gpu_pids.update(self._foreign_compute_pids())
            self._start_isolation_monitor()
            if self._gpu_counter_supported:
                try:
                    self._gpu_start_j = (
                        float(
                            self._pynvml.nvmlDeviceGetTotalEnergyConsumption(
                                self._gpu_handle
                            )
                        )
                        / 1000.0
                    )
                except Exception:
                    self._gpu_start_j = None
            elif self._gpu_power_sampler is not None:
                self._gpu_power_sampler.start()

    @staticmethod
    def _counter_delta(
        start: Optional[float],
        end: Optional[float],
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        if start is None or end is None:
            return None
        if end >= start:
            return end - start
        if maximum is not None and maximum > start:
            return (maximum - start) + end
        return None

    def _stop_cpu_measurement(self) -> Optional[float]:
        values: list[float] = []
        for domain, start in zip(self._rapl_domains, self._rapl_start):
            end = domain.read_joules()
            delta = self._counter_delta(start, end, domain.max_range_joules())
            if delta is not None:
                values.append(delta)
        return sum(values) if values else None

    def _stop_gpu_measurement(self) -> tuple[Optional[float], Optional[str]]:
        if self._gpu_handle is None:
            return None, None

        if self._gpu_counter_supported and self._gpu_start_j is not None:
            try:
                end_j = (
                    float(
                        self._pynvml.nvmlDeviceGetTotalEnergyConsumption(
                            self._gpu_handle
                        )
                    )
                    / 1000.0
                )
                delta = self._counter_delta(self._gpu_start_j, end_j)
                if delta is not None:
                    return delta, "nvml_total_energy"
            except Exception:
                pass

        if self._gpu_power_sampler is not None:
            sampled = self._gpu_power_sampler.stop()
            if sampled is not None:
                return sampled, "nvml_power_integration"

        return None, None

    def _process_utilization(self, elapsed_sec: float) -> Optional[float]:
        if (
            self._start_process_cpu_sec is None
            or elapsed_sec <= 0
        ):
            return None
        end_times = self._process.cpu_times()
        end_cpu_sec = float(end_times.user + end_times.system)
        used_cpu_sec = max(0.0, end_cpu_sec - self._start_process_cpu_sec)
        logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        return min(max(used_cpu_sec / (elapsed_sec * logical_cpus), 0.0), 1.0)

    def _modeled_cpu_energy(
        self,
        elapsed_sec: float,
        utilization: Optional[float],
    ) -> float:
        if self.cpu_power_model == "constant":
            power_w = self.cpu_active_power_w
        else:
            u = 0.0 if utilization is None else utilization
            if self.include_idle_power:
                power_w = self.cpu_idle_power_w + u * (
                    self.cpu_active_power_w - self.cpu_idle_power_w
                )
            else:
                # Attribute only the estimated active share to this process.
                power_w = u * self.cpu_active_power_w
        return max(0.0, power_w * elapsed_sec)

    def stop(self) -> EnergyReading:
        self._sync_device()
        if self._start_wall is None:
            raise RuntimeError("EnergyTracker.start() must be called first.")

        elapsed_sec = max(0.0, time.perf_counter() - self._start_wall)
        utilization = self._process_utilization(elapsed_sec)

        cpu_measured = self._stop_cpu_measurement()
        gpu_measured, gpu_method = self._stop_gpu_measurement()
        self._stop_isolation_monitor()

        measured_values = [
            value
            for value in (cpu_measured, gpu_measured)
            if value is not None
        ]

        modeled: Optional[float] = None
        source_parts: list[str] = []
        if cpu_measured is not None:
            source_parts.append("cpu:rapl")
        if gpu_measured is not None:
            source_parts.append(f"gpu:{gpu_method}")

        # Model CPU energy only when CPU hardware measurement is unavailable.
        # The hybrid total remains available descriptively, but measured GPU
        # energy is the primary accelerator metric for algorithm ranking.
        if cpu_measured is None and self.allow_modeled_fallback:
            modeled = self._modeled_cpu_energy(elapsed_sec, utilization)
            source_parts.append(f"cpu_modeled:{self.cpu_power_model}")

        reported_values = list(measured_values)
        if modeled is not None:
            reported_values.append(modeled)

        hybrid_total = sum(reported_values) if reported_values else None
        gpu_primary = gpu_measured
        is_estimated = modeled is not None
        source = "+".join(source_parts) or "unavailable"

        invalid_reasons: list[str] = []
        if self.energy_role == "ranking":
            if gpu_primary is None:
                invalid_reasons.append("gpu_energy_unavailable")
            if self._foreign_gpu_pids:
                invalid_reasons.append(
                    "concurrent_gpu_processes:"
                    + ",".join(str(pid) for pid in sorted(self._foreign_gpu_pids))
                )
            if self._gpu_counter_scope == "unavailable":
                invalid_reasons.append("gpu_counter_scope_unavailable")

        valid_for_ranking = not invalid_reasons and gpu_primary is not None

        return EnergyReading(
            cpu_energy_measured_j=cpu_measured,
            gpu_energy_measured_j=gpu_measured,
            compute_energy_modeled_j=modeled,
            compute_energy_reported_j=hybrid_total,
            gpu_energy_primary_j=gpu_primary,
            total_energy_hybrid_j=hybrid_total,
            cpu_utilization_fraction=utilization,
            elapsed_sec=elapsed_sec,
            energy_source=source,
            energy_is_estimated=is_estimated,
            energy_role=self.energy_role,
            energy_valid_for_ranking=valid_for_ranking,
            energy_invalid_reason=(";".join(invalid_reasons) or None),
            gpu_counter_scope=self._gpu_counter_scope,
            gpu_uuid=self._gpu_uuid,
            concurrent_gpu_processes_detected=bool(self._foreign_gpu_pids),
        )


def communication_metrics(
    *,
    download_bytes: int,
    upload_bytes: int,
    selected_clients: int,
    cfg: dict,
) -> dict[str, float]:
    """Return sequential and synchronous-parallel network models.

    `modeled_comm_time_sequential_sec` sums transfers for all clients.
    `modeled_comm_time_parallel_sec` approximates synchronous parallel links
    where all selected clients receive/upload concurrently at the configured
    per-client rates and share the same one-way latency.
    """
    comm_cfg = cfg.get("resources", {}).get("communication", {})

    down_mbps = float(comm_cfg.get("downlink_mbps", 100.0))
    up_mbps = float(comm_cfg.get("uplink_mbps", 20.0))
    latency_ms = float(comm_cfg.get("one_way_latency_ms", 0.0))
    down_j_per_byte = float(comm_cfg.get("downlink_j_per_byte", 0.0))
    up_j_per_byte = float(comm_cfg.get("uplink_j_per_byte", 0.0))

    down_time = (download_bytes * 8.0) / max(down_mbps * 1_000_000.0, 1e-12)
    up_time = (upload_bytes * 8.0) / max(up_mbps * 1_000_000.0, 1e-12)
    sequential = down_time + up_time + 2.0 * latency_ms / 1000.0

    clients = max(int(selected_clients), 1)
    per_client_download = download_bytes / clients
    per_client_upload = upload_bytes / clients
    parallel = (
        (per_client_download * 8.0) / max(down_mbps * 1_000_000.0, 1e-12)
        + (per_client_upload * 8.0) / max(up_mbps * 1_000_000.0, 1e-12)
        + 2.0 * latency_ms / 1000.0
    )

    communication_energy = (
        download_bytes * down_j_per_byte
        + upload_bytes * up_j_per_byte
    )

    return {
        "download_bytes": float(download_bytes),
        "upload_bytes": float(upload_bytes),
        "total_comm_bytes": float(download_bytes + upload_bytes),
        "modeled_comm_time_sequential_sec": sequential,
        "modeled_comm_time_parallel_sec": parallel,
        # Backward-compatible alias.
        "modeled_comm_time_sec": parallel,
        "modeled_comm_energy_j": communication_energy,
    }


def system_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_device_name": None,
    }
    if torch.cuda.is_available():
        try:
            index = device.index if device.index is not None else 0
            metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        except Exception:
            pass
    return metadata
