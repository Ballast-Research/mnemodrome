"""
CPU affinity utilities for resource isolation.

Pins the MCP server process to specific CPU cores to avoid contention
with other services (like job-manager) on the same host.
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_cpu_count() -> int:
    """Get total number of CPUs available."""
    return os.cpu_count() or 4


def set_cpu_affinity(cpu_cores: list[int] | None = None) -> bool:
    """
    Set CPU affinity for the current process.

    Binds this process to specific CPU cores to reduce contention
    with other services.

    Args:
        cpu_cores: List of CPU core indices to bind to.
                  If None, auto-allocates first half of available cores.
                  Example: [0, 1, 2, 3] for 8-core system (MCP)

    Returns:
        True if successful, False if not supported on this OS
    """
    try:
        import psutil

        total_cpus = get_cpu_count()

        # Default: use first half of cores (MCP gets first half)
        # For 8-core system: MCP gets [0,1,2,3], job-manager gets [4,5,6,7]
        if cpu_cores is None:
            half = total_cpus // 2
            cpu_cores = list(range(0, half))

        process = psutil.Process(os.getpid())
        process.cpu_affinity(cpu_cores)

        logger.info(
            f"✅ CPU affinity set: PID {os.getpid()} bound to cores {cpu_cores} "
            f"({len(cpu_cores)}/{total_cpus} CPUs)"
        )
        return True

    except ImportError:
        logger.warning(
            "psutil not installed - CPU affinity unavailable. "
            "Install with: pip install psutil"
        )
        return False
    except Exception as e:
        logger.warning(f"CPU affinity setup failed: {e}")
        return False
