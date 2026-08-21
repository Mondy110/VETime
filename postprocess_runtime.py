"""Runtime limits for CPU-bound TSB metric post-processing."""

import multiprocessing as mp
import os
from contextlib import contextmanager


DEFAULT_POSTPROCESS_WORKERS = 4


def resolve_postprocess_workers(requested_workers, cpu_count=None):
    """Return a conservative process count while reserving two logical CPUs."""
    available_cpus = mp.cpu_count() if cpu_count is None else cpu_count
    usable_cpus = max(1, available_cpus - 2)
    if requested_workers is None:
        return min(DEFAULT_POSTPROCESS_WORKERS, usable_cpus)
    if requested_workers <= 0:
        raise ValueError("postprocess worker count must be positive")
    return min(requested_workers, usable_cpus)


def configure_postprocess_worker(cpu_threads_per_worker=1):
    """Prevent each metric process from expanding into a PyTorch CPU thread pool."""
    if cpu_threads_per_worker <= 0:
        raise ValueError("cpu_threads_per_worker must be positive")

    import torch

    torch.set_num_threads(cpu_threads_per_worker)
    try:
        torch.set_num_interop_threads(cpu_threads_per_worker)
    except RuntimeError:
        # Inter-op threads can only be set once; a spawned worker may already
        # have initialized them while importing its dependencies.
        pass


@contextmanager
def postprocess_thread_limits(cpu_threads_per_worker=1):
    """Temporarily cap CPU math libraries before spawning metric workers."""
    if cpu_threads_per_worker <= 0:
        raise ValueError("cpu_threads_per_worker must be positive")

    thread_variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in thread_variables}
    try:
        for name in thread_variables:
            os.environ[name] = str(cpu_threads_per_worker)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
