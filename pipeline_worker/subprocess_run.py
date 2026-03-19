"""Run a function in a subprocess to isolate Fortran/C crashes.

xtb-python's Fortran library can call ``exit()`` or ``STOP``, which
terminates the entire process.  Running the computation in a child
process ensures that a crash only kills the child — the parent worker
stays alive, reports the error, and continues processing tasks.
"""

import multiprocessing
import traceback


def run_in_subprocess(fn, *args, **kwargs):
    """Run *fn* in a child process and return its result.

    Args:
        fn: Callable to execute.
        *args: Positional arguments for *fn*.
        **kwargs: Keyword arguments for *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        RuntimeError: If the child process crashes or returns an error.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_worker, args=(fn, result_queue, args, kwargs)
    )
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"Computation crashed (exit code {process.exitcode})"
        )

    if result_queue.empty():
        raise RuntimeError("Computation returned no result")

    ok, value = result_queue.get_nowait()
    if ok:
        return value
    raise RuntimeError(value)


def _worker(fn, result_queue, args, kwargs):
    """Child process entry point."""
    try:
        result = fn(*args, **kwargs)
        result_queue.put((True, result))
    except Exception:
        result_queue.put((False, traceback.format_exc()))
