"""Suppress C-level stdout/stderr output from Fortran libraries.

xtb-python's Fortran library writes directly to file descriptors 1 and 2,
bypassing Python's sys.stdout/sys.stderr. This module provides a context
manager to redirect those file descriptors to /dev/null during computation,
keeping Docker logs clean.
"""

import contextlib
import os


@contextlib.contextmanager
def suppress_fortran_output():
    """Redirect C-level stdout/stderr to /dev/null.

    Use this around xtb-python / ASE calculations to prevent Fortran
    output from flooding Docker logs.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stdout)
        os.close(old_stderr)
