"""
xtb geometry optimization worker.

Receives a 3D molfile, optimizes the geometry using xtb-python + ASE
(LBFGS optimizer), and returns the optimized molfile with the final energy.

Uses the same approach as xtbservice (cheminfo-py/xtbservice).

Requirements:
    pip install sseclient-py requests ase xtb-python rdkit

Environment variables:
    SERVER_URL  - Pipeline server URL (default: http://localhost:5172)
    TOKEN       - Authentication token for the pipeline server
    INSTANCES   - Number of worker instances to run (default: 1)

Usage:
    SERVER_URL=http://pipeline.cheminfo.org TOKEN=your-token python worker.py
"""

import os
import re
import shutil
import tempfile
from copy import deepcopy

from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from rdkit import Chem
from xtb.ase.calculator import XTB

# Force single-threaded math libraries so each worker instance uses exactly
# one core.  With INSTANCES=8 and 8 CPUs, every core runs one xtb process.
# Without this, each xtb spawns many OpenBLAS/OMP threads, exhausting the
# container PID limit and causing "[Errno 11] Resource temporarily unavailable".
for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

WORKER_NAME = "xtbOptimization"

# Regex for a V2000 atom line: x y z symbol ...
ATOM_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\S+)"
)


def molfile_to_ase(molfile):
    """Convert a V2000 molfile to an ASE Atoms object.

    Args:
        molfile: V2000 molfile string with 3D coordinates.

    Returns:
        ASE Atoms object.
    """
    mol = Chem.MolFromMolBlock(molfile, sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("Failed to parse molfile")
    mol.UpdatePropertyCache(strict=False)

    pos = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()
    species = [mol.GetAtomWithIdx(j).GetSymbol() for j in range(num_atoms)]
    atoms = Atoms(species, positions=pos)
    atoms.pbc = False
    return atoms


def update_molfile_coordinates(molfile, positions):
    """Update a V2000 molfile with new atom positions.

    Args:
        molfile: Original V2000 molfile string.
        positions: Numpy array of shape (num_atoms, 3) with new coordinates.

    Returns:
        Updated molfile string with new coordinates.
    """
    mol_lines = molfile.split("\n")
    counts_line = mol_lines[3].strip()
    num_atoms = int(counts_line[:3])

    for i in range(num_atoms):
        x, y, z = positions[i]
        old_line = mol_lines[4 + i]
        symbol_and_rest = old_line[31:]
        mol_lines[4 + i] = f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol_and_rest}"

    return "\n".join(mol_lines)


# --- Worker processing function ---


def optimize_geometry(data, parameters=None):
    """Run xtb geometry optimization on a 3D molfile using ASE LBFGS.

    Uses the same approach as xtbservice: ASE's LBFGS optimizer with
    the xtb-python calculator.

    Args:
        data: Task input dict containing ``molfile`` (a V2000/V3000 string).
        parameters: Optional dict with xtb parameters:
            - method: xtb method (default: "GFNFF").
                Options: "GFNFF", "GFN2xTB", "GFN1xTB".
            - fmax: Force convergence criterion (default: 0.000005 Eh/Angstrom)
            - maxIterations: Max optimization steps (default: 100)

    Returns:
        Dict with "molfile" (optimized) and "energy" (in eV, ASE units).
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    method = parameters.get("method", "GFNFF")
    fmax = parameters.get("fmax", 0.000005)
    max_iterations = parameters.get("maxIterations", 100)

    print(
        f"[{WORKER_NAME}] Parameters: method={method}, "
        f"fmax={fmax}, maxIter={max_iterations}"
    )

    atoms = molfile_to_ase(molfile)
    mol = deepcopy(atoms)
    mol.pbc = False
    mol.calc = XTB(method=method)

    # Work in a temp directory for xtb restart/scratch files
    original_dir = os.getcwd()
    work_dir = tempfile.mkdtemp(prefix="xtb_opt_")
    try:
        os.chdir(work_dir)

        from pipeline_worker.suppress_output import suppress_fortran_output
        with suppress_fortran_output():
            opt = LBFGS(mol, logfile=None)
            opt.run(fmax=fmax, steps=max_iterations)
            energy = float(mol.get_potential_energy())
        optimized_molfile = update_molfile_coordinates(molfile, mol.positions)

        return {"molfile": optimized_molfile, "energy": energy}
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, optimize_geometry)
    client.run()
