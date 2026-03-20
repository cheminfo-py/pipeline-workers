"""
Psi4 geometry optimization worker.

Receives a 3D molfile, optimizes the geometry using Psi4, and returns
the optimized molfile with the final energy.

Requirements:
    conda install -c conda-forge psi4 rdkit
    pip install sseclient-py requests

Environment variables:
    SERVER_URL  - Pipeline server URL (default: http://localhost:5172)
    TOKEN       - Authentication token for the pipeline server
    CPUS        - Number of CPUs for this container (default: 1)

Usage:
    SERVER_URL=http://pipeline.cheminfo.org TOKEN=your-token python worker.py
"""

import os
import re
import shutil
import tempfile

from rdkit import Chem

# Set math library thread count from CPUS env var.
_cpus = os.environ.get("CPUS", "1")
for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_var, _cpus)

WORKER_NAME = "psi4Optimization"

# Methods that include their own basis set (composite methods).
COMPOSITE_METHODS = {"r2scan3c"}

# Regex for a V2000 atom line: x y z symbol ...
ATOM_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\S+)"
)


def molfile_to_xyz(molfile, charge, multiplicity):
    """Convert a V2000 molfile to a Psi4-compatible XYZ string.

    Args:
        molfile: V2000 molfile string with 3D coordinates.
        charge: Molecular charge (integer).
        multiplicity: Spin multiplicity (integer).

    Returns:
        XYZ string for psi4.geometry().
    """
    mol = Chem.MolFromMolBlock(molfile, sanitize=True, removeHs=False)
    if mol is None:
        raise ValueError("Failed to parse molfile")
    mol.UpdatePropertyCache(strict=False)

    conformer = mol.GetConformer()
    lines = [f"{charge} {multiplicity}"]
    for i in range(mol.GetNumAtoms()):
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        position = conformer.GetAtomPosition(i)
        lines.append(f"{symbol}  {position.x:.6f}  {position.y:.6f}  {position.z:.6f}")

    return "\n".join(lines)


def update_molfile_coordinates(molfile, positions):
    """Update a V2000 molfile with new atom positions.

    Args:
        molfile: Original V2000 molfile string.
        positions: List of (x, y, z) tuples with new coordinates.

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
    """Run Psi4 geometry optimization on a 3D molfile.

    The computation runs in a subprocess so that a crash in Psi4
    does not take down the main worker process.

    Args:
        data: Task input dict containing ``molfile`` (a V2000/V3000 string).
        parameters: Optional dict with Psi4 parameters:
            - method: Calculation method (default: "b3lyp-d3bj").
            - basisSet: Basis set (default: "def2-svp"). Ignored for composite methods.
            - charge: Molecular charge (default: 0).
            - multiplicity: Spin multiplicity (default: 1).
            - maxIterations: Max optimization steps (default: 100).

    Returns:
        Dict with "molfile" (optimized) and "energy" (in Hartrees).
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    method = parameters.get("method", "b3lyp-d3bj")
    basis_set = parameters.get("basisSet", "def2-svp")
    charge = parameters.get("charge", 0)
    multiplicity = parameters.get("multiplicity", 1)
    max_iterations = parameters.get("maxIterations", 100)

    # Composite methods include their own basis set.
    if method in COMPOSITE_METHODS:
        basis_set = None

    print(
        f"[{WORKER_NAME}] Parameters: method={method}, basis={basis_set}, "
        f"charge={charge}, multiplicity={multiplicity}, maxIter={max_iterations}"
    )

    from pipeline_worker.subprocess_run import run_in_subprocess

    return run_in_subprocess(
        _run_optimization, molfile, method, basis_set, charge, multiplicity,
        max_iterations
    )


def _run_optimization(molfile, method, basis_set, charge, multiplicity,
                       max_iterations):
    """Run the Psi4 optimization in a subprocess.

    Args:
        molfile: V2000 molfile string.
        method: Psi4 method name.
        basis_set: Basis set name, or None for composite methods.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        max_iterations: Max optimization steps.

    Returns:
        Dict with "molfile" (optimized) and "energy" (in Hartrees).
    """
    cpus = int(os.environ.get("CPUS", "1"))

    # Psi4's C++ PSIOManager initializes scratch on import.
    # Create the work dir and set PSI_SCRATCH + CWD before importing psi4.
    work_dir = tempfile.mkdtemp(prefix="psi4_opt_")
    os.environ["PSI_SCRATCH"] = work_dir
    original_dir = os.getcwd()
    os.chdir(work_dir)

    try:
        import psi4

        psi4.set_num_threads(cpus)
        psi4.core.IOManager.shared_object().set_default_path(work_dir + "/")
        psi4.core.set_output_file(os.path.join(work_dir, "output.dat"), False)

        xyz_string = molfile_to_xyz(molfile, charge, multiplicity)
        molecule = psi4.geometry(xyz_string)
        molecule.set_molecular_charge(charge)
        molecule.set_multiplicity(multiplicity)

        options = {
            "geom_maxiter": max_iterations,
            "reference": "rhf" if multiplicity == 1 else "uhf",
        }
        if basis_set is not None:
            options["basis"] = basis_set

        psi4.set_options(options)

        method_spec = method
        energy = psi4.optimize(method_spec, molecule=molecule)

        # Extract optimized coordinates.
        geometry_matrix = molecule.geometry().np
        positions = [
            (geometry_matrix[i][0] * 0.529177249,
             geometry_matrix[i][1] * 0.529177249,
             geometry_matrix[i][2] * 0.529177249)
            for i in range(molecule.natom())
        ]

        optimized_molfile = update_molfile_coordinates(molfile, positions)

        return {"molfile": optimized_molfile, "energy": float(energy)}
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, optimize_geometry)
    client.run()
