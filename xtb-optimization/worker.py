"""
xtb geometry optimization worker.

Receives a 3D molfile with energy, optimizes the geometry using xtb (GFN2-xTB),
and returns the optimized molfile with the final energy.

Requirements:
    pip install sseclient-py requests
    xtb must be installed and available in PATH.

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
import subprocess
import tempfile

# Force single-threaded math libraries so each worker instance uses exactly
# one core.  With INSTANCES=8 and 8 CPUs, every core runs one xtb process.
# Without this, each xtb spawns many OpenBLAS/OMP threads, exhausting the
# container PID limit and causing "[Errno 11] Resource temporarily unavailable".
for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

WORKER_NAME = "xtbOptimization"

# --- Molfile / XYZ conversion ---

# Regex for a V2000 atom line: x y z symbol ...
ATOM_RE = re.compile(
    r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\S+)"
)


def molfile_to_xyz(molfile):
    """Convert a V2000 molfile to XYZ format using plain string parsing.

    Args:
        molfile: V2000 molfile string with 3D coordinates.

    Returns:
        XYZ file content as a string.
    """
    lines = molfile.split("\n")
    # Line 4 (index 3) is the counts line: "aaabbb..." where aaa = num atoms
    counts_line = lines[3].strip()
    num_atoms = int(counts_line[:3])

    xyz_lines = [str(num_atoms), ""]
    for i in range(num_atoms):
        atom_line = lines[4 + i]
        match = ATOM_RE.match(atom_line)
        if not match:
            raise ValueError(f"Cannot parse atom line {4 + i}: {atom_line}")
        x, y, z, symbol = match.group(1), match.group(2), match.group(3), match.group(4)
        xyz_lines.append(f"{symbol}  {x}  {y}  {z}")

    return "\n".join(xyz_lines) + "\n"


def update_molfile_coordinates(molfile, xyz_path):
    """Update a V2000 molfile with optimized coordinates from an XYZ file.

    Args:
        molfile: Original V2000 molfile string.
        xyz_path: Path to the optimized XYZ file from xtb.

    Returns:
        Updated molfile string with new coordinates.
    """
    with open(xyz_path) as f:
        xyz_lines = f.readlines()

    mol_lines = molfile.split("\n")
    counts_line = mol_lines[3].strip()
    num_atoms = int(counts_line[:3])

    for i in range(num_atoms):
        parts = xyz_lines[i + 2].split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        # V2000 atom line: positions 0-9 (x), 10-19 (y), 20-29 (z), 31-33 (symbol), rest
        old_line = mol_lines[4 + i]
        symbol_and_rest = old_line[31:]
        mol_lines[4 + i] = f"{x:10.4f}{y:10.4f}{z:10.4f} {symbol_and_rest}"

    return "\n".join(mol_lines)


def parse_xtb_energy(work_dir, stdout=""):
    """Parse the final energy from xtb output.

    Tries multiple sources in order: energy file, xtbopt.log, stdout.

    Args:
        work_dir: Directory where xtb was run.
        stdout: Captured stdout from the xtb process.

    Returns:
        Energy in Hartree as a float.
    """
    # 1. Parse from the xtb energy file (most reliable)
    energy_path = os.path.join(work_dir, "energy")
    if os.path.exists(energy_path):
        with open(energy_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue

    # 2. Fallback: parse from xtbopt.log
    log_path = os.path.join(work_dir, "xtbopt.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            energy = _parse_energy_from_text(f)
            if energy is not None:
                return energy

    # 3. Fallback: parse from stdout
    if stdout:
        energy = _parse_energy_from_text(stdout.splitlines())
        if energy is not None:
            return energy

    raise ValueError("Could not parse energy from xtb output")


def _parse_energy_from_text(lines):
    """Extract 'total energy' value from xtb text output.

    Args:
        lines: Iterable of text lines.

    Returns:
        Energy as float, or None if not found.
    """
    for line in lines:
        if "total energy" in line.lower():
            parts = line.split()
            for j, part in enumerate(parts):
                if part == "Eh" and j > 0:
                    try:
                        return float(parts[j - 1])
                    except (ValueError, IndexError):
                        continue
    return None


# --- Worker processing function ---


def optimize_geometry(data, parameters=None):
    """Run xtb geometry optimization on a 3D molfile.

    This is the processing function passed to :class:`WorkerClient`.

    Args:
        data: Task input dict containing ``molfile`` (a V2000/V3000 string).
        parameters: Optional dict with xtb parameters:
            - method: GFN method (default: "GFN2-xTB")
            - optimizationLevel: Optimization level (default: "normal")
            - charge: Molecular charge (default: 0)
            - multiplicity: Spin multiplicity (default: 1)
            - maxIterations: Max optimization steps (default: 200)
            - solvent: Solvent for ALPB model (e.g., "water", "methanol")

    Returns:
        Dict with "molfile" (optimized) and "energy" (in Hartree).
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    method = parameters.get("method", "GFN2-xTB")
    optimization_level = parameters.get("optimizationLevel", "normal")
    charge = parameters.get("charge", 0)
    multiplicity = parameters.get("multiplicity", 1)
    max_iterations = parameters.get("maxIterations", 200)
    solvent = parameters.get("solvent")

    print(
        f"[{WORKER_NAME}] Parameters: method={method}, "
        f"opt={optimization_level}, charge={charge}, "
        f"multiplicity={multiplicity}, maxIter={max_iterations}"
        + (f", solvent={solvent}" if solvent else "")
    )

    xyz_content = molfile_to_xyz(molfile)

    work_dir = tempfile.mkdtemp(prefix="xtb_")
    try:
        input_xyz = os.path.join(work_dir, "input.xyz")
        with open(input_xyz, "w") as f:
            f.write(xyz_content)

        # Build xtb command
        cmd = ["xtb", "input.xyz", "--opt", optimization_level]

        # Method flag
        method_lower = method.lower()
        if "gfn2" in method_lower:
            cmd.extend(["--gfn", "2"])
        elif "gfn1" in method_lower:
            cmd.extend(["--gfn", "1"])
        elif "gfn0" in method_lower:
            cmd.extend(["--gfn", "0"])
        elif "gfn-ff" in method_lower or "gfnff" in method_lower:
            cmd.append("--gfnff")

        cmd.extend(["--chrg", str(charge)])
        uhf = multiplicity - 1
        if uhf > 0:
            cmd.extend(["--uhf", str(uhf)])
        cmd.extend(["--iterations", str(max_iterations)])

        if solvent:
            cmd.extend(["--alpb", solvent])

        # Single-threaded: one xtb per core, no internal parallelism
        cmd.extend(["--parallel", "1"])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=500,
        )

        # Send subprocess output to the server log (not Docker logs)
        try:
            from pipeline_worker import log
            log(result.stdout)
            log(result.stderr)
        except ImportError:
            pass

        if result.returncode != 0:
            raise RuntimeError(
                f"xtb failed (exit {result.returncode}): {result.stderr[-500:]}"
            )

        # Read optimized coordinates
        opt_xyz = os.path.join(work_dir, "xtbopt.xyz")
        if not os.path.exists(opt_xyz):
            raise RuntimeError("xtb did not produce optimized geometry")

        optimized_molfile = update_molfile_coordinates(molfile, opt_xyz)
        energy = parse_xtb_energy(work_dir, result.stdout)

        return {"molfile": optimized_molfile, "energy": energy}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, optimize_geometry)
    client.run()
