"""
xtb IR spectroscopy worker.

Receives an optimized 3D molfile, computes vibrational frequencies and IR
intensities using xtb (--hess), and returns the spectrum data.

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

WORKER_NAME = "xtbIr"

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


def parse_vibspectrum(work_dir):
    """Parse the vibspectrum file produced by xtb --hess.

    The vibspectrum file has a header section (lines starting with '$' or '#')
    followed by data lines in one of two formats:

    With symmetry label (vibrational modes)::

        7        a            326.63         0.00000         NO

    Without symmetry label (translational/rotational modes)::

        1                      -0.00         0.00000          -

    The frequency and intensity columns shift depending on whether a
    symmetry label is present. We detect this by checking if the second
    field is numeric.

    Args:
        work_dir: Directory where xtb was run.

    Returns:
        List of dicts with 'frequency' and 'intensity' for each mode,
        excluding translational/rotational modes (frequency ~ 0).
    """
    vibspectrum_path = os.path.join(work_dir, "vibspectrum")
    if not os.path.exists(vibspectrum_path):
        raise RuntimeError("xtb did not produce vibspectrum file")

    modes = []
    with open(vibspectrum_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("$") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                # Detect whether a symmetry label is present by trying to
                # parse parts[1] as a float.  If it works, there is no
                # symmetry column; if it fails, skip over the label.
                try:
                    frequency = float(parts[1])
                    intensity = float(parts[2])
                except ValueError:
                    # parts[1] is a symmetry label like "a"
                    frequency = float(parts[2])
                    intensity = float(parts[3])
                # Skip translational/rotational modes (frequency ~ 0)
                if abs(frequency) < 1.0:
                    continue
                modes.append({"frequency": frequency, "intensity": intensity})
            except (ValueError, IndexError):
                continue

    if not modes:
        raise RuntimeError("No vibrational modes found in vibspectrum file")

    return modes


def parse_xtb_energy(work_dir, stdout=""):
    """Parse the final energy from xtb output.

    Tries multiple sources in order: energy file, stdout.

    Args:
        work_dir: Directory where xtb was run.
        stdout: Captured stdout from the xtb process.

    Returns:
        Energy in Hartree as a float, or None if not found.
    """
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

    if stdout:
        for line in stdout.splitlines():
            if "total energy" in line.lower():
                parts = line.split()
                for j, part in enumerate(parts):
                    if part == "Eh" and j > 0:
                        try:
                            return float(parts[j - 1])
                        except (ValueError, IndexError):
                            continue
    return None


def parse_zero_point_energy(stdout):
    """Parse the zero-point energy from xtb stdout.

    Args:
        stdout: Captured stdout from the xtb process.

    Returns:
        Zero-point energy in Hartree as a float, or None if not found.
    """
    for line in stdout.splitlines():
        if "zero point energy" in line.lower():
            parts = line.split()
            for j, part in enumerate(parts):
                if part == "Eh" and j > 0:
                    try:
                        return float(parts[j - 1])
                    except (ValueError, IndexError):
                        continue
    return None


# --- Worker processing function ---


def compute_ir(data, parameters=None):
    """Run xtb Hessian calculation to obtain IR spectrum from an optimized molfile.

    This is the processing function passed to :class:`WorkerClient`.

    Args:
        data: Task input dict containing ``molfile`` (an optimized V2000 string).
        parameters: Optional dict with xtb parameters:
            - method: GFN method (default: "GFN2-xTB")
            - charge: Molecular charge (default: 0)
            - multiplicity: Spin multiplicity (default: 1)
            - solvent: Solvent for ALPB model (e.g., "water", "methanol")

    Returns:
        Dict with:
            - "modes": list of {frequency, intensity} for each vibrational mode
            - "energy": total energy in Hartree (if available)
            - "zeroPointEnergy": zero-point energy in Hartree (if available)
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    method = parameters.get("method", "GFN2-xTB")
    charge = parameters.get("charge", 0)
    multiplicity = parameters.get("multiplicity", 1)
    solvent = parameters.get("solvent")

    print(
        f"[{WORKER_NAME}] Parameters: method={method}, "
        f"charge={charge}, multiplicity={multiplicity}"
        + (f", solvent={solvent}" if solvent else "")
    )

    xyz_content = molfile_to_xyz(molfile)

    work_dir = tempfile.mkdtemp(prefix="xtb_ir_")
    try:
        input_xyz = os.path.join(work_dir, "input.xyz")
        with open(input_xyz, "w") as f:
            f.write(xyz_content)

        # Build xtb command for Hessian calculation (produces vibspectrum)
        cmd = ["xtb", "input.xyz", "--hess"]

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

        if solvent:
            cmd.extend(["--alpb", solvent])

        # Single-threaded: one xtb per core, no internal parallelism
        cmd.extend(["--parallel", "1"])

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Send subprocess output to the server log (not Docker logs)
        from pipeline_worker import log
        log(result.stdout)
        log(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"xtb failed (exit {result.returncode}): {result.stderr[-500:]}"
            )

        modes = parse_vibspectrum(work_dir)
        energy = parse_xtb_energy(work_dir, result.stdout)
        zero_point_energy = parse_zero_point_energy(result.stdout)

        output = {"modes": modes}
        if energy is not None:
            output["energy"] = energy
        if zero_point_energy is not None:
            output["zeroPointEnergy"] = zero_point_energy

        return output
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, compute_ir)
    client.run()
