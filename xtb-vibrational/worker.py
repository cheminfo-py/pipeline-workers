"""
xtb vibrational spectroscopy worker.

Receives an optimized 3D molfile, computes vibrational frequencies, IR
intensities, and Raman activities using xtb-python + ASE, and returns the
full spectral analysis matching the xtbservice API format.

This worker uses the same computational approach as the xtbservice
(https://github.com/cheminfo-py/xtbservice) — ASE's Infrared class and
PlaczekStatic Raman calculator with BondPolarizability, not the xtb CLI.

Requirements:
    pip install sseclient-py requests ase xtb-python rdkit numpy scipy

Environment variables:
    SERVER_URL  - Pipeline server URL (default: http://localhost:5172)
    TOKEN       - Authentication token for the pipeline server
    INSTANCES   - Number of worker instances to run (default: 1)

Usage:
    SERVER_URL=http://pipeline.cheminfo.org TOKEN=your-token python worker.py
"""

import os
import shutil
import tempfile
from math import log, pi, sqrt

import numpy as np
from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.vibrations import Infrared
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.raman import StaticRamanCalculator
from rdkit import Chem
from scipy import spatial
from xtb.ase.calculator import XTB

# Force single-threaded math libraries so each worker instance uses exactly
# one core.
for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

WORKER_NAME = "xtbVibrational"

# Threshold for "large" imaginary frequency (cm-1)
IMAGINARY_FREQ_THRESHOLD = 10


# --- Molfile conversion ---


def molfile_to_ase(molfile):
    """Convert a V2000 molfile to an ASE Atoms object and RDKit Mol.

    Args:
        molfile: V2000 molfile string with 3D coordinates.

    Returns:
        Tuple of (Atoms, Mol).
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
    return atoms, mol


def get_bonds_from_mol(mol):
    """Extract bond list from an RDKit Mol.

    Args:
        mol: RDKit Mol object.

    Returns:
        List of (startAtom, endAtom) tuples (0-based).
    """
    bonds = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return bonds


# --- Spectrum folding (matching xtbservice fold()) ---


def fold(frequencies, intensities, start=800.0, end=4000.0, npts=None,
         width=4.0, folding_type="Gaussian", normalize=False):
    """Fold frequencies and intensities into a broadened spectrum.

    Matches the xtbservice fold() function exactly.

    Args:
        frequencies: Array of vibrational frequencies (cm-1).
        intensities: Array of corresponding intensities.
        start: Start of the wavenumber range (cm-1).
        end: End of the wavenumber range (cm-1).
        npts: Number of output points (auto-calculated if None).
        width: Peak width (cm-1).
        folding_type: "Gaussian" or "Lorentzian".
        normalize: Whether to normalize peaks.

    Returns:
        Tuple of (wavenumbers, spectrum) as numpy arrays.
    """
    if not npts:
        npts = int((end - start) / width * 10 + 1)

    prefactor = 1
    if folding_type.lower() == "lorentzian":
        intensities = np.array(intensities) * width * pi / 2.0
        if normalize:
            prefactor = 2.0 / width / pi
    else:
        sigma = width / 2.0 / sqrt(2.0 * log(2.0))
        if normalize:
            prefactor = 1.0 / sigma / sqrt(2 * pi)

    frequencies = np.array(frequencies)
    intensities = np.array(intensities)
    spectrum = np.empty(npts)
    energies = np.linspace(start, end, npts)

    for i, energy in enumerate(energies):
        if folding_type.lower() == "lorentzian":
            spectrum[i] = (
                intensities * 0.5 * width / pi
                / ((frequencies - energy) ** 2 + 0.25 * width ** 2)
            ).sum()
        else:
            spectrum[i] = (
                intensities * np.exp(-(frequencies - energy) ** 2 / 2.0 / sigma ** 2)
            ).sum()

    return energies, prefactor * spectrum


# --- Mode analysis (matching xtbservice compile_modes_info) ---


def clean_frequency(frequencies, n):
    """Extract frequency value and imaginary flag."""
    if frequencies[n].imag != 0:
        return frequencies[n].imag, "i"
    return frequencies[n].real, " "


def get_alignment(ir, mode_number):
    """Compute displacement alignment for a mode (xtbservice get_alignment)."""
    displacements = ir.get_mode(mode_number)
    dot_result = []
    for i, displ_i in enumerate(displacements):
        for j, displ_j in enumerate(displacements):
            if i < j:
                dot_result.append(spatial.distance.cosine(displ_i, displ_j))
    return np.mean(dot_result) if dot_result else 0.0


def get_displacement_xyz_for_mode(ir, frequencies, symbols, n):
    """Build displacement XYZ string for a mode (xtbservice format)."""
    xyz_file = []
    xyz_file.append("%6d\n" % len(ir.atoms))

    f, c = clean_frequency(frequencies, n)
    xyz_file.append("Mode #%d, f = %.1f%s cm^-1" % (n, float(f.real), c))

    if ir.ir:
        xyz_file.append(", I = %.4f (D/Å)^2 amu^-1.\n" % ir.intensities[n])
    else:
        xyz_file.append(".\n")

    mode = ir.get_mode(n)
    for i, pos in enumerate(ir.atoms.positions):
        xyz_file.append(
            "%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n"
            % (symbols[i], pos[0], pos[1], pos[2],
               mode[i, 0], mode[i, 1], mode[i, 2])
        )

    return "".join(xyz_file)


def get_bond_vector(positions, bond):
    """Get bond vector between two atoms."""
    return positions[bond[1]] - positions[bond[0]]


def get_displaced_positions(positions, mode):
    """Get displaced positions for a mode."""
    return positions + mode


def get_bond_displacements(mol, atoms, mode):
    """Compute bond displacements for a single mode."""
    bonds = get_bonds_from_mol(mol)
    positions = atoms.positions
    displaced_positions = get_displaced_positions(positions, mode) - mode.sum(axis=0)
    changes = []
    for bond in bonds:
        bond_disp = np.linalg.norm(
            get_bond_vector(positions, bond)
        ) - np.linalg.norm(get_bond_vector(displaced_positions, bond))
        changes.append(np.linalg.norm(bond_disp))
    return changes


def compile_all_bond_displacements(mol, atoms, ir):
    """Compile bond displacements for all modes."""
    bond_displacements = []
    for mode_number in range(3 * len(ir.indices)):
        bond_displacements.append(
            get_bond_displacements(mol, atoms, ir.get_mode(mode_number))
        )
    return np.vstack(bond_displacements)


def select_most_contributing_atoms(ir, mode, threshold=0.4):
    """Select atoms that contribute most to a mode."""
    displacements = ir.get_mode(mode)
    norms = np.linalg.norm(displacements, axis=1)
    relative_contribution = norms / norms.max() if norms.max() > 0 else norms
    diff = np.abs(np.diff(relative_contribution))
    max_diff = np.max(diff) if len(diff) > 0 else 1.0
    return np.where(relative_contribution > threshold * max_diff)[0]


def select_most_contributing_bonds(displacements, threshold=0.4):
    """Select bonds that contribute most to a mode."""
    if len(displacements) > 1:
        total = displacements.sum()
        if total > 0:
            relative_contribution = displacements / total
            diff = np.abs(np.diff(relative_contribution))
            max_diff = np.max(diff) if len(diff) > 0 else 1.0
            return np.where(relative_contribution > threshold * max_diff)[0]
    return np.array([0])


def get_moments_of_inertia(positions, masses):
    """Compute principal moments of inertia (xtbservice utils)."""
    com = masses @ positions / masses.sum()
    positions_ = positions - com

    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(positions_)):
        x, y, z = positions_[i]
        m = masses[i]
        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    I = np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]])
    evals, _ = np.linalg.eigh(I)
    return evals


def get_change_in_moi(atoms, ir, mode_number):
    """Compute change in moment of inertia for a mode."""
    return float(np.linalg.norm(
        np.linalg.norm(
            get_moments_of_inertia(
                get_displaced_positions(atoms.positions, ir.get_mode(mode_number)),
                atoms.get_masses(),
            )
        )
        - np.linalg.norm(
            get_moments_of_inertia(atoms.positions, atoms.get_masses())
        )
    ))


def get_max_displacements(ir, linear):
    """Get most relevant modes for each atom (xtbservice get_max_displacements)."""
    mode_abs_displacements = []
    for n in range(3 * len(ir.indices)):
        mode_abs_displacements.append(np.linalg.norm(ir.get_mode(n), axis=1))

    mode_abs_displacements = np.stack(mode_abs_displacements)
    if linear:
        mode_abs_displacements[:5, :] = 0
    else:
        mode_abs_displacements[:6, :] = 0

    return dict(
        zip(
            [int(i) for i in ir.indices],
            [list(int(x) for x in a)[::-1]
             for a in mode_abs_displacements.argsort(axis=0).T],
        )
    )


def compile_modes_info(ir, linear, alignments, bond_displacements, bonds,
                       raman_intensities):
    """Compile rich mode information (xtbservice compile_modes_info)."""
    frequencies = ir.get_frequencies()
    symbols = ir.atoms.get_chemical_symbols()
    modes = []
    sorted_alignments = sorted(alignments, reverse=True)
    mapping = dict(
        zip(np.arange(len(frequencies)), np.argsort(frequencies))
    )
    third_best_alignment = sorted_alignments[2] if len(sorted_alignments) > 2 else 0
    has_imaginary = False
    has_large_imaginary = False
    num_modes = 3 * len(ir.indices)

    if raman_intensities is None:
        raman_intensities = [None] * num_modes

    for n in range(num_modes):
        n = int(mapping[n])
        if n < 3:
            mode_type = "translation"
        elif n < 5:
            mode_type = "rotation"
        elif n == 5:
            if linear:
                mode_type = "vibration"
            elif alignments[n] >= third_best_alignment:
                mode_type = "translation"
            else:
                mode_type = "rotation"
        else:
            mode_type = "vibration"

        f, c = clean_frequency(frequencies, n)
        if c == "i":
            has_imaginary = True
            if f > IMAGINARY_FREQ_THRESHOLD:
                has_large_imaginary = True

        most_contributing_bonds = None
        mode = ir.get_mode(n)
        if bond_displacements is not None:
            most_contributing_bonds = select_most_contributing_bonds(
                bond_displacements[n, :]
            )
            most_contributing_bonds = [bonds[i] for i in most_contributing_bonds]

        raman_intensity = (
            float(raman_intensities[n])
            if raman_intensities[n] is not None
            else None
        )

        modes.append({
            "number": n,
            "displacements": get_displacement_xyz_for_mode(
                ir, frequencies, symbols, n
            ),
            "intensity": float(ir.intensities[n]),
            "ramanIntensity": raman_intensity,
            "wavenumber": float(f),
            "imaginary": c == "i",
            "mostDisplacedAtoms": [
                int(i) for i in np.argsort(
                    np.linalg.norm(mode - mode.sum(axis=0), axis=1)
                )
            ][::-1],
            "mostContributingAtoms": [
                int(i) for i in select_most_contributing_atoms(ir, n)
            ],
            "mostContributingBonds": most_contributing_bonds,
            "modeType": mode_type,
            "centerOfMassDisplacement": float(
                np.linalg.norm(ir.get_mode(n).sum(axis=0))
            ),
            "totalChangeOfMomentOfInteria": get_change_in_moi(
                ir.atoms, ir, n
            ),
            "displacementAlignment": alignments[n],
        })

    return modes, has_imaginary, has_large_imaginary


def get_spectrum(modes, frequencies, intensities, start=0, end=4000,
                 npts=None, width=4, folding_type="Gaussian", normalize=False):
    """Get broadened spectrum filtering only vibration modes."""
    filtered_frequencies = []
    filtered_intensities = []
    for freq, intensity, mode in zip(frequencies, intensities, modes):
        if mode["modeType"] == "vibration":
            filtered_frequencies.append(freq.real)
            filtered_intensities.append(intensity)
    return fold(
        filtered_frequencies, filtered_intensities,
        start, end, npts, width, folding_type, normalize,
    )


# --- Worker processing function ---


def compute_vibrational(data, parameters=None):
    """Run vibrational analysis on an optimized molfile using xtb-python + ASE.

    Uses the same approach as xtbservice: ASE Infrared class for IR,
    StaticRamanCalculator + PlaczekStatic for Raman.

    Args:
        data: Task input dict containing ``molfile`` (an optimized V2000 string).
        parameters: Optional dict with parameters:
            - method: xtb method (default: "GFNFF").
                Options: "GFNFF", "GFN2xTB", "GFN1xTB".
            - charge: Molecular charge (default: 0)
            - multiplicity: Spin multiplicity (default: 1)

    Returns:
        Dict matching the xtbservice IRResult format.
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    method = parameters.get("method", "GFNFF")

    print(f"[{WORKER_NAME}] Parameters: method={method}")

    atoms, mol = molfile_to_ase(molfile)
    atoms.calc = XTB(method=method)

    # Work in a temp directory for ASE cache files
    original_dir = os.getcwd()
    work_dir = tempfile.mkdtemp(prefix="xtb_vib_")
    try:
        os.chdir(work_dir)

        # Compute IR first (always works)
        raman_intensities = None
        raman_spectrum = None
        ir = Infrared(atoms, name="ir")
        ir.run()

        # Compute Raman separately (may fail for some molecules/methods)
        try:
            rm = StaticRamanCalculator(
                atoms, BondPolarizability, name="raman"
            )
            rm.ir = True
            rm.run()
            pz = PlaczekStatic(atoms, name="raman")
            raman_intensities = pz.get_absolute_intensities()
        except Exception as error:
            print(f"[{WORKER_NAME}] Raman calculation failed: {error}")

        zpe = float(ir.get_zero_point_energy())
        moi = atoms.get_moments_of_inertia()
        linear = bool(sum(moi > 0.01) == 2)

        bonds = get_bonds_from_mol(mol)

        # Bond displacements
        bond_displacements = compile_all_bond_displacements(mol, atoms, ir)
        mask = np.zeros_like(bond_displacements)
        if len(atoms) > 2:
            if linear:
                mask[:5, :] = 1
            else:
                mask[:6, :] = 1
            masked_bond_displacements = np.ma.masked_array(
                bond_displacements, mask
            )
        else:
            masked_bond_displacements = bond_displacements

        most_relevant_mode_for_bond_ = masked_bond_displacements.argmax(axis=0)
        most_relevant_mode_for_bond = []
        for i, mode_idx in enumerate(most_relevant_mode_for_bond_):
            most_relevant_mode_for_bond.append({
                "startAtom": bonds[i][0],
                "endAtom": bonds[i][1],
                "mode": int(mode_idx),
                "displacement": float(bond_displacements[mode_idx][i]),
            })

        # Displacement alignments
        alignments = [
            get_alignment(ir, n) for n in range(3 * len(ir.indices))
        ]

        # Compile mode info
        mode_info, has_imaginary, has_large_imaginary = compile_modes_info(
            ir, linear, alignments, bond_displacements, bonds,
            raman_intensities,
        )

        # Broadened spectra
        frequencies = ir.get_frequencies()
        ir_intensities_list = [m["intensity"] for m in mode_info]
        spectrum = get_spectrum(
            mode_info, frequencies, ir_intensities_list,
        )

        try:
            raman_intensities_list = [m["ramanIntensity"] for m in mode_info]
            raman_spectrum = list(
                get_spectrum(
                    mode_info, frequencies, raman_intensities_list,
                )[1]
            )
        except Exception:
            raman_spectrum = None

        result = {
            "wavenumbers": list(float(x) for x in spectrum[0]),
            "intensities": list(float(x) for x in spectrum[1]),
            "ramanIntensities": raman_spectrum,
            "zeroPointEnergy": zpe,
            "modes": mode_info,
            "hasImaginaryFrequency": has_imaginary,
            "mostRelevantModesOfAtoms": get_max_displacements(ir, linear),
            "mostRelevantModesOfBonds": most_relevant_mode_for_bond,
            "isLinear": linear,
            "momentsOfInertia": [float(i) for i in moi],
            "hasLargeImaginaryFrequency": has_large_imaginary,
        }

        return result
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, compute_vibrational)
    client.run()
