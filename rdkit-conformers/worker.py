"""
Conformer generation worker.

Receives a 3D molfile, generates multiple conformers using RDKit
(ETKDGv3 + force field minimization), and returns the conformer library
sorted by energy.

Requirements:
    pip install sseclient-py requests rdkit
    (rdkit can also be installed via conda: conda install -c conda-forge rdkit)

Environment variables:
    SERVER_URL  - Pipeline server URL (default: http://localhost:3000)
    TOKEN       - Authentication token for the pipeline server
    INSTANCES   - Number of worker instances to run (default: 1)

Usage:
    SERVER_URL=http://pipeline.cheminfo.org TOKEN=your-token python worker.py
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

WORKER_NAME = "rdkitConformers"


def generate_conformers(data, parameters=None):
    """Generate 3D conformers from a molfile using RDKit.

    This is the processing function passed to :class:`WorkerClient`.

    Args:
        data: Task input dict containing ``molfile`` (a V2000/V3000 string).
        parameters: Optional dict with conformer generation parameters:
            - maxConformers: Maximum number of conformers to return (default: 10)
            - rmsdThreshold: RMSD threshold for pruning similar conformers (default: 0.5)
            - forceField: Force field for minimization: "UFF", "MMFF94", "MMFF94s"
                (default: "MMFF94")
            - poolMultiplier: Multiplier for initial conformer pool size (default: 10)

    Returns:
        Dict with "conformers": list of {molfile, energy} sorted by energy.
    """
    molfile = data["molfile"]
    if parameters is None:
        parameters = {}

    max_conformers = parameters.get("maxConformers", 10)
    rmsd_threshold = parameters.get("rmsdThreshold", 0.5)
    force_field = parameters.get("forceField", "MMFF94")
    pool_multiplier = parameters.get("poolMultiplier", 10)

    print(
        f"[{WORKER_NAME}] Parameters: maxConformers={max_conformers}, "
        f"rmsd={rmsd_threshold}, forceField={force_field}, "
        f"poolMultiplier={pool_multiplier}"
    )

    mol = Chem.MolFromMolBlock(molfile, removeHs=False)
    if mol is None:
        raise ValueError("Failed to parse molfile")

    # Generate a pool of conformers using ETKDGv3
    pool_size = max_conformers * pool_multiplier
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = rmsd_threshold
    params.numThreads = 1
    params.randomSeed = 42

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=pool_size, params=params)
    if not conf_ids:
        raise RuntimeError("Failed to generate any conformers")

    print(f"[{WORKER_NAME}] Generated {len(conf_ids)} initial conformers")

    # Minimize each conformer with the selected force field and collect energies
    energies = {}
    if force_field.upper() == "UFF":
        for conf_id in conf_ids:
            result = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            if result == -1:
                continue
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is not None:
                energies[conf_id] = ff.CalcEnergy()
    else:
        # MMFF94 or MMFF94s
        variant = force_field.upper()
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if mmff_props is None:
            raise RuntimeError(
                f"Failed to compute {variant} properties for molecule"
            )
        for conf_id in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, mmff_props, confId=conf_id
            )
            if ff is None:
                continue
            ff.Minimize(maxIts=500)
            energies[conf_id] = ff.CalcEnergy()

    if not energies:
        raise RuntimeError("All conformer minimizations failed")

    # Sort by energy and prune by RMSD
    sorted_ids = sorted(energies, key=lambda cid: energies[cid])
    kept_ids = _prune_by_rmsd(mol, sorted_ids, rmsd_threshold)

    # Take only the requested number
    kept_ids = kept_ids[:max_conformers]

    print(
        f"[{WORKER_NAME}] Returning {len(kept_ids)} conformers "
        f"(after RMSD pruning)"
    )

    # Build output
    conformers = []
    for conf_id in kept_ids:
        conf_molfile = Chem.MolToMolBlock(mol, confId=conf_id)
        conformers.append({
            "molfile": conf_molfile,
            "energy": energies[conf_id],
        })

    return {"conformers": conformers}


def _prune_by_rmsd(mol, sorted_ids, threshold):
    """Remove conformers that are too similar to already-kept ones.

    Args:
        mol: RDKit Mol object with embedded conformers.
        sorted_ids: Conformer IDs sorted by energy (lowest first).
        threshold: RMSD threshold below which conformers are pruned.

    Returns:
        List of kept conformer IDs.
    """
    kept = []
    for conf_id in sorted_ids:
        is_duplicate = False
        for kept_id in kept:
            rmsd = rdMolAlign.GetBestRMS(mol, mol, kept_id, conf_id)
            if rmsd < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(conf_id)
    return kept


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, generate_conformers)
    client.run()
