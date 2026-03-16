"""
Run the conformer generation locally on a sample molfile.

This script demonstrates how to use the generate_conformers function
without connecting to the Pipeline server. Useful for development
and testing.

Requirements:
    rdkit must be installed.
    Install via conda: conda install -c conda-forge rdkit

Usage:
    cd rdkit-conformers
    python example.py
    python example.py input.mol                 # generate conformers for a specific molfile
    python example.py input.mol -o out          # write output to out.json
    python example.py --max-conformers 20       # generate more conformers
"""

import argparse
import json

from worker import generate_conformers

# Ethanol 3D molfile (V2000) with approximate coordinates
SAMPLE_MOLFILE = """\

  Actelion Java MolfileCreator 1.0

  9  8  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124    1.7400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1280    0.1800    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.9424    0.2200    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.5200    0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.5200   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    1.2200   -0.8900 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  2  7  1  0  0  0  0
  2  8  1  0  0  0  0
  3  9  1  0  0  0  0
M  END
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D conformers from a molfile using RDKit."
    )
    parser.add_argument(
        "molfile",
        nargs="?",
        help="Path to input .mol file (uses built-in ethanol if omitted)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output prefix: writes <prefix>.json",
    )
    parser.add_argument(
        "--max-conformers",
        type=int,
        default=10,
        help="Maximum number of conformers (default: 10)",
    )
    parser.add_argument(
        "--force-field",
        default="MMFF94",
        help="Force field: UFF, MMFF94, MMFF94s (default: MMFF94)",
    )
    parser.add_argument(
        "--rmsd-threshold",
        type=float,
        default=0.5,
        help="RMSD threshold for pruning (default: 0.5)",
    )
    arguments = parser.parse_args()

    # Read input molfile
    if arguments.molfile:
        with open(arguments.molfile) as f:
            molfile = f.read()
        print(f"Input: {arguments.molfile}")
    else:
        molfile = SAMPLE_MOLFILE
        print("Input: built-in ethanol molfile")

    parameters = {
        "maxConformers": arguments.max_conformers,
        "forceField": arguments.force_field,
        "rmsdThreshold": arguments.rmsd_threshold,
    }

    print(f"Force field: {parameters['forceField']}")
    print(f"Max conformers: {parameters['maxConformers']}")
    print(f"RMSD threshold: {parameters['rmsdThreshold']}")
    print("Generating conformers...")

    # Run the same function used by the pipeline worker
    data = {"molfile": molfile}
    result = generate_conformers(data, parameters)

    num_conformers = len(result["conformers"])
    print(f"Generated {num_conformers} conformer(s)")
    for i, conformer in enumerate(result["conformers"]):
        print(f"  Conformer {i + 1}: energy = {conformer['energy']:.4f} kcal/mol")

    # Write output files
    if arguments.output:
        json_path = f"{arguments.output}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written: {json_path}")
    else:
        # Print just the summary (full JSON would be very long with molfiles)
        summary = {
            "numConformers": num_conformers,
            "energies": [c["energy"] for c in result["conformers"]],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
