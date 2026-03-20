"""
Run the Psi4 optimization locally on a sample molfile.

This script demonstrates how to use the optimize_geometry function
without connecting to the Pipeline server. Useful for development
and testing.

Requirements:
    Install via conda: conda install -c conda-forge psi4 rdkit

Usage:
    cd psi4-optimization
    python example.py
    python example.py input.mol          # optimize a specific molfile
    python example.py input.mol -o out   # write output to out.mol and out.json
"""

import argparse
import json

from worker import optimize_geometry

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
        description="Run Psi4 geometry optimization on a molfile."
    )
    parser.add_argument(
        "molfile",
        nargs="?",
        help="Path to input .mol file (uses built-in ethanol if omitted)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output prefix: writes <prefix>.mol and <prefix>.json",
    )
    parser.add_argument(
        "--method",
        default="b3lyp-d3bj",
        help="Psi4 method (default: b3lyp-d3bj)",
    )
    parser.add_argument(
        "--basis",
        default="def2-svp",
        help="Basis set (default: def2-svp)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (default: 0)",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (default: 1)",
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
        "method": arguments.method,
        "basisSet": arguments.basis,
        "charge": arguments.charge,
        "multiplicity": arguments.multiplicity,
    }

    print(f"Method: {parameters['method']}")
    print(f"Basis: {parameters['basisSet']}")
    print(f"Charge: {parameters['charge']}, Multiplicity: {parameters['multiplicity']}")
    print("Running optimization...")

    data = {"molfile": molfile}
    result = optimize_geometry(data, parameters)

    print(f"Energy: {result['energy']:.10f} Hartrees")

    # Write output files
    if arguments.output:
        mol_path = f"{arguments.output}.mol"
        json_path = f"{arguments.output}.json"
        with open(mol_path, "w") as f:
            f.write(result["molfile"])
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written: {mol_path}, {json_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
