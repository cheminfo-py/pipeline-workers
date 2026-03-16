"""
Run the xtb IR calculation locally on a sample molfile.

This script demonstrates how to use the compute_ir function
without connecting to the Pipeline server. Useful for development
and testing.

Requirements:
    xtb must be installed and available in PATH.
    Install via conda: conda install -c conda-forge xtb

Usage:
    cd xtb-ir
    python example.py
    python example.py input.mol          # compute IR for a specific molfile
    python example.py input.mol -o out   # write output to out.json
"""

import argparse
import json

from worker import compute_ir

# Ethanol 3D molfile (V2000) — pre-optimized coordinates
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
        description="Run xtb IR calculation (Hessian) on a molfile."
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
        "--method",
        default="GFN2-xTB",
        help="GFN method (default: GFN2-xTB)",
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

    parameters = {"method": arguments.method}

    print(f"Method: {parameters['method']}")
    print("Running IR calculation (Hessian)...")

    # Run the same function used by the pipeline worker
    data = {"molfile": molfile}
    result = compute_ir(data, parameters)

    if "energy" in result:
        print(f"Energy: {result['energy']:.10f} Eh")
    if "zeroPointEnergy" in result:
        print(f"Zero-point energy: {result['zeroPointEnergy']:.10f} Eh")
    print(f"Vibrational modes: {len(result['modes'])}")

    # Write output files
    if arguments.output:
        json_path = f"{arguments.output}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Written: {json_path}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
