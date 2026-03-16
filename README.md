# Pipeline Workers (Python)

Python-based distributed workers for the [Pipeline](https://github.com/cheminfo/pipeline) system. Each worker connects via SSE, receives tasks, processes them, and posts results back. The shared `pipeline_worker` package handles all server communication — worker authors only write the processing function.

## Architecture

```
pipeline-workers/
├── pipeline_worker/         # Shared infrastructure package
│   ├── __init__.py
│   └── client.py            # WorkerClient class
├── xtb-optimization/   # xtb geometry optimization worker
│   ├── worker.py            # Processing logic + entry point
│   ├── example.py           # Standalone local test script
│   ├── Dockerfile
│   └── requirements.txt
├── xtb-vibrational/             # xtb IR spectroscopy worker
│   ├── worker.py
│   ├── example.py
│   ├── Dockerfile
│   └── requirements.txt
├── rdkit-conformers/    # RDKit conformer generation worker
│   ├── worker.py
│   ├── example.py
│   ├── Dockerfile
│   └── requirements.txt
└── README.md
```

## IR prediction pipeline

The workers can be chained together in the Pipeline server to build an IR prediction workflow:

```
molfile ──> rdkitConformers ──> xtbOptimization ──> xtbVibrational ──> IR spectrum
              │                        │                       │
              ▼                        ▼                       ▼
         conformers[]            optimized molfile      frequencies + intensities
         + energies              + energy               + zero-point energy
```

| Step | Worker | Input | Output |
| ---- | ------ | ----- | ------ |
| 1 | `rdkitConformers` | molfile | conformers[] (molfile + energy each) |
| 2 | `xtbOptimization` | molfile | optimized molfile + energy |
| 3 | `xtbVibrational` | optimized molfile | IR + Raman spectra, modes, moments of inertia |

## Creating a new worker

### 1. Create the worker directory

```bash
mkdir my-worker
```

### 2. Write the processing function

Create `my-worker/worker.py`:

```python
WORKER_NAME = "myWorker"


def process(data, parameters):
    """Process a task.

    Args:
        data: Task input (parsed JSON matching the worker's input schema).
        parameters: Worker parameters dict, or None.

    Returns:
        Result dict matching the worker's output schema.
    """
    # Your computation here. For example:
    value = data["inputField"]
    result = do_something(value, parameters)
    return {"outputField": result}


if __name__ == "__main__":
    from pipeline_worker import WorkerClient

    client = WorkerClient(WORKER_NAME, process)
    client.run()
```

The `data` argument contains the task input as a Python dict. Its structure matches the worker's input schema as defined in the Pipeline server.

The `parameters` argument contains the worker's parameters (from the pipeline configuration), or `None` if no parameters are set. These are the same for every task dispatched to this worker.

The return value must be a dict matching the worker's output schema.

If the function raises an exception, the error message is sent back to the server and the task is marked as failed.

### 3. Add a local test script (optional)

Create `my-worker/example.py` to test the processing function without connecting to the server:

```python
import json

from worker import process

sample_input = {"inputField": "test value"}
result = process(sample_input, parameters=None)
print(json.dumps(result, indent=2))
```

See [xtb-optimization/example.py](xtb-optimization/example.py) for a full example with argument parsing and file I/O.

### 4. Add requirements.txt

```
sseclient-py>=1.8
requests>=2.31
```

Add any additional dependencies your worker needs (e.g., `numpy`, `scipy`).

### 5. Add a Dockerfile

Create `my-worker/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY my-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_worker /app/pipeline_worker
COPY my-worker/worker.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
```

The Docker build context must be the repository root so that the shared `pipeline_worker` package is accessible.

### 6. Add to Docker Compose

In the Pipeline project's compose file:

```yaml
my-worker:
  build:
    context: /path/to/pipeline-workers
    dockerfile: my-worker/Dockerfile
  env_file:
    - .env
  environment:
    - SERVER_URL=http://pipeline:60313
    - INSTANCES=4
  depends_on:
    - pipeline
```

### 7. Register the worker in the Pipeline server

The worker must be registered in the database with its name, input schema, and output schema. This is done via the admin API or in the seed data.

## Prerequisites

Some workers require external tools. Install them before running locally.

### xtb (required by `xtb-optimization` and `xtb-vibrational`)

[xtb](https://github.com/grimme-lab/xtb) is a semi-empirical quantum chemistry program used for geometry optimization. Install it via conda-forge:

```bash
# Using mamba (recommended, faster than conda)
mamba install -c conda-forge xtb

# Or using conda
conda install -c conda-forge xtb
```

If you don't have conda/mamba, install [Miniforge](https://github.com/conda-forge/miniforge) first:

```bash
brew install miniforge    # macOS
conda init zsh            # or bash — restart your shell after this
```

Verify the installation:

```bash
xtb --version
```

> **Note:** Make sure `xtb` is in your PATH. If you installed it in a conda environment, activate that environment before running the worker.

### RDKit (required by `rdkit-conformers`)

[RDKit](https://www.rdkit.org/) is a cheminformatics toolkit used for conformer generation. Install via conda-forge:

```bash
mamba install -c conda-forge rdkit

# Or using conda
conda install -c conda-forge rdkit
```

Verify the installation:

```bash
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO').GetNumAtoms())"
```

## Running locally

### Development mode (connected to server)

```bash
SERVER_URL=http://localhost:5172 TOKEN=your-token python xtb-optimization/worker.py
```

### Standalone test (no server needed)

Each worker can include an `example.py` script that runs the processing function locally without connecting to the Pipeline server. This is useful for verifying that external tools are installed and working correctly.

#### xtb-optimization

```bash
cd xtb-optimization
python example.py                        # uses built-in ethanol molfile
python example.py input.mol              # optimize a specific molfile
python example.py input.mol -o output    # writes output.mol and output.json
python example.py --method GFN-FF        # use a faster force-field method
```

Expected output:

```
Input: built-in ethanol molfile
Method: GFN2-xTB
Optimization level: normal
Running optimization...
[xtbOptimization] Parameters: method=GFN2-xTB, opt=normal, charge=0, multiplicity=1, maxIter=200
Energy: -11.2111622673 Eh
{ "molfile": "...", "energy": -11.211162267257 }
```

#### xtb-vibrational

```bash
cd xtb-vibrational
python example.py                        # uses built-in ethanol molfile
python example.py input.mol              # compute IR for a specific molfile
python example.py input.mol -o output    # writes output.json
python example.py --method GFN-FF        # use a faster force-field method
```

#### rdkit-conformers

```bash
cd rdkit-conformers
python example.py                              # uses built-in ethanol molfile
python example.py input.mol                    # generate conformers for a specific molfile
python example.py input.mol -o output          # writes output.json
python example.py --max-conformers 20          # generate more conformers
python example.py --force-field UFF            # use UFF instead of MMFF94
```

## WorkerClient API

```python
from pipeline_worker import WorkerClient

client = WorkerClient(
    worker_name="myWorker",     # Must match the name in the Pipeline server
    process_fn=my_function,     # Callable(data, parameters) -> result
    server_url="...",           # Optional, defaults to SERVER_URL env var
    token="...",                # Optional, defaults to TOKEN env var
)
client.run()                    # Starts listening (blocks forever)
```

### Environment variables

| Variable     | Description                         | Default                                |
| ------------ | ----------------------------------- | -------------------------------------- |
| `SERVER_URL` | Pipeline server URL                 | `http://localhost:5172`                |
| `TOKEN`      | Authentication token                | `f47ac10b-58cc-4372-a567-0e02b2c3d479` |
| `INSTANCES`  | Number of concurrent worker threads | `1`                                    |

### What WorkerClient handles

- SSE connection with automatic reconnection
- Heartbeat messages every 30 seconds during task execution
- Result posting with exponential-backoff retries (3 attempts)
- Per-instance runner ID and system metadata
- Thread-safe statistics tracking (completed/failed tasks, average time)
- Multi-instance threading via the `INSTANCES` environment variable
