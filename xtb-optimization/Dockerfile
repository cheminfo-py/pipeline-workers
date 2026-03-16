# Force amd64 platform — xtb's ARM64 build has numerical bugs that cause
# GFN2-xTB geometry optimizations to diverge.
FROM --platform=linux/amd64 condaforge/miniforge3:latest

# Install xtb-python, ASE, and RDKit via conda
RUN conda install -y -c conda-forge xtb-python ase rdkit && \
    conda clean -afy

WORKDIR /app

COPY xtb-optimization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared pipeline_worker package
COPY pipeline_worker /app/pipeline_worker

COPY xtb-optimization/worker.py .

# Disable Python stdout buffering so logs appear immediately in Docker
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
