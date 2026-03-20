FROM condaforge/miniforge3:latest

# Install Psi4 and RDKit via conda
RUN conda install -y -c conda-forge psi4 rdkit dftd3-python && \
    conda clean -afy

WORKDIR /app

COPY psi4-optimization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared pipeline_worker package
COPY pipeline_worker /app/pipeline_worker

COPY psi4-optimization/worker.py .

# Disable Python stdout buffering so logs appear immediately in Docker
# Psi4's C++ PSIOManager reads PSI_SCRATCH before any Python API call.
# Point it to tmpfs so scratch files work in a read-only container.
ENV PSI_SCRATCH=/tmp
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
