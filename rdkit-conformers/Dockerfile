# RDKit is easiest to install via conda
FROM --platform=linux/amd64 condaforge/miniforge3:latest

# Install rdkit via conda, then pip dependencies
RUN conda install -y -c conda-forge rdkit && \
    conda clean -afy

WORKDIR /app

COPY rdkit-conformers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared pipeline_worker package
COPY pipeline_worker /app/pipeline_worker

COPY rdkit-conformers/worker.py .

# Disable Python stdout buffering so logs appear immediately in Docker
ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]
