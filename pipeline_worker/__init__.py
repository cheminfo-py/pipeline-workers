"""Reusable infrastructure for Pipeline distributed workers.

Handles SSE connection, heartbeat, result posting, and statistics tracking.
Worker authors only need to provide a processing function. Use Docker
Compose ``deploy.replicas`` to run multiple instances in parallel.

Usage::

    from pipeline_worker import WorkerClient

    def process(data, parameters):
        # Do the actual work here.
        return {"result_key": "result_value"}

    if __name__ == "__main__":
        client = WorkerClient("myWorkerName", process)
        client.run()
"""

from pipeline_worker.client import WorkerClient, log

__all__ = ["WorkerClient", "log"]
