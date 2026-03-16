"""Reusable infrastructure for Pipeline distributed workers.

Handles SSE connection, heartbeat, result posting, statistics tracking,
and multi-instance threading. Worker authors only need to provide a
processing function.

Usage::

    from pipeline_worker import WorkerClient

    def process(data, parameters):
        # Do the actual work here.
        return {"result_key": "result_value"}

    if __name__ == "__main__":
        client = WorkerClient("myWorkerName", process)
        client.run()
"""

from pipeline_worker.client import WorkerClient

__all__ = ["WorkerClient"]
