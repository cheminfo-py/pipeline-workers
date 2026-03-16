"""Pipeline worker client — SSE listener, heartbeat, result posting.

This module provides the :class:`WorkerClient` class that encapsulates all
communication with the Pipeline server so that individual workers only need
to implement a processing function.
"""

import io
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid

import requests
import sseclient

HEARTBEAT_INTERVAL = 30
MAX_RESULT_RETRIES = 3
MAX_LOG_SIZE = 1_048_576


class _TeeStream:
    """Write to both the original stream and a capture buffer."""

    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original.flush()
        self.buffer.flush()


def _get_system_info(runner_id):
    """Gather system metadata sent on each SSE connection.

    Args:
        runner_id: Unique runner UUID for this instance.

    Returns:
        Dict with runner metadata fields.
    """
    try:
        cpu_count = str(os.cpu_count() or 1)
    except Exception:
        cpu_count = "1"
    return {
        "runnerId": runner_id,
        "hostname": socket.gethostname(),
        "cpuCount": cpu_count,
        "totalMemory": "0",
        "freeMemory": "0",
    }


class WorkerClient:
    """Connects to the Pipeline server, receives tasks via SSE, and posts results.

    Args:
        worker_name: Name of the worker as registered in the Pipeline server.
        process_fn: Callable ``(data, parameters) -> result`` that performs
            the actual computation. *data* is the task input (parsed JSON),
            *parameters* is an optional dict of worker parameters (may be
            ``None``). The return value is sent back as the task result.
        server_url: Pipeline server URL. Defaults to ``SERVER_URL`` env var
            or ``http://localhost:3000``.
        token: Authentication token. Defaults to ``TOKEN`` env var or the
            default dev token.
    """

    def __init__(self, worker_name, process_fn, *, server_url=None, token=None):
        self.worker_name = worker_name
        self.process_fn = process_fn
        self.server_url = server_url or os.environ.get(
            "SERVER_URL", "http://localhost:3000"
        )
        self.token = token or os.environ.get(
            "TOKEN", "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        )
        self._auth_headers = {"Authorization": f"Bearer {self.token}"}

        # Shutdown flag — set by SIGTERM/SIGINT handler
        self._shutdown = threading.Event()

        # Thread-safe statistics shared across instances
        self._stats_lock = threading.Lock()
        self._stats = {
            "completedTasks": 0,
            "failedTasks": 0,
            "averageTaskTimeMs": 0,
            "freeMemory": 0,
        }
        self._total_time_ms = 0

    # --- Public API ---

    def run(self):
        """Start the worker, spawning multiple instances if configured.

        Reads the ``INSTANCES`` environment variable (default ``1``) to
        determine how many listener threads to start.  Handles SIGTERM and
        SIGINT for fast Docker shutdown.
        """

        def _handle_signal(signum, _frame):
            name = signal.Signals(signum).name
            print(f"[{self.worker_name}] Received {name}, shutting down...")
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        instances = int(os.environ.get("INSTANCES", "1"))
        print(
            f"[{self.worker_name}] Starting {instances} instance(s), "
            f"server={self.server_url}"
        )
        if instances > 1:
            threads = []
            for _ in range(instances):
                thread = threading.Thread(target=self._listen, daemon=True)
                thread.start()
                threads.append(thread)
            # Wait for shutdown signal instead of joining threads forever
            self._shutdown.wait()
        else:
            self._listen()

        print(f"[{self.worker_name}] Stopped.")
        sys.exit(0)

    # --- Internal methods ---

    def _listen(self):
        """Connect to the SSE endpoint and process tasks in a loop."""
        runner_id = str(uuid.uuid4())

        while not self._shutdown.is_set():
            try:
                params = {**_get_system_info(runner_id), "token": self.token}
                url = f"{self.server_url}/v1/worker/listen/{self.worker_name}"
                print(f"[{self.worker_name}] Connecting to {url} ...")
                response = requests.get(
                    url, params=params, stream=True, timeout=None
                )
                response.raise_for_status()
                client = sseclient.SSEClient(response)

                print(
                    f"[{self.worker_name}] Connected (runner {runner_id[:8]})"
                )

                for event in client.events():
                    if self._shutdown.is_set():
                        break

                    if event.event == "task":
                        message = json.loads(event.data)
                        task_id = message["taskId"]
                        data = message["data"]
                        parameters = message.get("parameters")

                        print(
                            f"[{self.worker_name}] Received task {task_id}"
                        )

                        stop_heartbeat = self._start_heartbeat(
                            task_id, runner_id
                        )

                        # Capture stdout/stderr during task processing
                        log_buffer = io.StringIO()
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        sys.stdout = _TeeStream(old_stdout, log_buffer)
                        sys.stderr = _TeeStream(old_stderr, log_buffer)

                        start = time.time()
                        try:
                            result = self.process_fn(data, parameters)
                            duration_ms = (time.time() - start) * 1000
                            with self._stats_lock:
                                self._stats["completedTasks"] += 1
                                self._total_time_ms += duration_ms
                                self._stats["averageTaskTimeMs"] = int(
                                    self._total_time_ms
                                    / self._stats["completedTasks"]
                                )
                                current_stats = dict(self._stats)
                            logs = log_buffer.getvalue()[:MAX_LOG_SIZE] or None
                            self._post_result(
                                task_id, result, runner_id, current_stats,
                                logs
                            )
                            print(
                                f"[{self.worker_name}] Task {task_id} "
                                f"completed ({duration_ms:.0f}ms)"
                            )
                        except Exception as error:
                            with self._stats_lock:
                                self._stats["failedTasks"] += 1
                                current_stats = dict(self._stats)
                            logs = log_buffer.getvalue()[:MAX_LOG_SIZE] or None
                            self._post_error(
                                task_id, str(error), runner_id, current_stats,
                                logs
                            )
                            print(
                                f"[{self.worker_name}] Task {task_id} "
                                f"failed: {error}"
                            )
                        finally:
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                            stop_heartbeat.set()

                        # Reconnect after each task
                        break

            except Exception as error:
                if self._shutdown.is_set():
                    break
                print(
                    f"[{self.worker_name}] Disconnected: {error}, "
                    f"retrying in 1s..."
                )
                # Use event wait instead of sleep so shutdown is immediate
                self._shutdown.wait(1)

    def _send_heartbeat(self, task_id, runner_id):
        """Send a single heartbeat to the server.

        Args:
            task_id: Task ID to heartbeat for.
            runner_id: Runner UUID for this instance.
        """
        try:
            url = f"{self.server_url}/v1/worker/{self.worker_name}/heartbeat"
            requests.post(
                url,
                json={"taskId": task_id, "runnerId": runner_id},
                headers=self._auth_headers,
                timeout=10,
            )
        except Exception:
            print(
                f"[{self.worker_name}] heartbeat failed for task {task_id}"
            )

    def _start_heartbeat(self, task_id, runner_id):
        """Start a background thread that sends heartbeats.

        Args:
            task_id: Task ID to heartbeat for.
            runner_id: Runner UUID for this instance.

        Returns:
            A threading.Event that, when set, stops the heartbeat loop.
        """
        stop_event = threading.Event()

        def heartbeat_loop():
            while not stop_event.wait(HEARTBEAT_INTERVAL):
                self._send_heartbeat(task_id, runner_id)

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        return stop_event

    def _post_result(self, task_id, result, runner_id, runner_stats, logs=None):
        """Post a successful result back to the server with retries.

        Args:
            task_id: Task ID.
            result: Task result payload.
            runner_id: Runner UUID for this instance.
            runner_stats: Runner statistics dict.
            logs: Optional captured log output string.
        """
        url = f"{self.server_url}/v1/worker/{self.worker_name}/result"
        payload = {
            "taskId": task_id,
            "result": result,
            "runnerId": runner_id,
            "runnerStats": runner_stats,
        }
        if logs:
            payload["logs"] = logs
        self._post_with_retries(url, payload, "result", task_id)

    def _post_error(self, task_id, error_message, runner_id, runner_stats,
                    logs=None):
        """Post an error back to the server with retries.

        Args:
            task_id: Task ID.
            error_message: Error description.
            runner_id: Runner UUID for this instance.
            runner_stats: Runner statistics dict.
            logs: Optional captured log output string.
        """
        url = f"{self.server_url}/v1/worker/{self.worker_name}/result"
        payload = {
            "taskId": task_id,
            "error": error_message,
            "runnerId": runner_id,
            "runnerStats": runner_stats,
        }
        if logs:
            payload["logs"] = logs
        self._post_with_retries(url, payload, "error", task_id)

    def _post_with_retries(self, url, payload, label, task_id):
        """POST JSON to *url* with exponential-backoff retries.

        Args:
            url: Target URL.
            payload: JSON payload dict.
            label: Human-readable label for log messages ("result" or "error").
            task_id: Task ID for logging.
        """
        for attempt in range(MAX_RESULT_RETRIES):
            try:
                requests.post(
                    url, json=payload, headers=self._auth_headers, timeout=30
                )
                return
            except Exception as error:
                if attempt < MAX_RESULT_RETRIES - 1:
                    wait = 2**attempt
                    print(
                        f"[{self.worker_name}] Failed to post {label} for "
                        f"task {task_id} (attempt {attempt + 1}/"
                        f"{MAX_RESULT_RETRIES}): {error}, "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"[{self.worker_name}] Failed to post {label} for "
                        f"task {task_id} after {MAX_RESULT_RETRIES} "
                        f"attempts: {error}"
                    )
