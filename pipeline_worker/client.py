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
# SSE read timeout — if no data (including keepalive pings) arrives within
# this many seconds, the connection is considered dead and will be retried.
# Must be longer than the server's SSE keepalive interval.
SSE_READ_TIMEOUT = 60
# Reconnection backoff parameters (in seconds)
RECONNECT_BASE_DELAY = 1
RECONNECT_MAX_DELAY = 30


# Global log buffer — single-threaded, no need for thread-local storage.
_log_buffer = None


def log(text):
    """Write text to the task log buffer without printing to stdout.

    Use this for verbose output (e.g. subprocess stdout/stderr) that should
    be sent to the server but not clutter Docker logs.

    Args:
        text: Text to append to the log buffer.
    """
    global _log_buffer
    if _log_buffer is not None and text:
        _log_buffer.write(text)


class _TeeStream:
    """Write to the original stream and the log buffer.

    Installed on ``sys.stdout``/``sys.stderr`` so that print output from the
    worker's processing function is captured and sent to the server as logs.
    """

    def __init__(self, original):
        self.original = original

    def write(self, text):
        self.original.write(text)
        if _log_buffer is not None:
            _log_buffer.write(text)

    def flush(self):
        self.original.flush()


def _read_host_hostname():
    """Read the host machine's hostname from a mounted file.

    Returns:
        Hostname string, or empty string if the file is not available.
    """
    try:
        with open("/etc/host_hostname") as f:
            return f.read().strip()
    except OSError:
        return ""


def _read_cgroup_int(path):
    """Read an integer value from a cgroup file.

    Args:
        path: Absolute path to the cgroup file.

    Returns:
        Integer value, or 0 if the file cannot be read or contains "max".
    """
    try:
        with open(path) as f:
            value = f.read().strip()
        if value == "max":
            return 0
        return int(value)
    except (OSError, ValueError):
        return 0


def _get_container_memory():
    """Read container memory limit and usage from cgroup v2 files.

    Returns:
        Tuple of (total_bytes, free_bytes). Returns (0, 0) when not
        running inside a container or cgroup files are unavailable.
    """
    total = _read_cgroup_int("/sys/fs/cgroup/memory.max")
    current = _read_cgroup_int("/sys/fs/cgroup/memory.current")
    free = max(total - current, 0) if total > 0 else 0
    return total, free


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
    total_memory, free_memory = _get_container_memory()
    return {
        "runnerId": runner_id,
        "hostname": os.environ.get("WORKER_HOSTNAME") or _read_host_hostname() or socket.gethostname(),
        "cpuCount": os.environ.get("CPUS", cpu_count),
        "totalMemory": str(total_memory),
        "freeMemory": str(free_memory),
    }


class WorkerClient:
    """Connects to the Pipeline server, receives tasks via SSE, and posts results.

    Each Docker container runs a single WorkerClient instance. Use Docker
    Compose ``deploy.replicas`` to run multiple instances in parallel.

    Args:
        worker_name: Name of the worker as registered in the Pipeline server.
        process_fn: Callable ``(data, parameters) -> result`` that performs
            the actual computation. *data* is the task input (parsed JSON),
            *parameters* is an optional dict of worker parameters (may be
            ``None``). The return value is sent back as the task result.
        server_url: Pipeline server URL. Defaults to ``SERVER_URL`` env var
            or ``http://localhost:5172``.
        token: Authentication token. Defaults to ``TOKEN`` env var or the
            default dev token.
    """

    def __init__(self, worker_name, process_fn, *, server_url=None, token=None):
        self.worker_name = worker_name
        self.process_fn = process_fn
        self.server_url = server_url or os.environ.get(
            "SERVER_URL", "http://localhost:5172"
        )
        self.token = token or os.environ.get(
            "TOKEN", "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        )
        self._auth_headers = {"Authorization": f"Bearer {self.token}"}

        # Shutdown flag — set by SIGTERM/SIGINT handler
        self._shutdown = threading.Event()

        # Statistics for this instance
        self._stats = {
            "completedTasks": 0,
            "failedTasks": 0,
            "averageTaskTimeMs": 0,
            "freeMemory": 0,
        }
        self._total_time_ms = 0

    # --- Public API ---

    def run(self):
        """Start the worker. Handles SIGTERM and SIGINT for fast Docker shutdown."""
        # Install TeeStream to capture print output during task processing.
        sys.stdout = _TeeStream(sys.stdout)
        sys.stderr = _TeeStream(sys.stderr)

        def _handle_signal(signum, _frame):
            name = signal.Signals(signum).name
            print(f"[{self.worker_name}] Received {name}, shutting down...")
            self._shutdown.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        cpus = os.environ.get("CPUS", "1")
        print(
            f"[{self.worker_name}] Starting, "
            f"{cpus} CPU(s), "
            f"server={self.server_url}"
        )
        self._listen()

        print(f"[{self.worker_name}] Stopped.")
        sys.exit(0)

    # --- Internal methods ---

    def _listen(self):
        """Connect to the SSE endpoint and process tasks in a loop."""
        global _log_buffer

        runner_id = str(uuid.uuid4())
        consecutive_failures = 0

        while not self._shutdown.is_set():
            try:
                params = {**_get_system_info(runner_id), "token": self.token}
                url = f"{self.server_url}/v1/worker/listen/{self.worker_name}"
                print(f"[{self.worker_name}] Connecting to {url} ...")
                response = requests.get(
                    url, params=params, stream=True,
                    timeout=(10, SSE_READ_TIMEOUT),
                )
                response.raise_for_status()
                client = sseclient.SSEClient(response)

                print(
                    f"[{self.worker_name}] Connected (runner {runner_id[:8]})"
                )
                # Reset backoff on successful connection
                consecutive_failures = 0

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

                        # Enable log capture
                        _log_buffer = io.StringIO()

                        start = time.time()
                        try:
                            result = self.process_fn(data, parameters)
                            duration_ms = (time.time() - start) * 1000
                            self._stats["completedTasks"] += 1
                            self._total_time_ms += duration_ms
                            self._stats["averageTaskTimeMs"] = int(
                                self._total_time_ms
                                / self._stats["completedTasks"]
                            )
                            _, free = _get_container_memory()
                            self._stats["freeMemory"] = free
                            logs = (
                                _log_buffer.getvalue()[:MAX_LOG_SIZE]
                                or None
                            )
                            self._post_result(
                                task_id, result, runner_id,
                                dict(self._stats), logs
                            )
                            print(
                                f"[{self.worker_name}] Task {task_id} "
                                f"completed ({duration_ms:.0f}ms)"
                            )
                        except Exception as error:
                            self._stats["failedTasks"] += 1
                            _, free = _get_container_memory()
                            self._stats["freeMemory"] = free
                            logs = (
                                _log_buffer.getvalue()[:MAX_LOG_SIZE]
                                or None
                            )
                            self._post_error(
                                task_id, str(error), runner_id,
                                dict(self._stats), logs
                            )
                            print(
                                f"[{self.worker_name}] Task {task_id} "
                                f"failed: {error}"
                            )
                        finally:
                            _log_buffer = None
                            stop_heartbeat.set()
                else:
                    # SSE stream ended without error (server closed
                    # connection) — reconnect immediately
                    if not self._shutdown.is_set():
                        print(
                            f"[{self.worker_name}] SSE stream ended, "
                            f"reconnecting..."
                        )
                    continue

            except Exception as error:
                if self._shutdown.is_set():
                    break
                consecutive_failures += 1
                delay = min(
                    RECONNECT_BASE_DELAY * 2 ** (consecutive_failures - 1),
                    RECONNECT_MAX_DELAY,
                )
                print(
                    f"[{self.worker_name}] Disconnected: {error}, "
                    f"retrying in {delay}s..."
                )
                # Use event wait instead of sleep so shutdown is immediate
                self._shutdown.wait(delay)

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
