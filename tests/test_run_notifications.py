import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "mco2_run_notify.py"


class CaptureServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, CaptureHandler)
        self.messages = []
        self.fail_requests = set()
        self.messages_lock = threading.Lock()


class CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size).decode("utf-8")
        with self.server.messages_lock:
            request_number = len(self.server.messages) + 1
            self.server.messages.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "title": self.headers.get("Title"),
                    "body": body,
                }
            )
        status = 503 if request_number in self.server.fail_requests else 200
        self.send_response(status)
        self.end_headers()

    def log_message(self, _format, *_args):
        pass


@contextmanager
def ntfy_stub():
    server = CaptureServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def ntfy_environment(server_url, heartbeat_seconds=None):
    env = os.environ.copy()
    for key in ("NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"):
        env.pop(key, None)
    env.update(
        NTFY_SERVER=server_url,
        NTFY_TOPIC="local-test-topic",
        NTFY_TOKEN="test-only-token",
    )
    if heartbeat_seconds is not None:
        env["NTFY_HEARTBEAT_SECONDS"] = str(heartbeat_seconds)
    return env


def test_missing_configuration_prevents_child_launch(tmp_path):
    marker = tmp_path / "child-started"
    child = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('child-started').write_text('started')",
    ]
    env = os.environ.copy()
    for key in ("NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"):
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, str(RUNNER), "--", *child],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert "NTFY_TOPIC" in result.stderr


def test_failed_preflight_prevents_child_launch(tmp_path):
    marker = tmp_path / "child-started"
    child = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('child-started').write_text('started')",
    ]
    with ntfy_stub() as (server, server_url):
        server.fail_requests.add(1)
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--", *child],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert not marker.exists()
    assert len(server.messages) == 1
    assert "preflight" in result.stderr.lower()
    assert "test-only-token" not in result.stderr


def test_failed_start_notification_prevents_child_launch(tmp_path):
    marker = tmp_path / "child-started"
    child = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('child-started').write_text('started')",
    ]
    with ntfy_stub() as (server, server_url):
        server.fail_requests.add(2)
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--", *child],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert not marker.exists()
    assert len(server.messages) == 2
    assert "start notification" in result.stderr.lower()
    assert "test-only-token" not in result.stderr


def test_check_option_sends_preflight_without_starting_a_command(tmp_path):
    with ntfy_stub() as (server, server_url):
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--check"],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0
    assert len(server.messages) == 1
    assert server.messages[0]["title"] == "MCO2 ntfy preflight"
    assert server.messages[0]["body"] == "MCO2 ntfy preflight succeeded."


def test_successful_run_sends_progress_and_terminal_summary(tmp_path):
    child_code = """
from pathlib import Path
import json
import os
import sys
import time

snapshot = Path('results/pilots/fake-run')
snapshot.mkdir(parents=True)
(snapshot / 'case_one.json').write_text('{}')
print('fake benchmark stdout', flush=True)
print('fake benchmark stderr', file=sys.stderr, flush=True)
time.sleep(1.2)
(snapshot / 'case_two.json').write_text('{}')
(snapshot / 'manifest.json').write_text(json.dumps({'cases': ['case_one', 'case_two']}))
Path('credential-forwarded').write_text(str('NTFY_TOKEN' in os.environ))
"""
    child = [sys.executable, "-c", child_code, "--input-family", "sparse"]
    with ntfy_stub() as (server, server_url):
        (tmp_path / ".env").write_text(
            "\n".join(
                (
                    f"NTFY_SERVER={server_url}",
                    "NTFY_TOPIC=local-test-topic",
                    "NTFY_TOKEN=test-only-token",
                    "NTFY_HEARTBEAT_SECONDS=1",
                )
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        for key in ("NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"):
            env.pop(key, None)
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--", *child],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0
    assert "fake benchmark stdout" in result.stdout
    assert "fake benchmark stderr" in result.stderr
    assert (tmp_path / "credential-forwarded").read_text() == "False"
    assert [message["title"] for message in server.messages] == [
        "MCO2 ntfy preflight",
        "MCO2 sparse started",
        "MCO2 sparse heartbeat",
        "MCO2 sparse succeeded",
    ]
    assert all(message["path"] == "/local-test-topic" for message in server.messages)
    assert all(message["authorization"] == "Bearer test-only-token" for message in server.messages)
    heartbeat = server.messages[2]["body"]
    assert "family: sparse" in heartbeat
    assert "results/pilots/fake-run" in heartbeat
    assert "cases completed: 1" in heartbeat
    assert "fake benchmark stdout" not in "\n".join(message["body"] for message in server.messages)
    final = server.messages[3]["body"]
    assert "exit code: 0" in final
    assert "cases: 2/2" in final


def test_delivery_failure_after_launch_preserves_benchmark_exit_code(tmp_path):
    child_code = """
import sys
import time

print('failed benchmark stdout', flush=True)
print('failed benchmark stderr', file=sys.stderr, flush=True)
time.sleep(1.2)
raise SystemExit(7)
"""
    child = [sys.executable, "-c", child_code]
    with ntfy_stub() as (server, server_url):
        server.fail_requests.update({3, 4})
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--", *child],
            cwd=tmp_path,
            env=ntfy_environment(server_url, heartbeat_seconds=1),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 7
    assert "failed benchmark stdout" in result.stdout
    assert "failed benchmark stderr" in result.stderr
    assert result.stderr.count("Warning: ntfy notification failed") == 2
    assert "test-only-token" not in result.stderr
    assert server.messages[0]["title"] == "MCO2 ntfy preflight"
    assert server.messages[1]["title"] == "MCO2 dense started"
    assert server.messages[2]["title"] == "MCO2 dense heartbeat"
    assert server.messages[3]["title"] == "MCO2 dense failed"
