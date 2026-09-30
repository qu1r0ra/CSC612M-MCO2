import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast


def runner(root: Path) -> list[str]:
    return [sys.executable, "-m", "stoquant.notify", "--root", str(root)]


class CaptureServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, CaptureHandler)
        self.messages = []
        self.fail_requests = set()
        self.redirect_requests = set()
        self.delay_requests = set()
        self.messages_lock = threading.Lock()


class CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = cast(CaptureServer, self.server)
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size).decode("utf-8")
        with server.messages_lock:
            request_number = len(server.messages) + 1
            server.messages.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "title": self.headers.get("Title"),
                    "body": body,
                }
            )
        if request_number in server.delay_requests:
            time.sleep(1.3)
        status = (
            302
            if request_number in server.redirect_requests
            else 503
            if request_number in server.fail_requests
            else 200
        )
        self.send_response(status)
        if status == 302:
            self.send_header("Location", "https://example.invalid/redirected")
        self.end_headers()

    def log_message(self, format, *args):
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
        [*runner(tmp_path), "--", *child],
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
            [*runner(tmp_path), "--", *child],
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


def test_non_local_http_server_is_refused_before_launch(tmp_path):
    result = subprocess.run(
        [*runner(tmp_path), "--", sys.executable, "-c", "pass"],
        cwd=tmp_path,
        env=ntfy_environment("http://example.invalid"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "https" in result.stderr


def test_redirect_is_refused_without_forwarding_authorization(tmp_path):
    with ntfy_stub() as (server, server_url):
        server.redirect_requests.add(1)
        result = subprocess.run(
            [*runner(tmp_path), "--check"],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 2
    assert len(server.messages) == 1
    assert server.messages[0]["authorization"] == "Bearer test-only-token"
    assert "302" in result.stderr


def test_python_child_uses_the_current_interpreter(tmp_path):
    child = ["python", "-c", "import sys; print(sys.executable)"]
    with ntfy_stub() as (_, server_url):
        result = subprocess.run(
            [*runner(tmp_path), "--", *child],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0
    assert str(Path(sys.executable).resolve()).lower() in result.stdout.lower()


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
            [*runner(tmp_path), "--", *child],
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
            [*runner(tmp_path), "--check"],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0
    assert len(server.messages) == 1
    assert server.messages[0]["title"] == "stoquant ntfy preflight"
    assert server.messages[0]["body"] == "stoquant ntfy preflight succeeded."


def test_elapsed_duration_excludes_slow_start_notification(tmp_path):
    child = [sys.executable, "-c", "import time; time.sleep(0.1)"]
    with ntfy_stub() as (server, server_url):
        server.delay_requests.add(2)
        result = subprocess.run(
            [*runner(tmp_path), "--", *child],
            cwd=tmp_path,
            env=ntfy_environment(server_url),
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0
    assert "duration: 00:00:00" in server.messages[-1]["body"]


def test_successful_run_sends_progress_and_terminal_summary(tmp_path):
    child_code = """
from pathlib import Path
import json
import os
import sys
import time

snapshot = Path('results/pilots/fake-run')
snapshot.mkdir(parents=True)
(snapshot / 'run-plan.json').write_text(json.dumps({'total_cases': 2, 'input_family': 'sparse'}))
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
            [*runner(tmp_path), "--", *child],
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
        "stoquant ntfy preflight",
        "stoquant benchmark started",
        "stoquant sparse heartbeat",
        "stoquant sparse succeeded",
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
            [*runner(tmp_path), "--", *child],
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
    assert server.messages[0]["title"] == "stoquant ntfy preflight"
    assert server.messages[1]["title"] == "stoquant benchmark started"
    assert server.messages[2]["title"] == "stoquant benchmark heartbeat"
    assert server.messages[3]["title"] == "stoquant benchmark failed"
    assert "event: failure" in server.messages[3]["body"]
    assert "exit code: 7" in server.messages[3]["body"]
