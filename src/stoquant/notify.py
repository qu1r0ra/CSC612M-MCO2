"""Run a benchmark command with ntfy lifecycle notifications."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from stoquant import layout
from stoquant.monitoring import (
    ConfigurationError,
    MonitoringConfig,
    parse_monitoring_config,
    with_heartbeat,
)
from stoquant.snapshot_store import SnapshotProgress, discover, new_since, progress


class NotificationError(RuntimeError):
    """An ntfy publish request was not accepted."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _configuration(root: Path) -> MonitoringConfig:
    return parse_monitoring_config(root, require_notifications=True)


def _publish(config: MonitoringConfig, title: str, message: str) -> None:
    if config.topic is None or config.token is None:
        raise ConfigurationError("Missing ntfy configuration: NTFY_TOPIC, NTFY_TOKEN.")
    topic = quote(config.topic, safe="")
    request = Request(
        f"{config.server}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "text/plain; charset=utf-8",
            "Title": title,
        },
        method="POST",
    )
    try:
        with build_opener(NoRedirects()).open(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise NotificationError(f"ntfy returned HTTP {response.status}.")
    except HTTPError as exc:
        raise NotificationError(f"ntfy returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise NotificationError("ntfy could not be reached.") from None


def _notify_best_effort(config: MonitoringConfig, title: str, message: str) -> None:
    try:
        _publish(config, title, message)
    except NotificationError as exc:
        print(f"Warning: ntfy notification failed: {exc}", file=sys.stderr, flush=True)


def _family(state: SnapshotProgress | None) -> str:
    return state.input_family if state is not None and state.input_family else "benchmark"


def _heartbeat_message(elapsed_seconds: float, state: SnapshotProgress | None) -> str:
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts = [
        "event: heartbeat",
        f"family: {_family(state)}",
        f"elapsed: {hours:02}:{minutes:02}:{seconds:02}",
    ]
    if state is not None and state.snapshot:
        parts.append(f"snapshot: {state.snapshot}")
    if state is not None:
        case_progress = f"cases completed: {state.completed_cases}"
        if state.total_cases is not None:
            case_progress += f"/{state.total_cases}"
        parts.append(case_progress)
    else:
        parts.append("cases completed: unavailable")
    return "\n".join(parts)


def _terminal_message(
    exit_code: int, elapsed_seconds: float, state: SnapshotProgress | None
) -> str:
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    outcome = "success" if exit_code == 0 else "failure"
    parts = [
        f"event: {outcome}",
        f"family: {_family(state)}",
        f"duration: {hours:02}:{minutes:02}:{seconds:02}",
        f"exit code: {exit_code}",
    ]
    if state is not None and state.snapshot:
        parts.append(f"snapshot: {state.snapshot}")
    if state is not None:
        case_summary = f"cases: {state.completed_cases}"
        if state.total_cases is not None:
            case_summary += f"/{state.total_cases}"
        parts.append(case_summary)
    return "\n".join(parts)


def _run_command(command: list[str], root: Path, config: MonitoringConfig) -> int:
    heartbeat_seconds = config.heartbeat_seconds
    previous_snapshots = discover(root)
    started_at = datetime.now(UTC)

    try:
        _publish(
            config,
            "stoquant benchmark started",
            "\n".join(
                (
                    "event: start",
                    "family: benchmark",
                    f"started: {started_at.isoformat(timespec='seconds')}",
                    f"heartbeat interval: {heartbeat_seconds} seconds",
                )
            ),
        )
    except NotificationError as exc:
        print(f"ntfy start notification failed: {exc}", file=sys.stderr, flush=True)
        return 2

    child_env = os.environ.copy()
    for key in ("NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"):
        child_env.pop(key, None)
    child_env["STOQUANT_MONITORING_PROVIDER"] = "ntfy"
    child_env["STOQUANT_MONITORING_HEARTBEAT_SECONDS"] = str(heartbeat_seconds)

    exit_code = 127
    child: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    try:
        if command[0] == "python":
            command = [sys.executable, *command[1:]]
        child = subprocess.Popen(command, cwd=root, env=child_env)
        next_heartbeat = time.monotonic() + heartbeat_seconds
        while True:
            remaining = max(0, next_heartbeat - time.monotonic())
            try:
                exit_code = child.wait(timeout=remaining)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                selected = new_since(root, previous_snapshots)
                state = progress(selected, root)
                _notify_best_effort(
                    config,
                    f"stoquant {_family(state)} heartbeat",
                    _heartbeat_message(elapsed, state),
                )
                next_heartbeat = time.monotonic() + heartbeat_seconds
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        exit_code = child.returncode if child is not None and child.returncode else 130
    except OSError as exc:
        print(f"Could not start the benchmark command: {exc}", file=sys.stderr, flush=True)

    elapsed = time.monotonic() - started
    snapshot = new_since(root, previous_snapshots)
    state = progress(snapshot, root)
    outcome = "succeeded" if exit_code == 0 else "failed"
    _notify_best_effort(
        config,
        f"stoquant {_family(state)} {outcome}",
        _terminal_message(exit_code, elapsed, state),
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a command with ntfy lifecycle notifications")
    parser.add_argument(
        "--check",
        action="store_true",
        help="send a preflight notification without starting a command",
    )
    parser.add_argument("--heartbeat-seconds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=layout.ROOT, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if args.check and command:
        parser.error("--check cannot be combined with a command")
    if not command and not args.check:
        parser.error("provide a command after --")

    try:
        config = _configuration(args.root)
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.heartbeat_seconds is not None:
        if args.heartbeat_seconds < 1:
            parser.error("--heartbeat-seconds must be a positive integer")
        config = with_heartbeat(config, args.heartbeat_seconds)

    try:
        _publish(
            config,
            "stoquant ntfy preflight",
            "stoquant ntfy preflight succeeded.",
        )
    except NotificationError as exc:
        print(f"ntfy preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.check:
        print("ntfy preflight succeeded.")
        return 0
    return _run_command(command, args.root, config)


if __name__ == "__main__":
    raise SystemExit(main())
