"""Run a benchmark command with ntfy lifecycle notifications."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from stoquant import layout

DEFAULT_SERVER = "https://ntfy.sh"
DEFAULT_HEARTBEAT_SECONDS = 3600


class ConfigurationError(ValueError):
    """The local ntfy configuration is missing or invalid."""


class NotificationError(RuntimeError):
    """An ntfy publish request was not accepted."""


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without exporting them to child processes."""
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigurationError("Could not read the local ntfy configuration file.") from exc

    values: dict[str, str] = {}
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in {"NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _configuration(root: Path, environ: Mapping[str, str] | None = None) -> dict[str, str | int]:
    file_values = _read_env_file(root / ".env")
    process_values = os.environ if environ is None else environ
    values = {
        **file_values,
        **{key: value for key, value in process_values.items() if key.startswith("NTFY_")},
    }

    missing = [key for key in ("NTFY_TOPIC", "NTFY_TOKEN") if not values.get(key, "").strip()]
    if missing:
        raise ConfigurationError(
            "Missing ntfy configuration: " + ", ".join(missing) + ". Set them in a local .env file."
        )

    try:
        heartbeat_seconds = int(values.get("NTFY_HEARTBEAT_SECONDS", DEFAULT_HEARTBEAT_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("NTFY_HEARTBEAT_SECONDS must be a positive integer.") from exc
    if heartbeat_seconds < 1:
        raise ConfigurationError("NTFY_HEARTBEAT_SECONDS must be a positive integer.")

    server = str(values.get("NTFY_SERVER", DEFAULT_SERVER)).rstrip("/")
    if not server.startswith(("https://", "http://")):
        raise ConfigurationError("NTFY_SERVER must be an http or https URL.")
    return {
        "server": server,
        "topic": values["NTFY_TOPIC"].strip(),
        "token": values["NTFY_TOKEN"].strip(),
        "heartbeat_seconds": heartbeat_seconds,
    }


def _publish(config: Mapping[str, str | int], title: str, message: str) -> None:
    topic = quote(str(config["topic"]), safe="")
    request = Request(
        f"{config['server']}/{topic}",
        data=message.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "text/plain; charset=utf-8",
            "Title": title,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise NotificationError(f"ntfy returned HTTP {response.status}.")
    except HTTPError as exc:
        raise NotificationError(f"ntfy returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise NotificationError("ntfy could not be reached.") from None


def _run_family(command: list[str]) -> str:
    for index, argument in enumerate(command):
        if argument == "--input-family" and index + 1 < len(command):
            return " ".join(command[index + 1].split())[:48] or "unknown"
        if argument.startswith("--input-family="):
            return " ".join(argument.partition("=")[2].split())[:48] or "unknown"
    return "dense"


def _snapshot_directories(root: Path) -> set[Path]:
    results = root / "results"
    if not results.is_dir():
        return set()
    snapshots: set[Path] = set()
    for path in results.iterdir():
        if not path.is_dir():
            continue
        if path.name == "pilots":
            snapshots.update(child for child in path.iterdir() if child.is_dir())
        else:
            snapshots.add(path)
    return snapshots


def _explicit_output_directory(root: Path, command: list[str]) -> Path | None:
    for index, argument in enumerate(command):
        value = None
        if argument == "--output-dir" and index + 1 < len(command):
            value = command[index + 1]
        elif argument.startswith("--output-dir="):
            value = argument.partition("=")[2]
        if value:
            output = Path(value)
            return output if output.is_absolute() else root / output
    return None


def _snapshot_for_run(
    root: Path,
    command: list[str],
    previous_snapshots: set[Path],
) -> Path | None:
    explicit = _explicit_output_directory(root, command)
    if explicit is not None and explicit.exists():
        return explicit
    candidates = _snapshot_directories(root) - previous_snapshots
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _snapshot_progress(
    snapshot: Path | None, root: Path
) -> tuple[str | None, int | None, int | None]:
    if snapshot is None or not snapshot.is_dir():
        return None, None, None
    try:
        relative = snapshot.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = None

    completed = len(list(snapshot.glob("case_*.json")))
    manifest_path = snapshot / "manifest.json"
    total = None
    if manifest_path.is_file():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cases = manifest.get("cases")
            if isinstance(cases, list):
                total = len(cases)
        except (OSError, ValueError):
            pass
    return relative, completed, total


def _notify_best_effort(config: Mapping[str, str | int], title: str, message: str) -> None:
    try:
        _publish(config, title, message)
    except NotificationError as exc:
        print(f"Warning: ntfy notification failed: {exc}", file=sys.stderr, flush=True)


def _heartbeat_message(
    family: str,
    elapsed_seconds: float,
    snapshot: Path | None,
    root: Path,
    command: list[str],
    previous_snapshots: set[Path],
) -> str:
    selected = snapshot or _snapshot_for_run(root, command, previous_snapshots)
    relative, completed, total = _snapshot_progress(selected, root)
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts = [
        "event: heartbeat",
        f"family: {family}",
        f"elapsed: {hours:02}:{minutes:02}:{seconds:02}",
    ]
    if relative:
        parts.append(f"snapshot: {relative}")
    if completed is not None:
        progress = f"cases completed: {completed}"
        if total is not None:
            progress += f"/{total}"
        parts.append(progress)
    else:
        parts.append("cases completed: unavailable")
    return "\n".join(parts)


def _terminal_message(
    family: str,
    exit_code: int,
    elapsed_seconds: float,
    snapshot: Path | None,
    root: Path,
) -> str:
    relative, completed, total = _snapshot_progress(snapshot, root)
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    outcome = "success" if exit_code == 0 else "failure"
    parts = [
        f"event: {outcome}",
        f"family: {family}",
        f"duration: {hours:02}:{minutes:02}:{seconds:02}",
        f"exit code: {exit_code}",
    ]
    if relative:
        parts.append(f"snapshot: {relative}")
    if completed is not None:
        case_summary = f"cases: {completed}"
        if total is not None:
            case_summary += f"/{total}"
        parts.append(case_summary)
    return "\n".join(parts)


def _run_command(command: list[str], root: Path, config: Mapping[str, str | int]) -> int:
    family = _run_family(command)
    heartbeat_seconds = int(config["heartbeat_seconds"])
    previous_snapshots = _snapshot_directories(root)
    started_at = datetime.now(UTC)

    try:
        _publish(
            config,
            f"stoquant {family} started",
            "\n".join(
                (
                    "event: start",
                    f"family: {family}",
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
        child = subprocess.Popen(command, cwd=root, env=child_env)
        next_heartbeat = time.monotonic() + heartbeat_seconds
        while True:
            remaining = max(0, next_heartbeat - time.monotonic())
            try:
                exit_code = child.wait(timeout=remaining)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                selected = _snapshot_for_run(root, command, previous_snapshots)
                _notify_best_effort(
                    config,
                    f"stoquant {family} heartbeat",
                    _heartbeat_message(
                        family,
                        elapsed,
                        selected,
                        root,
                        command,
                        previous_snapshots,
                    ),
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
    snapshot = _snapshot_for_run(root, command, previous_snapshots)
    outcome = "succeeded" if exit_code == 0 else "failed"
    _notify_best_effort(
        config,
        f"stoquant {family} {outcome}",
        _terminal_message(family, exit_code, elapsed, snapshot, root),
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
        config["heartbeat_seconds"] = args.heartbeat_seconds

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
