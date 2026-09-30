"""Parse the notifier settings once for notifications and run provenance."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

DEFAULT_SERVER = "https://ntfy.sh"
DEFAULT_HEARTBEAT_SECONDS = 3600
NTFY_KEYS = frozenset({"NTFY_SERVER", "NTFY_TOPIC", "NTFY_TOKEN", "NTFY_HEARTBEAT_SECONDS"})


class ConfigurationError(ValueError):
    """The local monitoring configuration is missing or invalid."""


class MonitoringMetadata(TypedDict):
    provider: str
    heartbeat_interval_seconds: int


@dataclass(frozen=True)
class MonitoringConfig:
    server: str
    topic: str | None
    token: str | None = field(repr=False)
    heartbeat_seconds: int
    provider: str | None

    def manifest_metadata(self) -> MonitoringMetadata | None:
        if self.provider is None:
            return None
        return {
            "provider": self.provider,
            "heartbeat_interval_seconds": self.heartbeat_seconds,
        }


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
        if key not in NTFY_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def parse_monitoring_config(
    root: Path,
    environ: Mapping[str, str] | None = None,
    *,
    require_notifications: bool = False,
) -> MonitoringConfig:
    file_values = _read_env_file(root / ".env")
    process_values = os.environ if environ is None else environ
    values = {
        **file_values,
        **{key: value for key, value in process_values.items() if key in NTFY_KEYS},
    }

    topic = values.get("NTFY_TOPIC", "").strip() or None
    token = values.get("NTFY_TOKEN", "").strip() or None
    missing = [key for key, value in (("NTFY_TOPIC", topic), ("NTFY_TOKEN", token)) if not value]
    if require_notifications and missing:
        raise ConfigurationError(
            "Missing ntfy configuration: " + ", ".join(missing) + ". Set them in a local .env file."
        )
    if not require_notifications and (topic is None) != (token is None):
        # A partially configured notifier is an error even when the caller only
        # needs to record metadata; both credentials are one setting.
        raise ConfigurationError("NTFY_TOPIC and NTFY_TOKEN must be configured together.")

    server = str(values.get("NTFY_SERVER", DEFAULT_SERVER)).rstrip("/")
    try:
        parsed = urlsplit(server)
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError(
            "NTFY_SERVER must use https (http is allowed only for localhost)."
        ) from exc
    if (
        not parsed.hostname
        or any(character.isspace() for character in parsed.netloc)
        or (port is not None and not 1 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or (
            parsed.scheme != "https"
            and not (
                parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
    ):
        raise ConfigurationError("NTFY_SERVER must use https (http is allowed only for localhost).")

    provider = process_values.get("STOQUANT_MONITORING_PROVIDER")
    provider = provider.strip() if provider is not None else None
    if provider == "":
        raise ConfigurationError("STOQUANT_MONITORING_PROVIDER must be 'ntfy'.")
    if provider is None and topic is not None and token is not None:
        provider = "ntfy"
    if provider not in (None, "ntfy"):
        raise ConfigurationError("STOQUANT_MONITORING_PROVIDER must be 'ntfy'.")

    heartbeat_value = process_values.get("STOQUANT_MONITORING_HEARTBEAT_SECONDS")
    if heartbeat_value is None:
        heartbeat_value = values.get("NTFY_HEARTBEAT_SECONDS", str(DEFAULT_HEARTBEAT_SECONDS))
    try:
        heartbeat_seconds = int(heartbeat_value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("NTFY_HEARTBEAT_SECONDS must be a positive integer.") from exc
    if heartbeat_seconds < 1:
        raise ConfigurationError("NTFY_HEARTBEAT_SECONDS must be a positive integer.")

    return MonitoringConfig(server, topic, token, heartbeat_seconds, provider)


def with_heartbeat(config: MonitoringConfig, heartbeat_seconds: int) -> MonitoringConfig:
    if heartbeat_seconds < 1:
        raise ConfigurationError("NTFY_HEARTBEAT_SECONDS must be a positive integer.")
    return replace(config, heartbeat_seconds=heartbeat_seconds)
