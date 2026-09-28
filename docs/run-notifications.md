# Run notifications

`just bench-matrix` uses ntfy to report whether a benchmark command started, periodic progress, and its terminal result. The notification wrapper performs a configuration and delivery check before starting the benchmark driver. Missing settings or a failed preflight or start notification stops the driver. After launch, notification outages are warnings and do not replace the benchmark command's exit code.

## One-time setup

1. Install the [ntfy phone app](https://docs.ntfy.sh/subscribe/phone/) or use the [ntfy web app](https://ntfy.sh/app).
2. Choose a fresh MCO2 topic name in the form `<project>-<family-name>-<random-suffix>`. Use a long, unpredictable suffix and keep the complete topic private: ntfy topic names are public and anyone who knows one can subscribe to it.
3. Subscribe to that topic in the phone app or web app. Topics do not need to be created separately.
4. Sign in to [ntfy.sh](https://ntfy.sh/) and create a dedicated access token in the web app's Account section, as described in [ntfy's publishing documentation](https://docs.ntfy.sh/publish/#access-tokens). ntfy.sh access tokens currently grant account-level access, so store this token only in the local configuration and revoke it from the account page if it is no longer needed.
5. From the repository root, copy `.env.example` to `.env` and replace the topic and token placeholders. `.env` is ignored by Git. Keep the real topic and token out of source files, benchmark outputs, screenshots, and issue reports.

```powershell
Copy-Item .env.example .env
```

The default server is `https://ntfy.sh` and the default heartbeat interval is one hour. Set `NTFY_HEARTBEAT_SECONDS` in `.env` to change the interval.

## Verify delivery

Run the notifier preflight before a benchmark. It sends one test notification and exits without starting a child process or building CUDA code:

```powershell
uv run python mco2_run_notify.py --check
```

Confirm that the notification arrives on the subscribed device. A missing configuration or failed send returns a nonzero exit code.

## Alerts and recorded metadata

Each `just bench-matrix` invocation sends:

- A preflight notification before the benchmark driver is started.
- A start notification with the input family, UTC start time, and heartbeat interval.
- A heartbeat at the configured interval with the input family, elapsed time, relative snapshot path when known, and completed case count when available.
- A terminal success or failure notification with the input family, duration, exit code, relative snapshot path when known, and completed/total case count when available.

Alerts do not include benchmark log output, absolute paths, the ntfy topic, or credentials. The benchmark manifest records only the monitoring provider (`ntfy`) and heartbeat interval; it contains no topic or token. The wrapper removes ntfy settings from the benchmark process environment.
