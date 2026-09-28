# Run notifications

`just bench-matrix` sends ntfy notifications for preflight, start, hourly progress, and the terminal result. The wrapper checks configuration and delivery before it starts the benchmark driver. A missing setting or failed preflight or start notification prevents the driver from starting. After launch, delivery failures are warnings; the benchmark command's exit code remains authoritative.

## One-time setup

1. Install the [ntfy phone app](https://docs.ntfy.sh/subscribe/phone/) or open the [ntfy web app](https://ntfy.sh/app). Continue when either interface is ready to subscribe.
2. Choose a fresh benchmark topic such as `stoquant-bunyi-<random-suffix>`. Use a long, unpredictable suffix. Topic names are public: anyone who knows the name can subscribe to or publish on that topic. Continue when you have the exact name to enter in `.env`.
3. Subscribe to the topic in the phone app or web app. ntfy creates topics on first subscription or publish. Continue when the topic appears in your subscriptions.
4. Sign in to [ntfy.sh](https://ntfy.sh/) and create a dedicated benchmark access token in the web app's Account section, following [ntfy's access-token documentation](https://docs.ntfy.sh/publish/#access-tokens). Continue when the token value is ready for step 5; revoke it from Account when it is no longer needed.
5. From the repository root, copy `.env.example` to `.env` and replace the topic and token placeholders. `.env` is ignored by Git. This step is complete when both values are set locally; keep them out of source files, benchmark outputs, screenshots, and issue reports.

```powershell
Copy-Item .env.example .env
```

The wrapper sends heartbeats once per hour by default. Set `NTFY_HEARTBEAT_SECONDS` in `.env` to change the interval.

## Verify delivery

Run the notifier preflight before a benchmark. It sends one test notification and exits without starting a child process or building CUDA code:

```powershell
uv run python -m stoquant.notify --check
```

Setup is complete when the command exits successfully and the test notification arrives on the subscribed device. A missing configuration or failed send returns a nonzero exit code.

## Alerts and recorded metadata

Each `just bench-matrix` invocation sends:

- A preflight notification before the benchmark driver is started.
- A start notification with the input family, UTC start time, and heartbeat interval.
- A heartbeat at the configured interval with the input family, elapsed time, relative snapshot path when known, and completed case count when available.
- A terminal success or failure notification with the input family, duration, exit code, relative snapshot path when known, and completed/total case count when available.

Alerts do not include benchmark log output, absolute paths, the ntfy topic, or credentials. The benchmark manifest records only the monitoring provider (`ntfy`) and heartbeat interval; it contains no topic or token. The wrapper removes ntfy settings from the benchmark process environment.
