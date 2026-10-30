#!/usr/bin/env bash
# Install (or remove with --remove) the Colab scale study watchdog as a user timer.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITS="$HOME/.config/systemd/user"
if [ "${1:-}" = "--remove" ]; then
  systemctl --user disable --now oe-colab-watchdog.timer || true
  rm -f "$UNITS/oe-colab-watchdog.service" "$UNITS/oe-colab-watchdog.timer"
  systemctl --user daemon-reload
  echo "watchdog removed"; exit 0
fi
mkdir -p "$UNITS"
cp "$DIR/watchdog.service" "$UNITS/oe-colab-watchdog.service"
cp "$DIR/watchdog.timer" "$UNITS/oe-colab-watchdog.timer"
systemctl --user daemon-reload
systemctl --user enable --now oe-colab-watchdog.timer
systemctl --user list-timers oe-colab-watchdog.timer --no-pager
