#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if pgrep -f "(uv run|python[0-9]*).*dictate\.py" > /dev/null; then
    echo "WhisperDictate läuft bereits (PIDs: $(pgrep -f '(uv run|python[0-9]*).*dictate\.py' | tr '\n' ' '))."
    exit 0
fi

nohup uv run dictate.py > /tmp/whisper-dictate.log 2>&1 &
disown
echo "WhisperDictate gestartet (PID $!). Log: /tmp/whisper-dictate.log"
