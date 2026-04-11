#!/usr/bin/env bash
cd "$(dirname "$(readlink -f "$0")")"

pids=$(pgrep -f "(uv run|python[0-9]*).*dictate\.py" || true)
if [ -z "$pids" ]; then
    echo "WhisperDictate läuft nicht."
    exit 0
fi

echo "Beende Prozesse: $(echo $pids | tr '\n' ' ')"
kill $pids 2>/dev/null || true
sleep 1

remaining=$(pgrep -f "(uv run|python[0-9]*).*dictate\.py" || true)
if [ -n "$remaining" ]; then
    echo "Erzwinge Beenden: $(echo $remaining | tr '\n' ' ')"
    kill -9 $remaining 2>/dev/null || true
    sleep 0.5
fi

if pgrep -f "(uv run|python[0-9]*).*dictate\.py" > /dev/null; then
    echo "Fehler: Prozesse konnten nicht beendet werden."
    exit 1
fi
echo "WhisperDictate beendet."
