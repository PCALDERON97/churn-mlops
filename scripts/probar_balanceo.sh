#!/usr/bin/env bash
# Igual que probar_balanceo.ps1, para Linux/macOS o Git Bash.
N=${1:-20}; URL=${2:-http://localhost:30080/info}
for i in $(seq 1 "$N"); do curl -s "$URL" | python -c "import sys,json;print(json.load(sys.stdin)['pod'])"; done | sort | uniq -c
