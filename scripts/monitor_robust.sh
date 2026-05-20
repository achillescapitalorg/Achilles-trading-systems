#!/bin/bash
export KAGGLE_USERNAME=achillescapital
export KAGGLE_KEY=6930acf6551ebd0053a9188088e4dad3
KERNEL="achillescapital/notebookf4e2a1d5b2"
OUTDIR="data/beta_testing/processed/models"
LOG="scripts/monitor_v6.log"
mkdir -p "$OUTDIR"
echo "$(date): ROBUST monitor started" >> "$LOG"
while true; do
  STATUS=$(venv/Scripts/kaggle.exe kernels status "$KERNEL" 2>&1)
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "$(date): API error (exit $EXIT_CODE), retrying in 60s..." >> "$LOG"
    sleep 60
    continue
  fi
  echo "$(date): $STATUS" >> "$LOG"
  if echo "$STATUS" | grep -q "COMPLETE"; then
    echo "$(date): COMPLETE - downloading" >> "$LOG"
    venv/Scripts/kaggle.exe kernels output "$KERNEL" -p "$OUTDIR" --quiet 2>&1 >> "$LOG"
    echo "$(date): DONE - files:" >> "$LOG"
    ls -lth "$OUTDIR" >> "$LOG"
    exit 0
  fi
  if echo "$STATUS" | grep -qi "error\|failed\|cancelled"; then
    echo "$(date): KERNEL FAILED" >> "$LOG"
    exit 1
  fi
  sleep 180
done
