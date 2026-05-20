#!/bin/bash
# Monitor Kaggle kernel and download outputs when complete

export KAGGLE_USERNAME=achillescapital
export KAGGLE_KEY=6930acf6551ebd0053a9188088e4dad3
KERNEL="achillescapital/notebookf4e2a1d5b2"
OUTDIR="data/beta_testing/processed/models"
LOG="scripts/kaggle_monitor.log"

mkdir -p "$OUTDIR"

echo "$(date): Starting monitor for $KERNEL" >> "$LOG"

while true; do
    STATUS=$(venv/Scripts/kaggle.exe kernels status "$KERNEL" 2>&1)
    echo "$(date): $STATUS" >> "$LOG"
    
    if echo "$STATUS" | grep -q "COMPLETE"; then
        echo "$(date): Kernel complete! Downloading outputs..." >> "$LOG"
        venv/Scripts/kaggle.exe kernels output "$KERNEL" -p "$OUTDIR" --quiet 2>&1 >> "$LOG"
        echo "$(date): Download finished. Files in $OUTDIR:" >> "$LOG"
        ls -la "$OUTDIR" >> "$LOG"
        echo "$(date): MONITOR DONE" >> "$LOG"
        exit 0
    fi
    
    if echo "$STATUS" | grep -q "ERROR\|FAILED\|CANCELLED"; then
        echo "$(date): Kernel failed!" >> "$LOG"
        exit 1
    fi
    
    sleep 120
done
