#!/bin/bash
# Daily stock screener scan — triggered by launchd
# Runs the full screening pipeline + tracker update + digest generation
#
# Logs go to: logs/scan_YYYYMMDD.log
# launchd stderr/stdout also captured separately in logs/

set -euo pipefail

PROJECT_DIR="/Users/andrew/Desktop/stock_screener"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
TODAY=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/scan_${TODAY}.log"

cd "$PROJECT_DIR"

echo "========================================" >> "$LOG_FILE"
echo "Scan started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Skip weekends (Sat=6, Sun=0)
DOW=$(date +%w)
if [ "$DOW" -eq 0 ] || [ "$DOW" -eq 6 ]; then
    echo "Weekend — skipping scan" >> "$LOG_FILE"
    exit 0
fi

# Run the full scan via scheduler.py (scan + digest + tracker update)
"$VENV_PYTHON" scheduler.py --capital 10000 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "Scan finished: $(date) (exit code: $EXIT_CODE)" >> "$LOG_FILE"

# Prune logs older than 30 days
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
