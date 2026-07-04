#!/bin/bash
# Daily stock screener scan — triggered by launchd
# Runs the full screening pipeline + tracker update + digest + email
#
# Logs: logs/scan_YYYYMMDD.log
# Skips weekends and duplicate runs (safe to trigger on wake + schedule)

set -euo pipefail

PROJECT_DIR="/Users/andrew/stock_screener"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
TODAY=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/scan_${TODAY}.log"
LOCK_FILE="$LOG_DIR/.scan_${TODAY}.done"

cd "$PROJECT_DIR"

# Load SMTP credentials
set -a
source "$PROJECT_DIR/.env"
set +a

# Skip weekends (Sat=6, Sun=0)
DOW=$(date +%w)
if [ "$DOW" -eq 0 ] || [ "$DOW" -eq 6 ]; then
    echo "$(date): Weekend — skipping scan" >> "$LOG_FILE"
    exit 0
fi

# Skip if already ran today (RunAtLoad + StartCalendarInterval can double-fire)
if [ -f "$LOCK_FILE" ]; then
    echo "$(date): Already ran today — skipping" >> "$LOG_FILE"
    exit 0
fi

echo "========================================" >> "$LOG_FILE"
echo "Scan started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run the full scan via scheduler.py with email digest
caffeinate -i "$VENV_PYTHON" scheduler.py --capital 10000 --email andrewjulolr10@gmail.com >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "Scan finished: $(date) (exit code: $EXIT_CODE)" >> "$LOG_FILE"

# Mark today as done
touch "$LOCK_FILE"

# Prune logs and lock files older than 30 days
find "$LOG_DIR" -name "scan_*.log" -mtime +30 -delete 2>/dev/null || true
find "$LOG_DIR" -name ".scan_*.done" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
