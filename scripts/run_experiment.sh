#!/usr/bin/env bash
# =====================================================================
# run_experiment.sh
# ---------------------------------------------------------------------
# Runs the GA repeatedly with a FIXED number of fitness workers each time
# (autoscaler OFF) so you get clean data points for your hypothesis:
#   "does adding workers improve convergence speed, and where does the
#    overhead stop being worth it?"
#
# For each worker count it:
#   1. starts the pool with exactly N workers
#   2. runs the GA to completion
#   3. copies result.json -> results/result_workers_N.json
#
# Usage:   ./scripts/run_experiment.sh "1 2 3 4 6"
# =====================================================================
set -euo pipefail

WORKER_COUNTS="${1:-1 2 3 4 6}"
cd "$(dirname "$0")/.."

echo "=== Building images ==="
docker compose build fitness-pool ga-service

for N in $WORKER_COUNTS; do
  echo ""
  echo "=========================================="
  echo "  RUN: $N fitness worker(s)"
  echo "=========================================="

  # Clean slate
  docker compose down --remove-orphans >/dev/null 2>&1 || true
  rm -f results/status.json results/result.json

  # Start the pool scaled to exactly N workers (scaler NOT started here,
  # so the worker count stays fixed for a clean measurement).
  docker compose up -d --scale fitness-pool="$N" fitness-pool

  # Wait for health
  echo "  waiting for pool to be healthy..."
  sleep 8

  # Run the GA to completion (foreground)
  docker compose run --rm ga-service

  # Save the labelled result
  cp results/result.json "results/result_workers_${N}.json"
  echo "  saved -> results/result_workers_${N}.json"

  docker compose down --remove-orphans >/dev/null 2>&1 || true
done

echo ""
echo "=== All runs complete. Analysing... ==="
python3 scripts/analyze_results.py
