#!/usr/bin/env bash
# Free smoke test: one short episode with the rule planners (no OPENROUTER_API_KEY, no network
# calls) to confirm that the environment, the CasADi/OSQP solver stack and the simulation code work
# end to end. Run ./setup_env.sh first to create the "multi_agent_mpc" conda environment.
set -euo pipefail

ENV_NAME="${ENV_NAME:-multi_agent_mpc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(mktemp -d)"

echo "Running smoke test: rule planner, 1 intruder, 1 seed, 100 steps (free, no API calls)..."
conda run -n "$ENV_NAME" --cwd "$SCRIPT_DIR" python3 -m scripts.run_red_count_grid \
    --planner rule \
    --red-counts 1 \
    --n-seeds 1 \
    --sim-steps 100 \
    --output-results-csv "$OUT_DIR/results.csv" \
    --output-reasoning-csv "$OUT_DIR/reasoning.csv"

echo
echo "Smoke test passed. Results written to $OUT_DIR"
