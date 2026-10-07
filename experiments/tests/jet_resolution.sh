#!/usr/bin/env bash
# Run the jet resolution/scheme test matrix (see experiments/tests/jet_resolution.jl). Run inside tmux after `source env.sh`.
#   experiments/tests/jet_resolution.sh [--benchmark | --smoke] [--years Y] [--budget-hours H] [--parallel N] [--cpu]
#   --benchmark  every run for 60 model days, one at a time, then the projected cost of the full matrix
#   --smoke      every scheme at 2° and 1° for 10 model days (quick local check)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MODE=full; YEARS=5; BUDGET_H=5; PARALLEL=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark) MODE=benchmark; shift ;;
        --smoke) MODE=smoke; shift ;;
        --years) YEARS="$2"; shift 2 ;;
        --budget-hours) BUDGET_H="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --cpu) export JETRES_CPU=1; shift ;;
        -h|--help) sed -n 2,5p "$0"; exit 0 ;;
        *) echo "Unknown option $1" >&2; exit 1 ;;
    esac
done

OUT="${EXO_OCEAN_OUTPUT:-$REPO/output}/ocean"
mkdir -p "$OUT"
SUMMARY="$OUT/jetres_summary.txt"
SCHEMES=(bih10 bih100 weno9 weno5 weno9_tr)
T0=$(date +%s)

run_one() {  # scheme resolution years
    local log="$OUT/jetres_$1_$2deg_D200.log" start status
    start=$(date +%s)
    julia --project="$REPO" "$REPO/experiments/tests/jet_resolution.jl" "$1" "$2" "$3" 2>&1 |
        while IFS= read -r line; do printf '%s %s\n' "$(date +%s)" "$line"; done > "$log"
    status=${PIPESTATUS[0]}
    printf '%s  %-9s %5s°  %-9s exit %s  %4d min\n' "$(date '+%F %T')" "$1" "$2" "$MODE" "$status" $(( ($(date +%s) - start) / 60 )) | tee -a "$SUMMARY"
}

within_budget() { (( $(date +%s) - T0 < BUDGET_H * 3600 )); }

# GPU minutes per model year from a run log (timestamped progress lines, skipping the first 10 model days of spin-up)
cost_per_year() {
    awk '/Time: [0-9.]+ (seconds|minutes|hours|days|years),/ {
            for (i = 1; i <= NF; i++) if ($i == "Time:") { v = $(i+1); u = $(i+2) }
            sub(",", "", u)
            d = (u == "seconds") ? v/86400 : (u == "minutes") ? v/1440 : (u == "hours") ? v/24 : (u == "years") ? v*365 : v
            if (d >= 10 && t0 == "") { t0 = $1; d0 = d }
            if (t0 != "") { t1 = $1; d1 = d }
         }
         END { if (t0 != "" && d1 > d0) printf "%.2f", (t1 - t0) / 60 / ((d1 - d0) / 365); else printf "nan" }' "$1"
}

case "$MODE" in
    smoke)
        for res in 2.0 1.0; do for s in "${SCHEMES[@]}"; do run_one "$s" "$res" "$(awk 'BEGIN { print 10/365 }')"; done; done ;;

    benchmark)
        printf '\n%-9s %6s  %16s  %20s\n' scheme res "min/model yr" "est. ${YEARS}-yr run (h)"
        total=0
        for res in 1.0 0.5 0.25; do
            for s in "${SCHEMES[@]}"; do
                [[ "$res" == 0.25 && "$s" != weno9 && "$s" != bih10 ]] && continue
                run_one "$s" "$res" "$(awk 'BEGIN { print 60/365 }')" > /dev/null
                c=$(cost_per_year "$OUT/jetres_${s}_${res}deg_D200.log")
                h=$(awk -v c="$c" -v y="$YEARS" 'BEGIN { print (c == "nan") ? 0 : c * y / 60 }')
                total=$(awk -v a="$total" -v b="$h" 'BEGIN { print a + b }')
                printf '%-9s %5s°  %16s  %20.2f\n' "$s" "$res" "$c" "$h" | tee -a "$SUMMARY"
            done
        done
        printf 'Projected GPU time for the full matrix (run one at a time): %.1f h\n' "$total" | tee -a "$SUMMARY" ;;

    full)
        echo "$(date '+%F %T')  full matrix: ${YEARS} model yr per run, budget ${BUDGET_H} h, ${PARALLEL} small runs at a time" | tee -a "$SUMMARY"
        run_one weno9 0.25 "$YEARS" &                      # longest run first
        big=$!
        small=()
        for res in 1.0 0.5; do
            for s in "${SCHEMES[@]}"; do
                if ! within_budget; then echo "budget reached: skipping $s ${res}°" | tee -a "$SUMMARY"; continue; fi
                while (( $(jobs -rp | grep -vc "^$big$") >= PARALLEL )); do sleep 20; done
                run_one "$s" "$res" "$YEARS" &
                small+=($!)
            done
        done
        (( ${#small[@]} )) && wait "${small[@]}"
        if within_budget; then run_one bih10 0.25 "$YEARS"; else echo "budget reached: skipping bih10 0.25°" | tee -a "$SUMMARY"; fi
        wait "$big"
        echo "$(date '+%F %T')  done in $(( ($(date +%s) - T0) / 60 )) min. Analyse with: python analysis/jet_resolution.py" | tee -a "$SUMMARY" ;;
esac
