#!/usr/bin/env bash
# Starts a Julia experiment in the background, set up by its "#launch" lines:  scripts/launch.sh [-n (dry run)] <script.jl> [script arguments]
set -euo pipefail

dry=0
if [[ ${1:-} == -n ]]; then dry=1; shift; fi
[[ -f ${1:-} ]] || { echo "Usage: scripts/launch.sh [-n] <script.jl> [script arguments]   (LIMIT=<duration> overrides the time limit)"; exit 1; }
root=$(cd "$(dirname "$0")/.." && pwd)
script=$(realpath --relative-to="$root" "$1"); shift
cd "$root"
name=$(basename "$script" .jl)
output="${EXO_OCEAN_OUTPUT:-output}"

fail() { echo "$1"; (( dry )) || exit 1; }
exists() { compgen -G "$output/$1" > /dev/null; }

limit="" gpu="" conflicts=$name size="" emulator="" niceness="" stamp=0 after="" threads=4
chain_labels=() chain_args=()
while read -r key value; do
    case $key in
        limit) limit=$value ;;                      # time limit for timeout, e.g. 12h
        gpu) gpu=$value ;;                          # GPU name that must be listed by nvidia-smi -L
        conflicts) conflicts+="|$value" ;;          # regex of other experiments that must not be running
        size) size=$value ;;                        # expected output size, shown beside the free space
        emulator) emulator=$value ;;                # emulator Python is "required" or "optional"
        nice) niceness=$value ;;                    # nice level for the run
        threads) threads=$value ;;                  # Julia threads (default 4); CPU runs want most of the cores
        timestamp) stamp=1 ;;                       # prefix each log line with the Unix time
        after) after=$value ;;                      # shell command run if the experiment succeeds, with its log in $LOG
        chain) chain_labels+=("${value%% *}"); chain_args+=("${value#* }") ;;  # <label> [args]: one process per chain, log <name>_<label>.log
        need) exists "$value" || fail "Nothing at $output/$value" ;;           # glob under the output folder that must exist
        warn) exists "${value%% *}" || echo "Warning: nothing at $output/${value%% *}; ${value#* }" ;;  # <glob> <consequence if missing>
        *) echo "Unknown directive in $script: #launch $key $value"; exit 1 ;;
    esac
done < <(sed -n 's/^#launch //p' "$script")

if [[ -n $gpu ]]; then
    grep -q "$gpu" <<< "$(nvidia-smi -L 2> /dev/null || true)" || fail "No $gpu on this machine"
else
    nvidia-smi > /dev/null 2>&1 || fail "nvidia-smi can't reach the GPU driver"
fi

if running=$(pgrep -af "julia.*($conflicts)\.jl"); then fail "Already running:"$'\n'"$running"; fi

if [[ -n $emulator ]]; then
    python=${EXO_EMULATOR_PYTHON:-$HOME/ocean-venv/bin/python}
    [[ -x $python || -n ${EXO_EMULATOR_PYTHON:-} ]] || python=/data/pt426/ocean-venv/bin/python
    if [[ -x $python ]]; then export EXO_EMULATOR_PYTHON=$python
    elif [[ $emulator == required ]]; then fail "No Python at $python (set EXO_EMULATOR_PYTHON)"
    else echo "Warning: no emulator Python at $python (set EXO_EMULATOR_PYTHON); wind tests fail unless their wind file is cached"
    fi
fi

echo "Free disk space for the output${size:+ (these runs can reach $size)}:"
df -h "$output/" 2> /dev/null | tail -1 || true

limit=${LIMIT:-$limit}
cmd=(julia --project=. -t "$threads" "$script")
if [[ -n $limit ]]; then cmd=(timeout "$limit" "${cmd[@]}"); fi
if [[ -n $niceness ]]; then cmd=(nice -n "$niceness" "${cmd[@]}"); fi

start() {  # log file, then the arguments for the experiment
    local log=$1; shift
    if (( dry )); then
        echo "Would run: ${cmd[*]} $* >> $log"
        if [[ -n $after ]]; then echo "Then: $after"; fi
        return 0
    fi
    if (( stamp )) || [[ -n $after ]]; then
        (
            trap '' HUP
            if (( stamp )); then
                "${cmd[@]}" "$@" 2>&1 | gawk '{ print systime(), $0; fflush() }' >> "$log"
            else
                "${cmd[@]}" "$@" >> "$log" 2>&1
            fi
            if [[ -n $after ]]; then LOG=$log bash -c "$after"; fi
        ) < /dev/null > /dev/null 2>&1 &
    else
        nohup "${cmd[@]}" "$@" >> "$log" 2>&1 < /dev/null &
    fi
    echo "Started $name $* (PID $!), log $log"
}

(( dry )) || mkdir -p logs
if (( ${#chain_labels[@]} )); then
    for i in "${!chain_labels[@]}"; do
        read -ra args <<< "${chain_args[i]}"
        start "logs/${name}_${chain_labels[i]}.log" "${args[@]}" "$@"
    done
else
    start "logs/$name.log" "$@"
fi
