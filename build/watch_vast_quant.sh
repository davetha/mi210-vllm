#!/usr/bin/env bash
# Watchdog for the rented quantisation job. Prints one status line per poll and
# ABORTS THE RENTAL if the box is burning money without making progress.
#
#   ./watch_vast_quant.sh <instance_id> <ssh_host> <ssh_port> [poll_seconds]
#
# The last run cost $13.55 and produced nothing, because two failures were
# invisible until the money was gone:
#   * hf_transfer was deprecated and silently ignored -> 6 MB/s for 20 minutes
#     (11h ETA on a 247 GB pull) while the log looked completely normal.
#   * the job died on an import error and the box sat idle, still billing.
# Both are cheap to detect and expensive to miss, so this checks for them every
# poll rather than trusting the log.
set -uo pipefail

ID=${1:?instance id}; HOST=${2:?ssh host}; PORT=${3:?ssh port}; POLL=${4:-60}
KEY_FILE=${KEY_FILE:-$HOME/.config/vastai/vast_api_key}
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ConnectTimeout=20 -o BatchMode=yes -p $PORT root@$HOST"

# Thresholds. MIN_DL_MBPS is set well under the ~155 MB/s Xet achieves but well
# over the ~6 MB/s slow path, so it separates the two without false alarms.
MIN_DL_MBPS=${MIN_DL_MBPS:-40}
STALL_POLLS=${STALL_POLLS:-5}          # polls with no progress before aborting
LOW_CREDIT=${LOW_CREDIT:-3.00}         # warn the user at this balance

api() { curl -s -m 30 "https://console.vast.ai/api/v0/$1" \
        -H "Authorization: Bearer $(cat "$KEY_FILE")" "${@:2}"; }

# Keep this free of nested quotes: an earlier version used an f-string with
# escaped double quotes inside single quotes inside a shell function, which
# python parsed as a line-continuation error. The watchdog then reported
# 'credit $?' every poll and the budget guard silently never fired -- exactly
# the class of failure this script exists to catch.
credit() { api users/current/ | python3 -c 'import sys,json
d=json.load(sys.stdin)
print("%.2f" % d.get("credit", 0))' 2>/dev/null || echo "?"; }

destroy() {
    echo ">>> DESTROYING instance $ID"
    api "instances/$ID/" -X DELETE -H "Content-Type: application/json" -d '{}' | head -c 60
    echo
}

prev_mb=0; stalls=0; warned=0
echo "watching $ID every ${POLL}s  (abort if <${MIN_DL_MBPS} MB/s or stalled ${STALL_POLLS} polls)"

while :; do
    ts=$(date +%H:%M:%S)
    cr=$(credit)

    # One round-trip for everything: bytes on disk, whether the job process is
    # alive, GPU utilisation, and the last meaningful log line.
    # OK= is the liveness marker for the ssh round-trip itself. Without it a
    # transient network blip returns empty fields, which read as "no process,
    # GPUs idle" -- i.e. indistinguishable from a dead job, and the watchdog
    # would destroy a perfectly healthy rental. Only act on data we actually got.
    probe=$($SSH '
        mb=$(du -sm /workspace/src /workspace/nemotron-heretic-w4a16 2>/dev/null \
             | awk "{s+=\$1} END {print s+0}")
        alive=$(pgrep -cf "run.sh|run_quant|dl.sh" 2>/dev/null || echo 0)
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
              | awk "{s+=\$1} END {print int(s/NR)}")
        echo "OK=$mb $alive $gpu"' 2>/dev/null)
    case "$probe" in
        OK=*) read -r mb alive gpu <<<"${probe#OK=}" ;;
        *)    ssh_fails=$((${ssh_fails:-0}+1))
              echo "$ts  ssh unreachable (${ssh_fails}/3) -- not judging the job on this"
              [ "$ssh_fails" -ge 3 ] && { echo ">>> ssh down 3 polls; check manually:"; \
                  echo "    ssh -p $PORT root@$HOST"; }
              sleep "$POLL"; continue ;;
    esac
    ssh_fails=0

    mb=${mb:-0}; alive=${alive:-0}; gpu=${gpu:-0}
    rate=$(( (mb - prev_mb) / (POLL > 0 ? POLL : 1) ))
    # Scan the WHOLE log, not the tail. The phase headers are printed once, and
    # calibration emits thousands of progress lines afterwards, so "=== 3.
    # quantise ===" scrolls out of any tail window within minutes. When that
    # happened the phase went blank, the quantise-aware branch stopped matching,
    # and the download-phase heuristics took over and began striking a healthy
    # run. grep over the full file is a few KB -- cheap, and it cannot go stale.
    phase=$($SSH 'grep -E "^=== [0-9]" /workspace/quant.log 2>/dev/null | tail -1' 2>/dev/null)

    printf "%s  %6d MB  %+4d MB/s  gpu %3d%%  proc %s  credit \$%s  %s\n" \
           "$ts" "$mb" "$rate" "$gpu" "$alive" "$cr" "${phase:-...}"

    # --- the job died but the box is still billing -----------------------
    if [ "$alive" -eq 0 ]; then
        if $SSH 'grep -q "DONE" /workspace/quant.log 2>/dev/null'; then
            echo ">>> JOB COMPLETE"; exit 0
        fi
        echo ">>> job process gone and no DONE marker -- last 15 lines:"
        $SSH 'tail -15 /workspace/quant.log 2>/dev/null | tr "\r" "\n" | grep -vE "^\s*$"'
        # Do NOT destroy on the first sighting. A crash is often followed by a
        # hands-on fix (install a missing dep, correct a pin, relaunch), and the
        # 247 GB already on that disk is worth far more than the few minutes of
        # idle billing. This exact race destroyed a box mid-repair once and threw
        # away a completed download. Touching /workspace/HOLD suppresses the
        # teardown entirely while you work.
        if $SSH 'test -f /workspace/HOLD' 2>/dev/null; then
            echo "    /workspace/HOLD present -- leaving the box alone"
            sleep "$POLL"; continue
        fi
        dead=$((${dead:-0}+1))
        if [ "$dead" -lt "${DEAD_GRACE:-5}" ]; then
            echo "    grace $dead/${DEAD_GRACE:-5} -- touch /workspace/HOLD to keep it"
            sleep "$POLL"; continue
        fi
        destroy; exit 1
    fi
    dead=0

    # --- stall detection, DOWNLOAD PHASE ONLY ----------------------------
    # These heuristics are valid while fetching and actively harmful during
    # GPTQ. Quantisation writes nothing to disk for hours (output is saved once,
    # at the end) and is layer-sequential, so one GPU works while the rest idle
    # and the 4-card AVERAGE sits at 5-9%. "bytes unchanged AND gpu<10%" is
    # therefore true for a perfectly healthy quantisation run -- which is
    # exactly how this watchdog destroyed a job at layer 1 of 89 while the log
    # was printing "Quantizing model.layers.1.mixer.experts.141.down_proj".
    #
    # The log is the only honest progress signal once compute starts, so switch
    # to it: healthy GPTQ emits a new "Quantizing ..." line every few minutes.
    if [ "$phase" = "=== 3. quantise ===" ] || echo "$phase" | grep -q "quantise"; then
        # Track the LOG BYTE COUNT, not any one keyword. llm-compressor's output
        # changes shape as it goes -- "(3/89): Calibrating: 61/256" during
        # calibration, "Quantizing model.layers.N..." during the solve, plus
        # metric lines in between. An earlier version counted only "Quantizing "
        # and read 0 for the entire calibration phase, i.e. it was about to kill
        # a run that was demonstrably alive at layer 3 of 89. Any healthy phase
        # appends to the log; a truly hung one does not.
        qbytes=$($SSH 'wc -c < /workspace/quant.log 2>/dev/null' 2>/dev/null)
        qbytes=${qbytes:-0}
        if [ "$qbytes" -gt "${prev_q:-0}" ]; then
            stalls=0
            layer=$($SSH 'tail -1 /workspace/quant.log | tr "\r" "\n" | tail -1 | cut -c1-60' 2>/dev/null)
            echo "    progressing: $layer"
        else
            stalls=$((stalls+1))
            echo "    log not growing, strike $stalls/$STALL_POLLS"
        fi
        prev_q=$qbytes
    elif [ "$rate" -gt 0 ] && [ "$rate" -lt "$MIN_DL_MBPS" ]; then
        stalls=$((stalls+1))
        echo "    slow transfer ($rate MB/s < $MIN_DL_MBPS), strike $stalls/$STALL_POLLS"
    elif [ "$rate" -eq 0 ] && [ "$gpu" -lt 10 ]; then
        stalls=$((stalls+1))
        echo "    no progress and GPUs idle, strike $stalls/$STALL_POLLS"
    else
        stalls=0
    fi
    if [ "$stalls" -ge "$STALL_POLLS" ]; then
        echo ">>> ABORT: $STALL_POLLS consecutive polls with no useful progress."
        $SSH 'tail -15 /workspace/quant.log 2>/dev/null | tr "\r" "\n" | grep -vE "^\s*$"'
        destroy; exit 1
    fi

    # --- budget ----------------------------------------------------------
    if [ "$warned" -eq 0 ] && [ "$(echo "$cr < $LOW_CREDIT" | bc -l 2>/dev/null)" = "1" ]; then
        echo ">>> CREDIT BELOW \$$LOW_CREDIT -- tell the user now"
        warned=1
    fi
    if [ "$(echo "$cr <= 0.20" | bc -l 2>/dev/null)" = "1" ]; then
        echo ">>> OUT OF CREDIT -- destroying before it strands mid-job"
        destroy; exit 1
    fi

    prev_mb=$mb
    sleep "$POLL"
done
