#!/usr/bin/env bash
# GPU benchmarks for bin/cg_gpu (MPI + OpenMP target offload, one GPU per MPI rank).
#   MACHINE=vanda_gpu bash scripts/run_gpu.sh
# Knobs: GPU_2D=1 (2D problem instead of steps 2-4), ONLY=validate (stop after step 1), VAL_P (rank counts for step 1), MACHINE, NGPU (GPUs of the job; default from CUDA_VISIBLE_DEVICES / nvidia-smi, capped
#        by the devices the OpenMP runtime sees), GPU_SIZES, GSOLVE_N (512), GSTRONG_N (512),
#        GWEAK_L (384), ITERS, REPS, MPIRUN, BIN.   Output: results/$MACHINE/gpu.csv
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=${MACHINE:-gpu}
BIN=${BIN:-bin}
ITERS=${ITERS:-50}
REPS=${REPS:-3}
MPIRUN=${MPIRUN:-"mpirun --bind-to none"}   # GPU nodes give the job a cpuset that core binding can trip over
GPU_SIZES=${GPU_SIZES:-"128 192 256 384 512 640 768"}
export OMP_TARGET_OFFLOAD=${OMP_TARGET_OFFLOAD:-MANDATORY}   # fail instead of silently running on the host
export CUDA_MODULE_LOADING=EAGER                              # no lazy kernel loading inside timed loops
csv() { grep '^CSV,' | sed 's/^CSV,//' || true; }

# GPUs granted to this job: PBS sets CUDA_VISIBLE_DEVICES; nvidia-smi may list the whole node
if [[ -z "${NGPU:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c . || true)
    else
        NGPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)
    fi
    (( NGPU >= 1 )) || NGPU=1
fi
# never more ranks than devices the OpenMP runtime can actually see
NDEV=$( ($MPIRUN -np 1 "$BIN/cg_gpu" -n 8 2>&1 >/dev/null || true) | sed -n 's/.*devices\/node=\([0-9]*\).*/\1/p')
if [[ -n $NDEV ]] && (( NDEV >= 1 && NDEV < NGPU )); then NGPU=$NDEV; fi

OUT=results/$MACHINE
mkdir -p "$OUT"
CSV=$OUT/gpu.csv
for f in "$CSV" "$OUT/validation.txt"; do [[ -f $f ]] && mv "$f" "$f.bak.$(date +%Y%m%d-%H%M%S)"; done
echo "exp,rep,$("$BIN/cg_serial" -H)" > "$CSV"
record() {
    local exp=$1 rep=$2 l n; shift 2
    l=$("$@" | csv) || true
    n=$(printf '%s' "$l" | grep -c '' || true)
    if [[ $n -eq 1 ]]; then echo "$exp,$rep,$l" | tee -a "$CSV"; else echo "WARNING: run failed: $*" >&2; fi
}
{
    echo "machine: $MACHINE   date: $(date)"
    echo "gpus: $NGPU (OpenMP devices: ${NDEV:-?})   ITERS=$ITERS REPS=$REPS SIZES=$GPU_SIZES"
    echo "toolchain module: ${GPU_MODULE:-unknown}"
    hostname
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv,noheader 2>/dev/null || true
    lscpu 2>/dev/null | grep -E "Model name|Socket" || true
    $MPIRUN --version 2>&1 | grep -m1 . || true
    ${MPICC:-mpicc} --version 2>&1 | grep -m1 . || true
} > "$OUT/env.txt"
cat "$OUT/env.txt"

# 1. correctness on the GPU, point-wise against the serial solution. P = 1, 2, 4 ranks: with
#    one GPU the ranks share it, which still runs the halo path (pack on the GPU, staging,
#    MPI, unpack) on real hardware.
#    P = 8 (2x2x2 ranks) exercises all three face directions. Ranks sharing a GPU are not
#    bound to cores (the job's cpuset on GPU nodes makes Open MPI's core binding fail).
VAL_P=${VAL_P:-"1 2 4 8"}
ser=/tmp/ser40.$$.bin gpu=/tmp/gpu40.$$.bin nfail=0
"$BIN/cg_serial" -n 40 -q -o $ser > /dev/null
for P in $VAL_P; do for a in cg cg1; do
    rm -f $gpu
    l=$($MPIRUN --oversubscribe -np "$P" "$BIN/cg_gpu" -n 40 -a $a -q -o $gpu | csv) || true
    d=nan; [[ -f $gpu ]] && d=$(python3 scripts/compare_bin.py $gpu $ser 2>/dev/null || echo nan)
    ok=$(python3 -c "import sys; d=float(sys.argv[1]); print('PASS' if d == d and d < 1e-10 else 'FAIL')" "$d" 2>/dev/null || echo FAIL)
    [[ $ok == PASS ]] || nfail=$((nfail + 1))
    [[ -n $l ]] && echo "validate,1,$l" >> "$CSV"
    echo "  $ok  max|x_gpu-x_ser|/max|x_ser| = $d  (P=$P, $a)" | tee -a "$OUT/validation.txt"
done; done
rm -f $ser $gpu
echo "validation: $nfail failure(s)" | tee -a "$OUT/validation.txt"
[[ ${ONLY:-} == validate ]] && { echo "done (validation only) -> $OUT/validation.txt"; exit 0; }

# 2. one GPU, growing problem: time per iteration and effective bandwidth
[[ ${GPU_2D:-0} == 1 ]] && GPU_SIZES=""          # 2D run: skip the 3D sweep
for rep in $(seq 1 "$REPS"); do for N in $GPU_SIZES; do for a in cg cg1; do
    record gsize "$rep" $MPIRUN -np 1 "$BIN/cg_gpu" -n "$N" -a $a -t 0 -m "$ITERS" -q
done; done; done

# 3. several GPUs (if the job has them): strong (512^3) and weak (~384^3 per GPU) scaling
if (( NGPU > 1 )); then
    for rep in $(seq 1 "$REPS"); do for P in $(seq 1 "$NGPU"); do
        record gstrong "$rep" $MPIRUN -np "$P" "$BIN/cg_gpu" -n "${GSTRONG_N:-512}" -t 0 -m "$ITERS" -q
        record gweak "$rep" $MPIRUN -np "$P" "$BIN/cg_gpu" -L "${GWEAK_L:-384}" -t 0 -m "$ITERS" -q
    done; done
fi

# 3b. 2D problem on one GPU (GPU_2D=1): size sweep, validation against the serial 2D solution, solve
if [[ ${GPU_2D:-0} == 1 ]]; then
    "$BIN/cg_serial" -D 2 -n 100 -q -o /tmp/ser2d.$$.bin > /dev/null
    for P in 1 2 4; do for a in cg cg1; do
        rm -f /tmp/gpu2d.$$.bin
        l=$($MPIRUN --oversubscribe -np "$P" "$BIN/cg_gpu" -D 2 -n 100 -a $a -q -o /tmp/gpu2d.$$.bin | csv) || true
        d=nan; [[ -f /tmp/gpu2d.$$.bin ]] && d=$(python3 scripts/compare_bin.py /tmp/gpu2d.$$.bin /tmp/ser2d.$$.bin 2>/dev/null || echo nan)
        ok=$(python3 -c "import sys; d=float(sys.argv[1]); print('PASS' if d == d and d < 1e-10 else 'FAIL')" "$d" 2>/dev/null || echo FAIL)
        [[ -n $l ]] && echo "validate2d,1,$l" >> "$CSV"
        echo "  $ok  2D max|x_gpu-x_ser|/max|x_ser| = $d  (P=$P, $a)" | tee -a "$OUT/validation.txt"
    done; done
    rm -f /tmp/ser2d.$$.bin /tmp/gpu2d.$$.bin
    for rep in $(seq 1 "$REPS"); do for N in ${GPU_SIZES2D:-1024 2048 4096 8192 12288 16384}; do
        record gsize2d "$rep" $MPIRUN -np 1 "$BIN/cg_gpu" -D 2 -n "$N" -t 0 -m "$ITERS" -q
    done; done
    record gsolve2d 1 $MPIRUN -np 1 "$BIN/cg_gpu" -D 2 -n "${GSOLVE2D_N:-8192}" -t 1e-8 -q
    echo "done (2D) -> $CSV"; exit 0
fi

# 4. complete solve on one GPU, same problem as the CPU time-to-solution table
for a in cg cg1; do record gsolve 1 $MPIRUN -np 1 "$BIN/cg_gpu" -n "${GSOLVE_N:-512}" -a $a -t 1e-8 -q; done
echo "done -> $CSV"
