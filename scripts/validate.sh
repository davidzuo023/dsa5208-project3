#!/usr/bin/env bash
# Correctness tests for the parallel solver.
#
#   bash scripts/validate.sh quick   # ~30 s, used by `make test`   -> results/validation_quick/
#   bash scripts/validate.sh full    # all variants x P x T x decompositions + edge cases + grid study
#                                    #                                 -> results/validation/
#
# 1. every parallel variant (alg x comm x decomposition x ranks x threads) is compared point-wise
#    with the serial solution (solutions are written with MPI-IO and compared in Python);
# 2. iteration counts / residuals are compared with the serial run;
# 3. grid-convergence study: the discretisation error must decrease as O(h^2) in serial
#    and in parallel;
# 4. the same three checks for the 2D problem (-D 2: 5-point stencil, files *2d.csv).
# Results go to results/validation/*.csv; scripts/check_validation.py turns them into PASS/FAIL.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-full}
MPIRUN=${MPIRUN:-mpirun}
if [[ -z "${MPIFLAGS+x}" ]]; then
    # Validation runs may use more ranks than cores (timings are irrelevant here).
    if $MPIRUN --version 2>&1 | grep -q "Open MPI"; then MPIFLAGS="--oversubscribe"; else MPIFLAGS=""; fi
fi
# quick mode never overwrites the full results that the report uses
if [[ $MODE == quick ]]; then OUT=${OUT:-results/validation_quick}; else OUT=${OUT:-results/validation}; fi
BIN=${BIN:-bin}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
export OMP_WAIT_POLICY=passive   # threads must not spin when cores are oversubscribed

NV=40   # 40 is not divisible by 3 or 6 -> uneven subdomains are exercised
if [[ $MODE == quick ]]; then
    PS="1 2 4"; TS="1"; DS="1 3"; NCONV="16 32 64"
else
    PS="1 2 3 4 6 8"; TS="1 2"; DS="1 2 3"; NCONV="16 24 32 48 64 96 128"
fi
ALGS="cg cg1 pipecg"; COMMS="blocking overlap"
HDR=$("$BIN/cg_serial" -H)
csv() { grep '^CSV,' | sed 's/^CSV,//'; }   # strip the "CSV," tag

# ---------------------------------------------------------------- serial reference
echo "$HDR" > "$OUT/serial_ref.csv"
OMP_NUM_THREADS=1 "$BIN/cg_serial" -n $NV -q -o "$TMP/serial.bin" | csv >> "$OUT/serial_ref.csv"

# ---------------------------------------------------------------- all variants
echo "$HDR,maxdiff_vs_serial" > "$OUT/variants.csv"
n=0
for P in $PS; do for T in $TS; do
    [[ $T -gt 1 && $P -gt 4 ]] && continue
    for d in $DS; do for a in $ALGS; do for c in $COMMS; do
        f="$TMP/p.bin"
        line=$(OMP_NUM_THREADS=$T $MPIRUN $MPIFLAGS -np $P "$BIN/cg_mpi" -n $NV -a $a -c $c -d $d -q -o "$f" | csv)
        diff=$(python3 scripts/compare_bin.py "$f" "$TMP/serial.bin")
        echo "$line,$diff" >> "$OUT/variants.csv"
        n=$((n+1))
    done; done; done
done; done
echo "variants: $n runs"

# ---------------------------------------------------------------- edge cases
# tiny / uneven local blocks: 1-point-thick slabs, 2-point blocks, non-divisible grids
echo "$HDR,maxdiff_vs_serial" > "$OUT/edge.csv"
EDGES="12:8:1 12:8:3 7:6:2 9:5:3 2:2:1"
[[ $MODE == quick ]] && EDGES="12:8:1 7:6:2"
for e in $EDGES; do
    IFS=: read -r N P d <<< "$e"
    OMP_NUM_THREADS=1 "$BIN/cg_serial" -n $N -q -o "$TMP/sref.bin" > /dev/null
    for a in $ALGS; do for c in $COMMS; do
        line=$(OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np $P "$BIN/cg_mpi" -n $N -a $a -c $c -d $d -q -o "$TMP/p.bin" | csv)
        diff=$(python3 scripts/compare_bin.py "$TMP/p.bin" "$TMP/sref.bin")
        echo "$line,$diff" >> "$OUT/edge.csv"
    done; done
done

# ---------------------------------------------------------------- grid convergence
echo "$HDR" > "$OUT/convergence.csv"
for N in $NCONV; do
    OMP_NUM_THREADS=1 "$BIN/cg_serial" -n $N -t 1e-10 -q | csv >> "$OUT/convergence.csv"
    OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np 4 "$BIN/cg_mpi" -n $N -t 1e-10 -a cg -c overlap -d 3 -q | csv >> "$OUT/convergence.csv"
    OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np 4 "$BIN/cg_mpi" -n $N -t 1e-10 -a pipecg -c overlap -d 3 -q | csv >> "$OUT/convergence.csv"
done

# ---------------------------------------------------------------- 2D problem (5-point stencil)
NV2=100   # 100 is not divisible by 3 -> uneven subdomains
if [[ $MODE == quick ]]; then DS2="1 2"; NCONV2="32 64 128"; EDGES2="12:8:1 7:6:2"
else DS2="1 2"; NCONV2="32 48 64 96 128 192 256"; EDGES2="12:8:1 12:8:2 7:6:2 9:5:2 2:2:1"; fi
echo "$HDR" > "$OUT/serial_ref2d.csv"
OMP_NUM_THREADS=1 "$BIN/cg_serial" -D 2 -n $NV2 -q -o "$TMP/serial2d.bin" | csv >> "$OUT/serial_ref2d.csv"
echo "$HDR,maxdiff_vs_serial" > "$OUT/variants2d.csv"
n=0
for P in $PS; do for T in $TS; do
    [[ $T -gt 1 && $P -gt 4 ]] && continue
    for d in $DS2; do for a in $ALGS; do for c in $COMMS; do
        line=$(OMP_NUM_THREADS=$T $MPIRUN $MPIFLAGS -np $P "$BIN/cg_mpi" -D 2 -n $NV2 -a $a -c $c -d $d -q -o "$TMP/p.bin" | csv)
        diff=$(python3 scripts/compare_bin.py "$TMP/p.bin" "$TMP/serial2d.bin")
        echo "$line,$diff" >> "$OUT/variants2d.csv"
        n=$((n+1))
    done; done; done
done; done
echo "2D variants: $n runs"
echo "$HDR,maxdiff_vs_serial" > "$OUT/edge2d.csv"
for e in $EDGES2; do
    IFS=: read -r N P d <<< "$e"
    OMP_NUM_THREADS=1 "$BIN/cg_serial" -D 2 -n $N -q -o "$TMP/sref.bin" > /dev/null
    for a in $ALGS; do for c in $COMMS; do
        line=$(OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np $P "$BIN/cg_mpi" -D 2 -n $N -a $a -c $c -d $d -q -o "$TMP/p.bin" | csv)
        diff=$(python3 scripts/compare_bin.py "$TMP/p.bin" "$TMP/sref.bin")
        echo "$line,$diff" >> "$OUT/edge2d.csv"
    done; done
done
echo "$HDR" > "$OUT/convergence2d.csv"
for N in $NCONV2; do
    OMP_NUM_THREADS=1 "$BIN/cg_serial" -D 2 -n $N -t 1e-10 -q | csv >> "$OUT/convergence2d.csv"
    OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np 4 "$BIN/cg_mpi" -D 2 -n $N -t 1e-10 -a cg -c overlap -d 2 -q | csv >> "$OUT/convergence2d.csv"
    OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np 4 "$BIN/cg_mpi" -D 2 -n $N -t 1e-10 -a pipecg -c overlap -d 2 -q | csv >> "$OUT/convergence2d.csv"
done

# ---------------------------------------------------------------- residual histories
# run far below the usual tolerance to expose the attainable accuracy of each CG variant
if [[ $MODE == full ]]; then
    for a in $ALGS; do
        OMP_NUM_THREADS=1 $MPIRUN $MPIFLAGS -np 4 "$BIN/cg_mpi" -n 64 -a $a -c overlap -d 3 -t 1e-15 -m 700 \
            -r "$OUT/hist_$a.csv" -e 5 -q > /dev/null
    done
fi

python3 scripts/check_validation.py "$OUT"
