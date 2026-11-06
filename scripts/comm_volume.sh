#!/usr/bin/env bash
# Halo-exchange volume per iteration for 1D/2D/3D decompositions as a function of P.
# The byte and message counts are deterministic (independent of the machine), so this can run
# oversubscribed on a laptop: every run does a single CG iteration and only the counters matter.
set -euo pipefail
cd "$(dirname "$0")/.."
MPIRUN=${MPIRUN:-mpirun}
BIN=${BIN:-bin}
N=${N:-256}
PS=${PS:-"2 4 8 16 32 64"}
if [[ -z "${MPIFLAGS+x}" ]]; then
    if $MPIRUN --version 2>&1 | grep -q "Open MPI"; then MPIFLAGS="--oversubscribe"; else MPIFLAGS=""; fi
fi
OUT=results/commvol.csv
echo "$("$BIN/cg_serial" -H)" > "$OUT"
for P in $PS; do for d in 1 2 3; do
    OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive $MPIRUN $MPIFLAGS -np "$P" "$BIN/cg_mpi" -n "$N" -d $d -m 1 -t 0 -q \
        | grep '^CSV,' | sed 's/^CSV,//' | tee -a "$OUT"
done; done
