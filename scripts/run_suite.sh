#!/usr/bin/env bash
# Benchmark suite. The same script is used on a workstation and inside the PBS jobs in pbs/.
#
#   MACHINE=local bash scripts/run_suite.sh
#
# Knobs (environment variables):
#   MACHINE       results go to results/$MACHINE/                               [local]
#   NS            global grid N (N^3 unknowns) for strong-scaling experiments    [256]
#   NW            ~NW^3 unknowns per rank for weak scaling                       [64]
#   ITERS         fixed number of CG iterations per timing run (tol = 0)         [100]
#   REPS          repetitions of every configuration (plots use the minimum)     [3]
#   PLIST         MPI process counts   [powers of 2 below a node, half node, whole nodes]
#   VARIANT_PMIN  smallest P for the decomposition/halo/algorithm variants        [2]
#   EXPERIMENTS   subset of: stream serial strong variants hybrid weak solve sizes
#                 strong2 strong2d weak2d sizes2d solve2d (2D problem: NS2D, NW2D, SIZES2D, SOLVE2D_MAX)
#                 (sizes = serial code on SIZES, fixed iterations and full solves;
#                 strong2 = strong scaling of NS2^3 from one node up)  [all but sizes, strong2]
#   SIZES         grids for the serial problem-size study           [64 96 128 192 256 384 512]
#   MPI_FLAVOR    openmpi | intelmpi | mpich                          [auto-detected]
#   MPIRUN        launcher                                                       [mpirun]
#   BIN           directory with the executables                                 [bin]
#   APPEND        1 = append to an existing bench.csv instead of starting a new one [0]
#
# Output: results/$MACHINE/bench.csv  (one row per run, columns: exp,rep,<solver CSV>)
#         results/$MACHINE/stream.csv (memory bandwidth vs threads)
#         results/$MACHINE/env.txt    (hardware / software description)
set -euo pipefail
cd "$(dirname "$0")/.."

MACHINE=${MACHINE:-local}
NS=${NS:-256}
NW=${NW:-64}
ITERS=${ITERS:-100}
REPS=${REPS:-3}
VARIANT_PMIN=${VARIANT_PMIN:-2}
EXPERIMENTS=${EXPERIMENTS:-"stream serial strong variants hybrid weak solve"}
MPIRUN=${MPIRUN:-mpirun}
BIN=${BIN:-bin}
OUT=results/$MACHINE
mkdir -p "$OUT"

# ------------------------------------------------------------------ resources
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
    NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
    CPN=$(( $(wc -l < "$PBS_NODEFILE") / NODES ))     # = mpiprocs per node in the select line
else
    NODES=1
    CPN=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)   # nproc obeys OMP_NUM_THREADS
fi
CORES=$(( NODES * CPN ))
if [[ -z "${PLIST:-}" ]]; then
    # 1, 2, 4, ... below one node, half a node (one socket when the job has whole nodes), then whole
    # nodes only: a P that splits unevenly across nodes (e.g. 128 = 72 + 56) is not used
    PLIST=$( { p=1; while (( p < CPN )); do echo $p; p=$(( p * 2 )); done
               (( CPN % 2 == 0 )) && echo $(( CPN / 2 ))
               for k in $(seq 1 "$NODES"); do echo $(( k * CPN )); done; } | sort -n | uniq | tr '\n' ' ')
fi

if [[ -z "${MPI_FLAVOR:-}" ]]; then
    v=$($MPIRUN --version 2>&1 || true)
    if   grep -q "Open MPI" <<< "$v"; then MPI_FLAVOR=openmpi
    elif grep -qi "intel"   <<< "$v"; then MPI_FLAVOR=intelmpi
    else MPI_FLAVOR=mpich; fi
fi
HOSTOPT=""
[[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]] && HOSTOPT="--hostfile $PBS_NODEFILE"

# environment for ranks started on other nodes (only variables that are set)
XENV=""; for v in PATH LD_LIBRARY_PATH; do [[ -n "${!v:-}" ]] && XENV="$XENV -x $v"; done
# Open MPI: start remote daemons from the same installation as this mpirun
PREFIX=""
if [[ ${MPI_FLAVOR:-} == openmpi ]] && command -v mpirun >/dev/null; then
    PREFIX="--prefix $(dirname "$(dirname "$(command -v mpirun)")")"
fi

# launch P T solver-args...   (pins every rank to T consecutive cores, threads close to it)
launch() {
    local P=$1 T=$2; shift 2
    local ppn=$(( CPN / T )); (( ppn > P )) && ppn=$P
    case $MPI_FLAVOR in
    openmpi)
        $MPIRUN -np "$P" $HOSTOPT $PREFIX --map-by "ppr:${ppn}:node:PE=${T}" --bind-to core \
            $XENV \
            -x OMP_NUM_THREADS="$T" -x OMP_PLACES=cores -x OMP_PROC_BIND=close "$BIN/cg_mpi" "$@" ;;
    intelmpi)
        # under PBS, Intel MPI ignores -ppn unless the scheduler's placement is switched off
        I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off I_MPI_PIN_DOMAIN=omp \
            OMP_NUM_THREADS=$T OMP_PLACES=cores OMP_PROC_BIND=close \
            $MPIRUN -np "$P" -ppn "$ppn" "$BIN/cg_mpi" "$@" ;;
    *)
        OMP_NUM_THREADS=$T OMP_PLACES=cores OMP_PROC_BIND=close \
            $MPIRUN -n "$P" -ppn "$ppn" -bind-to "core:$T" "$BIN/cg_mpi" "$@" ;;
    esac
}

csv() { grep '^CSV,' | sed 's/^CSV,//'; }
BENCH=$OUT/bench.csv
# a new job starts a new file (the old one is kept as a backup), so runs with different
# settings never mix; APPEND=1 adds to the existing file instead
if [[ -f $BENCH && "${APPEND:-0}" != 1 ]]; then mv "$BENCH" "$BENCH.bak.$(date +%Y%m%d-%H%M%S)"; fi
[[ -f $BENCH ]] || echo "exp,rep,$("$BIN/cg_serial" -H)" > "$BENCH"
record() {
    local exp=$1 rep=$2 l n; shift 2
    l=$("$@" | csv) || true
    n=$(printf '%s' "$l" | grep -c '' || true)
    if [[ $n -eq 1 ]]; then echo "$exp,$rep,$l" | tee -a "$BENCH"
    elif [[ $n -eq 0 ]]; then echo "WARNING: run failed: $*" >&2
    else echo "WARNING: $n result lines (MPI library mismatch? every rank ran alone): $*" >&2; fi
}
has() { [[ " $EXPERIMENTS " == *" $1 "* ]]; }

{
    echo "machine: $MACHINE   date: $(date)"
    echo "nodes: $NODES  cores/node: $CPN  MPI flavour: $MPI_FLAVOR"
    echo "NS=$NS NW=$NW ITERS=$ITERS REPS=$REPS PLIST=$PLIST"
    hostname; lscpu 2>/dev/null | grep -E "Model name|Socket|Core|Thread|NUMA node\(|L2|L3" || true
    $MPIRUN --version 2>&1 | head -2 || true
    ${MPICC:-mpicc} --version 2>&1 | head -1 || true   # the compiler that built all binaries
    command -v ompi_info >/dev/null && ompi_info 2>/dev/null | grep -E "MCA (pml|btl|mtl):" | sed 's/^ */  /' || true
} > "$OUT/env.txt"
cat "$OUT/env.txt"

# multi-node jobs: one quick run on all cores first. Remote daemons are started through PBS
# (pbs_tmrsh: ranks stay inside the job's cpuset) if possible, else with ssh. If neither works,
# measure on the first node only instead of failing for the whole walltime.
multi_ok() { local o; o=$(launch "$CORES" 1 -n 64 -t 0 -m 2 -q 2>&1 || true); grep -q '^CSV,' <<< "$o"; }
if (( NODES > 1 )) && [[ $MPI_FLAVOR == openmpi ]]; then
    agent=none
    TMRSH=$(command -v pbs_tmrsh || ls /opt/pbs/bin/pbs_tmrsh 2>/dev/null || true)
    if [[ -n $TMRSH ]]; then
        export OMPI_MCA_plm_rsh_agent=$TMRSH
        multi_ok && agent=$TMRSH || unset OMPI_MCA_plm_rsh_agent
    fi
    if [[ $agent == none ]] && multi_ok; then agent=ssh; fi
    if [[ $agent == none ]]; then
        echo "WARNING: multi-node launch failed; restricting to one node (P <= $CPN)" | tee -a "$OUT/env.txt"
        PLIST=$(for p in $PLIST; do if (( p <= CPN )); then echo $p; fi; done | tr '\n' ' ')
        NODES=1; CORES=$CPN
        first=$(head -1 "$PBS_NODEFILE"); grep -x -F "$first" "$PBS_NODEFILE" > "$OUT/hostfile.node1"
        HOSTOPT="--hostfile $OUT/hostfile.node1"
    else
        {
            echo "multi-node launch test on $CORES ranks: OK (remote start: $agent)"
            echo "placement (one line per node: host, cores allowed to this job, CPU):"
            $MPIRUN -np "$NODES" $HOSTOPT $PREFIX --map-by ppr:1:node --bind-to none \
                sh -c 'echo "  $(hostname)  cpus=$(grep Cpus_allowed_list /proc/self/status | cut -f2)  $(lscpu | grep "Model name" | sed "s/.*: *//")"' \
                2>&1 | sort || true
        } | tee -a "$OUT/env.txt"
    fi
fi

# ------------------------------------------------------------------ experiments
if has stream; then
    # same placement as the CG ranks: cores filled in order (one socket first)
    OMP_PROC_BIND=close OMP_PLACES=cores "$BIN/stream_omp" 80 "$CPN" | tee "$OUT/stream.csv"
fi

# serial problem-size study: matrix dimension N^3 and nonzeros ~7N^3 both vary
if has sizes; then
    for N in ${SIZES:-64 96 128 192 256 384 512}; do
        for rep in $(seq 1 "$REPS"); do
            record sizes_iter "$rep" "$BIN/cg_serial" -n "$N" -t 0 -m "$ITERS" -q
        done
        record sizes_solve 1 "$BIN/cg_serial" -n "$N" -t 1e-8 -q
    done
fi

for rep in $(seq 1 "$REPS"); do
    if has serial; then
        record serial "$rep" "$BIN/cg_serial" -n "$NS" -t 0 -m "$ITERS" -q
    fi
    for P in $PLIST; do
        # baseline: pure MPI, 3D blocks, overlapped halos, classic CG
        if has strong; then
            record strong "$rep" launch "$P" 1 -n "$NS" -t 0 -m "$ITERS" -a cg -c overlap -d 3 -q
        fi
        # one factor at a time around the baseline
        if has variants && (( P >= VARIANT_PMIN )); then
            for d in 1 2; do
                record decomp "$rep" launch "$P" 1 -n "$NS" -t 0 -m "$ITERS" -a cg -c overlap -d $d -q
            done
            record comm "$rep" launch "$P" 1 -n "$NS" -t 0 -m "$ITERS" -a cg -c blocking -d 3 -q
            for a in cg1 pipecg; do
                record alg "$rep" launch "$P" 1 -n "$NS" -t 0 -m "$ITERS" -a $a -c overlap -d 3 -q
            done
            record alg "$rep" launch "$P" 1 -n "$NS" -t 0 -m "$ITERS" -a pipecg -c blocking -d 3 -q
        fi
        # strong scaling of a larger grid (NS2^3), from one full node upwards
        if has strong2 && (( P >= CPN )); then
            record strong2 "$rep" launch "$P" 1 -n "${NS2:-768}" -t 0 -m "$ITERS" -a cg -c overlap -d 3 -q
            record strong2_pipe "$rep" launch "$P" 1 -n "${NS2:-768}" -t 0 -m "$ITERS" -a pipecg -c overlap -d 3 -q
        fi
        if has weak; then
            record weak "$rep" launch "$P" 1 -L "$NW" -t 0 -m "$ITERS" -a cg -c overlap -d 3 -q
        fi
        # 2D problem (5-point stencil): strong scaling of NS2D^2, slabs vs 2D blocks, weak scaling
        if has strong2d; then
            record strong2d "$rep" launch "$P" 1 -D 2 -n "${NS2D:-8192}" -t 0 -m "$ITERS" -a cg -c overlap -d 2 -q
            if (( P >= VARIANT_PMIN )); then
                record decomp2d "$rep" launch "$P" 1 -D 2 -n "${NS2D:-8192}" -t 0 -m "$ITERS" -a cg -c overlap -d 1 -q
                record comm2d "$rep" launch "$P" 1 -D 2 -n "${NS2D:-8192}" -t 0 -m "$ITERS" -a cg -c blocking -d 2 -q
                record alg2d "$rep" launch "$P" 1 -D 2 -n "${NS2D:-8192}" -t 0 -m "$ITERS" -a cg1 -c overlap -d 2 -q
            fi
        fi
        if has weak2d; then
            record weak2d "$rep" launch "$P" 1 -D 2 -L "${NW2D:-2048}" -t 0 -m "$ITERS" -a cg -c overlap -d 2 -q
        fi
    done
    # hybrid: all cores of the job, ranks x threads, threads never span nodes
    if has hybrid; then
        for T in 1 2 3 4 6 8 9 12 16 18 24 32 36 48 64 72 96 128; do
            (( T > CPN || CPN % T != 0 )) && continue
            P=$(( CORES / T ))
            record hybrid "$rep" launch "$P" "$T" -n "$NS" -t 0 -m "$ITERS" -a cg -c overlap -d 3 -q
        done
    fi
done

# serial problem-size study of the 2D problem: N^2 unknowns, 5N^2-4N nonzeros
if has sizes2d; then
    for N in ${SIZES2D:-256 512 1024 2048 4096 8192}; do
        for rep in $(seq 1 "$REPS"); do
            record sizes2d_iter "$rep" "$BIN/cg_serial" -D 2 -n "$N" -t 0 -m "$ITERS" -q
        done
        (( N <= ${SOLVE2D_MAX:-4096} )) && record sizes2d_solve 1 "$BIN/cg_serial" -D 2 -n "$N" -t 1e-8 -q
    done
fi

# 2D time to solution on all cores
if has solve2d; then
    for a in cg cg1; do
        record solve2d 1 launch "$CORES" 1 -D 2 -n "${NS2D:-8192}" -t 1e-8 -a $a -c overlap -d 2 -q
    done
fi

# time to solution (tol = 1e-8) of the three CG variants on all cores
if has solve; then
    for a in cg cg1 pipecg; do
        record solve 1 launch "$CORES" 1 -n "$NS" -t 1e-8 -a $a -c overlap -d 3 -q
    done
fi
echo "done -> $BENCH"
