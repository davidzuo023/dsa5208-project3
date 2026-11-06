# Conjugate Gradients for the 2D and 3D Poisson equation with MPI, OpenMP and GPU offload

DSA5208 project. A matrix-free Conjugate Gradient (CG) solver for

    -Δu = f  on (0,1)^d (d = 2 or 3),  u = 0 on the boundary,

discretised with the 5-point stencil on an N² grid or the 7-point stencil on an N³ grid
(`-D 2|3`, all three codes). The project has a serial
reference code, a hybrid MPI+OpenMP code in which every parallelisation decision can be switched
at run time, and an MPI + OpenMP target-offload code for GPUs (one GPU per MPI rank):

| switch | choices | what it changes |
|---|---|---|
| `-d` | `1`, `2`, `3` | slab / pencil / block domain decomposition (MPI Cartesian communicator) |
| `-c` | `blocking`, `overlap` | `MPI_Sendrecv` halo, or `Isend/Irecv` overlapped with the interior stencil |
| `-a` | `cg`, `cg1`, `pipecg` | classic CG (2 Allreduce/iter), Chronopoulos–Gear (1 Allreduce), pipelined CG (1 Iallreduce hidden behind the SpMV) |
| `OMP_NUM_THREADS` | T | OpenMP threads per MPI rank |

The report is `report/report.pdf`. Authors: Wu Yuhan, Zuo Dewei, Zhu Chen. Licence: MIT (`LICENSE`).

Headline results (NUS Vanda, Xeon Platinum 8452Y, 32 cores per node; NVIDIA A40). 3D:
512³ CG iteration in 9.23 ms on 256 cores / 8 nodes (speedup 59 over the serial code,
81 % strong-scaling efficiency relative to one node, 83 % for 768³), 80 % node-level weak-scaling
efficiency on 1.8 × 10⁹ unknowns, one A40 2.4 times faster than a 32-core socket
(25.0 ms per 512³ iteration). 2D: 8192² iteration in 4.56 ms on 256 cores, 90 % strong and
93 % weak node-level efficiency; slabs take 18 % less time than 2D blocks.

## Quick start: from source to report

Run these from the project root, in this order.

```bash
# 0. (containers running as root with Open MPI only)
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 1. build
make                                      # -> bin/cg_serial  bin/cg_mpi  bin/stream_omp

# 2. a first run: serial and 4 MPI ranks on a 128^3 grid
#    (Open MPI: add --oversubscribe if the machine has fewer than 4 cores)
./bin/cg_serial -n 128
OMP_NUM_THREADS=1 mpirun -np 4 ./bin/cg_mpi -n 128

# 3. correctness (parallel vs serial, edge cases, O(h^2) study, residual histories)
bash scripts/validate.sh full             # -> results/validation/summary.txt  (RESULT: PASS)

# 4. halo-volume measurement for 1D/2D/3D decompositions (machine independent)
bash scripts/comm_volume.sh               # -> results/commvol.csv

# 5. benchmarks on this machine (any subset of experiments, see scripts/run_suite.sh)
MACHINE=local NS=256 NW=128 REPS=3 bash scripts/run_suite.sh    # -> results/local/

# 6. on Vanda (see the section below): the four jobs behind the report
qsub -P <project> pbs/vanda256.pbs  ...    # -> results/vanda256/ etc.

# 7. figures, tables and the PDF report from results/vanda*/
make report                               # -> figures/*.pdf, report/report.pdf
```

## Layout

```
src/cg_serial.c        serial reference solver (classic CG)
src/cg_mpi.c           hybrid MPI+OpenMP solver (all variants)
src/common.h           problem definition (manufactured solution), CSV format
src/cg_gpu.c           MPI + OpenMP target-offload solver (classic CG and CG1, one GPU per rank)
src/stream_omp.c       STREAM-triad bandwidth probe (for the roofline/performance model)
scripts/validate.sh    correctness suite (parallel vs serial, edge cases, O(h^2) study, residual histories)
scripts/check_validation.py   PASS/FAIL evaluation of the validation CSVs
scripts/run_suite.sh   benchmark suite (strong, weak, variants, hybrid, time-to-solution, STREAM)
scripts/comm_volume.sh halo volume per iteration for 1D/2D/3D decompositions
scripts/make_plots.py  figures + LaTeX tables + number macros for the report
scripts/run_gpu.sh     GPU benchmarks (validation vs serial, size sweep, multi-GPU, time to solution)
scripts/gpu_job.sh     builds with the NVIDIA HPC SDK on a Vanda GPU node and runs run_gpu.sh
pbs/vanda256.pbs       PBS job: 8 nodes x 32 cores, strong/weak scaling, variants, hybrid, solves (256 processes)
pbs/vanda_node.pbs     PBS job: one 32-core chunk: STREAM, serial baseline, quick validation
pbs/vanda_sizes.pbs    PBS job: serial problem-size study (N = 64 ... 512) on one core
pbs/vanda_gpu.pbs      PBS job: one A40 GPU (calls scripts/gpu_job.sh; -v GPU_2D=1 for the 2D problem)
pbs/vanda2d.pbs        PBS job: 2D problem on 8 x 32 cores (strong, weak, variants, solves)
pbs/vanda_sizes2d.pbs  PBS job: serial 2D problem-size study on one core
pbs/vanda_suite.pbs    generic PBS job: build, quick validation, full suite (any node count)
pbs/atlas_suite.pbs    same for Atlas (queue parallel24)
pbs/validate.pbs       full validation as a PBS job
results/               measurements behind the report (see "Results included")
figures/, report/      generated figures, report source and PDF
```

## Requirements

* a C11 compiler with OpenMP (GCC ≥ 9; tested with 13.3), reached through the MPI wrapper
  `mpicc`: all three programs are built with it, so the serial baseline and the parallel code
  always use the same compiler
* an MPI-3 library (`MPI_Iallreduce`); tested with Open MPI 4.1.6. Intel MPI and MPICH launch
  lines are also supported by `scripts/run_suite.sh` (`MPI_FLAVOR=intelmpi|mpich`)
* Python ≥ 3.6 with `pandas` and `matplotlib` ≥ 3.3 for the plots
  (`pip install --user pandas matplotlib`), `pdflatex` for the report. The validation scripts
  need only the Python standard library.

## Build

```bash
make                       # -> bin/cg_serial  bin/cg_mpi  bin/stream_omp
make OPT="-O3 -march=x86-64-v3"   # if login and compute nodes have different CPUs
make BIN=bin_vanda         # build into another directory (the PBS scripts do this)
```

## Run

```bash
# serial reference, 128^3 grid, relative residual 1e-8
./bin/cg_serial -n 128

# 8 MPI ranks x 1 thread, 3D blocks, overlapped halo, classic CG
OMP_NUM_THREADS=1 mpirun -np 8 ./bin/cg_mpi -n 256 -d 3 -c overlap -a cg

# 2 ranks x 4 threads, pipelined CG, pinned
mpirun -np 2 --map-by ppr:2:node:PE=4 --bind-to core \
       -x OMP_NUM_THREADS=4 -x OMP_PLACES=cores -x OMP_PROC_BIND=close \
       ./bin/cg_mpi -n 256 -a pipecg

# benchmark mode: exactly 100 iterations (tol 0), print only the CSV line
mpirun -np 4 ./bin/cg_mpi -n 256 -t 0 -m 100 -q

# weak scaling: ~128^3 points per rank
mpirun -np 8 ./bin/cg_mpi -L 128 -t 0 -m 50
```

All options (`./bin/cg_mpi -h`):

```
-D DIM      problem dimension: 2 (N^2 grid, 5-point) | 3 (N^3 grid, 7-point) [3]
-n N        global interior grid N^DIM (strong scaling)          [128]
-L n        ~n^3 points per rank: global N = round(n*P^(1/3)) (weak scaling)
-a ALG      cg | cg1 | pipecg                                    [cg]
-c COMM     blocking | overlap                                   [overlap]
-d D        decomposition dimensionality 1 | 2 | 3 (at most DIM)  [DIM]
-t TOL      relative residual tolerance (0 = run maxit iterations) [1e-8]
-m MAXIT    maximum iterations                                   [50N+100]
-k CHUNKS   interior chunks between MPI progress polls           [4]
-o FILE     write solution (raw float64, C order) with MPI-IO
-r FILE     write residual history (recursive and true) to FILE
-e EVERY    history every EVERY iterations                       [1]
-q          quiet: only print the CSV line
-H          print CSV header and exit
```

Every run prints one line starting with `CSV,` (columns: `./bin/cg_mpi -H`): configuration,
iterations, recursive and true residual, max/L2 error against the exact solution, time split into
stencil / halo / reduction / vector work (max over ranks), halo bytes and messages per iteration.

> Running as root inside a container with Open MPI needs
> `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.

## Validate

```bash
make test                          # quick: ~30 s -> results/validation_quick/
bash scripts/validate.sh full      # 180 variant runs + 30 edge cases + grid study (~3 min on 2 cores)
cat results/validation/summary.txt
```

Checks: every parallel configuration reproduces the serial solution point-wise
(≤ 1e-10 relative for cg/cg1, ≤ 1e-8 for pipecg; observed 4e-15 and 2e-12), same iteration count,
true residual ≤ 1.5·tol, and the discretisation error decreases as O(h²) (observed order 2.000)
in serial and in parallel. Ranks may exceed the number of cores here (Open MPI `--oversubscribe`
is added automatically).

## Benchmarks on Vanda (NUS HPC, PBS Pro)

1. Clone the repository on Vanda (`git clone https://github.com/davidzuo023/dsa5208-project3.git`).
   The PBS scripts build the code on the compute node, so nothing needs to be compiled on the
   login node.
2. The free queue (`auto_free`) stops at 160 cores per user; 256 cores need a project
   (`hpc project`, e.g. `personal-<nusnet>`). Jobs with a project are submitted without a queue
   name; the scripts that set `#PBS -q auto_free` are submitted with that line removed:

   ```bash
   sed '/#PBS -q/d' pbs/vanda256.pbs   | qsub -P <project>   # 8 x 32 cores, ~1 h   -> results/vanda256/
   qsub -P <project> pbs/vanda_node.pbs                       # 32 cores, ~15 min    -> results/vanda_node/
   sed '/#PBS -q/d' pbs/vanda_sizes.pbs | qsub -P <project>   # 1 core, ~40 min      -> results/vanda_sizes/
   sed '/#PBS -q/d' pbs/vanda_gpu.pbs   | qsub -P <project>   # 1 A40, ~15 min       -> results/vanda_gpu/
   sed '/#PBS -q/d' pbs/vanda_gpu.pbs   | qsub -P <project> -v ONLY=validate,MACHINE=vanda_gpu_val
   ```
3. `make report` rebuilds every figure, table and number in the text from these directories.

Notes from running on Vanda:

* CPU jobs load `gcc/13.1.0 openmpi/gcc/64/4.1.8` and set
  `OMPI_MCA_btl_vader_single_copy_mechanism=none` (the nodes forbid the CMA single-copy path).
* Multi-node runs start the remote ranks through PBS (`pbs_tmrsh`, picked automatically by
  `scripts/run_suite.sh`), which keeps them inside the job's CPU set. Before the experiments the
  suite tests a launch on all cores and falls back to one node if it fails; `env.txt` records the
  remote-start method and the cores and CPU model of every node.
* The GPU job tries `nvhpc-hpcx-cuda12/26.1`, `nvhpc-hpcx-2.20-cuda12/26.1`, `nvhpc/24.5` and
  `nvhpc/26.1` in turn and keeps the first one whose build runs on the GPU (`mpicc` wraps `nvc`,
  flags `-mp=gpu -gpu=cc86`); `OMP_TARGET_OFFLOAD=MANDATORY` makes a run fail instead of silently
  falling back to the host. The same script works in an interactive GPU session
  (`qsub -I -l select=1:ncpus=36:ngpus=1 -l walltime=01:00:00`, then `bash scripts/gpu_job.sh`).
  Without a GPU compiler the code builds and runs on the host:
  `make gpu GPUFLAGS="-O2 -fopenmp -foffload=disable"`.

Knobs of `scripts/run_suite.sh` (environment variables): `MACHINE`, `NS`, `NW`, `ITERS`, `REPS`,
`PLIST`, `VARIANT_PMIN`, `EXPERIMENTS`, `MPI_FLAVOR`, `MPIRUN`, `BIN`, `APPEND` (see the header of
the script). Every run starts a new `results/$MACHINE/bench.csv` and keeps the previous one as
`bench.csv.bak.<date>`; set `APPEND=1` to add to the existing file instead (e.g. to re-run one
experiment with `EXPERIMENTS=weak`).

**What is timed.** `t_iter` is the time per iteration of the steady-state CG loop. Work vectors
are allocated and first-touched before the clock starts, and the one-off setup SpMV of `cg1` and
`pipecg` is counted in the time to solution (`t_solve`) but not in `t_iter`, so the three
variants are compared iteration against iteration.

## Results included

All measurements in the report:

* `results/vanda256/` — 8 nodes × 32 cores (one socket of 2 × Intel Xeon Platinum 8452Y per node),
  Open MPI 4.1.8, GCC 13.1.0: strong scaling (512³ from 1 to 256 processes, 768³ from 32), weak
  scaling (192³ per process), decomposition / halo / algorithm variants, process × thread splits
  and complete solves. `bench.csv` has every repetition; `make report` writes the best-of-3 table
  `bench_best.csv`.
* `results/vanda_node/` — STREAM triad, serial baseline and the quick validation suite on one
  32-core chunk of the same CPU type.
* `results/vanda_sizes/` — serial problem-size study, N = 64 … 512.
* `results/vanda_gpu/`, `results/vanda_gpu_val/` — one NVIDIA A40: size sweep and complete solves;
  validation against the serial solution with 1, 2, 4 and 8 ranks sharing the GPU is in
  `vanda_gpu_val/` (all PASS). The two FAIL lines at P = 4 in `vanda_gpu/validation.txt` are runs
  that never started (an Open MPI core-binding error, since fixed by `--bind-to none`).
* `results/vanda2d/` — 2D problem on 8 × 32 cores: strong scaling of 8192² from 1 to 256 processes,
  slab / blocking-halo / CG1 variants, weak scaling with 2048² per process, complete solves.
* `results/vanda_sizes2d/` — serial 2D problem-size study, N = 256 … 8192.
* `results/vanda_gpu2d/` — one A40, 2D problem: validation, size sweep 1024² … 16384², solve.
* `results/validation/` — the full correctness suite for 3D and 2D (`*2d.csv`);
  `results/commvol.csv` — halo volumes.

## Use of AI tools

We used Claude (Anthropic) as a coding and writing assistant. Under our direction it wrote most of
the implementation, the test and benchmark scripts and a first draft of the report. We chose the
problem and the parallelisation questions, obtained access to Vanda and ran all jobs there, checked
the results against the validation suite, and reviewed and edited the report. All numbers in the
report come from our runs.
