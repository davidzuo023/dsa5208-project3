#!/usr/bin/env bash
# Build and benchmark the GPU code on a Vanda GPU node (batch: pbs/vanda_gpu.pbs;
# interactive: qsub -I -q auto_free -l select=1:ncpus=36:ngpus=1 -l walltime=01:00:00,
# then  cd <project dir> && bash scripts/gpu_job.sh).
# Tries the NVIDIA HPC SDK modules in order and keeps the first one whose build runs on the GPU.
cd "$(dirname "$0")/.."
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export UCX_TLS=^cma      # HPC-X/UCX: same ptrace restriction as Open MPI's CMA on the CPU nodes
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# `module` is a shell function of the job shell; make sure this child shell has it too
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh \
             /usr/share/lmod/lmod/init/bash; do [[ -f $f ]] && { . "$f"; break; }; done
fi
type module >/dev/null 2>&1 || { echo "module command not available"; exit 1; }
BIN=bin_gpu; mkdir -p $BIN
ok=0; prev=""
for mod in nvhpc-hpcx-cuda12/26.1 nvhpc-hpcx-2.20-cuda12/26.1 nvhpc/24.5 nvhpc/26.1; do
    # unload only what the previous attempt loaded (a full purge could drop site module paths)
    [[ -n $prev ]] && module unload "$prev" >/dev/null 2>&1
    prev=$mod
    module load "$mod" >/dev/null 2>&1 || { echo "-- $mod: module not loadable"; continue; }
    command -v nvc >/dev/null && command -v mpicc >/dev/null || { echo "-- $mod: no nvc/mpicc"; continue; }
    mpicc --version 2>&1 | grep -qi -E "nvc|nvidia" || { echo "-- $mod: mpicc does not wrap nvc"; continue; }
    echo "== trying $mod: $(command -v mpicc)"
    nvc -O3 -o $BIN/cg_serial src/cg_serial.c -lm || continue
    make -B BIN=$BIN GPUCC=mpicc GPUFLAGS="-O3 -mp=gpu -gpu=cc86 -Minfo=mp" $BIN/cg_gpu || continue
    # smoke test: OMP_TARGET_OFFLOAD=MANDATORY refuses to fall back to the host
    if OMP_TARGET_OFFLOAD=MANDATORY mpirun --bind-to none -np 1 $BIN/cg_gpu -n 64; then ok=1; export GPU_MODULE=$mod; echo "== using $mod"; break; fi
    echo "-- $mod: GPU run failed"
done
(( ok )) || { echo "no working GPU toolchain found"; exit 1; }
MACHINE=${MACHINE:-vanda_gpu} BIN=$BIN ITERS=50 REPS=3 GPU_SIZES="128 192 256 384 512 640 768" bash scripts/run_gpu.sh
