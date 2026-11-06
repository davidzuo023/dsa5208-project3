# DSA5208 project: hybrid MPI+OpenMP Conjugate Gradient for the 3D Poisson equation
#
# All three programs are compiled through the MPI wrapper, so the serial baseline and the
# parallel code always use the same underlying compiler (speedups compare code, not compilers).
#
#   make            build bin/cg_serial, bin/cg_mpi, bin/stream_omp
#   make test       quick correctness check (needs mpirun)
#   make report     rebuild figures/tables from results/vanda*/ and compile report/report.pdf
#
# Override the compiler wrappers / flags if needed, e.g. on a cluster whose login node
# has a different CPU than the compute nodes:
#   make OPT="-O3 -march=x86-64-v3"
MPICC  ?= mpicc
OPT    ?= -O3 -march=native
CFLAGS ?= $(OPT) -std=c11 -Wall -Wextra -Wno-unknown-pragmas
OMPFLAG ?= -fopenmp
LDLIBS = -lm

BIN ?= bin
SRC = src

all: $(BIN)/cg_serial $(BIN)/cg_mpi $(BIN)/stream_omp

$(BIN):
	mkdir -p $(BIN)

# serial: SIMD pragmas only, no threads
$(BIN)/cg_serial: $(SRC)/cg_serial.c $(SRC)/common.h | $(BIN)
	$(MPICC) $(CFLAGS) -fopenmp-simd -o $@ $< $(LDLIBS)

$(BIN)/cg_mpi: $(SRC)/cg_mpi.c $(SRC)/common.h | $(BIN)
	$(MPICC) $(CFLAGS) $(OMPFLAG) -o $@ $< $(LDLIBS)

$(BIN)/stream_omp: $(SRC)/stream_omp.c | $(BIN)
	$(MPICC) $(CFLAGS) $(OMPFLAG) -o $@ $< $(LDLIBS)

# GPU build (MPI + OpenMP target offload). Default flags: NVIDIA HPC SDK for an A40 (sm_86).
# Host-fallback test build without a GPU compiler:  make gpu GPUFLAGS="-O2 -fopenmp -foffload=disable"
GPUCC    ?= mpicc
GPUFLAGS ?= -O3 -mp=gpu -gpu=cc86 -Minfo=mp
# (always rebuilt, so a host-only build is never mistaken for a GPU build; scripts/gpu_job.sh
#  builds the matching serial reference with `nvc -O3`)
gpu: | $(BIN)
	$(GPUCC) $(GPUFLAGS) -o $(BIN)/cg_gpu $(SRC)/cg_gpu.c $(LDLIBS)
$(BIN)/cg_gpu: $(SRC)/cg_gpu.c $(SRC)/common.h | $(BIN)
	$(GPUCC) $(GPUFLAGS) -o $@ $< $(LDLIBS)

# Word version of the report (needs pandoc and python-docx; run after `make report`)
docx:
	python3 scripts/make_docx.py

test: all
	bash scripts/validate.sh quick

report:
	python3 scripts/make_plots.py
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null && \
	             pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
	@echo "report/report.pdf rebuilt"

clean:
	rm -rf $(BIN) *.bin

.PHONY: all gpu test report docx clean
