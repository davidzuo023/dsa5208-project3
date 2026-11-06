/*
 * cg_serial.c -- serial reference implementation.
 *
 * Unpreconditioned Conjugate Gradient (Hestenes-Stiefel) for the 7-point 3D or the 5-point 2D
 * Poisson problem described in common.h.  Matrix-free: A is applied as a stencil, never stored.
 * The 2D grid (-D 2) is stored as N x 1 x N, exactly like the parallel code.
 *
 * This code is the baseline for all speedups.  It uses the same kernel fusions as the
 * parallel code (p.Ap is computed inside the stencil sweep, r.r inside the x/r update) and the
 * same `omp simd` vectorisation of the inner loops (compiled with -fopenmp-simd: SIMD only,
 * no threads), so that the speedup measures parallelism, not a better serial kernel.
 *
 * Usage: ./cg_serial [-D 2|3] [-n N] [-t tol] [-m maxit] [-o dump.bin] [-q] [-H]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "common.h"

static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int N;           /* interior points per direction */
static int NJ;          /* points in the j direction: N (3D) or 1 (2D) */
static int DIM = 3;
static size_t sI, sJ;   /* strides of i and j (k is contiguous) */
#define IDX(i, j, k) ((size_t)(i) * sI + (size_t)(j) * sJ + (size_t)(k))

/* y = A x on the interior, returns x.y */
static double stencil(const double *restrict x, double *restrict y)
{
    double dot = 0.0;
    if (DIM == 2) {
        for (int i = 1; i <= N; i++) {
            const size_t b = IDX(i, 1, 0);
#pragma omp simd reduction(+ : dot)
            for (int k = 1; k <= N; k++) {
                const size_t c = b + k;
                const double v = 4.0 * x[c] - x[c - sI] - x[c + sI] - x[c - 1] - x[c + 1];
                y[c] = v;
                dot += x[c] * v;
            }
        }
        return dot;
    }
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= NJ; j++) {
            const size_t b = IDX(i, j, 0);
#pragma omp simd reduction(+ : dot)
            for (int k = 1; k <= N; k++) {
                const size_t c = b + k;
                const double v = 6.0 * x[c] - x[c - sI] - x[c + sI] - x[c - sJ] - x[c + sJ]
                                 - x[c - 1] - x[c + 1];
                y[c] = v;
                dot += x[c] * v;
            }
        }
    return dot;
}

int main(int argc, char **argv)
{
    N = 64;
    double tol = 1e-8;
    long maxit = -1;
    const char *dump = NULL;
    int quiet = 0, opt;
    while ((opt = getopt(argc, argv, "D:n:t:m:o:qH")) != -1) {
        switch (opt) {
        case 'D': DIM = atoi(optarg); break;
        case 'n': N = atoi(optarg); break;
        case 't': tol = atof(optarg); break;
        case 'm': maxit = atol(optarg); break;
        case 'o': dump = optarg; break;
        case 'q': quiet = 1; break;
        case 'H': printf("%s\n", CSV_HEADER); return 0;
        default:
            fprintf(stderr, "usage: %s [-D 2|3] [-n N] [-t tol] [-m maxit] [-o dump.bin] [-q] [-H]\n", argv[0]);
            return 1;
        }
    }
    if (N < 1) { fprintf(stderr, "N must be >= 1\n"); return 1; }
    if (DIM != 2 && DIM != 3) { fprintf(stderr, "-D must be 2 or 3\n"); return 1; }
    NJ = DIM == 3 ? N : 1;
    if (maxit < 0) maxit = 50L * N + 100;

    double ts0 = wtime();
    sJ = (size_t)N + 2;
    sI = sJ * ((size_t)NJ + 2);
    const size_t ntot = sI * ((size_t)N + 2);
    double *x  = calloc(ntot, sizeof(double));
    double *r  = calloc(ntot, sizeof(double));
    double *p  = calloc(ntot, sizeof(double));
    double *Ap = calloc(ntot, sizeof(double));
    double *g  = malloc((size_t)(N + 2) * sizeof(double));
    double *gd = malloc((size_t)(N + 2) * sizeof(double));
    if (!x || !r || !p || !Ap || !g || !gd) { fprintf(stderr, "out of memory\n"); return 1; }

    const double h = 1.0 / (N + 1);
    for (int i = 0; i <= N + 1; i++) { g[i] = g_fun(i * h); gd[i] = g_dd(i * h); }
    /* j factor of u = g(x)g(y)g(z): g(y) in 3D, 1 in 2D (then u = g(x)g(z)) */
    double *gj = malloc((size_t)(NJ + 2) * sizeof(double)), *gdj = malloc((size_t)(NJ + 2) * sizeof(double));
    if (!gj || !gdj) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int j = 0; j <= NJ + 1; j++) { gj[j] = DIM == 3 ? g[j] : 1.0; gdj[j] = DIM == 3 ? gd[j] : 0.0; }

    /* x0 = 0  =>  r0 = b = h^2 f,  p0 = r0 */
    double rr = 0.0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= NJ; j++)
            for (int k = 1; k <= N; k++) {
                const size_t c = IDX(i, j, k);
                const double b = -h * h * (gd[i] * gj[j] * g[k] + g[i] * gdj[j] * g[k] + g[i] * gj[j] * gd[k]);
                r[c] = b;
                p[c] = b;
                rr += b * b;
            }
    const double bnorm = sqrt(rr);
    /* calloc'ed pages are only mapped on first write: touch x and Ap now, outside the clock */
    memset(x, 0, ntot * sizeof(double));
    memset(Ap, 0, ntot * sizeof(double));
    double t_setup = wtime() - ts0;

    /* ---------------- CG iterations ---------------- */
    double t_spmv = 0.0, t_vec = 0.0;
    long it = 0;
    double t0 = wtime();
    while (it < maxit && sqrt(rr) > tol * bnorm) {
        double ta = wtime();
        const double pAp = stencil(p, Ap);
        double tb = wtime();
        t_spmv += tb - ta;

        const double alpha = rr / pAp;
        double rr_new = 0.0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= NJ; j++) {
                const size_t b = IDX(i, j, 0);
#pragma omp simd reduction(+ : rr_new)
                for (int k = 1; k <= N; k++) {
                    const size_t c = b + k;
                    x[c] += alpha * p[c];
                    r[c] -= alpha * Ap[c];
                    rr_new += r[c] * r[c];
                }
            }
        const double beta = rr_new / rr;
        rr = rr_new;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= NJ; j++) {
                const size_t b = IDX(i, j, 0);
#pragma omp simd
                for (int k = 1; k <= N; k++) p[b + k] = r[b + k] + beta * p[b + k];
            }
        t_vec += wtime() - tb;
        it++;
    }
    const double t_solve = wtime() - t0;
    const int converged = sqrt(rr) <= tol * bnorm;

    /* ---------------- verification (not timed) ---------------- */
    stencil(x, Ap); /* Ap := A x */
    double res2 = 0.0, err2 = 0.0, errmax = 0.0;
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= NJ; j++)
            for (int k = 1; k <= N; k++) {
                const size_t c = IDX(i, j, k);
                const double b = -h * h * (gd[i] * gj[j] * g[k] + g[i] * gdj[j] * g[k] + g[i] * gj[j] * gd[k]);
                const double d = b - Ap[c];
                res2 += d * d;
                const double e = fabs(x[c] - g[i] * gj[j] * g[k]);
                err2 += e * e;
                if (e > errmax) errmax = e;
            }
    const double true_relres = sqrt(res2) / bnorm;
    const double err_l2 = sqrt(err2 * pow(h, DIM));

    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) { perror(dump); return 1; }
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= NJ; j++) fwrite(&x[IDX(i, j, 1)], sizeof(double), (size_t)N, f);
        fclose(f);
    }

    const double npts = (double)N * NJ * N;
    if (!quiet) {
        fprintf(stderr, "[serial] %dD N=%d iters=%ld converged=%d relres=%.3e true_relres=%.3e "
                        "err_max=%.3e err_l2=%.3e t_solve=%.3fs t_iter=%.3fms\n",
                DIM, N, it, converged, sqrt(rr) / bnorm, true_relres, errmax, err_l2, t_solve,
                it ? 1e3 * t_solve / it : 0.0);
    }
    printf("CSV,serial,cg,none,none,1,1,1,1,1,%d,%d,%d,%.0f,%.0f,%ld,%d,%.6e,%.6e,%.6e,%.6e,0,0,%.6e,"
           "%.6e,%.6e,%.6e,%.6e,0,0,0,%.3f\n",
           N, NJ, N, npts, npts, it, converged, t_setup, t_solve, it ? t_solve / it : 0.0, t_spmv,
           t_vec, sqrt(rr) / bnorm, true_relres, errmax, err_l2,
           t_solve > 0 ? npts * it / t_solve / 1e6 : 0.0);

    free(x); free(r); free(p); free(Ap); free(g); free(gd); free(gj); free(gdj);
    return 0;
}
