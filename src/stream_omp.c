/*
 * stream_omp.c -- minimal STREAM-triad style memory-bandwidth probe (OpenMP).
 *
 * The CG iteration is memory-bandwidth bound, so the attainable node bandwidth is the
 * natural roofline for the solver.  Prints one CSV line per thread count:
 *     stream,threads,triad_GBs
 *
 * Usage: ./stream_omp [Mdoubles_per_array=40] [max_threads=omp_get_max_threads()]
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    const size_t n = (size_t)(argc > 1 ? atof(argv[1]) : 40.0) * 1000000;
    const int tmax = argc > 2 ? atoi(argv[2]) : omp_get_max_threads();
    printf("stream,threads,triad_GBs\n");
    /* thread counts 1,2,4,8,... and always tmax itself */
    for (int t = 1;; t = (2 * t < tmax) ? 2 * t : tmax) {
        omp_set_num_threads(t);
        /* fresh arrays for every thread count, first-touched by these t threads, so that
         * on multi-socket nodes the pages are spread like the solver's (not all on the
         * socket of whichever thread touched them first in an earlier round)            */
        double *a = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double)), *c = malloc(n * sizeof(double));
        if (!a || !b || !c) { fprintf(stderr, "out of memory\n"); return 1; }
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
        double best = 1e30;
        for (int rep = 0; rep < 8; rep++) {
            const double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
            const double dt = omp_get_wtime() - t0;
            if (dt < best) best = dt;
        }
        /* 2 reads + 1 write (+1 write-allocate read not counted, as in STREAM) */
        printf("stream,%d,%.2f\n", t, 3.0 * n * sizeof(double) / best / 1e9);
        fflush(stdout);
        free(a); free(b); free(c);
        if (t == tmax) break;
    }
    return 0;
}
