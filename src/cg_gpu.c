/*
 * cg_gpu.c -- MPI + OpenMP target offload Conjugate Gradient solver (one GPU per MPI rank).
 *
 * Same problem, decomposition and CSV output as cg_mpi.c. All vectors live in GPU memory for
 * the whole solve; every kernel (stencil, fused vector updates, inner products) is an
 * `omp target teams distribute parallel for` loop. The halo is packed on the GPU, staged
 * through host buffers (`omp target update`), exchanged with MPI_Sendrecv and unpacked on
 * the GPU. Reductions: the local part is an OpenMP reduction on the GPU, the global part an
 * MPI_Allreduce on the host. Algorithms: classic CG (-a cg) and Chronopoulos-Gear (-a cg1).
 * Problems: 3D 7-point (default) and 2D 5-point (-D 2, stored as N x 1 x N like cg_mpi.c).
 *
 * Build (NVIDIA HPC SDK):  mpicc -O3 -mp=gpu -gpu=cc86 -o cg_gpu cg_gpu.c -lm
 * Host fallback (testing): mpicc -O2 -fopenmp -o cg_gpu cg_gpu.c -lm
 *
 * Usage: mpirun -np P ./cg_gpu [-D 2|3] [-n N | -L nloc] [-a cg|cg1] [-t tol] [-m maxit] [-o dump] [-q] [-H]
 */
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "common.h"

typedef struct {
    int N, Nj, dim, dims[3], coords[3], n[3], off[3], nbr[3][2];
    MPI_Comm cart;
    int rank, size, device, ndev;
    size_t sI, sJ, ntot, fsz[3];
    double *sbuf[3][2], *rbuf[3][2];
    double *g[3], *gd[3], h;
    double t_spmv, t_halo, t_reduce, t_vec;
    long long halo_bytes;
    int halo_msgs;
} Ctx;

static void die(Ctx *c, const char *msg)
{
    if (!c || c->rank == 0) fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* device-resident vector: host shadow + device copy, zeroed on the device */
static double *dvec(Ctx *c)
{
    const size_t n = c->ntot;
    double *v = malloc(n * sizeof(double));
    if (!v) die(c, "out of memory");
#pragma omp target enter data map(alloc : v[0:n])
#pragma omp target teams distribute parallel for map(alloc : v[0:n])
    for (size_t q = 0; q < n; q++) v[q] = 0.0;
    return v;
}

static void dfree(Ctx *c, double *v)
{
    const size_t n = c->ntot;
#pragma omp target exit data map(delete : v[0:n])
    free(v);
}

/* ------------------------------------------------------------------ halo exchange */
static void pack(Ctx *c, const double *v, int d, int s)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot, fs = c->fsz[d];
    double *buf = c->sbuf[d][s];
    if (d == 0) {
        const int i = s ? n0 : 1;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) buf[(size_t)(j - 1) * n2 + (k - 1)] = v[i * sI + j * sJ + k];
    } else if (d == 1) {
        const int j = s ? n1 : 1;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int i = 1; i <= n0; i++)
            for (int k = 1; k <= n2; k++) buf[(size_t)(i - 1) * n2 + (k - 1)] = v[i * sI + j * sJ + k];
    } else {
        const int k = s ? n2 : 1;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int i = 1; i <= n0; i++)
            for (int j = 1; j <= n1; j++) buf[(size_t)(i - 1) * n1 + (j - 1)] = v[i * sI + j * sJ + k];
    }
#pragma omp target update from(buf[0:fs])
}

static void unpack(Ctx *c, double *v, int d, int s)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot, fs = c->fsz[d];
    double *buf = c->rbuf[d][s];
#pragma omp target update to(buf[0:fs])
    if (d == 0) {
        const int i = s ? n0 + 1 : 0;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) v[i * sI + j * sJ + k] = buf[(size_t)(j - 1) * n2 + (k - 1)];
    } else if (d == 1) {
        const int j = s ? n1 + 1 : 0;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int i = 1; i <= n0; i++)
            for (int k = 1; k <= n2; k++) v[i * sI + j * sJ + k] = buf[(size_t)(i - 1) * n2 + (k - 1)];
    } else {
        const int k = s ? n2 + 1 : 0;
#pragma omp target teams distribute parallel for collapse(2) map(alloc : v[0:nt], buf[0:fs])
        for (int i = 1; i <= n0; i++)
            for (int j = 1; j <= n1; j++) v[i * sI + j * sJ + k] = buf[(size_t)(i - 1) * n1 + (j - 1)];
    }
}

static void halo(Ctx *c, double *v)
{
    for (int d = 0; d < 3; d++)
        for (int s = 0; s < 2; s++)
            if (c->nbr[d][s] != MPI_PROC_NULL) pack(c, v, d, s);
    for (int d = 0; d < 3; d++) {
        const int cnt = (int)c->fsz[d];
        MPI_Sendrecv(c->sbuf[d][1], cnt, MPI_DOUBLE, c->nbr[d][1], 2 * d + 1, c->rbuf[d][0], cnt, MPI_DOUBLE,
                     c->nbr[d][0], 2 * d + 1, c->cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(c->sbuf[d][0], cnt, MPI_DOUBLE, c->nbr[d][0], 2 * d, c->rbuf[d][1], cnt, MPI_DOUBLE,
                     c->nbr[d][1], 2 * d, c->cart, MPI_STATUS_IGNORE);
    }
    for (int d = 0; d < 3; d++)
        for (int s = 0; s < 2; s++)
            if (c->nbr[d][s] != MPI_PROC_NULL) unpack(c, v, d, s);
}

/* ------------------------------------------------------------------ kernels */
/* y = A x (after halo exchange of x); returns local x.y */
static double spmv(Ctx *c, double *x, double *y)
{
    double t = MPI_Wtime();
    halo(c, x);
    c->t_halo += MPI_Wtime() - t;
    t = MPI_Wtime();
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
    double dot = 0.0;
    if (c->dim == 2) { /* 5-point stencil in the (i,k) plane, j = 1 only */
#pragma omp target teams distribute parallel for collapse(2) reduction(+ : dot) map(alloc : x[0:nt], y[0:nt])
        for (int i = 1; i <= n0; i++)
            for (int k = 1; k <= n2; k++) {
                const size_t m = i * sI + sJ + k;
                const double v = 4.0 * x[m] - x[m - sI] - x[m + sI] - x[m - 1] - x[m + 1];
                y[m] = v;
                dot += x[m] * v;
            }
        c->t_spmv += MPI_Wtime() - t;
        return dot;
    }
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : dot) map(alloc : x[0:nt], y[0:nt])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const size_t m = i * sI + j * sJ + k;
                const double v = 6.0 * x[m] - x[m - sI] - x[m + sI] - x[m - sJ] - x[m + sJ] - x[m - 1] - x[m + 1];
                y[m] = v;
                dot += x[m] * v;
            }
    c->t_spmv += MPI_Wtime() - t;
    return dot;
}

static double upd_xr(Ctx *c, double *x, double *r, double *p, double *Ap, double a)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
    double s = 0.0;
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : s) \
    map(alloc : x[0:nt], r[0:nt], p[0:nt], Ap[0:nt])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const size_t m = i * sI + j * sJ + k;
                x[m] += a * p[m];
                const double rk = r[m] - a * Ap[m];
                r[m] = rk;
                s += rk * rk;
            }
    return s;
}

static void upd_p(Ctx *c, double *p, double *r, double b)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
#pragma omp target teams distribute parallel for collapse(3) map(alloc : p[0:nt], r[0:nt])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const size_t m = i * sI + j * sJ + k;
                p[m] = r[m] + b * p[m];
            }
}

/* Chronopoulos-Gear: p = r + b p ; s = w + b s ; x += a p ; r -= a s ; returns r.r */
static double upd_cg1(Ctx *c, double *x, double *r, double *p, double *sv, double *w, double a, double bt)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
    double s = 0.0;
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : s) \
    map(alloc : x[0:nt], r[0:nt], p[0:nt], sv[0:nt], w[0:nt])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const size_t m = i * sI + j * sJ + k;
                const double pk = r[m] + bt * p[m], sk = w[m] + bt * sv[m];
                p[m] = pk;
                sv[m] = sk;
                x[m] += a * pk;
                const double rk = r[m] - a * sk;
                r[m] = rk;
                s += rk * rk;
            }
    return s;
}

static void copy_vec(Ctx *c, double *dst, double *src)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
#pragma omp target teams distribute parallel for collapse(3) map(alloc : dst[0:nt], src[0:nt])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) dst[i * sI + j * sJ + k] = src[i * sI + j * sJ + k];
}

/* r = b = h^2 f from the separable 1D tables (copied to the device for this kernel) ; returns b.b */
static double set_rhs(Ctx *c, double *r)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
    const double h2 = c->h * c->h;
    const double *gx = c->g[0], *gy = c->g[1], *gz = c->g[2], *dx = c->gd[0], *dy = c->gd[1], *dz = c->gd[2];
    double s = 0.0;
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : s) map(alloc : r[0:nt]) \
    map(to : gx[0:n0 + 2], gy[0:n1 + 2], gz[0:n2 + 2], dx[0:n0 + 2], dy[0:n1 + 2], dz[0:n2 + 2])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const double b = -h2 * (dx[i] * gy[j] * gz[k] + gx[i] * dy[j] * gz[k] + gx[i] * gy[j] * dz[k]);
                r[i * sI + j * sJ + k] = b;
                s += b * b;
            }
    return s;
}

/* true residual and error vs the exact solution, on the device */
static void verify(Ctx *c, double *x, double *Ax, double bnorm, double *tr, double *em, double *el)
{
    spmv(c, x, Ax);
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    const size_t sI = c->sI, sJ = c->sJ, nt = c->ntot;
    const double h2 = c->h * c->h;
    const double *gx = c->g[0], *gy = c->g[1], *gz = c->g[2], *dx = c->gd[0], *dy = c->gd[1], *dz = c->gd[2];
    /* three single-variable reductions: multi-variable reductions need 16-byte atomics that
     * some offload toolchains (GCC nvptx) do not provide; this runs once per solve */
    double s1 = 0.0, s2 = 0.0, mx = 0.0;
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : s1) map(alloc : Ax[0:nt]) \
    map(to : gx[0:n0 + 2], gy[0:n1 + 2], gz[0:n2 + 2], dx[0:n0 + 2], dy[0:n1 + 2], dz[0:n2 + 2])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const double b = -h2 * (dx[i] * gy[j] * gz[k] + gx[i] * dy[j] * gz[k] + gx[i] * gy[j] * dz[k]);
                const double d = b - Ax[i * sI + j * sJ + k];
                s1 += d * d;
            }
#pragma omp target teams distribute parallel for collapse(3) reduction(+ : s2) map(alloc : x[0:nt]) \
    map(to : gx[0:n0 + 2], gy[0:n1 + 2], gz[0:n2 + 2])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const double e = x[i * sI + j * sJ + k] - gx[i] * gy[j] * gz[k];
                s2 += e * e;
            }
#pragma omp target teams distribute parallel for collapse(3) reduction(max : mx) map(alloc : x[0:nt]) \
    map(to : gx[0:n0 + 2], gy[0:n1 + 2], gz[0:n2 + 2])
    for (int i = 1; i <= n0; i++)
        for (int j = 1; j <= n1; j++)
            for (int k = 1; k <= n2; k++) {
                const double e = fabs(x[i * sI + j * sJ + k] - gx[i] * gy[j] * gz[k]);
                mx = e > mx ? e : mx;
            }
    double sums[2] = {s1, s2};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, c->cart);
    *tr = sqrt(sums[0]) / bnorm;
    *em = mx;
    *el = sqrt(sums[1] * pow(c->h, c->dim));
}

static void allreduce(Ctx *c, double *buf, int n)
{
    const double t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, buf, n, MPI_DOUBLE, MPI_SUM, c->cart);
    c->t_reduce += MPI_Wtime() - t;
}

static void dump_solution(Ctx *c, double *x, const char *fname)
{
    const size_t nt = c->ntot;
#pragma omp target update from(x[0:nt])
    MPI_Datatype ft, mt;
    int gs[3] = {c->N, c->Nj, c->N}, ls[3] = {c->n[0], c->n[1], c->n[2]}, st[3] = {c->off[0], c->off[1], c->off[2]};
    int ms[3] = {c->n[0] + 2, c->n[1] + 2, c->n[2] + 2}, mst[3] = {1, 1, 1};
    MPI_Type_create_subarray(3, gs, ls, st, MPI_ORDER_C, MPI_DOUBLE, &ft);
    MPI_Type_create_subarray(3, ms, ls, mst, MPI_ORDER_C, MPI_DOUBLE, &mt);
    MPI_Type_commit(&ft);
    MPI_Type_commit(&mt);
    if (c->rank == 0) MPI_File_delete(fname, MPI_INFO_NULL);
    MPI_Barrier(c->cart);
    MPI_File fh;
    if (MPI_File_open(c->cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        die(c, "cannot open dump file");
    MPI_File_set_view(fh, 0, MPI_DOUBLE, ft, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, x, 1, mt, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&ft);
    MPI_Type_free(&mt);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Ctx C;
    memset(&C, 0, sizeof C);
    Ctx *c = &C;
    MPI_Comm_rank(MPI_COMM_WORLD, &c->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c->size);
    int N = 128, L = 0, quiet = 0, opt, cg1 = 0;
    c->dim = 3;
    double tol = 1e-8;
    long maxit = -1;
    const char *dump = NULL;
    while ((opt = getopt(argc, argv, "D:n:L:a:t:m:o:qH")) != -1) {
        switch (opt) {
        case 'D': c->dim = atoi(optarg); break;
        case 'n': N = atoi(optarg); break;
        case 'L': L = atoi(optarg); break;
        case 'a': cg1 = !strcmp(optarg, "cg1"); break;
        case 't': tol = atof(optarg); break;
        case 'm': maxit = atol(optarg); break;
        case 'o': dump = optarg; break;
        case 'q': quiet = 1; break;
        case 'H':
            if (c->rank == 0) printf("%s\n", CSV_HEADER);
            MPI_Finalize();
            return 0;
        default:
            if (c->rank == 0) fprintf(stderr, "usage: %s [-D 2|3] [-n N|-L n] [-a cg|cg1] [-t tol] [-m maxit] [-o f] [-q]\n", argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    /* one GPU per rank: ranks on the same node take consecutive devices */
    MPI_Comm node;
    int lrank;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &lrank);
    MPI_Comm_free(&node);
    c->ndev = omp_get_num_devices();
    c->device = c->ndev > 0 ? lrank % c->ndev : -1;
    if (c->ndev > 0) omp_set_default_device(c->device);

    MPI_Barrier(MPI_COMM_WORLD);
    const double ts0 = MPI_Wtime();
    if (c->dim != 2 && c->dim != 3) die(c, "-D must be 2 or 3");
    int dims[3] = {0, 0, 0}, periods[3] = {0, 0, 0};
    if (c->dim == 2) dims[1] = 1;
    MPI_Dims_create(c->size, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &c->cart);
    MPI_Comm_rank(c->cart, &c->rank);
    MPI_Cart_coords(c->cart, c->rank, 3, c->coords);
    if (L > 0) N = (int)lround(L * (c->dim == 3 ? cbrt((double)c->size) : sqrt((double)c->size)));
    c->N = N;
    c->Nj = c->dim == 3 ? N : 1;
    if (maxit < 0) maxit = 50L * N + 100;
    c->h = 1.0 / (N + 1);
    for (int d = 0; d < 3; d++) {
        c->dims[d] = dims[d];
        const int Nd = d == 1 ? c->Nj : N;
        const int q = Nd / dims[d], rem = Nd % dims[d];
        c->n[d] = q + (c->coords[d] < rem);
        c->off[d] = c->coords[d] * q + (c->coords[d] < rem ? c->coords[d] : rem);
        MPI_Cart_shift(c->cart, d, 1, &c->nbr[d][0], &c->nbr[d][1]);
        if (c->n[d] < 1) die(c, "too many ranks for this grid");
    }
    c->sJ = (size_t)c->n[2] + 2;
    c->sI = c->sJ * ((size_t)c->n[1] + 2);
    c->ntot = c->sI * ((size_t)c->n[0] + 2);
    c->fsz[0] = (size_t)c->n[1] * c->n[2];
    c->fsz[1] = (size_t)c->n[0] * c->n[2];
    c->fsz[2] = (size_t)c->n[0] * c->n[1];
    for (int d = 0; d < 3; d++)
        for (int s = 0; s < 2; s++) {
            const size_t fs = c->fsz[d];
            double *sb = malloc(fs * sizeof(double)), *rb = malloc(fs * sizeof(double));
            c->sbuf[d][s] = sb;
            c->rbuf[d][s] = rb;
#pragma omp target enter data map(alloc : sb[0:fs], rb[0:fs])
            if (c->nbr[d][s] != MPI_PROC_NULL) {
                c->halo_bytes += (long long)fs * 8;
                c->halo_msgs++;
            }
        }
    for (int d = 0; d < 3; d++) {
        c->g[d] = malloc((size_t)(c->n[d] + 2) * sizeof(double));
        c->gd[d] = malloc((size_t)(c->n[d] + 2) * sizeof(double));
        for (int l = 0; l <= c->n[d] + 1; l++) {
            const double t = (c->off[d] + l) * c->h;
            c->g[d][l] = (c->dim == 2 && d == 1) ? 1.0 : g_fun(t);   /* 2D: u = g(x) g(z) */
            c->gd[d][l] = (c->dim == 2 && d == 1) ? 0.0 : g_dd(t);
        }
    }

    double *x = dvec(c), *r = dvec(c), *p = dvec(c), *q = dvec(c), *s = cg1 ? dvec(c) : NULL;
    double bb = set_rhs(c, r);
    MPI_Allreduce(MPI_IN_PLACE, &bb, 1, MPI_DOUBLE, MPI_SUM, c->cart);
    const double bnorm = sqrt(bb);
    /* warm-up, untimed: launch every loop kernel (and the halo path) once so that one-off
     * kernel loading is not charged to the solve. x = 0 and the step sizes are 0, so no
     * vector changes value: q = A*0 = 0, x += 0, r -= 0, p = r (re-set by the solver anyway) */
    (void)spmv(c, x, q);
    (void)upd_xr(c, x, r, p, q, 0.0);
    upd_p(c, q, q, 0.0);
    if (cg1) (void)upd_cg1(c, x, r, p, s, q, 0.0, 0.0);
    c->t_spmv = c->t_halo = c->t_reduce = c->t_vec = 0.0;
    MPI_Barrier(c->cart);
    const double t_setup = MPI_Wtime() - ts0;

    /* ---------------- solve ---------------- */
    long it = 0;
    double rr = bb, t0, t_loop0;
    MPI_Barrier(c->cart);
    t0 = MPI_Wtime();
    if (!cg1) {
        copy_vec(c, p, r);
        t_loop0 = MPI_Wtime();
        while (it < maxit && sqrt(rr) > tol * bnorm) {
            double pAp = spmv(c, p, q); /* q = A p */
            allreduce(c, &pAp, 1);
            const double alpha = rr / pAp;
            double t = MPI_Wtime();
            double rrn = upd_xr(c, x, r, p, q, alpha);
            c->t_vec += MPI_Wtime() - t;
            allreduce(c, &rrn, 1);
            const double beta = rrn / rr;
            rr = rrn;
            t = MPI_Wtime();
            upd_p(c, p, r, beta);
            c->t_vec += MPI_Wtime() - t;
            it++;
        }
    } else { /* w = q, s = s */
        double buf[2];
        buf[1] = spmv(c, r, q);
        allreduce(c, &buf[1], 1);
        double gamma = rr, alpha = gamma / buf[1], beta = 0.0;
        c->t_spmv = c->t_halo = c->t_reduce = c->t_vec = 0.0;
        t_loop0 = MPI_Wtime();
        while (it < maxit && sqrt(gamma) > tol * bnorm) {
            double t = MPI_Wtime();
            buf[0] = upd_cg1(c, x, r, p, s, q, alpha, beta);
            c->t_vec += MPI_Wtime() - t;
            buf[1] = spmv(c, r, q);
            allreduce(c, buf, 2);
            beta = buf[0] / gamma;
            alpha = buf[0] / (buf[1] - beta * buf[0] / alpha);
            gamma = buf[0];
            it++;
        }
        rr = gamma;
    }
    MPI_Barrier(c->cart);
    const double t_end = MPI_Wtime(), t_solve = t_end - t0;
    double t_loop = t_end - t_loop0;
    MPI_Allreduce(MPI_IN_PLACE, &t_loop, 1, MPI_DOUBLE, MPI_MAX, c->cart);
    double tm[4] = {c->t_spmv, c->t_halo, c->t_reduce, c->t_vec};
    MPI_Allreduce(MPI_IN_PLACE, tm, 4, MPI_DOUBLE, MPI_MAX, c->cart);
    const int converged = sqrt(rr) <= tol * bnorm;

    double tr, em, el;
    verify(c, x, q, bnorm, &tr, &em, &el);
    if (dump) dump_solution(c, x, dump);

    long long hb[2] = {c->halo_bytes, c->halo_bytes}, nl = (long long)c->n[0] * c->n[1] * c->n[2], nlm = nl;
    int msgs = c->halo_msgs;
    MPI_Allreduce(MPI_IN_PLACE, &hb[0], 1, MPI_LONG_LONG, MPI_MAX, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &hb[1], 1, MPI_LONG_LONG, MPI_SUM, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &nlm, 1, MPI_LONG_LONG, MPI_MAX, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &msgs, 1, MPI_INT, MPI_MAX, c->cart);
    if (c->rank == 0) {
        const double npts = (double)N * c->Nj * N;
        if (!quiet)
            fprintf(stderr, "[cg_gpu] problem=%dD alg=%s P=%d devices/node=%d N=%d iters=%ld conv=%d relres=%.3e true_relres=%.3e "
                            "err_max=%.3e t_solve=%.3fs t_iter=%.3fms (spmv %.3f halo %.3f reduce %.3f vec %.3f)\n",
                    c->dim, cg1 ? "cg1" : "cg", c->size, c->ndev, N, it, converged, sqrt(rr) / bnorm, tr, em, t_solve,
                    it ? 1e3 * t_loop / it : 0.0, tm[0], tm[1], tm[2], tm[3]);
        printf("CSV,gpu,%s,staged,%dD,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%.1f,%ld,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
               "%.6e,%.6e,%.6e,%.6e,%lld,%lld,%d,%.3f\n",
               cg1 ? "cg1" : "cg", c->dim, c->size, c->ndev > 0 ? 1 : 0, dims[0], dims[1], dims[2], N, c->Nj, N, nlm,
               npts / c->size, it, converged, t_setup, t_solve, it ? t_loop / it : 0.0, tm[0], tm[1], tm[2], tm[3],
               sqrt(rr) / bnorm, tr, em, el, hb[0], hb[1], msgs, t_loop > 0 ? npts * it / t_loop / 1e6 : 0.0);
        fflush(stdout);
    }
    dfree(c, x); dfree(c, r); dfree(c, p); dfree(c, q);
    if (s) dfree(c, s);
    MPI_Comm_free(&c->cart);
    MPI_Finalize();
    return 0;
}
