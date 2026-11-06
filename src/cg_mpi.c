/*
 * cg_mpi.c -- hybrid MPI + OpenMP Conjugate Gradient solver for the 2D and 3D Poisson problem.
 *
 *   -D 2|3         problem dimension: 5-point stencil on an N x N grid or 7-point stencil on
 *                  N x N x N. The 2D grid is stored as N x 1 x N (i and contiguous k), so every
 *                  kernel, the halo exchange and the decomposition are shared by both problems.
 *
 * Parallelisation choices that can be switched at run time (see usage()):
 *
 *   -d 1|2|3       domain decomposition: slabs (1D), pencils (2D) or blocks (3D) on an
 *                  MPI Cartesian communicator; uneven sizes are allowed.
 *   -c blocking    halo exchange with MPI_Sendrecv, then the whole stencil sweep
 *   -c overlap     non-blocking halo exchange (Irecv/Isend) overlapped with the stencil on the
 *                  interior; MPI progress is driven by MPI_Testall between interior chunks;
 *                  the boundary shell is computed after MPI_Waitall
 *   -a cg          classic Hestenes-Stiefel CG: 2 blocking MPI_Allreduce per iteration
 *   -a cg1         Chronopoulos-Gear CG: both inner products fused into 1 MPI_Allreduce
 *   -a pipecg      Ghysels-Vanroose pipelined CG: 1 MPI_Iallreduce overlapped with the
 *                  matrix-vector product
 *   OMP_NUM_THREADS  OpenMP threads per MPI rank (MPI_THREAD_FUNNELED, all MPI calls are
 *                  made by the master thread outside parallel regions)
 *
 * Usage: mpirun -np P ./cg_mpi [-D 2|3] [-n N | -L nloc] [-a alg] [-c comm] [-d dim]
 *                              [-t tol] [-m maxit] [-k chunks] [-o dump.bin] [-q] [-H]
 */
#define _POSIX_C_SOURCE 200809L
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "common.h"

enum { ALG_CG = 0, ALG_CG1 = 1, ALG_PIPE = 2 };
enum { COMM_BLOCKING = 0, COMM_OVERLAP = 1 };
static const char *alg_name[] = {"cg", "cg1", "pipecg"};
static const char *comm_name[] = {"blocking", "overlap"};

typedef struct {
    int N[3];        /* global interior points per dimension          */
    int dims[3];     /* process grid                                  */
    int coords[3];   /* my coordinates in the process grid            */
    int n[3];        /* local interior points per dimension           */
    int off[3];      /* global index (0-based) of my first point      */
    int nbr[3][2];   /* neighbours [dim][0=low,1=high] (or PROC_NULL) */
    MPI_Comm cart;
    int rank, size, nthreads;
    int dim;         /* problem dimension: 3 (7-point) or 2 (5-point, N[1] = 1)    */
    size_t sI, sJ, ntot; /* strides: idx = i*sI + j*sJ + k, k contiguous */
    /* halo buffers */
    size_t fsz[3];
    double *sbuf[3][2], *rbuf[3][2];
    MPI_Request req[12];
    int nreq;
    int comm_mode, nchunks;
    /* local 1D tables of the separable manufactured solution */
    double *g[3], *gd[3];
    double h;
    /* optional residual history (validation only, not timed) */
    int hist_every;
    FILE *hist;
    double *scratch, bnorm;
    /* instrumentation: phase timers cover the iteration loop only (t_loop0 = its start) */
    double t_spmv, t_halo, t_reduce, t_vec, t_loop0;
    long long halo_bytes; /* bytes this rank sends per halo exchange */
    int halo_msgs;        /* messages this rank sends per halo exchange */
} Ctx;

#define IDX(c, i, j, k) ((size_t)(i) * (c)->sI + (size_t)(j) * (c)->sJ + (size_t)(k))

static void die(Ctx *c, const char *msg)
{
    if (c == NULL || c->rank == 0) fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* ------------------------------------------------------------------------------------ */
/* memory: 64-byte aligned, first-touch initialised by the same threads that compute     */
/* ------------------------------------------------------------------------------------ */
static double *vec_alloc(Ctx *c)
{
    size_t bytes = c->ntot * sizeof(double);
    bytes = (bytes + 63) / 64 * 64;
    double *v = aligned_alloc(64, bytes);
    if (!v) die(c, "out of memory");
    const int ni = c->n[0] + 2;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ni; i++) memset(v + (size_t)i * c->sI, 0, c->sI * sizeof(double));
    return v;
}

/* ------------------------------------------------------------------------------------ */
/* halo exchange                                                                         */
/* ------------------------------------------------------------------------------------ */
/* Face (d, side) of the interior: side 0 -> first interior layer, side 1 -> last.
 * Ghost (d, side): side 0 -> layer 0, side 1 -> layer n+1.                             */
static void pack_all(Ctx *c, const double *restrict v)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
#pragma omp parallel
    {
        for (int s = 0; s < 2; s++) {
            if (c->nbr[0][s] != MPI_PROC_NULL) { /* i-face: rows of contiguous k */
                const int i = s ? n0 : 1;
                double *buf = c->sbuf[0][s];
#pragma omp for nowait schedule(static)
                for (int j = 1; j <= n1; j++)
                    memcpy(buf + (size_t)(j - 1) * n2, v + IDX(c, i, j, 1), (size_t)n2 * sizeof(double));
            }
            if (c->nbr[1][s] != MPI_PROC_NULL) { /* j-face: rows of contiguous k */
                const int j = s ? n1 : 1;
                double *buf = c->sbuf[1][s];
#pragma omp for nowait schedule(static)
                for (int i = 1; i <= n0; i++)
                    memcpy(buf + (size_t)(i - 1) * n2, v + IDX(c, i, j, 1), (size_t)n2 * sizeof(double));
            }
            if (c->nbr[2][s] != MPI_PROC_NULL) { /* k-face: fully strided gather */
                const int k = s ? n2 : 1;
                double *buf = c->sbuf[2][s];
#pragma omp for nowait schedule(static)
                for (int i = 1; i <= n0; i++)
                    for (int j = 1; j <= n1; j++) buf[(size_t)(i - 1) * n1 + (j - 1)] = v[IDX(c, i, j, k)];
            }
        }
    }
}

static void unpack_all(Ctx *c, double *restrict v)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
#pragma omp parallel
    {
        for (int s = 0; s < 2; s++) {
            if (c->nbr[0][s] != MPI_PROC_NULL) {
                const int i = s ? n0 + 1 : 0;
                const double *buf = c->rbuf[0][s];
#pragma omp for nowait schedule(static)
                for (int j = 1; j <= n1; j++)
                    memcpy(v + IDX(c, i, j, 1), buf + (size_t)(j - 1) * n2, (size_t)n2 * sizeof(double));
            }
            if (c->nbr[1][s] != MPI_PROC_NULL) {
                const int j = s ? n1 + 1 : 0;
                const double *buf = c->rbuf[1][s];
#pragma omp for nowait schedule(static)
                for (int i = 1; i <= n0; i++)
                    memcpy(v + IDX(c, i, j, 1), buf + (size_t)(i - 1) * n2, (size_t)n2 * sizeof(double));
            }
            if (c->nbr[2][s] != MPI_PROC_NULL) {
                const int k = s ? n2 + 1 : 0;
                const double *buf = c->rbuf[2][s];
#pragma omp for nowait schedule(static)
                for (int i = 1; i <= n0; i++)
                    for (int j = 1; j <= n1; j++) v[IDX(c, i, j, k)] = buf[(size_t)(i - 1) * n1 + (j - 1)];
            }
        }
    }
}

/* tags: data travelling towards the high neighbour uses tag 2d+1, towards the low one 2d */
static void halo_blocking(Ctx *c, double *v)
{
    pack_all(c, v);
    for (int d = 0; d < 3; d++) {
        const int cnt = (int)c->fsz[d];
        MPI_Sendrecv(c->sbuf[d][1], cnt, MPI_DOUBLE, c->nbr[d][1], 2 * d + 1,
                     c->rbuf[d][0], cnt, MPI_DOUBLE, c->nbr[d][0], 2 * d + 1, c->cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(c->sbuf[d][0], cnt, MPI_DOUBLE, c->nbr[d][0], 2 * d,
                     c->rbuf[d][1], cnt, MPI_DOUBLE, c->nbr[d][1], 2 * d, c->cart, MPI_STATUS_IGNORE);
    }
    unpack_all(c, v);
}

static void halo_start(Ctx *c, double *v)
{
    c->nreq = 0;
    for (int d = 0; d < 3; d++) {
        const int cnt = (int)c->fsz[d];
        if (c->nbr[d][0] != MPI_PROC_NULL)
            MPI_Irecv(c->rbuf[d][0], cnt, MPI_DOUBLE, c->nbr[d][0], 2 * d + 1, c->cart, &c->req[c->nreq++]);
        if (c->nbr[d][1] != MPI_PROC_NULL)
            MPI_Irecv(c->rbuf[d][1], cnt, MPI_DOUBLE, c->nbr[d][1], 2 * d, c->cart, &c->req[c->nreq++]);
    }
    pack_all(c, v);
    for (int d = 0; d < 3; d++) {
        const int cnt = (int)c->fsz[d];
        if (c->nbr[d][0] != MPI_PROC_NULL)
            MPI_Isend(c->sbuf[d][0], cnt, MPI_DOUBLE, c->nbr[d][0], 2 * d, c->cart, &c->req[c->nreq++]);
        if (c->nbr[d][1] != MPI_PROC_NULL)
            MPI_Isend(c->sbuf[d][1], cnt, MPI_DOUBLE, c->nbr[d][1], 2 * d + 1, c->cart, &c->req[c->nreq++]);
    }
}

static void halo_finish(Ctx *c, double *v)
{
    MPI_Waitall(c->nreq, c->req, MPI_STATUSES_IGNORE);
    c->nreq = 0;
    unpack_all(c, v);
}

/* drive MPI progress for outstanding halo and (optionally) reduction requests */
static void poll(Ctx *c, int halo_active, MPI_Request *extra)
{
    int flag;
    if (halo_active && c->nreq > 0) MPI_Testall(c->nreq, c->req, &flag, MPI_STATUSES_IGNORE);
    if (extra && *extra != MPI_REQUEST_NULL) MPI_Test(extra, &flag, MPI_STATUS_IGNORE);
}

/* ------------------------------------------------------------------------------------ */
/* stencil kernels                                                                       */
/* ------------------------------------------------------------------------------------ */
/* y = A x on the box [i0,i1]x[j0,j1]x[k0,k1]; returns the local part of x.y */
static double stencil_box(Ctx *c, const double *restrict x, double *restrict y,
                          int i0, int i1, int j0, int j1, int k0, int k1)
{
    if (i1 < i0 || j1 < j0 || k1 < k0) return 0.0;
    const size_t sI = c->sI, sJ = c->sJ;
    double dot = 0.0;
    if (c->dim == 2) { /* 5-point stencil in the (i,k) plane; the j "direction" has one layer */
#pragma omp parallel for schedule(static) reduction(+ : dot)
        for (int i = i0; i <= i1; i++) {
            const size_t b = (size_t)i * sI + sJ;
            const double *restrict xc = x + b;
            double *restrict yc = y + b;
            double dl = 0.0;
#pragma omp simd reduction(+ : dl)
            for (int k = k0; k <= k1; k++) {
                const double v = 4.0 * xc[k] - xc[k - sI] - xc[k + sI] - xc[k - 1] - xc[k + 1];
                yc[k] = v;
                dl += xc[k] * v;
            }
            dot += dl;
        }
        return dot;
    }
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : dot)
    for (int i = i0; i <= i1; i++)
        for (int j = j0; j <= j1; j++) {
            const size_t b = (size_t)i * sI + (size_t)j * sJ;
            const double *restrict xc = x + b;
            double *restrict yc = y + b;
            double dl = 0.0;
#pragma omp simd reduction(+ : dl)
            for (int k = k0; k <= k1; k++) {
                const double v = 6.0 * xc[k] - xc[k - sI] - xc[k + sI] - xc[k - sJ] - xc[k + sJ]
                                 - xc[k - 1] - xc[k + 1];
                yc[k] = v;
                dl += xc[k] * v;
            }
            dot += dl;
        }
    return dot;
}

/* stencil on an i-range split into chunks, polling MPI between chunks */
static double stencil_chunked(Ctx *c, const double *x, double *y, int i0, int i1, int j0, int j1,
                              int k0, int k1, int halo_active, MPI_Request *extra)
{
    if (i1 < i0) return 0.0;
    const int len = i1 - i0 + 1;
    int nch = c->nchunks < 1 ? 1 : c->nchunks;
    if (nch > len) nch = len;
    if (!halo_active && !extra) nch = 1;
    double dot = 0.0;
    for (int q = 0; q < nch; q++) {
        const int a = i0 + (int)((long)len * q / nch);
        const int b = i0 + (int)((long)len * (q + 1) / nch) - 1;
        dot += stencil_box(c, x, y, a, b, j0, j1, k0, k1);
        poll(c, halo_active, extra);
    }
    return dot;
}

/* y = A x (x's ghost layers are updated); returns local x.y.
 * extra: an outstanding non-blocking reduction to progress while computing (pipelined CG). */
static double spmv(Ctx *c, double *x, double *y, MPI_Request *extra)
{
    const int n0 = c->n[0], n1 = c->n[1], n2 = c->n[2];
    double dot = 0.0, t;
    if (c->comm_mode == COMM_BLOCKING) {
        t = MPI_Wtime();
        halo_blocking(c, x);
        c->t_halo += MPI_Wtime() - t;
        t = MPI_Wtime();
        dot = stencil_chunked(c, x, y, 1, n0, 1, n1, 1, n2, 0, extra);
        c->t_spmv += MPI_Wtime() - t;
    } else {
        t = MPI_Wtime();
        halo_start(c, x);
        c->t_halo += MPI_Wtime() - t;

        /* Only layers adjacent to a real neighbour depend on incoming halo data; layers on
         * the physical boundary (ghost = Dirichlet 0) belong to the "interior" sweep.     */
        int lo[3], hi[3];
        for (int d = 0; d < 3; d++) {
            lo[d] = c->nbr[d][0] != MPI_PROC_NULL ? 2 : 1;
            hi[d] = c->nbr[d][1] != MPI_PROC_NULL ? c->n[d] - 1 : c->n[d];
        }
        t = MPI_Wtime(); /* interior: needs no ghost values */
        dot = stencil_chunked(c, x, y, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], 1, extra);
        c->t_spmv += MPI_Wtime() - t;

        t = MPI_Wtime();
        halo_finish(c, x);
        c->t_halo += MPI_Wtime() - t;

        /* boundary shell: a disjoint cover of the remaining points (i-planes, then j-rows
         * inside the remaining i-range, then k-columns inside the remaining i,j-range)    */
        t = MPI_Wtime();
        const int has0 = c->nbr[0][0] != MPI_PROC_NULL, has1 = c->nbr[0][1] != MPI_PROC_NULL;
        if (has0) dot += stencil_box(c, x, y, 1, 1, 1, n1, 1, n2);
        if (has1 && !(n0 == 1 && has0)) dot += stencil_box(c, x, y, n0, n0, 1, n1, 1, n2);
        const int hj0 = c->nbr[1][0] != MPI_PROC_NULL, hj1 = c->nbr[1][1] != MPI_PROC_NULL;
        if (hj0) dot += stencil_box(c, x, y, lo[0], hi[0], 1, 1, 1, n2);
        if (hj1 && !(n1 == 1 && hj0)) dot += stencil_box(c, x, y, lo[0], hi[0], n1, n1, 1, n2);
        const int hk0 = c->nbr[2][0] != MPI_PROC_NULL, hk1 = c->nbr[2][1] != MPI_PROC_NULL;
        if (hk0) dot += stencil_box(c, x, y, lo[0], hi[0], lo[1], hi[1], 1, 1);
        if (hk1 && !(n2 == 1 && hk0)) dot += stencil_box(c, x, y, lo[0], hi[0], lo[1], hi[1], n2, n2);
        c->t_spmv += MPI_Wtime() - t;
        poll(c, 0, extra);
    }
    return dot;
}

/* ------------------------------------------------------------------------------------ */
/* fused vector kernels (one sweep over memory each)                                     */
/* ------------------------------------------------------------------------------------ */
#define FOR_INTERIOR(c)                                                                  \
    _Pragma("omp parallel for collapse(2) schedule(static) reduction(+ : s1, s2)")      \
    for (int i = 1; i <= (c)->n[0]; i++)                                                 \
        for (int j = 1; j <= (c)->n[1]; j++)

/* classic CG: x += a p ; r -= a Ap ; return r.r */
static double upd_cg_xr(Ctx *c, double *restrict x, double *restrict r, const double *restrict p,
                        const double *restrict Ap, double a)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        double l = 0.0;
#pragma omp simd reduction(+ : l)
        for (int k = 1; k <= n2; k++) {
            x[b + k] += a * p[b + k];
            const double rk = r[b + k] - a * Ap[b + k];
            r[b + k] = rk;
            l += rk * rk;
        }
        s1 += l;
    }
    (void)s2;
    return s1;
}

/* classic CG: p = r + beta p */
static void upd_cg_p(Ctx *c, double *restrict p, const double *restrict r, double beta)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
#pragma omp simd
        for (int k = 1; k <= n2; k++) p[b + k] = r[b + k] + beta * p[b + k];
    }
    (void)s1; (void)s2;
}

/* Chronopoulos-Gear: p = r + b p ; s = w + b s ; x += a p ; r -= a s ; return r.r */
static double upd_cg1(Ctx *c, double *restrict x, double *restrict r, double *restrict p,
                      double *restrict s, const double *restrict w, double a, double bt)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        double l = 0.0;
#pragma omp simd reduction(+ : l)
        for (int k = 1; k <= n2; k++) {
            const size_t q = b + k;
            const double pk = r[q] + bt * p[q];
            const double sk = w[q] + bt * s[q];
            p[q] = pk;
            s[q] = sk;
            x[q] += a * pk;
            const double rk = r[q] - a * sk;
            r[q] = rk;
            l += rk * rk;
        }
        s1 += l;
    }
    (void)s2;
    return s1;
}

/* Ghysels-Vanroose: z = q + b z ; s = w + b s ; p = r + b p ;
 *                   x += a p ; r -= a s ; w -= a z ;  returns r.r and w.r              */
static void upd_pipe(Ctx *c, double *restrict x, double *restrict r, double *restrict w,
                     double *restrict p, double *restrict s, double *restrict z,
                     const double *restrict q, double a, double bt, double *gam, double *del)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        double l1 = 0.0, l2 = 0.0;
#pragma omp simd reduction(+ : l1, l2)
        for (int k = 1; k <= n2; k++) {
            const size_t m = b + k;
            const double zk = q[m] + bt * z[m];
            const double sk = w[m] + bt * s[m];
            const double pk = r[m] + bt * p[m];
            z[m] = zk;
            s[m] = sk;
            p[m] = pk;
            x[m] += a * pk;
            const double rk = r[m] - a * sk;
            const double wk = w[m] - a * zk;
            r[m] = rk;
            w[m] = wk;
            l1 += rk * rk;
            l2 += wk * rk;
        }
        s1 += l1;
        s2 += l2;
    }
    *gam = s1;
    *del = s2;
}

static double dot_local(Ctx *c, const double *restrict u, const double *restrict v)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        double l = 0.0;
#pragma omp simd reduction(+ : l)
        for (int k = 1; k <= n2; k++) l += u[b + k] * v[b + k];
        s1 += l;
    }
    (void)s2;
    return s1;
}

static void copy_vec(Ctx *c, double *restrict dst, const double *restrict src)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        memcpy(dst + b + 1, src + b + 1, (size_t)n2 * sizeof(double));
    }
    (void)s1; (void)s2;
}

/* r = b = h^2 f evaluated from separable 1D tables (cheap, no exp per point) */
static void set_rhs(Ctx *c, double *restrict r)
{
    double s1 = 0.0, s2 = 0.0;
    const int n2 = c->n[2];
    const double h2 = c->h * c->h;
    const double *gx = c->g[0], *gy = c->g[1], *gz = c->g[2];
    const double *dx = c->gd[0], *dy = c->gd[1], *dz = c->gd[2];
    FOR_INTERIOR(c)
    {
        const size_t b = IDX(c, i, j, 0);
        for (int k = 1; k <= n2; k++)
            r[b + k] = -h2 * (dx[i] * gy[j] * gz[k] + gx[i] * dy[j] * gz[k] + gx[i] * gy[j] * dz[k]);
    }
    (void)s1; (void)s2;
}

static void allreduce_sum(Ctx *c, double *buf, int cnt)
{
    const double t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, buf, cnt, MPI_DOUBLE, MPI_SUM, c->cart);
    c->t_reduce += MPI_Wtime() - t;
}

/* ------------------------------------------------------------------------------------ */
/* verification: true residual ||b - A x|| / ||b|| and the error against the exact      */
/* solution (collective; Ax is scratch)                                                  */
/* ------------------------------------------------------------------------------------ */
static void verify(Ctx *c, double *x, double *Ax, double *true_relres, double *err_max, double *err_l2)
{
    spmv(c, x, Ax, NULL);
    const double h2 = c->h * c->h;
    const double *gx = c->g[0], *gy = c->g[1], *gz = c->g[2];
    const double *dx = c->gd[0], *dy = c->gd[1], *dz = c->gd[2];
    double s1 = 0.0, s2 = 0.0, emax = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+ : s1, s2) reduction(max : emax)
    for (int i = 1; i <= c->n[0]; i++)
        for (int j = 1; j <= c->n[1]; j++)
            for (int k = 1; k <= c->n[2]; k++) {
                const size_t m = IDX(c, i, j, k);
                const double b = -h2 * (dx[i] * gy[j] * gz[k] + gx[i] * dy[j] * gz[k] + gx[i] * gy[j] * dz[k]);
                const double d = b - Ax[m];
                s1 += d * d;
                const double e = fabs(x[m] - gx[i] * gy[j] * gz[k]);
                s2 += e * e;
                if (e > emax) emax = e;
            }
    double sums[2] = {s1, s2};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &emax, 1, MPI_DOUBLE, MPI_MAX, c->cart);
    *true_relres = sqrt(sums[0]) / c->bnorm;
    *err_max = emax;
    *err_l2 = sqrt(sums[1] * pow(c->h, c->dim));
}

/* residual history: recursive residual (free) and true residual (one extra, untimed spmv) */
static void hist_point(Ctx *c, long it, double rec_relres, double *x, int force)
{
    if (c->hist_every <= 0 || (!force && it % c->hist_every != 0)) return;
    const double saved[4] = {c->t_spmv, c->t_halo, c->t_reduce, c->t_vec};
    double tr, em, el;
    verify(c, x, c->scratch, &tr, &em, &el);
    c->t_spmv = saved[0]; c->t_halo = saved[1]; c->t_reduce = saved[2]; c->t_vec = saved[3];
    if (c->hist) fprintf(c->hist, "%ld,%.6e,%.6e\n", it, rec_relres, tr);
}

/* start of the steady-state iteration loop: setup work done before it (the initial SpMV
 * of CG1 and pipelined CG) counts in the time to solution but not in the time per
 * iteration, so the variants are compared iteration against iteration                  */
static void loop_start(Ctx *c)
{
    c->t_spmv = c->t_halo = c->t_reduce = c->t_vec = 0.0;
    c->t_loop0 = MPI_Wtime();
}

/* ------------------------------------------------------------------------------------ */
/* solvers. All start from x0 = 0 so r0 = b. Return the number of iterations.            */
/* Work vectors are allocated and first-touched by the caller, outside the timed region. */
/* *rr_out receives the final recursively-updated ||r||^2.                               */
/* ------------------------------------------------------------------------------------ */
static long solve_cg(Ctx *c, double *x, double *r, double **wk, double tol, double bnorm, long maxit,
                     double *rr_out)
{
    double *p = wk[0], *Ap = wk[1];
    double t = MPI_Wtime();
    copy_vec(c, p, r);
    c->t_vec += MPI_Wtime() - t;
    double rr = bnorm * bnorm;
    long it = 0;
    loop_start(c);
    while (it < maxit && sqrt(rr) > tol * bnorm) {
        hist_point(c, it, sqrt(rr) / bnorm, x, 0);
        double pAp = spmv(c, p, Ap, NULL);
        allreduce_sum(c, &pAp, 1);                  /* reduction #1 */
        const double alpha = rr / pAp;
        t = MPI_Wtime();
        double rrn = upd_cg_xr(c, x, r, p, Ap, alpha);
        c->t_vec += MPI_Wtime() - t;
        allreduce_sum(c, &rrn, 1);                  /* reduction #2 */
        const double beta = rrn / rr;
        rr = rrn;
        t = MPI_Wtime();
        upd_cg_p(c, p, r, beta);
        c->t_vec += MPI_Wtime() - t;
        it++;
    }
    hist_point(c, it, sqrt(rr) / bnorm, x, 1);
    *rr_out = rr;
    return it;
}

static long solve_cg1(Ctx *c, double *x, double *r, double **wk, double tol, double bnorm, long maxit,
                      double *rr_out)
{
    double *p = wk[0], *s = wk[1], *w = wk[2];
    double buf[2];
    buf[1] = spmv(c, r, w, NULL);           /* w0 = A r0, delta0 = (w0, r0) */
    buf[0] = bnorm * bnorm;                 /* gamma0 = (r0, r0)            */
    allreduce_sum(c, &buf[1], 1);
    double gamma = buf[0], alpha = gamma / buf[1], beta = 0.0;
    long it = 0;
    loop_start(c);
    while (it < maxit && sqrt(gamma) > tol * bnorm) {
        hist_point(c, it, sqrt(gamma) / bnorm, x, 0);
        double t = MPI_Wtime();
        buf[0] = upd_cg1(c, x, r, p, s, w, alpha, beta); /* r_{i+1}, (r,r) local */
        c->t_vec += MPI_Wtime() - t;
        buf[1] = spmv(c, r, w, NULL);                     /* w = A r, (w,r) local */
        allreduce_sum(c, buf, 2);                         /* the single reduction */
        beta = buf[0] / gamma;
        alpha = buf[0] / (buf[1] - beta * buf[0] / alpha);
        gamma = buf[0];
        it++;
    }
    hist_point(c, it, sqrt(gamma) / bnorm, x, 1);
    *rr_out = gamma;
    return it;
}

static long solve_pipe(Ctx *c, double *x, double *r, double **wk, double tol, double bnorm, long maxit,
                       double *rr_out)
{
    double *w = wk[0], *p = wk[1], *s = wk[2], *z = wk[3], *q = wk[4];
    double loc[2];
    loc[1] = spmv(c, r, w, NULL); /* w0 = A r0, local (r0,w0) */
    loc[0] = dot_local(c, r, r);  /* local (r0,r0) */
    double gamma = 0.0, gamma_old = 1.0, alpha = 1.0, beta = 0.0;
    long it = 0;
    loop_start(c);
    for (;;) {
        double buf[2] = {loc[0], loc[1]};
        if (it >= maxit) { /* iteration budget used up: no need for one more (wasted) SpMV */
            allreduce_sum(c, buf, 2);
            gamma = buf[0];
            break;
        }
        MPI_Request req;
        double t = MPI_Wtime();
        MPI_Iallreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, c->cart, &req);
        c->t_reduce += MPI_Wtime() - t;
        spmv(c, w, q, &req);        /* q = A w, overlapped with the reduction */
        t = MPI_Wtime();
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        c->t_reduce += MPI_Wtime() - t;
        gamma = buf[0];
        const double delta = buf[1];
        if (sqrt(gamma) <= tol * bnorm) break;
        hist_point(c, it, sqrt(gamma) / bnorm, x, 0);
        if (it > 0) {
            beta = gamma / gamma_old;
            alpha = gamma / (delta - beta * gamma / alpha);
        } else {
            beta = 0.0;
            alpha = gamma / delta;
        }
        t = MPI_Wtime();
        upd_pipe(c, x, r, w, p, s, z, q, alpha, beta, &loc[0], &loc[1]);
        c->t_vec += MPI_Wtime() - t;
        gamma_old = gamma;
        it++;
    }
    hist_point(c, it, sqrt(gamma) / bnorm, x, 1);
    *rr_out = gamma;
    return it;
}

/* ------------------------------------------------------------------------------------ */
/* parallel output of the solution (global natural order, raw doubles) with MPI-IO      */
/* ------------------------------------------------------------------------------------ */
static void dump_solution(Ctx *c, const double *x, const char *fname)
{
    MPI_Datatype ftype, mtype;
    int gs[3] = {c->N[0], c->N[1], c->N[2]};
    int ls[3] = {c->n[0], c->n[1], c->n[2]};
    int st[3] = {c->off[0], c->off[1], c->off[2]};
    int ms[3] = {c->n[0] + 2, c->n[1] + 2, c->n[2] + 2};
    int mst[3] = {1, 1, 1};
    MPI_Type_create_subarray(3, gs, ls, st, MPI_ORDER_C, MPI_DOUBLE, &ftype);
    MPI_Type_create_subarray(3, ms, ls, mst, MPI_ORDER_C, MPI_DOUBLE, &mtype);
    MPI_Type_commit(&ftype);
    MPI_Type_commit(&mtype);
    MPI_File fh;
    if (c->rank == 0) MPI_File_delete(fname, MPI_INFO_NULL);
    MPI_Barrier(c->cart);
    if (MPI_File_open(c->cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        die(c, "cannot open dump file");
    MPI_File_set_view(fh, 0, MPI_DOUBLE, ftype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, x, 1, mtype, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&ftype);
    MPI_Type_free(&mtype);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: mpirun -np P %s [options]\n"
            "  -D DIM      problem dimension 2 (N^2 grid, 5-point) | 3 (N^3, 7-point) [3]\n"
            "  -n N        global interior grid N^DIM (strong scaling)          [128]\n"
            "  -L n        ~n^DIM points per rank: N = round(n*P^(1/DIM)) (weak scaling)\n"
            "  -a ALG      cg | cg1 | pipecg                                    [cg]\n"
            "  -c COMM     blocking | overlap                                   [overlap]\n"
            "  -d D        decomposition dimensionality 1 | 2 | 3 (at most DIM)  [DIM]\n"
            "  -t TOL      relative residual tolerance (0 = run maxit iterations) [1e-8]\n"
            "  -m MAXIT    maximum iterations                                   [50N+100]\n"
            "  -k CHUNKS   interior chunks between MPI progress polls           [4]\n"
            "  -o FILE     write solution (raw float64, C order) with MPI-IO\n"
            "  -r FILE     write residual history (recursive and true) to FILE\n"
            "  -e EVERY    history every EVERY iterations                       [1]\n"
            "  -q          quiet: only print the CSV line\n"
            "  -H          print CSV header and exit\n",
            prog);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Ctx C;
    memset(&C, 0, sizeof C);
    Ctx *c = &C;
    MPI_Comm_rank(MPI_COMM_WORLD, &c->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c->size);
    if (provided < MPI_THREAD_FUNNELED) die(c, "MPI library does not provide MPI_THREAD_FUNNELED");

    int Nglob = 128, Lloc = 0, decomp = 0, quiet = 0, opt;
    c->dim = 3;
    double tol = 1e-8;
    long maxit = -1;
    const char *dump = NULL, *histfile = NULL;
    int hist_every = 1;
    c->comm_mode = COMM_OVERLAP;
    c->nchunks = 4;
    int alg = ALG_CG;
    while ((opt = getopt(argc, argv, "D:n:L:a:c:d:t:m:k:o:r:e:qHh")) != -1) {
        switch (opt) {
        case 'D': c->dim = atoi(optarg); break;
        case 'n': Nglob = atoi(optarg); break;
        case 'L': Lloc = atoi(optarg); break;
        case 'a':
            if (!strcmp(optarg, "cg")) alg = ALG_CG;
            else if (!strcmp(optarg, "cg1")) alg = ALG_CG1;
            else if (!strcmp(optarg, "pipecg")) alg = ALG_PIPE;
            else die(c, "unknown algorithm (cg|cg1|pipecg)");
            break;
        case 'c':
            if (!strcmp(optarg, "blocking")) c->comm_mode = COMM_BLOCKING;
            else if (!strcmp(optarg, "overlap")) c->comm_mode = COMM_OVERLAP;
            else die(c, "unknown comm mode (blocking|overlap)");
            break;
        case 'd': decomp = atoi(optarg); break;
        case 't': tol = atof(optarg); break;
        case 'm': maxit = atol(optarg); break;
        case 'k': c->nchunks = atoi(optarg); break;
        case 'o': dump = optarg; break;
        case 'r': histfile = optarg; break;
        case 'e': hist_every = atoi(optarg); break;
        case 'q': quiet = 1; break;
        case 'H':
            if (c->rank == 0) printf("%s\n", CSV_HEADER);
            MPI_Finalize();
            return 0;
        default:
            if (c->rank == 0) usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    if (c->dim != 2 && c->dim != 3) die(c, "-D must be 2 or 3");
    if (decomp == 0 || decomp > c->dim) decomp = c->dim;   /* default: as many as the problem has */
    if (decomp < 1) die(c, "-d must be 1, 2 or 3");
    c->nthreads = omp_get_max_threads();

    MPI_Barrier(MPI_COMM_WORLD);
    const double ts0 = MPI_Wtime();

    /* ---------------- process grid ---------------- */
    int dims[3] = {0, 0, 0};
    if (decomp == 1) { dims[0] = c->size; dims[1] = 1; dims[2] = 1; }
    else if (c->dim == 2) dims[1] = 1;                     /* 2D problem: split i and k   */
    else if (decomp == 2) dims[2] = 1;
    MPI_Dims_create(c->size, 3, dims);
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &c->cart);
    MPI_Comm_rank(c->cart, &c->rank);
    MPI_Cart_coords(c->cart, c->rank, 3, c->coords);
    /* weak scaling: keep ~Lloc^DIM points per rank on a square / cubic global grid */
    if (Lloc > 0) Nglob = (int)lround(Lloc * (c->dim == 3 ? cbrt((double)c->size) : sqrt((double)c->size)));
    for (int d = 0; d < 3; d++) {
        c->dims[d] = dims[d];
        c->N[d] = (c->dim == 2 && d == 1) ? 1 : Nglob;
        const int q = c->N[d] / dims[d], rem = c->N[d] % dims[d];
        c->n[d] = q + (c->coords[d] < rem ? 1 : 0);
        c->off[d] = c->coords[d] * q + (c->coords[d] < rem ? c->coords[d] : rem);
        MPI_Cart_shift(c->cart, d, 1, &c->nbr[d][0], &c->nbr[d][1]);
    }
    int nmin = c->n[0] < c->n[1] ? c->n[0] : c->n[1];
    nmin = nmin < c->n[2] ? nmin : c->n[2];
    MPI_Allreduce(MPI_IN_PLACE, &nmin, 1, MPI_INT, MPI_MIN, c->cart);
    if (nmin < 1) die(c, "too many ranks for this grid (a rank has an empty subdomain)");
    if (maxit < 0) maxit = 50L * c->N[0] + 100;
    c->h = 1.0 / (c->N[0] + 1);

    c->sJ = (size_t)c->n[2] + 2;
    c->sI = c->sJ * ((size_t)c->n[1] + 2);
    c->ntot = c->sI * ((size_t)c->n[0] + 2);
    c->fsz[0] = (size_t)c->n[1] * c->n[2];
    c->fsz[1] = (size_t)c->n[0] * c->n[2];
    c->fsz[2] = (size_t)c->n[0] * c->n[1];
    for (int d = 0; d < 3; d++)
        for (int s = 0; s < 2; s++) {
            c->sbuf[d][s] = malloc(c->fsz[d] * sizeof(double));
            c->rbuf[d][s] = malloc(c->fsz[d] * sizeof(double));
            if (c->nbr[d][s] != MPI_PROC_NULL) {
                c->halo_bytes += (long long)c->fsz[d] * (long long)sizeof(double);
                c->halo_msgs++;
            }
        }
    for (int d = 0; d < 3; d++) {
        c->g[d] = malloc((size_t)(c->n[d] + 2) * sizeof(double));
        c->gd[d] = malloc((size_t)(c->n[d] + 2) * sizeof(double));
        for (int l = 0; l <= c->n[d] + 1; l++) {
            const double t = (c->off[d] + l) * c->h; /* local l=1 is global index off+1 */
            /* 2D: the j factor of u = g(x)g(y)g(z) is 1, so u = g(x)g(z), f = -(g''g + g g'') */
            c->g[d][l] = (c->dim == 2 && d == 1) ? 1.0 : g_fun(t);
            c->gd[d][l] = (c->dim == 2 && d == 1) ? 0.0 : g_dd(t);
        }
    }

    double *x = vec_alloc(c), *r = vec_alloc(c);
    set_rhs(c, r);
    double bb = dot_local(c, r, r);
    MPI_Allreduce(MPI_IN_PLACE, &bb, 1, MPI_DOUBLE, MPI_SUM, c->cart);
    const double bnorm = sqrt(bb);
    c->bnorm = bnorm;
    if (histfile) {
        c->hist_every = hist_every < 1 ? 1 : hist_every;
        c->scratch = vec_alloc(c);
        if (c->rank == 0) {
            c->hist = fopen(histfile, "w");
            if (!c->hist) die(c, "cannot open history file");
            fprintf(c->hist, "iter,relres_recursive,relres_true\n");
        }
    }
    MPI_Barrier(c->cart);
    const double t_setup = MPI_Wtime() - ts0;

    /* ---------------- solve ---------------- */
    /* work vectors: allocated (zeroed, first-touched) before the clock starts, so that
     * the variants are compared on their iterations, not on page faults                 */
    const int nwork = alg == ALG_CG ? 2 : alg == ALG_CG1 ? 3 : 5;
    double *wk[5] = {NULL, NULL, NULL, NULL, NULL};
    for (int q = 0; q < nwork; q++) wk[q] = vec_alloc(c);
    double rr = 0.0;
    long it = 0;
    MPI_Barrier(c->cart);
    const double t0 = MPI_Wtime();
    switch (alg) {
    case ALG_CG: it = solve_cg(c, x, r, wk, tol, bnorm, maxit, &rr); break;
    case ALG_CG1: it = solve_cg1(c, x, r, wk, tol, bnorm, maxit, &rr); break;
    default: it = solve_pipe(c, x, r, wk, tol, bnorm, maxit, &rr); break;
    }
    MPI_Barrier(c->cart);
    const double t_end = MPI_Wtime();
    const double t_solve = t_end - t0;   /* time to solution, including setup SpMV */
    double t_loop = t_end - c->t_loop0;  /* steady-state iterations only            */
    MPI_Allreduce(MPI_IN_PLACE, &t_loop, 1, MPI_DOUBLE, MPI_MAX, c->cart);
    const int converged = sqrt(rr) <= tol * bnorm;
    double tm[4] = {c->t_spmv, c->t_halo, c->t_reduce, c->t_vec};
    MPI_Allreduce(MPI_IN_PLACE, tm, 4, MPI_DOUBLE, MPI_MAX, c->cart);

    /* ---------------- verification (not timed) ---------------- */
    double *Ax = vec_alloc(c);
    double true_relres, emax, err_l2;
    verify(c, x, Ax, &true_relres, &emax, &err_l2);

    long long hb[2] = {c->halo_bytes, c->halo_bytes};
    long long nloc = (long long)c->n[0] * c->n[1] * c->n[2];
    long long nl[2] = {nloc, nloc};
    int msgs = c->halo_msgs;
    MPI_Allreduce(MPI_IN_PLACE, &hb[0], 1, MPI_LONG_LONG, MPI_MAX, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &hb[1], 1, MPI_LONG_LONG, MPI_SUM, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &nl[0], 1, MPI_LONG_LONG, MPI_MAX, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &nl[1], 1, MPI_LONG_LONG, MPI_SUM, c->cart);
    MPI_Allreduce(MPI_IN_PLACE, &msgs, 1, MPI_INT, MPI_MAX, c->cart);

    if (dump) dump_solution(c, x, dump);

    if (c->rank == 0) {
        const double npts = (double)c->N[0] * c->N[1] * c->N[2];
        if (!quiet)
            fprintf(stderr,
                    "[cg_mpi] problem=%dD alg=%s comm=%s decomp=%dD P=%d T=%d grid=%dx%dx%d N=%d iters=%ld conv=%d "
                    "relres=%.3e true_relres=%.3e err_max=%.3e t_solve=%.3fs t_iter=%.3fms "
                    "(spmv %.3f halo %.3f reduce %.3f vec %.3f)\n",
                    c->dim, alg_name[alg], comm_name[c->comm_mode], decomp, c->size, c->nthreads, c->dims[0],
                    c->dims[1], c->dims[2], c->N[0], it, converged, sqrt(rr) / bnorm, true_relres, emax,
                    t_solve, it ? 1e3 * t_loop / it : 0.0, tm[0], tm[1], tm[2], tm[3]);
        printf("CSV,mpi,%s,%s,%dD,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%.1f,%ld,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
               "%.6e,%.6e,%.6e,%.6e,%lld,%lld,%d,%.3f\n",
               alg_name[alg], comm_name[c->comm_mode], decomp, c->size, c->nthreads, c->dims[0], c->dims[1],
               c->dims[2], c->N[0], c->N[1], c->N[2], nl[0], (double)nl[1] / c->size, it, converged, t_setup,
               t_solve, it ? t_loop / it : 0.0, tm[0], tm[1], tm[2], tm[3], sqrt(rr) / bnorm, true_relres,
               emax, err_l2, hb[0], hb[1], msgs, t_loop > 0 ? npts * it / t_loop / 1e6 : 0.0);
        fflush(stdout);
    }

    if (c->hist) fclose(c->hist);
    free(c->scratch);
    for (int q = 0; q < nwork; q++) free(wk[q]);
    free(x); free(r); free(Ax);
    for (int d = 0; d < 3; d++) {
        free(c->g[d]); free(c->gd[d]);
        for (int s = 0; s < 2; s++) { free(c->sbuf[d][s]); free(c->rbuf[d][s]); }
    }
    MPI_Comm_free(&c->cart);
    MPI_Finalize();
    return 0;
}
