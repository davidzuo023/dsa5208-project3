/*
 * common.h -- problem definition shared by the serial and the MPI+OpenMP solver.
 *
 * We solve the Poisson problem  -Laplace(u) = f  on (0,1)^3 (or (0,1)^2 with -D 2), u = 0 on the boundary,
 * discretised with the standard second-order 7-point finite-difference stencil on a uniform
 * grid with N interior points per direction (h = 1/(N+1)).  After multiplying by h^2 the
 * linear system is
 *
 *        A u = b,   (A u)_{ijk} = 6 u_{ijk} - u_{i+-1,j,k} - u_{i,j+-1,k} - u_{i,j,k+-1},
 *                   b_{ijk}     = h^2 f(x_i, y_j, z_k),
 *
 * where A is symmetric positive definite, so Conjugate Gradients (CG) applies.
 *
 * Manufactured solution (NOT an eigenfunction of the discrete Laplacian, so CG really has to
 * iterate and the discretisation error is a genuine O(h^2)):
 *
 *        u(x,y,z) = g(x) g(y) g(z),   g(t) = t (1-t) e^t,   g''(t) = -t (3+t) e^t
 *        f        = -(g''(x) g(y) g(z) + g(x) g''(y) g(z) + g(x) g(y) g''(z)).
 */
#ifndef CG_COMMON_H
#define CG_COMMON_H

#include <math.h>
#include <stddef.h>

static inline double g_fun(double t) { return t * (1.0 - t) * exp(t); }
static inline double g_dd(double t)  { return -t * (3.0 + t) * exp(t); }

/* One CSV line per run; both executables print exactly these columns. */
#define CSV_HEADER                                                                  \
    "code,alg,comm,decomp,P,T,px,py,pz,N0,N1,N2,nloc_max,nloc_avg,iters,converged,"  \
    "t_setup,t_solve,t_iter,t_spmv,t_halo,t_reduce,t_vec,relres,true_relres,"        \
    "err_max,err_l2,halo_bytes_iter_max,halo_bytes_iter_tot,msgs_iter_max,mlups"

#endif
