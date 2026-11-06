#!/usr/bin/env python3
"""Turn the Vanda measurements into the report's figures, LaTeX tables and number macros.

    python3 scripts/make_plots.py            (defaults below; `make report` calls this)

Inputs (all under results/):
  vanda256/     multi-node pure-MPI and hybrid runs, 8 nodes x 32 cores    (pbs/vanda256.pbs)
  vanda_node/   STREAM, serial baseline and quick validation on 32 cores     (pbs/vanda_node.pbs)
  vanda_sizes/  serial problem-size study                                    (pbs/vanda_sizes.pbs)
  vanda_gpu/    GPU size sweep and time to solution on one A40               (pbs/vanda_gpu.pbs)
  vanda_gpu_val/ GPU validation, P = 1, 2, 4, 8 ranks on one A40
  validation/   full correctness suite;  commvol.csv  halo volumes
Outputs: figures/*.pdf and report/generated/*.tex. Every number quoted in the report text comes
from report/generated/macros.tex, so re-running this script updates figures, tables and text.
"""
import argparse
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
GEN = os.path.join(ROOT, "report", "generated")
RES = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- style
# NeurIPS-style figures: the document's sans-serif face (Liberation Sans ~ Helvetica), open axes
# (no top/right spines), faint horizontal grid only, frameless legends, thin marks, left-aligned
# bold panel labels; figure titles live in the LaTeX captions, not in the figures.
# muted palette in the manner of recent NeurIPS papers (soft blue, orange, teal, purple; grey for
# references), in fixed order; adjacent slots pass the colour-vision-deficiency check of the dataviz
# palette validator, and every series also carries its own marker or line style
C = ["#3A6FB0", "#E08A3C", "#2E9C8A", "#8E5BAE", "#C44E52", "#9A9A9A", "#86B4DA", "#E5C35C"]
INK, INK2, GRID = "#1a1a1a", "#555555", "#ebebeb"
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 5, "axes.labelsize": 7.5, "legend.fontsize": 6.8, "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 1.3, "lines.markersize": 3.6, "lines.markeredgewidth": 0.8,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
})


BYTES_CPU = 96.0   # classic CG per point and iteration: 88 B of loads/stores + 8 B write-allocate
BYTES_GPU = 88.0   # GPUs write full sectors without reading them first: no write-allocate
A40_PEAK = 696.0   # GB/s, NVIDIA A40 data sheet


def suptitle(fig, text, sub=None):
    """figure titles are carried by the LaTeX captions (NeurIPS style): nothing is drawn"""
    return


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def log2_axis(ax, values, axis="x"):
    vals = sorted(set(int(v) for v in values))
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=2)
    a.set_major_locator(FixedLocator(vals))
    a.set_minor_locator(NullLocator())
    # label a tick only if it is at most 1/1.4 of the last labelled one (walking down), so
    # close values (192/224/256) do not collide; the marks themselves all stay
    labels, last = [""] * len(vals), None
    for k in range(len(vals) - 1, -1, -1):
        if last is None or last / vals[k] >= 1.4:
            labels[k] = str(vals[k])
            last = vals[k]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)
    a.set_minor_formatter(NullFormatter())


def plain_log(ax, axis="y", lo=-3, hi=5):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator([m * 10.0 ** e for e in range(lo, hi) for m in (1, 2, 5)]))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_locator(NullLocator())


def tex_escape(s):
    return str(s).replace("(R)", "").replace("(TM)", "").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def sci(x, nd=1):
    """1.2e-05 -> 1.2\\times10^{-5} (math mode)"""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, nd)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.{nd}f}\times10^{{{e}}}"


# ---------------------------------------------------------------- data
def best_of(df, keys):
    """fastest repetition per configuration (with that run's phase breakdown), plus the spread"""
    idx = df.groupby(keys)["t_iter"].idxmin()
    best = df.loc[idx].copy()
    agg = df.groupby(keys)["t_iter"].agg(spread=lambda s: (s.max() - s.min()) / s.min(), nreps="size",
                                         t_med="median").reset_index()
    best = best.merge(agg, on=keys)
    for c in ["t_spmv", "t_halo", "t_reduce", "t_vec"]:
        best[c + "_it"] = best[c] / best["iters"]
    return best.reset_index(drop=True)


KEYS = ["exp", "code", "alg", "comm", "decomp", "P", "T", "N0"]


def env_info(d):
    info = {}
    p = os.path.join(d, "env.txt")
    if not os.path.exists(p):
        return info
    txt = open(p).read()
    for key, pat in [("cpu", r"Model name:\s*(.+)"), ("mpi", r"\(Open MPI\)\s*([\w.]+)"),
                     ("gcc", r"gcc \(GCC\)\s*([\d.]+)"), ("gpu", r"(NVIDIA [^,]+), (\d+) MiB, ([\d.]+)"),
                     ("nvc", r"nvc ([\d.]+-\d+)"), ("module", r"toolchain module:\s*(\S+)"),
                     ("cps", r"Core\(s\) per socket:\s*(\d+)"), ("sockets", r"Socket\(s\):\s*(\d+)")]:
        m = re.search(pat, txt)
        if m:
            info[key] = m.groups() if key == "gpu" else m.group(1).strip()
    m = re.search(r"nodes:\s*(\d+)\s+cores/node:\s*(\d+)", txt)
    if m:
        info["nodes"], info["cpn"] = int(m.group(1)), int(m.group(2))
    info["tmrsh"] = "pbs_tmrsh" in txt
    return info


# ---------------------------------------------------------------- serial problem size
def load_sizes(d, pre, dim):
    p = os.path.join(d, "bench.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    it = s[s.exp == pre + "_iter"].groupby("N0").agg(t=("t_iter", "min"), t_med=("t_iter", "median")).reset_index()
    so = s[s.exp == pre + "_solve"].groupby("N0").agg(iters=("iters", "max"), ts=("t_solve", "min"),
                                                      err=("err_max", "min"), tr=("true_relres", "min"),
                                                      t_long=("t_iter", "min")).reset_index()
    z = it.merge(so, on="N0", how="left")
    # a 20-iteration run of a small grid lasts milliseconds and catches clock ramp-up and neighbours
    # on the shared node; the complete solve runs the same loop for hundreds of iterations.
    # Time per iteration = the faster of the two.
    z["t"] = z[["t", "t_long"]].min(axis=1)
    N = z.N0.astype(float)
    z["n"] = N ** dim
    z["nnz"] = 7 * N ** 3 - 6 * N ** 2 if dim == 3 else 5 * N ** 2 - 4 * N
    z["nspt"] = 1e9 * z.t / z.n
    z["bw"] = z.n * BYTES_CPU / z.t / 1e9
    # x, r, p, Ap with ghost layers (the 2D grid is stored as N x 1 x N with three j-planes)
    z["mib"] = 4 * (N + 2) ** 2 * (N + 2 if dim == 3 else 3) * 8 / 2 ** 20
    return z


def fig_sizes(d3, d2, macros, tables):
    import numpy as np
    z3, z2 = load_sizes(d3, "sizes", 3), load_sizes(d2, "sizes2d", 2)
    if z3 is None:
        return None, None
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.35))
    for z, lab, col, mk in [(z3, "3D, 7-point", C[0], "o"), (z2, "2D, 5-point", C[1], "s")]:
        if z is None:
            continue
        ax[0].loglog(z.n / 1e6, 1e3 * z.t, color=col, marker=mk, label=lab)
        zz = z.dropna(subset=["iters"])
        ax[1].loglog(zz.N0, zz.iters, color=col, marker=mk, label=lab)
        ax[2].loglog(zz.n / 1e6, zz.ts, color=col, marker=mk, label=lab)
    n_all = np.array(sorted(set(z3.n) | (set(z2.n) if z2 is not None else set())))
    ref = 1e3 * float(z3.t.iloc[-1]) * n_all / float(z3.n.iloc[-1])
    ax[0].loglog(n_all / 1e6, ref, color=INK2, ls="--", lw=1.0, label="$\\propto$ unknowns")
    ax[0].set_xlabel("unknowns (millions)")
    ax[0].set_ylabel("ms per iteration")
    ax[0].set_title("(a) Cost of one iteration")
    plain_log(ax[0], "y", -2, 4)
    ax[0].xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].legend(loc="upper left")
    Ns = sorted(set(z3.N0) | (set(z2.dropna(subset=["iters"]).N0) if z2 is not None else set()))
    k = float((z3.iters / z3.N0).iloc[-1])
    ax[1].loglog(Ns, [k * n for n in Ns], color=INK2, ls="--", lw=1.0, label=rf"${k:.1f}\,N$")
    ax[1].set_xscale("log", base=2)
    ax[1].xaxis.set_major_locator(FixedLocator([2 ** e for e in range(6, 14) if Ns[0] <= 2 ** e <= Ns[-1]]))
    ax[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax[1].xaxis.set_minor_locator(NullLocator())
    plain_log(ax[1], "y", 1, 6)
    ax[1].set_xlabel("grid points per direction $N$")
    ax[1].set_ylabel(r"iterations to $\varepsilon=10^{-8}$")
    ax[1].set_title("(b) Iterations")
    ax[1].legend(loc="upper left")
    ax[2].set_xlabel("unknowns (millions)")
    ax[2].set_ylabel("time to solution (s)")
    ax[2].set_title("(c) Time to solution")
    ax[2].xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[2].xaxis.set_minor_formatter(NullFormatter())
    plain_log(ax[2], "y", -2, 5)
    ax[2].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "sizes")

    def rows(z):
        return [f"{int(r.N0)} & {r.n / 1e6:.2f} & {r.nnz / 1e6:.1f} & {r.mib:.0f} & "
                f"{1e3 * r.t:.2f} & {r.nspt:.2f} & {r.bw:.1f} & "
                + (f"{int(r.iters)} & {r.ts:.1f} & ${sci(r.err, 2)}$" if r.iters == r.iters else "-- & -- & --")
                + " \\\\" for _, r in z.iterrows()]
    tables["sizes"] = rows(z3)
    tables["sizes2d"] = rows(z2) if z2 is not None else []

    def stats(z, tag):
        zz = z.dropna(subset=["iters"])
        lx = np.log(z.N0.astype(float))
        st = float(np.polyfit(lx, np.log(z.t), 1)[0])
        ss = float(np.polyfit(np.log(zz.N0.astype(float)), np.log(zz.ts), 1)[0])
        ipn = zz.iters / zz.N0
        a, e, ze = z.iloc[0], z.iloc[-1], zz.iloc[-1]
        m = {"Nmin": str(int(a.N0)), "Nmax": str(int(e.N0)), "NsSmall": fmt(a.nspt, 2), "NsLarge": fmt(e.nspt, 2),
             "NsMin": fmt(z.nspt.min(), 2), "NsMax": fmt(z.nspt.max(), 2), "BwLarge": fmt(e.bw, 1),
             "SlopeIter": fmt(st, 2), "SlopeSolve": fmt(ss, 2), "ItPerN": fmt(float(ipn.iloc[-1]), 2),
             "ItMin": str(int(zz.iloc[0].iters)), "ItMax": str(int(ze.iters)), "SolveMin": fmt(zz.iloc[0].ts, 2),
             "SolveMax": fmt(ze.ts, 0), "SolveMaxMin": fmt(ze.ts / 60, 1), "TiterMax": fmt(1e3 * e.t, 0),
             "BwMin": fmt(z.bw.min(), 0), "BwMax": fmt(z.bw.max(), 0), "ItPerNmin": fmt(ipn.min(), 2),
             "ItPerNmax": fmt(ipn.max(), 2), "MiBSmall": fmt(a.mib, 1), "GiBLarge": fmt(e.mib / 1024, 1),
             "NsolveMax": str(int(ze.N0)), "DofsMax": fmt(e.n / 1e6, 0),
             "Doubling": fmt(ze.ts / float(zz[zz.N0 == ze.N0 // 2].ts.iloc[0]), 1) if (zz.N0 == ze.N0 // 2).any() else "--"}
        macros.update({tag + k2: v for k2, v in m.items()})
    stats(z3, "Sz")
    if z2 is not None:
        stats(z2, "SzTwo")
        r2 = z2.iloc[1] if len(z2) > 1 else z2.iloc[0]
        macros.update({"SzTwoNsSecond": fmt(r2.nspt, 2), "SzTwoNsecond": str(int(r2.N0))})
    return z3, z2


# ---------------------------------------------------------------- multi-node strong scaling
def fig_strong(b, t_ser, cpn, macros, tables):
    s = b[b.exp == "strong"].sort_values("P").copy()
    s2 = b[b.exp == "strong2"].sort_values("P").copy()
    if s.empty:
        return
    s["S"] = t_ser / s.t_iter
    s["E"] = s.S / s.P
    t32 = float(s[s.P == cpn].t_iter.iloc[0])
    s["En"] = (t32 * cpn) / (s.t_iter * s.P)
    t32b = float(s2[s2.P == cpn].t_iter.iloc[0]) if not s2.empty else float("nan")
    s2["Sn"] = cpn * t32b / s2.t_iter
    s2["En"] = s2.Sn / s2.P
    Ps = s.P.values

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.65))
    ax[0].plot(Ps, Ps, color=INK2, lw=1.0, ls="--", label="ideal")
    ax[0].plot(Ps, s.S, color=C[0], marker="o", label=r"$512^3$, vs. serial code")
    if not s2.empty:
        ax[0].plot(s2.P, s2.Sn, color=C[1], marker="s", label=r"$768^3$, vs. one node $\times$ 32")
    ax[0].axvline(cpn, color=C[5], lw=0.8, ls=":")
    ax[0].text(cpn * 1.06, 1.3, "one node", fontsize=6.5, color=C[5], rotation=90, va="bottom")
    log2_axis(ax[0], Ps)
    log2_axis(ax[0], [1, 2, 4, 8, 16, 32, 64, 128, 256], axis="y")
    ax[0].set_ylim(0.8, 330)
    ax[0].set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("speedup")
    ax[0].set_title("(a) Speedup")
    ax[0].legend(loc="upper left", fontsize=6.5)
    ax[1].plot(Ps, 100 * s.E, color=C[0], marker="o", label=r"$512^3$, vs. serial code")
    sn = s[s.P >= cpn]
    ax[1].plot(sn.P, 100 * sn.En, color=C[0], marker="o", ls="--", mfc="white", label=r"$512^3$, vs. one node")
    if not s2.empty:
        ax[1].plot(s2.P, 100 * s2.En, color=C[1], marker="s", ls="--", mfc="white", label=r"$768^3$, vs. one node")
    ax[1].axhline(100, color=INK2, lw=1.0, ls="--")
    ax[1].axvline(cpn, color=C[5], lw=0.8, ls=":")
    log2_axis(ax[1], Ps)
    ax[1].set_ylim(0, 110)
    ax[1].set_xlabel("MPI processes $P$")
    ax[1].set_ylabel("parallel efficiency (%)")
    ax[1].set_title("(b) Efficiency")
    ax[1].legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    suptitle(fig, "Strong scaling from one core to eight nodes", f"pure MPI, 3D blocks, overlapped halo, "
             f"classic CG; {cpn} processes per node, {int(s.iters.iloc[0])} iterations, best of "
             f"{int(s.nreps.max())} runs")
    save(fig, "strong")

    # where the time goes: share of each phase
    fig, ax = plt.subplots(figsize=(6.6, 2.35))
    comps = [("t_spmv_it", "stencil $A\\mathbf{p}$", C[0]), ("t_vec_it", "vector updates + local dots", C[2]),
             ("t_halo_it", "halo exchange", C[1]), ("t_reduce_it", "MPI_Allreduce (incl. waiting)", C[3])]
    tot = sum(s[c] for c, _, _ in comps)
    xs = range(len(s))
    bottom = [0.0] * len(s)
    for col, lab, colr in comps:
        vals = (100 * s[col] / tot).tolist()
        ax.bar(xs, vals, bottom=bottom, color=colr, width=0.62, label=lab, edgecolor="white", linewidth=1.0)
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(int(p)) for p in Ps])
    ax.axvline(list(Ps).index(cpn) + 0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("MPI processes $P$ (grid $%d^3$); dotted line: one node" % int(s.N0.iloc[0]))
    ax.set_ylabel("share of iteration time (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout()
    suptitle(fig, "Where a CG iteration spends its time")
    save(fig, "breakdown")

    s2i = s2.set_index("P")
    rows = []
    for _, r in s.iterrows():
        P = int(r.P)
        comm = (r.t_halo_it + r.t_reduce_it) / r.t_iter * 100
        nodes = max(1, P // cpn)
        en = f"{100 * r.En:.0f}" if P >= cpn else "--"
        if P in s2i.index:
            t2, e2 = f"{1e3 * s2i.loc[P, 't_iter']:.1f}", f"{100 * s2i.loc[P, 'En']:.0f}"
        else:
            t2, e2 = "--", "--"
        rows.append(f"{P} & {nodes} & {int(r.px)}$\\times${int(r.py)}$\\times${int(r.pz)} & {1e3 * r.t_iter:.2f} & "
                    f"{r.S:.1f} & {100 * r.E:.0f} & {en} & {comm:.0f} & {r.halo_bytes_iter_max / 2 ** 10:.0f} & "
                    f"{t2} & {e2} \\\\")
    tables["strong"] = rows
    last, n1 = s.iloc[-1], s[s.P == cpn].iloc[0]
    l2 = s2.iloc[-1] if not s2.empty else None
    pm = int(last.P)
    macros.update({
        "SerialTiter": fmt(1e3 * t_ser, 0), "PoneTiter": fmt(1e3 * float(s.iloc[0].t_iter), 0),
        "PoneVsSerial": fmt(100 * abs(1 - float(s.iloc[0].t_iter) / t_ser), 1),
        "PoneFaster": "faster" if float(s.iloc[0].t_iter) < t_ser else "slower",
        "Pmax": str(pm), "NodesMax": str(pm // cpn), "SpeedupPmax": fmt(last.S, 1), "EffPmax": fmt(100 * last.E, 0),
        "TiterPmax": fmt(1e3 * last.t_iter, 2), "TiterNode": fmt(1e3 * n1.t_iter, 1),
        "SpeedupNode": fmt(n1.S, 1), "EffNode": fmt(100 * n1.E, 0), "NodeEffPmax": fmt(100 * last.En, 0),
        "NodeSpeedPmax": fmt(t32 / float(last.t_iter), 2),
        "NodeEffSixtyFour": fmt(100 * float(s[s.P == 2 * cpn].En.iloc[0]), 0),
        "TiterSixtyFour": fmt(1e3 * float(s[s.P == 2 * cpn].t_iter.iloc[0]), 1),
        "SpeedupEight": fmt(float(s[s.P == 8].S.iloc[0]), 1),
        "SpeedupFour": fmt(float(s[s.P == 4].S.iloc[0]), 2), "EffFour": fmt(100 * float(s[s.P == 4].E.iloc[0]), 0),
        "CommFracPmax": fmt((last.t_halo_it + last.t_reduce_it) / last.t_iter * 100, 0),
        "CommFracNode": fmt((n1.t_halo_it + n1.t_reduce_it) / n1.t_iter * 100, 0),
        "RedFracPmax": fmt(100 * last.t_reduce_it / last.t_iter, 0), "RedMsPmax": fmt(1e3 * last.t_reduce_it, 2),
        "HaloFracPmax": fmt(100 * last.t_halo_it / last.t_iter, 0), "HaloMsPmax": fmt(1e3 * last.t_halo_it, 2),
        "RedMsNode": fmt(1e3 * n1.t_reduce_it, 2), "HaloKiBPmax": fmt(last.halo_bytes_iter_max / 2 ** 10, 0),
        "HaloKiBNode": fmt(n1.halo_bytes_iter_max / 2 ** 10, 0),
        "VecFracSerial": fmt(100 * float(s.iloc[0].t_vec_it / s.iloc[0].t_iter), 0),
        "NS": str(int(s.N0.iloc[0])), "NSdofs": fmt(int(s.N0.iloc[0]) ** 3 / 1e6, 0), "Iters": str(int(s.iters.iloc[0])),
        "Reps": str(int(b.nreps.mode().iloc[0])), "MedSpread": fmt(100 * b[b.exp != "solve"].spread.median(), 1),
        "MaxSpread": fmt(100 * b[b.exp != "solve"].spread.max(), 0),
    })
    if l2 is not None:
        macros.update({"NStwo": str(int(s2.N0.iloc[0])), "NStwoDofs": fmt(int(s2.N0.iloc[0]) ** 3 / 1e6, 0),
                       "NodeEffTwoPmax": fmt(100 * l2.En, 0), "NodeSpeedTwoPmax": fmt(l2.Sn / cpn, 2),
                       "TiterTwoNode": fmt(1e3 * t32b, 1), "TiterTwoPmax": fmt(1e3 * l2.t_iter, 1)})
    return s


def fig_bandwidth(s, stream, cpn, macros):
    if s is None or stream is None:
        return
    st = stream.assign(gbs=stream.triad_GBs * 32.0 / 24.0)   # STREAM counts 24 B, moves 32 B
    bw = s.N0.astype(float) ** 3 * BYTES_CPU / s.t_iter / 1e9
    sock = float(st.gbs.iloc[-1])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    nodes = sorted(set(int(p) for p in s.P if p >= cpn))
    ax.plot(nodes, [sock * p / cpn for p in nodes], color=INK2, ls=":", lw=1.0,
            label="STREAM of one 32-core socket $\\times$ nodes")
    ax.plot(st.threads, st.gbs, color=INK2, marker="s", ls="--", label="STREAM triad, one socket")
    ax.plot(s.P, bw, color=C[0], marker="o", label="CG solver, $512^3$")
    log2_axis(ax, s.P)
    ax.set_yscale("log")
    plain_log(ax, "y", 0, 4)
    ax.set_xlabel("cores ($%d$ per node)" % cpn)
    ax.set_ylabel("DRAM bandwidth (GB/s)")
    ax.legend(loc="upper left", fontsize=6.3)
    fig.tight_layout()
    save(fig, "bandwidth")
    i32 = list(s.P).index(cpn)
    macros.update({"StreamOne": fmt(float(st.gbs.iloc[0]), 1), "StreamAll": fmt(sock, 0),
                   "StreamThreads": str(int(st.threads.iloc[-1])), "StreamEight": fmt(float(st[st.threads == 8].gbs.iloc[0]), 0),
                   "CGbwOne": fmt(float(bw.iloc[0]), 1), "CGbwNode": fmt(float(bw.iloc[i32]), 0),
                   "CGbwFrac": fmt(100 * float(bw.iloc[i32]) / sock, 0), "CGbwAll": fmt(float(bw.iloc[-1]), 0),
                   "CGbwTB": fmt(float(bw.iloc[-1]) / 1e3, 2),
                   "CGbwPerNodeMax": fmt(float(bw.iloc[-1]) / (int(s.P.iloc[-1]) // cpn), 0),
                   "CGbwFracNodes": fmt(100 * float(bw.iloc[-1]) / ((int(s.P.iloc[-1]) // cpn) * float(bw.iloc[i32])), 0),
                   "CGbwFracStreamNodes": fmt(100 * float(bw.iloc[-1]) / ((int(s.P.iloc[-1]) // cpn) * sock), 0),
                   "StreamSixteen": fmt(float(st[st.threads == 16].gbs.iloc[0]), 0),
                   "CGbwEight": fmt(float(bw[s.P == 8].iloc[0]), 0), "CGbwSixteen": fmt(float(bw[s.P == 16].iloc[0]), 0)})


def fig_weak(b, cpn, macros):
    w = b[b.exp == "weak"].sort_values("P")
    if w.empty:
        return
    tpp = w.t_iter / w.nloc_max                 # time per point owned by the busiest process
    e1 = float(tpp.iloc[0]) / tpp
    t32 = float(tpp[w.P == cpn].iloc[0])
    wn = w[w.P >= cpn]
    en = t32 / (wn.t_iter / wn.nloc_max)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(w.P, 100 * e1, color=C[0], marker="o", label="relative to one process")
    ax.plot(wn.P, 100 * en, color=C[1], marker="s", ls="--", mfc="white", label="relative to one node")
    ax.axhline(100, color=INK2, ls="--", lw=1.0)
    ax.axvline(cpn, color=C[5], lw=0.8, ls=":")
    log2_axis(ax, w.P)
    ax.set_ylim(0, 110)
    ax.set_xlabel("MPI processes $P$")
    ax.set_ylabel("weak-scaling efficiency (%)")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "weak")
    last = w.iloc[-1]
    two = w[w.P == 2 * cpn]
    mid = w[(w.P > 2 * cpn) & (w.P < last.P)]
    macros.update({"WeakTtwo": fmt(1e3 * float(two.t_iter.iloc[0]), 1) if not two.empty else "--",
                   "WeakTmidMin": fmt(1e3 * mid.t_iter.min(), 0), "WeakTmidMax": fmt(1e3 * mid.t_iter.max(), 0)})
    macros.update({"WeakEffPmax": fmt(100 * float(e1.iloc[-1]), 0), "WeakPmax": str(int(last.P)),
                   "WeakNodeEff": fmt(100 * float(en.iloc[-1]), 0), "WeakEffNode": fmt(100 * float(e1[w.P == cpn].iloc[0]), 0),
                   "WeakLocal": str(round(w.nloc_avg.iloc[0] ** (1 / 3))),
                   "WeakTone": fmt(1e3 * float(w.t_iter.iloc[0]), 1),
                   "WeakTnode": fmt(1e3 * float(w[w.P == cpn].t_iter.iloc[0]), 1),
                   "WeakTmax": fmt(1e3 * float(last.t_iter), 1), "WeakNmax": str(int(last.N0)),
                   "WeakDofsMax": fmt(int(last.N0) ** 3 / 1e9, 2),
                   "WeakBwOne": fmt(float(w.nloc_max.iloc[0]) * BYTES_CPU / float(w.t_iter.iloc[0]) / 1e9, 1),
                   "WeakBwNode": fmt(float(w[w.P == cpn].nloc_max.iloc[0]) * BYTES_CPU /
                                     float(w[w.P == cpn].t_iter.iloc[0]) / 1e9, 1)})


def fig_variants(b, macros, tables):
    base = b[(b.exp == "strong")][["P", "t_iter"]].rename(columns={"t_iter": "t_base"})
    v = b[b.exp.isin(["decomp", "comm", "alg"])].merge(base, on="P")
    if v.empty:
        return

    def label(r):
        if r.exp == "decomp":
            return f"{r.decomp} decomposition"
        if r.exp == "comm":
            return "blocking halo"
        return {"cg1": "CG1 (1 Allreduce)", "pipecg": "pipelined CG"}[r.alg] + \
            (" + blocking halo" if r.comm == "blocking" else "")

    v = v.assign(label=v.apply(label, axis=1), rel=v.t_iter / v.t_base)
    Ps = sorted(v.P.unique())
    panels = [("(a) Data distribution and halo strategy", ["1D decomposition", "2D decomposition", "blocking halo"]),
              ("(b) CG formulation (3D blocks)", ["CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, (title, labs) in zip(axs, panels):
        w = 0.8 / len(labs)
        for k, lab in enumerate(labs):
            d = v[v.label == lab].set_index("P").reindex(Ps)
            xs = [i + (k - (len(labs) - 1) / 2) * w for i in range(len(Ps))]
            ax.bar(xs, d.rel, width=w * 0.92, color=C[k], label=lab, edgecolor="white", linewidth=0.6)
        ax.axhline(1.0, color=INK, lw=1.0)
        ax.set_xticks(range(len(Ps)))
        ax.set_xticklabels([str(int(p)) for p in Ps])
        ax.set_xlabel("MPI processes $P$ (%s per node)" % 32)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel("time / baseline time")
    axs[0].set_ylim(0.8, max(1.25, v.rel.max() * 1.22))
    fig.tight_layout()
    suptitle(fig, "One design decision at a time, one to eight nodes", "time per iteration relative to the "
             "baseline (3D blocks, overlapped halo, classic CG) at the same $P$; below 1 is faster")
    save(fig, "variants")

    order = ["3D, overlap, CG (baseline)", "1D decomposition", "2D decomposition", "blocking halo",
             "CG1 (1 Allreduce)", "pipelined CG", "pipelined CG + blocking halo"]
    b0 = b[b.exp == "strong"].set_index("P")
    Pt = [p for p in (32, 64, 128, 256) if p in Ps] or Ps[:4]
    rows = []
    for lab in order:
        cells = []
        for P in Pt:
            if lab.endswith("(baseline)"):
                cells.append(f"{1e3 * b0.loc[P, 't_iter']:.2f}")
            else:
                d = v[(v.label == lab) & (v.P == P)]
                cells.append("--" if d.empty else f"{1e3 * d.t_iter.iloc[0]:.2f} ({d.rel.iloc[0]:.2f})")
        rows.append(tex_escape(lab) + " & " + " & ".join(cells) + r" \\")
    tables["variants"] = rows
    tables["variants_cols"] = Pt

    def rel(lab, P):
        d = v[(v.label == lab) & (v.P == P)]
        return float(d.rel.iloc[0]) if not d.empty else float("nan")

    def rng(lab):
        d = v[v.label == lab].rel
        return fmt(d.min(), 2) + "--" + fmt(d.max(), 2)

    pm, p0 = max(Ps), min(Ps)
    macros.update({"VarP": str(int(pm)), "VarPmin": str(int(p0)),
                   "RelOneD": fmt(rel("1D decomposition", pm)), "RelOneDmin": fmt(rel("1D decomposition", p0)),
                   "RelTwoD": fmt(rel("2D decomposition", pm)), "RelTwoDRange": rng("2D decomposition"),
                   "RelBlocking": fmt(rel("blocking halo", pm)), "RelBlockingRange": rng("blocking halo"),
                   "RelCGone": fmt(rel("CG1 (1 Allreduce)", pm)), "RelCGonemin": fmt(rel("CG1 (1 Allreduce)", p0)), "RelCGoneRange": rng("CG1 (1 Allreduce)"),
                   "RelPipe": fmt(rel("pipelined CG", pm)), "RelPipemin": fmt(rel("pipelined CG", p0)),
                   "RelPipeBlock": fmt(rel("pipelined CG + blocking halo", pm)),
                   "RelPipeTwo": fmt(float(b[b.exp == "strong2_pipe"].t_iter.iloc[-1] /
                                           b[b.exp == "strong2"].t_iter.iloc[-1]), 2)
                   if (b.exp == "strong2_pipe").any() else "--"})


def fig_commvol(path, macros):
    if not os.path.exists(path):
        return
    cv = pd.read_csv(path)
    N = int(cv.N0.iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for k, d in enumerate(["1D", "2D", "3D"]):
        s = cv[cv.decomp == d].sort_values("P")
        ax[0].plot(s.P, s.halo_bytes_iter_max / 2 ** 20, color=C[k], marker=MARK[k], label=d)
        ax[1].plot(s.P, s.halo_bytes_iter_tot / 2 ** 20, color=C[k], marker=MARK[k], label=d)
    for a in ax:
        log2_axis(a, cv.P)
        a.set_yscale("log")
        plain_log(a)
        a.set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("MiB sent per iteration")
    ax[0].set_title("(a) Busiest process")
    ax[1].set_title("(b) All processes together")
    ax[1].legend(title="decomposition", loc="upper left")
    fig.tight_layout()
    suptitle(fig, "Halo-exchange volume per CG iteration", f"$N={N}^3$, measured by the solver's byte counters")
    save(fig, "commvol")
    s64 = cv[cv.P == cv.P.max()].set_index("decomp")
    macros.update({"CVP": str(int(cv.P.max())), "CVN": str(N),
                   "CVoneD": fmt(s64.loc["1D", "halo_bytes_iter_max"] / 2 ** 20, 2),
                   "CVthreeD": fmt(s64.loc["3D", "halo_bytes_iter_max"] / 2 ** 20, 2),
                   "CVratio": fmt(s64.loc["1D", "halo_bytes_iter_max"] / s64.loc["3D", "halo_bytes_iter_max"], 1),
                   "CVmsgsOneD": str(int(s64.loc["1D", "msgs_iter_max"])),
                   "CVmsgsThreeD": str(int(s64.loc["3D", "msgs_iter_max"]))})


def fig_two_d(b2, b3, cpn, macros, tables):
    """2D problem on 1-256 processes: strong, weak and variants, next to the 3D curves"""
    s = b2[b2.exp == "strong2d"].sort_values("P").copy()
    if s.empty:
        return None
    t1 = float(s[s.P == 1].t_iter.iloc[0])
    s["S"] = t1 / s.t_iter
    s["E"] = s.S / s.P
    t32 = float(s[s.P == cpn].t_iter.iloc[0])
    s["En"] = t32 * cpn / (s.t_iter * s.P)
    w = b2[b2.exp == "weak2d"].sort_values("P").copy()
    w["tpp"] = w.t_iter / w.nloc_max
    w32 = float(w[w.P == cpn].tpp.iloc[0])
    s3 = b3[b3.exp == "strong"].sort_values("P")
    w3 = b3[b3.exp == "weak"].sort_values("P")
    t32_3 = float(s3[s3.P == cpn].t_iter.iloc[0])
    wtp3 = w3.t_iter / w3.nloc_max
    w32_3 = float(wtp3[w3.P == cpn].iloc[0])
    Ps = s.P.values
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax[0].plot(Ps, Ps, color=INK2, lw=1.0, ls="--", label="ideal")
    ax[0].plot(Ps, s.S, color=C[1], marker="s", label=f"2D ${int(s.N0.iloc[0])}^2$")
    s3S = float(s3[s3.P == 1].t_iter.iloc[0]) / s3.t_iter
    ax[0].plot(s3.P, s3S, color=C[0], marker="o", label="3D $512^3$")
    ax[0].axvline(cpn, color=C[5], lw=0.8, ls=":")
    ax[0].set_xscale("log", base=2)
    ax[0].xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax[0].xaxis.set_minor_locator(NullLocator())
    log2_axis(ax[0], [1, 4, 16, 64, 256], axis="y")
    ax[0].set_ylim(0.8, 330)
    ax[0].set_xlabel("MPI processes $P$")
    ax[0].set_ylabel("speedup over one process")
    ax[0].set_title("(a) Strong scaling")
    ax[0].legend(loc="upper left", fontsize=6.3)
    sn = s[s.P >= cpn]
    ax[1].plot(sn.P, 100 * sn.En, color=C[1], marker="s", label="2D strong")
    s3n = s3[s3.P >= cpn]
    ax[1].plot(s3n.P, 100 * t32_3 * cpn / (s3n.t_iter * s3n.P), color=C[0], marker="o", label="3D strong")
    wn = w[w.P >= cpn]
    ax[1].plot(wn.P, 100 * w32 / wn.tpp, color=C[1], marker="s", ls="--", mfc="white", label="2D weak")
    w3n = w3[w3.P >= cpn]
    ax[1].plot(w3n.P, 100 * w32_3 / (w3n.t_iter / w3n.nloc_max), color=C[0], marker="o", ls="--", mfc="white",
               label="3D weak")
    ax[1].axhline(100, color=INK2, lw=1.0, ls="--")
    log2_axis(ax[1], sn.P)
    ax[1].set_ylim(50, 110)
    ax[1].set_xlabel("MPI processes $P$ (32 per node)")
    ax[1].set_ylabel("efficiency vs. one node (%)")
    ax[1].set_title("(b) Across nodes")
    ax[1].legend(loc="lower left", fontsize=6.3)
    base = s.set_index("P").t_iter
    labs = [("decomp2d", "slabs (1D)"), ("comm2d", "blocking halo"), ("alg2d", "CG1")]
    Pv = [p for p in Ps if p >= cpn]
    wd = 0.8 / len(labs)
    rel = {}
    for k, (e, lab) in enumerate(labs):
        d = b2[b2.exp == e].set_index("P").t_iter.reindex(Pv)
        rel[e] = d / base.reindex(Pv)
        xs = [i + (k - (len(labs) - 1) / 2) * wd for i in range(len(Pv))]
        ax[2].bar(xs, rel[e], width=wd * 0.92, color=C[[3, 2, 4][k]], label=lab, edgecolor="white", linewidth=0.6)
    ax[2].axhline(1.0, color=INK, lw=1.0)
    ax[2].set_xticks(range(len(Pv)))
    ax[2].set_xticklabels([str(int(p)) for p in Pv])
    ax[2].set_ylim(0.7, 1.2)
    ax[2].set_xlabel("MPI processes $P$")
    ax[2].set_ylabel("time / baseline time")
    ax[2].set_title("(c) 2D design choices")
    ax[2].grid(axis="x", visible=False)
    ax[2].legend(loc="upper left", fontsize=6.3, ncol=3)
    fig.tight_layout()
    save(fig, "twod")
    wi = w.set_index("P")
    rows = []
    for _, r in s.iterrows():
        P = int(r.P)
        comm = (r.t_halo_it + r.t_reduce_it) / r.t_iter * 100
        en = f"{100 * r.En:.0f}" if P >= cpn else "--"
        wt = f"{1e3 * wi.loc[P, 't_iter']:.1f}" if P in wi.index else "--"
        we = f"{100 * w32 / wi.loc[P, 'tpp']:.0f}" if P in wi.index and P >= cpn else "--"
        rows.append(f"{P} & {max(1, P // cpn)} & {int(r.px)}$\\times${int(r.pz)} & {1e3 * r.t_iter:.2f} & {r.S:.1f} & "
                    f"{100 * r.E:.0f} & {en} & {comm:.0f} & {r.halo_bytes_iter_max / 2 ** 10:.0f} & {wt} & {we} \\\\")
    tables["strong2d"] = rows
    last, n1 = s.iloc[-1], s[s.P == cpn].iloc[0]
    lw = w.iloc[-1]
    pm = int(last.P)
    sol = b2[b2.exp == "solve2d"]
    rr = {e: (float(rel[e].min()), float(rel[e].max()), float(rel[e].iloc[-1]), float(rel[e].iloc[0])) for e in rel}
    macros.update({
        "TwoN": str(int(s.N0.iloc[0])), "TwoDofs": fmt(int(s.N0.iloc[0]) ** 2 / 1e6, 0),
        "TwoPoneT": fmt(1e3 * t1, 0), "TwoTnode": fmt(1e3 * n1.t_iter, 1), "TwoTmax": fmt(1e3 * last.t_iter, 2),
        "TwoSpeedNode": fmt(n1.S, 1), "TwoEffNode": fmt(100 * n1.E, 0), "TwoSpeedMax": fmt(last.S, 1),
        "TwoEffMax": fmt(100 * last.E, 0), "TwoNodeEffMax": fmt(100 * last.En, 0),
        "TwoNodeSpeedMax": fmt(t32 / float(last.t_iter), 2),
        "TwoSpeedFour": fmt(float(s[s.P == 4].S.iloc[0]), 2),
        "TwoCommMax": fmt((last.t_halo_it + last.t_reduce_it) / last.t_iter * 100, 0),
        "TwoRedMsMax": fmt(1e3 * last.t_reduce_it, 2), "TwoHaloMsMax": fmt(1e3 * last.t_halo_it, 2),
        "TwoRedFracMax": fmt(100 * last.t_reduce_it / last.t_iter, 0),
        "TwoHaloFracMax": fmt(100 * last.t_halo_it / last.t_iter, 0),
        "TwoHaloKiBMax": fmt(last.halo_bytes_iter_max / 2 ** 10, 0), "TwoMsgsMax": str(int(last.msgs_iter_max)),
        "TwoLocalMax": f"{int(s.N0.iloc[0]) // int(last.px)}$\\times${int(s.N0.iloc[0]) // int(last.pz)}",
        "TwoBwNode": fmt(int(s.N0.iloc[0]) ** 2 * BYTES_CPU / float(n1.t_iter) / 1e9, 0),
        "TwoBwMax": fmt(int(s.N0.iloc[0]) ** 2 * BYTES_CPU / float(last.t_iter) / 1e9, 0),
        "TwoWeakLocal": str(round(float(w.nloc_avg.iloc[0]) ** 0.5)), "TwoWeakTone": fmt(1e3 * float(w.t_iter.iloc[0]), 1),
        "TwoWeakTnode": fmt(1e3 * float(w[w.P == cpn].t_iter.iloc[0]), 1), "TwoWeakTmax": fmt(1e3 * lw.t_iter, 1),
        "TwoWeakNodeEff": fmt(100 * w32 / lw.tpp, 0), "TwoWeakNmax": str(int(lw.N0)),
        "TwoWeakDofsMax": fmt(int(lw.N0) ** 2 / 1e9, 2),
        "TwoWeakEffNode": fmt(100 * float(w.tpp.iloc[0]) / w32, 0),
        "TwoSlabMax": fmt(rr["decomp2d"][2], 2), "TwoSlabMin": fmt(rr["decomp2d"][3], 2),
        "TwoSlabBest": fmt(rr["decomp2d"][0], 2), "TwoSlabPct": fmt(100 * (1 - rr["decomp2d"][2]), 0),
        "TwoBlockRange": fmt(rr["comm2d"][0], 2) + "--" + fmt(rr["comm2d"][1], 2), "TwoBlockMax": fmt(rr["comm2d"][2], 2),
        "TwoCGoneRange": fmt(rr["alg2d"][0], 2) + "--" + fmt(rr["alg2d"][1], 2), "TwoCGoneMax": fmt(rr["alg2d"][2], 2),
        "TwoSlabHaloKiB": fmt(2 * int(s.N0.iloc[0]) * 8 / 1024, 0),
    })
    if not sol.empty:
        c0, c1 = sol[sol.alg == "cg"].iloc[0], sol[sol.alg == "cg1"]
        macros.update({"TwoSolveIters": str(int(c0.iters)), "TwoSolveCG": fmt(float(c0.t_solve), 0),
                       "TwoSolveTrue": f"${sci(float(c0.true_relres), 1)}$",
                       "TwoSolveSerialEst": fmt(int(c0.iters) * t1, 0),
                       "TwoSolveSpeedup": fmt(int(c0.iters) * t1 / float(c0.t_solve), 0)})
        if not c1.empty:
            macros["TwoSolveCGone"] = fmt(float(c1.t_solve.iloc[0]), 0)
            macros["TwoSolveCGoneGain"] = fmt(100 * (1 - float(c1.t_solve.iloc[0]) / float(c0.t_solve)), 0)
    return s, t1


def fig_hybrid(b, macros, tables):
    h = b[b.exp == "hybrid"].sort_values("T")
    if h.empty:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    labels = [f"{int(p)}$\\times${int(t)}" for p, t in zip(h.P, h['T'])]
    ax.bar(range(len(h)), 1e3 * h.t_iter, color=C[0], width=0.62, label="best run", edgecolor="none")
    ax.plot(range(len(h)), 1e3 * h.t_med, color=C[1], marker="D", ls="none", ms=3.6, label="median run")
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("MPI processes $\\times$ OpenMP threads")
    ax.set_ylabel("ms per iteration")
    ax.set_ylim(0, 1e3 * h.t_med.max() * 1.25)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    fig.tight_layout()
    save(fig, "hybrid")
    best, pure = h.loc[h.t_iter.idxmin()], h[h["T"] == 1].iloc[0]
    worst = h.loc[h.t_iter.idxmax()]
    hs = h.sort_values("t_iter")
    second = hs.iloc[-2]
    tables['hybrid'] = [f"{int(r.P)} & {int(r['T'])} & {1e3 * r.t_iter:.2f} & {1e3 * r.t_med:.2f} & "
                        f"{1e3 * r.t_halo_it:.2f} & {1e3 * r.t_reduce_it:.2f} \\\\" for _, r in h.iterrows()]
    macros.update({"HybBest": f"{int(best.P)}$\\times${int(best['T'])}", "HybBestT": fmt(1e3 * best.t_iter, 2),
                   "HybPureT": fmt(1e3 * pure.t_iter, 2), "HybWorst": f"{int(worst.P)}$\\times${int(worst['T'])}",
                   "HybWorstT": fmt(1e3 * worst.t_iter, 2),
                   "HybRange": fmt(100 * (worst.t_iter / best.t_iter - 1), 0),
                   "HybSecond": f"{int(second.P)}$\\times${int(second['T'])}", "HybSecondT": fmt(1e3 * second.t_iter, 2),
                   "HybCores": str(int(h.P.iloc[0] * h["T"].iloc[0])),
                   "HybGain": fmt(100 * (1 - best.t_iter / pure.t_iter), 0),
                   "HybRedBest": fmt(1e3 * best.t_reduce_it, 2), "HybRedPure": fmt(1e3 * pure.t_reduce_it, 2)})


# ---------------------------------------------------------------- GPU
def fig_gpu(gdir, s, t_ser, macros, tables, g2dir=None, s2=None):
    p = os.path.join(gdir, "gpu.csv")
    if not os.path.exists(p):
        return
    g = pd.read_csv(p)
    g2 = None
    if g2dir and os.path.exists(os.path.join(g2dir, "gpu.csv")):
        g2 = pd.read_csv(os.path.join(g2dir, "gpu.csv"))
        g2s = best_of(g2[g2.exp == "gsize2d"], KEYS).sort_values("N0").set_index("N0")
        bw2 = g2s.index.to_series().astype(float) ** 2 * BYTES_GPU / g2s.t_iter / 1e9
    gs = best_of(g[g.exp == "gsize"], KEYS).sort_values(["alg", "N0"])
    cg, c1 = gs[gs.alg == "cg"].set_index("N0"), gs[gs.alg == "cg1"].set_index("N0")
    Ns = list(cg.index)
    bw = {a: d.N0.astype(float) ** 3 * BYTES_GPU / d.t_iter / 1e9 for a, d in [("cg", cg.reset_index()),
                                                                              ("cg1", c1.reset_index())]}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].axhline(A40_PEAK, color=INK2, ls="--", lw=1.0, label="A40 peak (696 GB/s)")
    n3 = [N ** 3 / 1e6 for N in Ns]
    ax[0].plot(n3, bw["cg"].values, color=C[0], marker="o", label="3D, classic CG")
    ax[0].plot(n3, bw["cg1"].values, color=C[0], marker="o", ls="--", mfc="white", label="3D, CG1")
    if g2 is not None:
        ax[0].plot(g2s.index.astype(float) ** 2 / 1e6, bw2.values, color=C[1], marker="s", label="2D, classic CG")
    ax[0].set_xscale("log")
    ax[0].xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_ylim(0, 800)
    ax[0].set_xlabel("unknowns (millions, one GPU)")
    ax[0].set_ylabel("DRAM bandwidth (GB/s, 88 B/point)")
    ax[0].set_title("(a) One A40 across problem sizes")
    ax[0].legend(loc="lower right", fontsize=6.5)
    n512 = 512
    plat = [("1 core", t_ser), ("1 socket\n(32 cores)", float(s[s.P == 32].t_iter.iloc[0])),
            ("1 A40", float(cg.loc[n512, "t_iter"])),
            ("8 nodes\n(256 cores)", float(s[s.P == s.P.max()].t_iter.iloc[0]))]
    cols = [C[5], C[0], C[2], C[3]]
    ax[1].bar(range(4), [1e3 * t for _, t in plat], color=cols, width=0.6, edgecolor="none")
    for i, (_, t) in enumerate(plat):
        ax[1].text(i, 1e3 * t * 1.12, f"{1e3 * t:.1f}", ha="center", va="bottom", fontsize=6.8)
    ax[1].set_yscale("log")
    plain_log(ax[1], "y", 0, 4)
    ax[1].set_ylim(5, 2000)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([n for n, _ in plat], fontsize=6.8)
    ax[1].set_ylabel("ms per iteration")
    ax[1].set_title("(b) $512^3$: GPU against CPU")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    suptitle(fig, "MPI + OpenMP target offload on an NVIDIA A40", "classic CG, all vectors resident in GPU "
             "memory, 50 iterations, best of 3 runs")
    save(fig, "gpu")
    rows = []
    for N in Ns:
        r, q = cg.loc[N], c1.loc[N]
        sp = 100 * r.t_spmv_it / r.t_iter
        mem = 4 * (N + 2) ** 3 * 8 / 2 ** 30
        rows.append(f"{N} & {N ** 3 / 1e6:.0f} & {mem:.1f} & {1e3 * r.t_iter:.2f} & {1e3 * q.t_iter:.2f} & "
                    f"{bw['cg'][Ns.index(N)]:.0f} & {sp:.0f} & {r.mlups / 1e3:.2f} \\\\")
    tables["gpu"] = rows
    r = cg.loc[n512]
    t_sock = float(s[s.P == 32].t_iter.iloc[0])
    macros.update({"GpuTfiveTwelve": fmt(1e3 * r.t_iter, 1), "GpuBwFiveTwelve": fmt(bw["cg"][Ns.index(n512)], 0),
                   "GpuBwFrac": fmt(100 * bw["cg"][Ns.index(n512)] / A40_PEAK, 0),
                   "GpuBwMin": fmt(bw["cg"].min(), 0), "GpuBwMax": fmt(bw["cg"].max(), 0),
                   "GpuNmin": str(Ns[0]), "GpuNmax": str(Ns[-1]), "GpuDofsMax": fmt(Ns[-1] ** 3 / 1e6, 0),
                   "GpuVsSocket": fmt(t_sock / r.t_iter, 1), "GpuVsCore": fmt(t_ser / r.t_iter, 0),
                   "GpuVsEight": fmt(r.t_iter / float(s[s.P == s.P.max()].t_iter.iloc[0]), 1),
                   "GpuSpmvFrac": fmt(100 * r.t_spmv_it / r.t_iter, 0),
                   "GpuCGoneRel": fmt(100 * (c1.loc[n512, "t_iter"] / r.t_iter - 1), 0),
                   "GpuSpmvBw": fmt(n512 ** 3 * 16 / r.t_spmv_it / 1e9, 0),      # read p, write q
                   "GpuSpmvBwFrac": fmt(100 * n512 ** 3 * 16 / r.t_spmv_it / 1e9 / A40_PEAK, 0),
                   "GpuVecBw": fmt(n512 ** 3 * 72 / r.t_vec_it / 1e9, 0),        # 6 + 3 vector accesses
                   "GpuVecBwFrac": fmt(100 * n512 ** 3 * 72 / r.t_vec_it / 1e9 / A40_PEAK, 0),
                   "GpuBwAboveMin": fmt(bw["cg"][1:].min(), 0),
                   "GpuBwRatio": fmt(bw["cg"][Ns.index(n512)] / (n512 ** 3 * BYTES_CPU / t_sock / 1e9), 1),
                   "GpuMLUPS": fmt(r.mlups / 1e3, 2)})
    if g2 is not None:
        tables["gpu2d"] = [f"{int(N)} & {N ** 2 / 1e6:.0f} & {3 * 4 * (N + 2) ** 2 * 8 / 2 ** 30:.1f} & "
                           f"{1e3 * r2.t_iter:.2f} & -- & {bw2[N]:.0f} & {100 * r2.t_spmv_it / r2.t_iter:.0f} & "
                           f"{r2.mlups / 1e3:.2f} \\\\" for N, r2 in g2s.iterrows()]
        n2 = 8192 if 8192 in g2s.index else int(g2s.index[-2])
        r2 = g2s.loc[n2]
        macros.update({"GpuTwoN": str(n2), "GpuTwoT": fmt(1e3 * r2.t_iter, 2), "GpuTwoBw": fmt(bw2[n2], 0),
                       "GpuTwoBwMin": fmt(bw2.min(), 0), "GpuTwoBwMax": fmt(bw2.max(), 0),
                       "GpuTwoNmin": str(int(g2s.index[0])), "GpuTwoNmax": str(int(g2s.index[-1])),
                       "GpuTwoDofsMax": fmt(int(g2s.index[-1]) ** 2 / 1e6, 0),
                       "GpuTwoSpmvBw": fmt(n2 ** 2 * 16 / r2.t_spmv_it / 1e9, 0),
                       "GpuTwoVecBw": fmt(n2 ** 2 * 72 / r2.t_vec_it / 1e9, 0)})
        if s2 is not None:
            macros.update({"GpuTwoVsSocket": fmt(float(s2[s2.P == 32].t_iter.iloc[0]) / r2.t_iter, 1),
                           "GpuTwoVsEight": fmt(r2.t_iter / float(s2[s2.P == s2.P.max()].t_iter.iloc[0]), 1)})
        gs2 = g2[g2.exp == "gsolve2d"]
        if not gs2.empty:
            macros.update({"GpuTwoSolve": fmt(float(gs2.t_solve.iloc[0]), 0)})
        v = g2[g2.exp == "validate2d"]
    sol = g[g.exp == "gsolve"]
    if g2 is not None:
        sol = pd.concat([sol, g2[g2.exp == "gsolve2d"]], ignore_index=True)
    return sol


def gpu_validation(vdir, macros):
    p = os.path.join(vdir, "validation.txt")
    if not os.path.exists(p):
        return
    txt = open(p).read()
    vals = [float(x) for x in re.findall(r"PASS\s+max\S+\s*=\s*([\d.e+-]+)", txt)]
    ps = sorted(set(int(x) for x in re.findall(r"\(P=(\d+)", txt)))
    m = re.search(r"validation:\s*(\d+) failure", txt)
    macros.update({"GpuValRuns": str(len(vals)), "GpuValWorst": f"${sci(max(vals), 1)}$" if vals else "--",
                   "GpuValP": ", ".join(str(p) for p in ps), "GpuValFail": m.group(1) if m else "--"})


def gpu_validation2d(vdir, macros):
    p = os.path.join(vdir, "validation.txt")
    if not os.path.exists(p):
        return
    txt = open(p).read()
    vals = [float(x) for x in re.findall(r"PASS\s+2D max\S+\s*=\s*([\d.e+-]+)", txt)]
    ps = sorted(set(int(x) for x in re.findall(r"2D max\S+\s*=\s*\S+\s+\(P=(\d+)", txt)))
    fails = len(re.findall(r"FAIL\s+2D", txt))
    macros.update({"GpuValTwoRuns": str(len(vals)), "GpuValTwoWorst": f"${sci(max(vals), 1)}$" if vals else "--",
                   "GpuValTwoP": ", ".join(str(p) for p in ps), "GpuValTwoFail": str(fails)})


# ---------------------------------------------------------------- validation
def fig_validation(vdir, macros, tables):
    conv = os.path.join(vdir, "convergence.csv")
    if os.path.exists(conv):
        cv = pd.read_csv(conv)
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
        groups = [("serial", "cg", "serial CG", C[0], "o", "-"), ("mpi", "cg", "MPI CG, $P=4$", C[1], "x", "none"),
                  ("mpi", "pipecg", "MPI pipelined CG, $P=4$", C[2], "+", "none")]
        for code, alg, lab, col, mk, ls in groups:
            d = cv[(cv.code == code) & (cv.alg == alg)].sort_values("N0")
            h = 1.0 / (d.N0 + 1)
            ax[0].plot(h, d.err_l2, color=col, marker=mk, ls=ls, label=lab, markersize=6 if mk != "o" else 4)
        c2p = os.path.join(vdir, "convergence2d.csv")
        cv2 = pd.read_csv(c2p) if os.path.exists(c2p) else None
        if cv2 is not None:
            for code, alg, lab, mk, ls in [("serial", "cg", "2D serial CG", "s", "-"),
                                           ("mpi", "cg", "2D MPI CG, $P=4$", "x", "none")]:
                d = cv2[(cv2.code == code) & (cv2.alg == alg)].sort_values("N0")
                ax[0].plot(1.0 / (d.N0 + 1), d.err_l2, color=C[3], marker=mk, ls=ls, label=lab,
                           markersize=6 if mk == "x" else 3.6, mfc="white" if mk == "s" else None)
        d = cv[(cv.code == "serial")].sort_values("N0")
        h = 1.0 / (d.N0 + 1)
        ref = d.err_l2.iloc[-1] * (h / h.iloc[-1]) ** 2
        ax[0].plot(h, ref * 0.5, color=INK2, ls="--", lw=1.0, label="$O(h^2)$ reference")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05]))
        ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax[0].xaxis.set_minor_formatter(NullFormatter())
        ax[0].set_xlabel("grid spacing $h$")
        ax[0].set_ylabel("$\\|u_h - u\\|_{L^2}$")
        ax[0].set_title("(a) Discretisation error")
        ax[0].legend(loc="upper left", fontsize=6.3)
        for k, alg in enumerate(["cg", "cg1", "pipecg"]):
            p = os.path.join(vdir, f"hist_{alg}.csv")
            if not os.path.exists(p):
                continue
            hh = pd.read_csv(p)
            name = {"cg": "CG", "cg1": "CG1", "pipecg": "pipelined CG"}[alg]
            lw = 2.6 if alg == "cg" else 1.3   # CG and CG1 coincide: draw CG wide underneath
            ax[1].semilogy(hh.iter, hh.relres_true, color=C[k], lw=lw, label=f"{name}: true", zorder=3 - k / 10)
            ax[1].semilogy(hh.iter, hh.relres_recursive, color=C[k], ls=":", lw=lw * 0.8, label=f"{name}: recursive")
        ax[1].set_xlabel("iteration")
        ax[1].set_ylabel("$\\|r\\|/\\|b\\|$")
        ax[1].set_title("(b) Residual histories, $N=64^3$, $P=4$")
        ax[1].set_ylim(1e-16, 10)
        ax[1].legend(fontsize=6.3, ncol=1, loc="upper right")
        fig.tight_layout()
        suptitle(fig, "Correctness of the parallel solver")
        save(fig, "validation")
        ser = cv[cv.code == "serial"].set_index("N0")
        par = cv[(cv.code == "mpi") & (cv.alg == "cg")].set_index("N0")
        rows, prev = [], None
        for N in ser.index:
            e = ser.loc[N, "err_l2"]
            order = "--" if prev is None else f"{math.log(prev[1] / e) / math.log((prev[0] + 1) / (N + 1)) * -1:.3f}"
            pe = par.loc[N, "err_l2"] if N in par.index else float("nan")
            rows.append(f"{N} & ${sci(1 / (N + 1), 2)}$ & {int(ser.loc[N, 'iters'])} & ${sci(e, 3)}$ & "
                        f"${sci(pe, 3)}$ & {order} \\\\")
            prev = (N, e)
        tables["convergence"] = rows
        ords = [float(r.split("&")[-1].strip(" \\")) for r in rows[1:]]
        macros.update({"OrderMin": f"{min(ords):.3f}", "OrderMax": f"{max(ords):.3f}"})
        if cv2 is not None:
            s2 = cv2[cv2.code == "serial"].sort_values("N0")
            o2 = [math.log(a / b) / math.log((m + 1) / (n + 1)) for (n, a), (m, b) in
                  zip(zip(s2.N0, s2.err_l2), zip(s2.N0[1:], s2.err_l2[1:]))]
            macros.update({"OrderTwoMin": f"{min(o2):.3f}", "OrderTwoMax": f"{max(o2):.3f}",
                           "ConvTwoNmin": str(int(s2.N0.iloc[0])), "ConvTwoNmax": str(int(s2.N0.iloc[-1]))})
    var = os.path.join(vdir, "variants.csv")
    if os.path.exists(var):
        v = pd.read_csv(var)
        ref = pd.read_csv(os.path.join(vdir, "serial_ref.csv")).iloc[0]
        rows = []
        v2p = os.path.join(vdir, "variants2d.csv")
        v2 = pd.read_csv(v2p) if os.path.exists(v2p) else None
        for prob, vv, nn in [("3D", v, int(ref.N0)), ("2D", v2, 100)]:
          if vv is None:
            continue
          for (alg, comm), d in vv.groupby(["alg", "comm"], sort=False):
            rows.append(f"{prob} ${nn}^{prob[0]}$ & { {'cg': 'CG', 'cg1': 'CG1', 'pipecg': 'pipelined CG'}[alg]} & {comm} & {len(d)} & "
                        f"{d.P.min()}--{d.P.max()} & {d['T'].min()}--{d['T'].max()} & "
                        f"{int(d.iters.min())}--{int(d.iters.max())} & ${sci(d.maxdiff_vs_serial.max(), 1)}$ & "
                        f"${sci(d.true_relres.max(), 2)}$ \\\\")
        tables["validation"] = rows
        edge = os.path.join(vdir, "edge.csv")
        ne = len(pd.read_csv(edge)) if os.path.exists(edge) else 0
        cg = v[v.alg != "pipecg"].maxdiff_vs_serial.max()
        pp = v[v.alg == "pipecg"].maxdiff_vs_serial.max()
        macros.update({"ValRuns": str(len(v)), "ValEdge": str(ne), "ValN": str(int(ref.N0)),
                       "ValIters": str(int(ref.iters)), "ValWorstCG": f"${sci(cg, 1)}$",
                       "ValWorstPipe": f"${sci(pp, 1)}$", "ValErrMax": f"${sci(ref.err_max, 2)}$"})
        if v2 is not None:
            r2 = pd.read_csv(os.path.join(vdir, "serial_ref2d.csv")).iloc[0]
            e2 = os.path.join(vdir, "edge2d.csv")
            macros.update({"ValTwoRuns": str(len(v2)), "ValTwoEdge": str(len(pd.read_csv(e2))) if os.path.exists(e2) else "0",
                           "ValTwoIters": str(int(r2.iters)),
                           "ValTwoWorstCG": f"${sci(v2[v2.alg != 'pipecg'].maxdiff_vs_serial.max(), 1)}$",
                           "ValTwoWorstPipe": f"${sci(v2[v2.alg == 'pipecg'].maxdiff_vs_serial.max(), 1)}$",
                           "ValTwoErrMax": f"${sci(r2.err_max, 2)}$"})
    for alg in ["cg", "pipecg"]:
        p = os.path.join(vdir, f"hist_{alg}.csv")
        if os.path.exists(p):
            hh = pd.read_csv(p)
            tag = "Pipe" if alg == "pipecg" else "CG"
            macros["Floor" + tag] = f"${sci(hh.relres_true.min(), 0)}$"
            gap = hh[hh.relres_true > 1.5 * hh.relres_recursive]
            if not gap.empty:
                macros["Split" + tag] = f"${sci(gap.relres_recursive.iloc[0], 0)}$"
            macros["Final" + tag] = f"${sci(hh.relres_true.iloc[-1], 0)}$"
            after = hh.loc[hh.relres_true.idxmin():]
            macros["Drift" + tag] = f"${sci(after.relres_true.max(), 0)}$"
            macros["Last" + tag] = str(int(hh.iter.iloc[-1]))
    if "FloorCG" in macros and "FloorPipe" in macros:
        a = pd.read_csv(os.path.join(vdir, "hist_cg.csv")).relres_true.min()
        b = pd.read_csv(os.path.join(vdir, "hist_pipecg.csv")).relres_true.min()
        macros['FloorRatio'] = f"{b / a:.0f}"


def node_validation(path, macros):
    if not os.path.exists(path):
        return
    txt = open(path).read()
    w = {a: float(x) for a, x in re.findall(r"^\s*(\w+)\s+\d+ runs: worst \S+ = ([\d.e+-]+)", txt, re.M)}
    n = sum(int(x) for x in re.findall(r"^\s*\w+\s+(\d+) runs:", txt, re.M))
    e = re.search(r"edge cases:\s*(\d+) runs, worst diff ([\d.e+-]+)", txt)
    macros.update({"NodeValRuns": str(n), "NodeValEdge": e.group(1) if e else "--",
                   "NodeValCG": f"${sci(max(w.get('cg', 0), w.get('cg1', 0)), 1)}$",
                   "NodeValPipe": f"${sci(w.get('pipecg', 0), 1)}$",
                   "NodeValResult": "passes" if "RESULT: PASS" in txt else "fails"})


# ---------------------------------------------------------------- time to solution
def tab_solve(b, sizes, gsol, cpn, macros, tables, t_ser_iter=None, b2=None, t1_2d=None):
    name = {"cg": "CG", "cg1": "CG1", "pipecg": "pipelined CG"}
    rows = []
    p3 = "3D $512^3$"
    t_ser = None
    if sizes is not None and (sizes.N0 == 512).any():
        r = sizes[sizes.N0 == 512].iloc[0]
        # speedups use the serial baseline rate of all other comparisons (iterations x t_ser_iter);
        # the size study's own solve ran on a shared node and is listed as measured
        t_ser = int(r.iters) * t_ser_iter if t_ser_iter else float(r.ts)
        rows.append(f"{p3} & 1 core (shared node) & CG & {int(r.iters)} & {r.ts:.0f} & {1e3 * r.ts / r.iters:.0f} & "
                    f"${sci(r.tr, 2)}$ & ${sci(r.err, 2)}$ & -- \\\\")
        macros.update({"SolveSerialEst": fmt(t_ser, 0), "SolveSerialMeas": fmt(float(r.ts), 0)})
    s = b[b.exp == "solve"]
    nodes = int(s.P.max()) // cpn if not s.empty else 0
    for _, r in s.iterrows():
        sp = f"{t_ser / r.t_solve:.0f}" if t_ser else "--"
        rows.append(f"{p3} & {int(r.P)} cores ({nodes} nodes) & {name[r.alg]} & {int(r.iters)} & {r.t_solve:.1f} & "
                    f"{1e3 * r.t_solve / r.iters:.2f} & ${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ & {sp} \\\\")
    g3 = gsol[gsol.N1 > 1] if gsol is not None else None
    if g3 is not None:
        for _, r in g3.iterrows():
            sp = f"{t_ser / r.t_solve:.0f}" if t_ser else "--"
            rows.append(f"{p3} & 1 A40 GPU & {name[r.alg]} & {int(r.iters)} & {r.t_solve:.1f} & "
                        f"{1e3 * r.t_solve / r.iters:.2f} & ${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ & {sp} \\\\")
    # 2D rows: speedup relative to the one-process rate of the same job (iterations x t(P=1))
    if b2 is not None:
        s2 = b2[b2.exp == "solve2d"]
        g2 = gsol[gsol.N1 == 1] if gsol is not None else pd.DataFrame()
        rows.append("\\midrule")
        for _, r in pd.concat([s2, g2], ignore_index=True).iterrows():
            ref = int(r.iters) * t1_2d if t1_2d else None
            sp = f"{ref / r.t_solve:.0f}" if ref else "--"
            plat = f"{int(r.P)} cores ({int(r.P) // cpn} nodes)" if r.code == "mpi" else "1 A40 GPU"
            rows.append(f"2D ${int(r.N0)}^2$ & {plat} & {name[r.alg]} & {int(r.iters)} & {r.t_solve:.1f} & "
                        f"{1e3 * r.t_solve / r.iters:.2f} & ${sci(r.true_relres, 2)}$ & ${sci(r.err_max, 2)}$ & {sp} \\\\")
    tables["solve"] = rows
    gsol = g3
    cg, pp, c1 = s[s.alg == "cg"], s[s.alg == "pipecg"], s[s.alg == "cg1"]
    if not cg.empty:
        macros.update({"SolveIters": str(int(cg.iters.iloc[0])), "SolveCG": fmt(float(cg.t_solve.iloc[0]), 1),
                       "SolveCGTrue": f"${sci(float(cg.true_relres.iloc[0]), 1)}$"})
        if t_ser:
            macros["SolveSpeedup"] = fmt(t_ser / float(cg.t_solve.iloc[0]), 0)
    if not c1.empty:
        macros["SolveCGone"] = fmt(float(c1.t_solve.iloc[0]), 1)
    if not pp.empty:
        macros.update({"SolvePipe": fmt(float(pp.t_solve.iloc[0]), 1),
                       "SolvePipeTrue": f"${sci(float(pp.true_relres.iloc[0]), 1)}$",
                       "SolvePipeRatio": fmt(float(pp.true_relres.iloc[0]) / 1e-8, 0)})
    if gsol is not None and not gsol.empty:
        gc = gsol[gsol.alg == "cg"].iloc[0]
        macros.update({"SolveGpu": fmt(float(gc.t_solve), 1), "SolveGpuIters": str(int(gc.iters))})
        if t_ser:
            macros["SolveGpuSpeedup"] = fmt(t_ser / float(gc.t_solve), 0)


# every macro the report uses defaults to "--" if its experiment is missing; the list is read
# from report.tex (capitalised control sequences minus LaTeX's own)
FIGURES = ["sizes", "strong", "breakdown", "variants", "commvol", "hybrid", "weak", "bandwidth", "validation", "gpu",
           "twod"]
LATEX_OWN = {"LARGE", "Cref", "Omega", "Comment", "State", "For", "EndFor", "While", "EndWhile", "If", "EndIf"}


def report_macros():
    path = os.path.join(ROOT, "report", "report.tex")
    if not os.path.exists(path):
        return []
    return sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", open(path).read())) - LATEX_OWN)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=os.path.join(RES, "vanda256"))
    ap.add_argument("--node", default=os.path.join(RES, "vanda_node"))
    ap.add_argument("--sizes", default=os.path.join(RES, "vanda_sizes"))
    ap.add_argument("--gpu", default=os.path.join(RES, "vanda_gpu"))
    ap.add_argument("--gpuval", default=os.path.join(RES, "vanda_gpu_val"))
    ap.add_argument("--validation", default=os.path.join(RES, "validation"))
    ap.add_argument("--main2d", default=os.path.join(RES, "vanda2d"))
    ap.add_argument("--sizes2d", default=os.path.join(RES, "vanda_sizes2d"))
    ap.add_argument("--gpu2d", default=os.path.join(RES, "vanda_gpu2d"))
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    for f in FIGURES:
        p = os.path.join(FIG, f + ".pdf")
        if os.path.exists(p):
            os.remove(p)
    macros, tables = {}, {}
    mi, ni, si, gi = env_info(a.main), env_info(a.node), env_info(a.sizes), env_info(a.gpu)
    cpn = mi.get("cpn", 32)
    macros.update({"MachineCPU": tex_escape(mi.get("cpu", "?")), "MachineNodes": str(mi.get("nodes", "?")),
                   "MachineCPN": str(cpn), "MachineCores": str(mi.get("nodes", 0) * cpn),
                   "MachineCPS": str(mi.get("cps", "?")), "MachineMPI": "Open MPI " + mi.get("mpi", "?"),
                   "MachineGCC": mi.get("gcc", "?"), "NodeCPU": tex_escape(ni.get("cpu", "?")),
                   "SizesCPU": tex_escape(si.get("cpu", "?"))})
    if "gpu" in gi:
        macros.update({"GpuName": gi["gpu"][0], "GpuMem": fmt(int(gi["gpu"][1]) / 1024, 0) + "\\,GiB usable", "GpuDriver": gi["gpu"][2],
                       "GpuNVC": gi.get("nvc", "?"), "GpuModule": tex_escape(gi.get("module", "?"))})
    raw = pd.read_csv(os.path.join(a.main, "bench.csv"))
    # the hybrid sweep's P x 1 runs are the strong-scaling baseline configuration at P = all cores:
    # pool them, so the baseline at that P is the best of all 6 runs of the identical configuration
    pure = raw[(raw.exp == "hybrid") & (raw["T"] == 1)].assign(exp="strong")
    b = best_of(pd.concat([raw, pure], ignore_index=True), KEYS)
    hyb1 = b[(b.exp == "strong") & (b.P == int(pure.P.max()))] if not pure.empty else None
    if hyb1 is not None:   # hybrid table/figure: P x 1 shows the pooled best and pooled median
        hb = b.index[(b.exp == "hybrid") & (b["T"] == 1)]
        for c in ["t_iter", "t_med", "t_spmv_it", "t_halo_it", "t_reduce_it", "t_vec_it"]:
            b.loc[hb, c] = float(hyb1[c].iloc[0])
    b.to_csv(os.path.join(a.main, "bench_best.csv"), index=False)
    nb = os.path.join(a.node, "bench.csv")
    t_ser = float(best_of(pd.read_csv(nb), KEYS).query("exp == 'serial'").t_iter.iloc[0]) if os.path.exists(nb) \
        else float(b[(b.exp == "strong") & (b.P == 1)].t_iter.iloc[0])
    stp = os.path.join(a.node, "stream.csv")
    stream = pd.read_csv(stp) if os.path.exists(stp) else None
    sizes, sizes2d = fig_sizes(a.sizes, a.sizes2d, macros, tables)
    s = fig_strong(b, t_ser, cpn, macros, tables)
    fig_bandwidth(s, stream, cpn, macros)
    fig_weak(b, cpn, macros)
    fig_variants(b, macros, tables)
    fig_hybrid(b, macros, tables)
    b2 = s2 = t1_2d = None
    if os.path.exists(os.path.join(a.main2d, "bench.csv")):
        b2 = best_of(pd.read_csv(os.path.join(a.main2d, "bench.csv")), KEYS)
        b2.to_csv(os.path.join(a.main2d, "bench_best.csv"), index=False)
        s2, t1_2d = fig_two_d(b2, b, cpn, macros, tables)
    gsol = fig_gpu(a.gpu, s, t_ser, macros, tables, g2dir=a.gpu2d, s2=s2)
    gpu_validation(a.gpuval, macros)
    gpu_validation2d(a.gpu2d, macros)
    tab_solve(b, sizes, gsol, cpn, macros, tables, t_ser_iter=t_ser, b2=b2, t1_2d=t1_2d)
    fig_commvol(os.path.join(RES, "commvol.csv"), macros)
    fig_validation(a.validation, macros, tables)
    node_validation(os.path.join(a.node, "validation", "summary.txt"), macros)

    defaults = {k for k in report_macros() if k not in macros}
    for f in os.listdir(GEN):
        if f.endswith(".tex"):
            os.remove(os.path.join(GEN, f))
    with open(os.path.join(GEN, "macros.tex"), "w") as f:
        f.write("% generated by scripts/make_plots.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        for k in sorted(defaults):
            f.write(f"\\providecommand{{\\{k}}}{{--}}\n")
    for t in ["validation", "convergence", "strong", "variants", "hybrid", "solve", "sizes", "gpu", "sizes2d",
              "strong2d", "gpu2d"]:
        with open(os.path.join(GEN, f"tab_{t}.tex"), "w") as f:
            f.write("\n".join(tables.get(t, [])) + "\n")
    Ps = tables.get("variants_cols", [])
    with open(os.path.join(GEN, "tab_variants_head.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * max(len(Ps), 1) + "}\n\\toprule\n")
        f.write("variant & " + (" & ".join(f"$P={p}$" for p in Ps) if Ps else "not measured") + " \\\\\n\\midrule\n")
    for f in FIGURES:
        if not os.path.exists(os.path.join(FIG, f + ".pdf")):
            fig, ax = plt.subplots(figsize=(3.4, 1.2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{f}: no data", ha="center", va="center", color=INK2)
            save(fig, f)
    if defaults:
        print("WARNING: macros without data:", " ".join(sorted(defaults)))
    print("figures ->", FIG, "| tables/macros ->", GEN)
    for k in sorted(macros):
        print(f"  {k} = {macros[k]}")


if __name__ == "__main__":
    main()
