# figures.py
"""Generate the four PDFs referenced by paper/main.tex from REAL results.

Inputs (produced by sano_py.experiment):
  benchmarks/results/runs.csv
  benchmarks/results/latency_samples.csv

Outputs:
  paper/figures/architecture.pdf      (static schematic)
  paper/figures/latency_cdf.pdf       (from per-second p95 samples)
  paper/figures/reaction_time.pdf     (mean served-RPS curve under burst)
  paper/figures/autonomy.pdf          (mean success ratio during CP outage)
"""
from __future__ import annotations
import csv
import os
from collections import defaultdict
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


SYS_ORDER = ("HPA", "KEDA", "Istio+HPA", "SANO")
COLORS = {"HPA": "#cc4c4c", "KEDA": "#d28a3a",
          "Istio+HPA": "#3a7fb8", "SANO": "#2ca02c"}


plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
def _ensure_outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p


# ---------------------------------------------------------------------------
def fig_architecture(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    pod = FancyBboxPatch((0.3, 0.6), 5.0, 4.0,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         linewidth=1.2, edgecolor="#333", facecolor="#f5f5f5")
    ax.add_patch(pod)
    ax.text(2.8, 4.35, "Pod (microservice instance)",
            ha="center", fontweight="bold")
    app = FancyBboxPatch((0.6, 1.0), 1.9, 2.8,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         edgecolor="#1f4e79", facecolor="#dbe7f3")
    ax.add_patch(app)
    ax.text(1.55, 3.4, "App\n(catalogue)", ha="center", fontweight="bold")
    ax.text(1.55, 2.3, "/metrics\n/proc\ncgroups",
            ha="center", fontsize=7.5, style="italic", color="#1f4e79")
    agent = FancyBboxPatch((2.9, 1.0), 2.2, 2.8,
                           boxstyle="round,pad=0.03,rounding_size=0.1",
                           edgecolor="#1b6b1b", facecolor="#dff0d8")
    ax.add_patch(agent)
    ax.text(4.0, 3.45, "SANO Agent", ha="center",
            fontweight="bold", color="#1b6b1b")
    for i, lbl in enumerate(("Monitor", "Analyze", "Plan (TARS)", "Execute")):
        y = 2.95 - i * 0.42
        ax.add_patch(Rectangle((3.05, y - 0.16), 1.9, 0.32,
                               edgecolor="#1b6b1b", facecolor="white",
                               linewidth=0.6))
        ax.text(4.0, y, lbl, ha="center", va="center", fontsize=7.8)
    ax.text(4.0, 1.18, "Knowledge: SQLite",
            ha="center", fontsize=7.5, style="italic", color="#1b6b1b")
    ax.add_patch(FancyArrowPatch((2.5, 2.4), (2.9, 2.4),
                                 arrowstyle="->", mutation_scale=12))
    ax.add_patch(FancyArrowPatch((2.9, 1.6), (2.5, 1.6),
                                 arrowstyle="->", mutation_scale=12))
    api = FancyBboxPatch((6.2, 3.3), 3.4, 1.3,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         edgecolor="#7a4ea8", facecolor="#ece4f5")
    ax.add_patch(api)
    ax.text(7.9, 4.15, "Kubernetes API server",
            ha="center", fontweight="bold", color="#5b2a8a")
    ax.text(7.9, 3.65, "(opportunistic)", ha="center",
            fontsize=7.5, style="italic", color="#5b2a8a")
    cri = FancyBboxPatch((6.2, 1.7), 3.4, 1.0,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         edgecolor="#a64b00", facecolor="#fce5c8")
    ax.add_patch(cri)
    ax.text(7.9, 2.2, "CRI socket (fallback)",
            ha="center", fontweight="bold", color="#a64b00")
    peer = FancyBboxPatch((6.2, 0.2), 3.4, 1.1,
                          boxstyle="round,pad=0.03,rounding_size=0.1",
                          edgecolor="#1b6b1b", facecolor="#eaf5ea")
    ax.add_patch(peer)
    ax.text(7.9, 0.95, "Peer SANO agents",
            ha="center", fontweight="bold", color="#1b6b1b")
    ax.text(7.9, 0.5, "gossip (memberlist, UDP/7946)",
            ha="center", fontsize=7.5, style="italic", color="#1b6b1b")
    for xy in ((6.2, 3.95), (6.2, 2.2), (6.2, 0.75)):
        ax.add_patch(FancyArrowPatch((5.1, 2.4), xy,
                                     arrowstyle="->", mutation_scale=12,
                                     connectionstyle="arc3,rad=0.15"))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300); plt.close(fig)


# ---------------------------------------------------------------------------
def _read_samples(path: str):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["scenario"] != "S1_burst":
                continue
            by[row["system"]].append(float(row["p95_ms"]))
    return by


def fig_latency_cdf(samples_csv: str, out_path: str) -> None:
    data = _read_samples(samples_csv)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for name in SYS_ORDER:
        vals = np.sort(np.array(data.get(name, [])))
        if vals.size == 0:
            continue
        cdf = np.arange(1, vals.size + 1) / vals.size
        ax.plot(vals, cdf, label=name, color=COLORS[name], lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("Per-second p95 latency (ms, log scale)")
    ax.set_ylabel("CDF"); ax.set_ylim(0, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.axhline(0.99, ls="--", color="#888", lw=0.7)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_reaction_time(samples_csv: str, out_path: str) -> None:
    by_t = defaultdict(lambda: defaultdict(list))   # sys -> t -> [served_rps]
    with open(samples_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["scenario"] != "S1_burst":
                continue
            t = int(float(row["t"]))
            if t > 60:
                continue
            by_t[row["system"]][t].append(float(row["served_rps"]))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ts = sorted(next(iter(by_t.values())).keys())
    demand = [200.0 if t < 0 else (2000.0 if 0 <= t < 30 else 200.0)
              for t in ts]
    ax.plot(ts, demand, color="#444", ls="--", lw=1.0, label="Demand")
    for name in SYS_ORDER:
        if name not in by_t:
            continue
        ys = [np.mean(by_t[name].get(t, [0])) for t in ts]
        ax.plot(ts, ys, color=COLORS[name], lw=1.4, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Throughput (RPS)")
    ax.set_xlim(0, 60); ax.set_ylim(0, 2400)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_autonomy(samples_csv: str, out_path: str) -> None:
    """Use S4_apiserver_down: mean success rate per second across reps."""
    by_t = defaultdict(lambda: defaultdict(list))
    with open(samples_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["scenario"] != "S4_apiserver_down":
                continue
            t = int(float(row["t"]))
            ok = (row["action"] != "blocked")
            by_t[row["system"]][t].append(1.0 if ok else 0.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axvspan(180, 300, color="#fde2e2", alpha=0.7, label="API server down")
    ts = sorted(next(iter(by_t.values())).keys()) if by_t else []
    for name in SYS_ORDER:
        if name not in by_t:
            continue
        ys = [100.0 * np.mean(by_t[name].get(t, [1.0])) for t in ts]
        ax.plot(ts, ys, color=COLORS[name], lw=1.6, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Successful adaptations (%)")
    ax.set_ylim(-5, 108)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="center right", frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def make_all(results_dir: str, out_dir: str) -> None:
    _ensure_outdir(out_dir)
    samples = os.path.join(results_dir, "latency_samples.csv")
    fig_architecture(os.path.join(out_dir, "architecture.png"))
    fig_latency_cdf(samples, os.path.join(out_dir, "latency_cdf.png"))
    fig_reaction_time(samples, os.path.join(out_dir, "reaction_time.png"))
    fig_autonomy(samples, os.path.join(out_dir, "autonomy.png"))