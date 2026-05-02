# experiment.py
"""Run the full experiment matrix: 4 systems x N scenarios x 30 reps.

Outputs:
  benchmarks/results/runs.csv     one row per (system, scenario, rep)
  benchmarks/results/stats.csv    Mann-Whitney U + Cliff's delta
  benchmarks/results/latency_samples.csv  per-second p95 (for CDF figure)
"""
from __future__ import annotations
import csv
import os
from dataclasses import asdict
from typing import Iterable

from .cluster import (RunResult, ChaosPlan, WorkloadConfig, run_simulation)
from .stats import compare


SYSTEMS = ("HPA", "KEDA", "Istio+HPA", "SANO")

SCENARIOS = {
    "S1_burst":           dict(chaos=ChaosPlan()),
    "S2_pod_kills":       dict(chaos=ChaosPlan(pod_kill_period_s=60.0)),
    "S3_latency_inject":  dict(chaos=ChaosPlan(
        latency_inject_ms=((180.0, 300.0), 100.0))),
    "S4_apiserver_down":  dict(chaos=ChaosPlan(
        apiserver_down=(180.0, 300.0))),
}


def run_matrix(out_dir: str, *, reps: int = 30, duration_s: int = 600,
               seed0: int = 42) -> list[RunResult]:
    os.makedirs(out_dir, exist_ok=True)
    all_results: list[RunResult] = []
    runs_path = os.path.join(out_dir, "runs.csv")
    samples_path = os.path.join(out_dir, "latency_samples.csv")

    with open(runs_path, "w", newline="", encoding="utf-8") as fr, \
         open(samples_path, "w", newline="", encoding="utf-8") as fs:
        rw = csv.writer(fr)
        rw.writerow(["system", "scenario", "rep",
                     "p50_ms", "p95_ms", "p99_ms",
                     "throughput_rps", "reaction_time_s", "mttr_s",
                     "autonomy_ratio", "autonomy_index",
                     "overhead_cpu_pct", "overhead_added_ms"])
        sw = csv.writer(fs)
        sw.writerow(["system", "scenario", "rep", "t",
                     "rps", "served_rps", "replicas",
                     "cpu", "p95_ms", "error_rate", "api_up", "action"])

        for sys_name in SYSTEMS:
            for sc_name, sc_kwargs in SCENARIOS.items():
                for rep in range(reps):
                    res = run_simulation(
                        system=sys_name, scenario=sc_name, rep=rep,
                        duration_s=duration_s,
                        wcfg=WorkloadConfig(seed=seed0 + rep),
                        chaos=sc_kwargs["chaos"],
                        seed=seed0)
                    all_results.append(res)
                    rw.writerow([res.system, res.scenario, res.rep,
                                 round(res.p50_ms, 3), round(res.p95_ms, 3),
                                 round(res.p99_ms, 3),
                                 round(res.throughput_rps, 2),
                                 round(res.reaction_time_s, 3),
                                 round(res.mttr_s, 3),
                                 round(res.autonomy_ratio, 4),
                                 round(res.autonomy_index, 4),
                                 round(res.overhead_cpu_pct, 3),
                                 round(res.overhead_added_ms, 3)])
                    for r in res.trace:
                        sw.writerow([res.system, res.scenario, res.rep,
                                     r.t, round(r.rps, 2),
                                     round(r.served_rps, 2), r.replicas,
                                     round(r.cpu, 3), round(r.p95_ms, 3),
                                     round(r.error_rate, 4),
                                     int(r.api_up), r.action])

    write_stats(all_results, out_dir)
    return all_results


def write_stats(results: list[RunResult], out_dir: str) -> None:
    stats_path = os.path.join(out_dir, "stats.csv")
    by = {(r.system, r.scenario): [] for r in results}
    for r in results:
        by[(r.system, r.scenario)].append(r)

    metrics = [
        ("p99_ms",         lambda r: r.p99_ms),
        ("throughput_rps", lambda r: -r.throughput_rps),  # higher better
        ("mttr_s",         lambda r: r.mttr_s),
        ("autonomy_ratio", lambda r: -r.autonomy_ratio),
        ("overhead_cpu_pct", lambda r: r.overhead_cpu_pct),
    ]
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "metric",
                    "system_a", "system_b", "n_a", "n_b",
                    "u", "p_value", "cliffs_delta", "magnitude"])
        scenarios = sorted({r.scenario for r in results})
        for sc in scenarios:
            for mname, fn in metrics:
                base = [fn(r) for r in by[("HPA", sc)]]
                for sysn in ("KEDA", "Istio+HPA", "SANO"):
                    other = [fn(r) for r in by[(sysn, sc)]]
                    if not (base and other):
                        continue
                    sr = compare(other, base,
                                 metric=mname,
                                 system_a=sysn, system_b="HPA")
                    w.writerow([sc, mname, sr.system_a, sr.system_b,
                                sr.n_a, sr.n_b,
                                round(sr.u, 2),
                                f"{sr.p_value:.3e}",
                                round(sr.cliffs_delta, 4),
                                sr.magnitude])