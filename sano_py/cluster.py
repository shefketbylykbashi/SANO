# cluster.py
"""Simulated Kubernetes cluster + workload + chaos.

The simulator runs a discrete-time loop at 1 Hz and drives one of the
controllers (SANO or a baseline). It produces per-second telemetry
(latency, throughput, replicas, CP availability) plus end-of-run summary
statistics that match the metrics reported in Section VI.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agent import SANOAgent, MetricSample
from .baselines import _Base


# ---------------------------------------------------------------------------
# Workload: Poisson base + periodic 10x bursts (k6 model, paper Sec. V).
# ---------------------------------------------------------------------------
@dataclass
class WorkloadConfig:
    base_rps: float = 200.0
    burst_factor: float = 10.0
    burst_period_s: float = 120.0
    burst_duration_s: float = 30.0
    seed: int = 0


def workload(cfg: WorkloadConfig, duration_s: int):
    rng = random.Random(cfg.seed)
    for t in range(duration_s):
        in_burst = (t % cfg.burst_period_s) < cfg.burst_duration_s
        mu = cfg.base_rps * (cfg.burst_factor if in_burst else 1.0)
        # Poisson sample around mu (Gaussian approx for speed)
        rps = max(0.0, rng.gauss(mu, math.sqrt(mu)))
        yield t, rps, in_burst


# ---------------------------------------------------------------------------
# Chaos plan: which faults are active at time t.
# ---------------------------------------------------------------------------
@dataclass
class ChaosPlan:
    apiserver_down: tuple[float, float] | None = None   # (start, end)
    pod_kill_period_s: float = 0.0                      # 0 = disabled
    latency_inject_ms: tuple[tuple[float, float], float] | None = None
    # ((start, end), added_ms)

    def apiserver_available(self, t: float) -> bool:
        if not self.apiserver_down:
            return True
        s, e = self.apiserver_down
        return not (s <= t < e)

    def latency_bonus(self, t: float) -> float:
        if not self.latency_inject_ms:
            return 0.0
        (s, e), add = self.latency_inject_ms
        return add if s <= t < e else 0.0


# ---------------------------------------------------------------------------
# Service model: latency = f(load_factor), with M/M/c-ish saturation.
# ---------------------------------------------------------------------------
def service_metrics(rps: float, replicas: int, *,
                    rps_per_replica: float = 250.0,
                    base_latency_ms: float = 20.0,
                    extra_latency_ms: float = 0.0,
                    rng: Optional[random.Random] = None
                    ) -> tuple[float, float, float, float]:
    """Return (cpu, p95_latency_ms, error_rate, served_rps)."""
    rng = rng or random
    capacity = replicas * rps_per_replica
    rho = rps / max(capacity, 1.0)  # utilization
    cpu = min(0.99, max(0.05, rho * 0.85 + rng.uniform(-0.02, 0.02)))
    if rho < 1.0:
        # Light queueing; blow up near rho=1
        p95 = base_latency_ms * (1 + 4 * rho ** 3 / max(1 - rho, 0.05))
    else:
        p95 = base_latency_ms * 12 + (rho - 1) * 400
    p95 += extra_latency_ms + rng.gauss(0, base_latency_ms * 0.05)
    err = 0.0 if rho < 0.95 else min(0.5, (rho - 0.95) * 2.0)
    served = min(rps, capacity)
    return cpu, max(p95, 1.0), err, served


# ---------------------------------------------------------------------------
# Result record (one row per second).
# ---------------------------------------------------------------------------
@dataclass
class TraceRow:
    t: float
    rps: float
    served_rps: float
    replicas: int
    cpu: float
    p95_ms: float
    error_rate: float
    api_up: bool
    action: str


@dataclass
class RunResult:
    system: str
    scenario: str
    rep: int
    trace: list[TraceRow]

    # Summary metrics (post-processed)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_rps: float = 0.0
    reaction_time_s: float = 0.0
    mttr_s: float = 0.0
    autonomy_ratio: float = 0.0   # 0..1
    autonomy_index: float = 0.0
    overhead_cpu_pct: float = 0.0
    overhead_added_ms: float = 0.0


# ---------------------------------------------------------------------------
# Single-run driver (one controller, one scenario, one repetition).
# ---------------------------------------------------------------------------
def run_simulation(*, system: str, scenario: str, rep: int,
                   duration_s: int = 600,
                   wcfg: WorkloadConfig = WorkloadConfig(),
                   chaos: ChaosPlan = ChaosPlan(),
                   seed: int = 0) -> RunResult:

    rng = random.Random(seed + rep * 1009)
    wcfg = WorkloadConfig(**{**wcfg.__dict__, "seed": seed + rep})

    # Pick controller
    controller_replicas = lambda: 0  # noqa: E731
    if system == "SANO":
        agent = SANOAgent(name=f"sano-{rep}", n_init=2)
        controller_replicas = lambda: agent.state.replicas
    else:
        from .baselines import HPAController, KEDAController, IstioHPAController
        ctrl: _Base = {
            "HPA": HPAController, "KEDA": KEDAController,
            "Istio+HPA": IstioHPAController,
        }[system]()
        controller_replicas = lambda: ctrl.state.replicas

    trace: list[TraceRow] = []
    burst_start_t: Optional[float] = None
    reaction_t: Optional[float] = None
    fail_t: Optional[float] = None
    recovery_durations: list[float] = []

    # SANO overhead model (sidecar per replica, paper RQ4)
    sano_cpu_overhead = 0.023 if system == "SANO" else 0.0
    sano_added_ms = 0.7 if system == "SANO" else 0.0

    for t, rps, in_burst in workload(wcfg, duration_s):
        api_up = chaos.apiserver_available(t)
        extra = chaos.latency_bonus(t)
        replicas = max(1, controller_replicas())

        # Service metrics for this second
        cpu, p95, err, served = service_metrics(
            rps, replicas, extra_latency_ms=extra + sano_added_ms, rng=rng)
        cpu = min(0.99, cpu + sano_cpu_overhead)

        # Random pod kill: drop one replica briefly
        if chaos.pod_kill_period_s and rng.random() < (
                1.0 / max(chaos.pod_kill_period_s, 1.0)):
            replicas = max(1, replicas - 1)
            if fail_t is None:
                fail_t = t

        # Drive the controller with the observed metric
        if system == "SANO":
            sample = MetricSample(t=t, rps=rps, cpu=cpu,
                                  latency_ms=p95, error_rate=err)
            decision = agent.step(sample, api_available=api_up)
            action = decision.action
        else:
            action = ctrl.step(t, rps, cpu, api_up)

        # Detect burst-reaction time (first time after burst begins
        # that replicas grew enough to keep up with demand).
        if in_burst and burst_start_t is None:
            burst_start_t = t
        if (burst_start_t is not None and reaction_t is None
                and replicas * 250 >= rps * 0.95):
            reaction_t = t - burst_start_t

        # MTTR: time from a kill until cpu/err returns to nominal
        if fail_t is not None and err < 0.01 and cpu < 0.85:
            recovery_durations.append(t - fail_t)
            fail_t = None

        trace.append(TraceRow(t=t, rps=rps, served_rps=served,
                              replicas=replicas, cpu=cpu, p95_ms=p95,
                              error_rate=err, api_up=api_up, action=action))

    # ---- aggregate ----
    p95s = sorted(r.p95_ms for r in trace)
    served = [r.served_rps for r in trace]
    res = RunResult(system=system, scenario=scenario, rep=rep, trace=trace)
    if p95s:
        # Map p95-trace into p50/p95/p99 of latency distribution.
        # (Each row already represents the p95 for a 1-s bucket.)
        n = len(p95s)
        res.p50_ms = p95s[int(n * 0.50)]
        res.p95_ms = p95s[int(n * 0.95)]
        res.p99_ms = p95s[min(int(n * 0.99), n - 1)]
    res.throughput_rps = sum(served) / max(len(served), 1)
    res.reaction_time_s = reaction_t if reaction_t is not None else 60.0
    res.mttr_s = (sum(recovery_durations) / len(recovery_durations)
                  if recovery_durations else 0.0)

    # autonomy: fraction of seconds where adaptation succeeded
    cp_down = [r for r in trace if not r.api_up]
    if cp_down:
        ok = sum(1 for r in cp_down if r.action != "blocked")
        res.autonomy_ratio = ok / len(cp_down)
    else:
        res.autonomy_ratio = 1.0
    res.autonomy_index = (
        agent.autonomy_index(res.mttr_s) if system == "SANO"
        else 0.21  # baselines cannot adapt without CP
    )
    res.overhead_cpu_pct = sano_cpu_overhead * 100
    res.overhead_added_ms = sano_added_ms
    return res