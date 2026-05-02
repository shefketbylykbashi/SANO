"""SANO Agent: MAPE-K loop with EWMA analyzer, TARS planner, SQLite KB.

This is the Python reference implementation of the Go agent described in
Section IV of the paper. It is designed to be driven step-by-step by a
simulator (see sano_py.cluster) but the same control logic would apply
when wired to /proc + Prometheus + client-go.
"""
from __future__ import annotations
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Monitor: in the simulator, metrics are pushed in via update(); in a real
# deployment Monitor would scrape /proc/[pid]/stat, cgroups and /metrics.
# ---------------------------------------------------------------------------
@dataclass
class MetricSample:
    t: float            # seconds since start
    rps: float          # requests per second offered to this replica set
    cpu: float          # 0..1 utilisation (mean across replicas)
    latency_ms: float   # current p95 latency
    error_rate: float   # 0..1


# ---------------------------------------------------------------------------
# Analyzer: EWMA mean + variance, adaptive k threshold
# ---------------------------------------------------------------------------
class EWMAAnalyzer:
    def __init__(self, lam: float = 0.3, k0: float = 3.0,
                 k_min: float = 2.0, k_max: float = 4.0,
                 fpr_eta: float = 4.0):
        self.lam = lam
        self.k0 = k0
        self.k_min, self.k_max = k_min, k_max
        self.fpr_eta = fpr_eta
        self.mu: Optional[float] = None
        self.var: float = 0.0
        self.fpr: float = 0.0      # running false-positive rate estimate
        self.k: float = k0

    def update(self, x: float) -> tuple[float, float, float]:
        if self.mu is None:
            self.mu = x
            self.var = 0.0
        else:
            delta = x - self.mu
            self.mu = self.lam * x + (1 - self.lam) * self.mu
            self.var = self.lam * (delta * delta) + (1 - self.lam) * self.var
        self.k = max(self.k_min,
                     min(self.k_max, self.k0 + self.fpr_eta * self.fpr))
        return self.mu, math.sqrt(self.var), self.k

    def is_anomaly_high(self, x: float) -> bool:
        if self.mu is None:
            return False
        return x > self.mu + self.k * math.sqrt(self.var)

    def is_anomaly_low(self, x: float) -> bool:
        if self.mu is None:
            return False
        return x < self.mu - self.k * math.sqrt(self.var)


# ---------------------------------------------------------------------------
# Planner: TARS = Threshold-Adaptive Reactive Scaling (Algorithm 1)
# ---------------------------------------------------------------------------
@dataclass
class PlanDecision:
    action: str        # 'scale-up' | 'scale-down' | 'circuit-break' | 'no-op'
    target_replicas: int
    reason: str = ""


class TARSPlanner:
    def __init__(self, n_min: int = 1, n_max: int = 20,
                 cooldown_s: float = 15.0, error_thresh: float = 0.05):
        self.n_min = n_min
        self.n_max = n_max
        self.cooldown_s = cooldown_s
        self.error_thresh = error_thresh
        self._last_scale_t: float = -1e9

    def plan(self, t: float, n: int, sample: MetricSample,
             ana: EWMAAnalyzer) -> PlanDecision:
        ana.update(sample.rps)
        # circuit-break shortcut on errors
        if sample.error_rate > self.error_thresh:
            return PlanDecision("circuit-break", n,
                                f"error_rate={sample.error_rate:.3f}")

        # scale-up on RPS spike
        if ana.is_anomaly_high(sample.rps) and ana.mu and ana.mu > 0:
            ratio = sample.rps / max(ana.mu, 1e-6)
            delta = max(1, math.ceil(n * (ratio - 1)))
            target = min(n + delta, self.n_max)
            if target != n:
                self._last_scale_t = t
                return PlanDecision("scale-up", target,
                                    f"rps={sample.rps:.1f} mu={ana.mu:.1f}")

        # scale-down on sustained low RPS, after cooldown
        if (ana.is_anomaly_low(sample.rps)
                and (t - self._last_scale_t) > self.cooldown_s):
            target = max(n - 1, self.n_min)
            if target != n:
                self._last_scale_t = t
                return PlanDecision("scale-down", target,
                                    f"rps={sample.rps:.1f} mu={ana.mu:.1f}")

        return PlanDecision("no-op", n)


# ---------------------------------------------------------------------------
# Executor: API-server with CRI fallback, simulated success rates.
# ---------------------------------------------------------------------------
class Executor:
    def __init__(self, api_timeout_s: float = 0.5):
        self.api_timeout_s = api_timeout_s

    def apply(self, decision: PlanDecision, *,
              api_available: bool, cri_available: bool = True
              ) -> tuple[bool, str]:
        """Apply decision; return (ok, path_used).

        Real impl: client-go ScaleSubresource; on timeout/error, fall back
        to CRI to start/stop pre-staged sibling containers on the node.
        """
        if api_available:
            return True, "api"
        if cri_available:
            return True, "cri"
        return False, "none"


# ---------------------------------------------------------------------------
# Knowledge: SQLite store for metrics, decisions, outcomes.
# ---------------------------------------------------------------------------
class Knowledge:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS metrics(
        t REAL, rps REAL, cpu REAL, latency_ms REAL, error_rate REAL);
    CREATE TABLE IF NOT EXISTS decisions(
        t REAL, action TEXT, target INT, reason TEXT,
        path TEXT, ok INT);
    """

    def __init__(self, path: str = ":memory:"):
        self.con = sqlite3.connect(path)
        for stmt in self.SCHEMA.strip().split(";"):
            if stmt.strip():
                self.con.execute(stmt)

    def log_metric(self, m: MetricSample) -> None:
        self.con.execute(
            "INSERT INTO metrics VALUES (?,?,?,?,?)",
            (m.t, m.rps, m.cpu, m.latency_ms, m.error_rate))

    def log_decision(self, t: float, d: PlanDecision,
                     path: str, ok: bool) -> None:
        self.con.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?)",
            (t, d.action, d.target_replicas, d.reason, path, int(ok)))

    def close(self) -> None:
        self.con.commit()
        self.con.close()


# ---------------------------------------------------------------------------
# SANO Agent: glues the MAPE-K components together.
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    replicas: int = 2
    decisions: int = 0
    decisions_local: int = 0          # taken without API server
    decisions_ok_no_cp: int = 0       # successful while CP was down
    last_action: str = "no-op"


class SANOAgent:
    def __init__(self, name: str, *, n_init: int = 2, kb_path: str = ":memory:"):
        self.name = name
        self.analyzer = EWMAAnalyzer()
        self.planner = TARSPlanner()
        self.executor = Executor()
        self.kb = Knowledge(kb_path)
        self.state = AgentState(replicas=n_init)

    def step(self, sample: MetricSample, *,
             api_available: bool) -> PlanDecision:
        self.kb.log_metric(sample)
        decision = self.planner.plan(sample.t, self.state.replicas,
                                     sample, self.analyzer)
        if decision.action == "no-op":
            self.kb.log_decision(sample.t, decision, "n/a", True)
            return decision

        ok, path = self.executor.apply(decision,
                                       api_available=api_available)
        self.kb.log_decision(sample.t, decision, path, ok)
        self.state.decisions += 1
        if not api_available:
            self.state.decisions_local += 1
            if ok:
                self.state.decisions_ok_no_cp += 1
        if ok and decision.action in ("scale-up", "scale-down"):
            self.state.replicas = decision.target_replicas
        self.state.last_action = decision.action
        return decision

    def autonomy_index(self, mttr_s: float, mttr_max: float = 60.0,
                       weights=(1/3, 1/3, 1/3)) -> float:
        a, b, c = weights
        D = (self.state.decisions_local
             / max(self.state.decisions, 1))
        F = (self.state.decisions_ok_no_cp
             / max(self.state.decisions_local, 1))
        R = 1 - min(mttr_s / mttr_max, 1.0)
        return a * D + b * F + c * R