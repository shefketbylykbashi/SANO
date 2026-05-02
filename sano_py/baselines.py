# baselines.py
"""Baseline orchestrators: HPA, KEDA, Istio+HPA.

All three are control-plane-bound: when api_available=False they cannot
mutate replica counts. This mirrors the behaviour described in the paper.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BaselineState:
    replicas: int = 2
    decisions: int = 0
    decisions_ok_no_cp: int = 0
    last_action: str = "no-op"


class _Base:
    name = "base"
    react_delay_s: float = 0.0
    cooldown_s: float = 30.0

    def __init__(self, n_init: int = 2, n_min: int = 1, n_max: int = 20):
        self.state = BaselineState(replicas=n_init)
        self.n_min, self.n_max = n_min, n_max
        self._last_check = -1e9
        self._pending_until = -1e9       # when reaction lag completes
        self._pending_target: int = n_init

    def _maybe_apply(self, t: float, api_available: bool) -> str:
        if t < self._pending_until:
            return "pending"
        if self._pending_target == self.state.replicas:
            return "no-op"
        # Apply the queued decision
        self.state.decisions += 1
        if not api_available:
            self.state.last_action = "blocked"
            return "blocked"
        self.state.replicas = self._pending_target
        self.state.last_action = "scale"
        return "scale"

    def step(self, t: float, rps: float, cpu: float,
             api_available: bool) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class HPAController(_Base):
    """Vanilla HPA on CPU @ 60% target; ~15s loop + ~3s lag."""
    name = "HPA"
    react_delay_s = 18.2
    target_cpu = 0.60

    def step(self, t, rps, cpu, api_available):
        if t - self._last_check >= 15.0:
            self._last_check = t
            ratio = cpu / self.target_cpu
            target = max(self.n_min,
                         min(self.n_max,
                             max(1, round(self.state.replicas * ratio))))
            if target != self.state.replicas and t >= self._pending_until:
                self._pending_target = target
                self._pending_until = t + self.react_delay_s
        return self._maybe_apply(t, api_available)


class KEDAController(_Base):
    """KEDA Prometheus scaler on RPS, faster polling than HPA."""
    name = "KEDA"
    react_delay_s = 11.5
    target_rps_per_replica = 250.0

    def step(self, t, rps, cpu, api_available):
        if t - self._last_check >= 5.0:
            self._last_check = t
            target = max(self.n_min,
                         min(self.n_max,
                             max(1, round(rps / self.target_rps_per_replica))))
            if target != self.state.replicas and t >= self._pending_until:
                self._pending_target = target
                self._pending_until = t + self.react_delay_s
        return self._maybe_apply(t, api_available)


class IstioHPAController(_Base):
    """Istio outlier detection + HPA. Slightly faster than HPA alone."""
    name = "Istio+HPA"
    react_delay_s = 9.8
    target_cpu = 0.55

    def step(self, t, rps, cpu, api_available):
        if t - self._last_check >= 10.0:
            self._last_check = t
            ratio = cpu / self.target_cpu
            target = max(self.n_min,
                         min(self.n_max,
                             max(1, round(self.state.replicas * ratio))))
            if target != self.state.replicas and t >= self._pending_until:
                self._pending_target = target
                self._pending_until = t + self.react_delay_s
        return self._maybe_apply(t, api_available)