"""SANO end-to-end Python implementation.

This package mirrors the paper's pipeline:
  - sano_py.agent       MAPE-K loop, EWMA, TARS planner, knowledge store
  - sano_py.gossip      gossip-based peer coordination (simulated)
  - sano_py.cluster     simulated Kubernetes cluster + workload + chaos
  - sano_py.baselines   HPA / KEDA / Istio+HPA controllers
  - sano_py.experiment  scenario runner, 30 repetitions, CSV output
  - sano_py.stats       Mann-Whitney U, Cliff's delta
  - sano_py.figures     produces paper/figures/*.pdf

Entry point: python -m sano_py  (see sano_py/__main__.py).
"""
from __future__ import annotations
__version__ = "0.1.0"