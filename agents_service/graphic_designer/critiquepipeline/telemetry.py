"""Cost + time telemetry for the critique pipeline.

Thread-safe tracker: every LLM/image call is wrapped in ``tele.span(phase, model)``;
at the end ``tele.summary()`` prints a per-phase table with tokens / cost / seconds,
plus totals and the parallel speed-up, and returns ``{"cost", "wall_s"}`` so the value
can be surfaced to the frontend via state.

Mirrors the ``@profile`` style already used in ``nodes.py`` but aggregates across the
whole critique instead of one node.
"""

import time
import threading
from contextlib import contextmanager


class Telemetry:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.rows = []                       # list of (phase, model, tokens, cost, seconds)
        self.notes = {}                      # non-timing diagnostics (see .note)
        self._lock = threading.Lock()

    @contextmanager
    def span(self, phase, model):
        """Time a call. The caller fills ``box['tokens']`` and ``box['cost']``.

        Usage::

            with tele.span("integrity", model_name) as box:
                resp = model.invoke(...)
                box["tokens"] = resp.usage_metadata["total_tokens"]
                box["cost"]   = calculateCost(model_name, box["tokens"])
        """
        start = time.perf_counter()
        box = {"tokens": 0, "cost": 0.0}
        try:
            yield box
        finally:
            with self._lock:
                self.rows.append(
                    (phase, model, box["tokens"], box["cost"], time.perf_counter() - start)
                )

    def record(self, phase, model, tokens, cost, seconds):
        """Record a call that was timed elsewhere (e.g. a reused external agent)."""
        with self._lock:
            self.rows.append((phase, model, tokens, cost, seconds))

    def note(self, key, value):
        """Attach a non-timing diagnostic to the job record.

        Used for critique observability — e.g. ``pre_arbiter_findings`` (what every lens actually
        produced, before curation) and ``curation_dropped`` (what the quality floor removed).
        Together these answer the question a final suggestion list cannot: when an obvious defect
        is missing from the output, was it never detected, or was it detected and then dropped?
        """
        with self._lock:
            self.notes[key] = value

    def summary(self, extra_cost=0.0, do_print=True):
        wall = time.perf_counter() - self.t0
        billed = sum(r[3] for r in self.rows) + extra_cost
        serial = sum(r[4] for r in self.rows)
        tokens = sum(r[2] for r in self.rows)
        if do_print:
            print("\n-- Critique telemetry -------------------------------------------")
            print(f"{'phase':<24}{'model':<24}{'tokens':>8}{'cost$':>10}{'sec':>8}")
            for p, m, tk, c, s in self.rows:
                print(f"{p:<24}{m:<24}{tk:>8}{c:>10.4f}{s:>8.2f}")
            print("-----------------------------------------------------------------")
            print(f"{'TOTAL billed cost ($)':<56}{billed:>10.4f}")
            print(f"{'wall-clock time (s)':<56}{wall:>10.2f}")
            print(f"{'serial call time (s)':<56}{serial:>10.2f}")
            print(f"{'parallel speedup':<56}{(serial / wall if wall else 1):>9.2f}x")
        # Compact job record for the DB (totals + per-phase breakdown + diagnostics).
        return {
            "cost": round(billed, 6),
            "wall_s": round(wall, 2),
            "tokens": tokens,
            "phases": [{"phase": p, "model": m, "tokens": tk, "cost": round(c, 6), "sec": round(s, 2)}
                       for p, m, tk, c, s in self.rows],
            "notes": dict(self.notes),
        }
