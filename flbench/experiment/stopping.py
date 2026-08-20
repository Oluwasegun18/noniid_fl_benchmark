from __future__ import annotations


class StoppingController:
    """Decide when an experiment should terminate.

    For controlled-search and confirmation runs we use validation-accuracy
    convergence.  Patience is counted in *evaluation events* rather than raw
    communication rounds, because validation may be evaluated only every N
    rounds to reduce overhead.

    The controller deliberately separates:
      * ``min_delta``: smallest validation improvement considered observable;
      * ``patience_evaluations``: number of consecutive evaluations without
        such improvement before declaring convergence; and
      * ``min_rounds``: minimum amount of training before convergence is
        allowed to stop the run.

    ``federation.communication_rounds`` remains a safety ceiling so a run
    cannot continue forever when the validation metric keeps fluctuating.
    """

    def __init__(self, cfg):
        self.cfg = cfg["stopping"]
        self.best = float("-inf")
        self.best_round = 0
        self.wait = 0
        self.num_observations = 0
        self.last_improved = False

    def update(self, round_idx, metric, elapsed, max_rounds):
        """Return ``(stop, reason)`` after one communication round."""
        mode = str(self.cfg["mode"]).lower()

        # Fixed-round runs stop exactly at the configured round ceiling.
        if mode == "fixed_rounds":
            reached = round_idx >= max_rounds
            return reached, "fixed_rounds" if reached else None

        if mode == "target_accuracy":
            if metric is None:
                return False, None
            reached = metric >= float(self.cfg["target_accuracy"])
            if reached:
                return True, "target_accuracy"
            return (True, "max_rounds") if round_idx >= max_rounds else (False, None)

        if mode == "runtime_budget":
            reached = elapsed >= float(self.cfg["runtime_budget_sec"])
            if reached:
                return True, "runtime_budget"
            return (True, "max_rounds") if round_idx >= max_rounds else (False, None)

        if mode != "convergence":
            raise ValueError(f"Unsupported stopping mode: {mode}")

        # No new validation observation on this round.  Do not increment
        # patience because convergence is defined over observed validation
        # measurements, not over unevaluated communication rounds.
        self.last_improved = False
        if metric is None:
            return False, None

        self.num_observations += 1
        min_delta = float(
            self.cfg.get(
                "min_delta",
                self.cfg.get("convergence_tolerance", 1e-4),
            )
        )
        patience = int(
            self.cfg.get(
                "patience_evaluations",
                self.cfg.get("patience", 10),
            )
        )
        min_rounds = int(self.cfg.get("min_rounds", 0))

        # Improvement must exceed min_delta.  Tiny metric fluctuations are
        # therefore treated as a plateau rather than meaningful progress.
        if metric > self.best + min_delta:
            self.best = float(metric)
            self.best_round = int(round_idx)
            self.wait = 0
            self.last_improved = True
        else:
            self.wait += 1

        converged = round_idx >= min_rounds and self.wait >= patience
        if converged:
            return True, "convergence"

        # For convergence runs, process the final validation observation first
        # (so a new best model at the safety ceiling is not lost), then stop.
        if round_idx >= max_rounds:
            return True, "max_rounds"
        return False, None

    def state_dict(self):
        """State saved in checkpoints so resumed runs preserve patience."""
        return {
            "best": self.best,
            "best_round": self.best_round,
            "wait": self.wait,
            "num_observations": self.num_observations,
        }

    def load_state_dict(self, state):
        self.best = float(state.get("best", float("-inf")))
        self.best_round = int(state.get("best_round", 0))
        self.wait = int(state.get("wait", 0))
        self.num_observations = int(state.get("num_observations", 0))
        self.last_improved = False
