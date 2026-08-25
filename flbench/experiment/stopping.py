from __future__ import annotations

from collections import deque


class StoppingController:
    """Decide when an experiment should terminate.

    Convergence runs use a *smoothed validation metric* so that the normal
    oscillations seen in non-IID federated learning do not continually reset
    patience.  Importantly, smoothing is used **only for deciding when to
    stop**.  Model selection still uses the best *raw* validation accuracy.

    The convergence rule has four main controls:

    ``smoothing_window``
        Number of consecutive validation observations used in the moving
        average.  A value of 3 is the default for the controlled search.

    ``min_delta``
        Minimum increase in the *smoothed* validation metric that is treated
        as meaningful progress.  Small fluctuations below this value count as
        part of the plateau.

    ``patience_evaluations``
        Number of consecutive validation evaluations without meaningful
        smoothed improvement before convergence is declared.

    ``min_rounds``
        Minimum communication rounds that must be completed before the
        patience rule is allowed to terminate training.

    ``federation.communication_rounds`` remains a hard safety ceiling.  Thus,
    even a pathological run whose metric never settles cannot loop forever.
    """

    def __init__(self, cfg):
        self.cfg = cfg["stopping"]

        # Best *smoothed* score controls convergence/patience.
        self.best = float("-inf")
        self.best_round = 0
        self.wait = 0
        self.num_observations = 0
        self.last_improved = False

        # Best *raw* score controls checkpoint/model selection.  Keeping this
        # separate prevents smoothing from changing which model is ultimately
        # reported as the best observed validation checkpoint.
        self.raw_best = float("-inf")
        self.raw_best_round = 0
        self.raw_last_improved = False

        self.smoothing_window = max(
            1, int(self.cfg.get("smoothing_window", 1))
        )
        self.metric_history = deque(maxlen=self.smoothing_window)
        self.current_smoothed = None

    def update(self, round_idx, metric, elapsed, max_rounds):
        """Return ``(stop, reason)`` after one communication round."""
        mode = str(self.cfg["mode"]).lower()

        if mode == "fixed_rounds":
            reached = round_idx >= max_rounds
            return reached, "fixed_rounds" if reached else None

        if mode == "target_accuracy":
            if metric is None:
                return False, None
            reached = metric >= float(self.cfg["target_accuracy"])
            if reached:
                return True, "target_accuracy"
            return (
                (True, "max_rounds")
                if round_idx >= max_rounds
                else (False, None)
            )

        if mode == "runtime_budget":
            reached = elapsed >= float(self.cfg["runtime_budget_sec"])
            if reached:
                return True, "runtime_budget"
            return (
                (True, "max_rounds")
                if round_idx >= max_rounds
                else (False, None)
            )

        if mode != "convergence":
            raise ValueError(f"Unsupported stopping mode: {mode}")

        # Patience is counted only at validation events.  Communication rounds
        # without validation must not consume patience.
        self.last_improved = False
        self.raw_last_improved = False
        if metric is None:
            return False, None

        metric = float(metric)
        self.num_observations += 1

        # Track the raw best independently.  The experiment runner uses this
        # flag to save ``best_validation_model.pt``.
        if metric > self.raw_best:
            self.raw_best = metric
            self.raw_best_round = int(round_idx)
            self.raw_last_improved = True

        # Add the newest raw observation to the moving window.  For the first
        # few evaluations, before a full window exists, we intentionally do
        # not consume patience.  This gives the smoothing statistic time to
        # become representative rather than comparing partial windows.
        self.metric_history.append(metric)
        self.current_smoothed = sum(self.metric_history) / len(self.metric_history)

        if len(self.metric_history) < self.smoothing_window:
            # Still enforce the safety ceiling even during warm-up.
            if round_idx >= max_rounds:
                return True, "max_rounds"
            return False, None

        min_delta = float(
            self.cfg.get(
                "min_delta",
                self.cfg.get("convergence_tolerance", 1e-3),
            )
        )
        patience = int(
            self.cfg.get(
                "patience_evaluations",
                self.cfg.get("patience", 10),
            )
        )
        min_rounds = int(self.cfg.get("min_rounds", 0))

        # Only a sufficiently large increase in the moving-average validation
        # score resets patience.  Small alternating up/down changes therefore
        # count as a plateau instead of keeping the run alive indefinitely.
        if self.current_smoothed > self.best + min_delta:
            self.best = float(self.current_smoothed)
            self.best_round = int(round_idx)
            self.wait = 0
            self.last_improved = True
        else:
            self.wait += 1

        converged = round_idx >= min_rounds and self.wait >= patience
        if converged:
            return True, "convergence"

        # The maximum-round value is only a safety ceiling, but it guarantees
        # finite execution even if a configuration never satisfies the plateau
        # criterion.
        if round_idx >= max_rounds:
            return True, "max_rounds"
        return False, None

    def state_dict(self):
        """State saved in checkpoints so resumed runs preserve convergence."""
        return {
            "best": self.best,
            "best_round": self.best_round,
            "wait": self.wait,
            "num_observations": self.num_observations,
            "raw_best": self.raw_best,
            "raw_best_round": self.raw_best_round,
            "smoothing_window": self.smoothing_window,
            "metric_history": list(self.metric_history),
            "current_smoothed": self.current_smoothed,
        }

    def load_state_dict(self, state):
        self.best = float(state.get("best", float("-inf")))
        self.best_round = int(state.get("best_round", 0))
        self.wait = int(state.get("wait", 0))
        self.num_observations = int(state.get("num_observations", 0))
        self.raw_best = float(state.get("raw_best", float("-inf")))
        self.raw_best_round = int(state.get("raw_best_round", 0))

        # Prefer the current configuration's window size, while restoring as
        # much recent history as fits.  This is robust to older checkpoints
        # that pre-date smoothing and contain no history at all.
        self.smoothing_window = max(
            1, int(self.cfg.get("smoothing_window", state.get("smoothing_window", 1)))
        )
        history = list(state.get("metric_history", []))[-self.smoothing_window :]
        self.metric_history = deque(history, maxlen=self.smoothing_window)
        self.current_smoothed = state.get("current_smoothed")
        if self.current_smoothed is None and self.metric_history:
            self.current_smoothed = sum(self.metric_history) / len(self.metric_history)

        self.last_improved = False
        self.raw_last_improved = False
