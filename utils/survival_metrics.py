"""Metrics for right-censored survival outcomes (higher score means higher risk)."""

import numpy as np


def validate_outcomes(times, events):
    """Validate finite nonnegative follow-up times and binary observed events."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    if times.ndim != 1 or events.ndim != 1 or times.shape != events.shape:
        raise ValueError("Times and events must be equally sized one-dimensional arrays.")
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Follow-up times must be finite and nonnegative.")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("Events must be 1 (observed event) or 0 (right censored).")
    return times, events.astype(bool)


def concordance_index(times, risks, events, tied_tol=1e-8):
    """Return Harrell's C-index; tied risks earn 0.5 and no pairs yields NaN.

    A pair is comparable when the earlier time is an observed event. At equal
    times an event and a censoring are comparable; two events are not. Higher
    risk predicts shorter survival. This is a discrimination metric, not a
    calibrated survival probability or an IPCW/Uno estimator.
    """
    times, events = validate_outcomes(times, events)
    risks = np.asarray(risks, dtype=float)
    if risks.shape != times.shape or not np.isfinite(risks).all():
        raise ValueError("Risks must be a finite vector matching the outcomes.")
    if tied_tol < 0:
        raise ValueError("tied_tol must be nonnegative.")
    concordant = comparable = 0
    for index in np.flatnonzero(events):
        mask = (times > times[index]) | ((times == times[index]) & ~events)
        differences = risks[index] - risks[mask]
        comparable += len(differences)
        concordant += np.sum(differences > tied_tol) + 0.5 * np.sum(np.abs(differences) <= tied_tol)
    return float(concordant / comparable) if comparable else float("nan")
