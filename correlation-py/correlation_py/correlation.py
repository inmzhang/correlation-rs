from typing import Iterable, Optional

import numpy as np

from ._internal import cal_2nd_order_correlation_rs, cal_high_order_correlations_rs
from correlation_py.result import CorrelationResult
from correlation_py.utils import HyperEdge


def cal_2nd_order_correlations(detection_events: np.ndarray) -> CorrelationResult:
    """Calculate the 2nd order correlation analytically.

    Args:
        detection_events: The detection events.

    Returns:
        The correlation result.
    """
    detection_events = detection_events.astype(np.float64)
    data = cal_2nd_order_correlation_rs(detection_events)
    return CorrelationResult(data)


def cal_high_order_correlations(
    detection_events: np.ndarray,
    hyperedges: Optional[Iterable[HyperEdge]] = None,
    num_threads: Optional[int] = None,
    max_iters: Optional[int] = None,
) -> CorrelationResult:
    """Calculate the high order correlation numerically.

    Args:
        detection_events: The detection events.
        hyperedges: The hyperedges to take into account, excluding the 1st and 2nd
            order edges.
        num_threads: The number of threads to use during solving hyperedge clusters.
        max_iters: The maximum number of iterations to use during solving hyperedge
            clusters.

    Returns:
        The correlation result.
    """
    detection_events = detection_events.astype(np.float64)
    hyperedges_pass = hyperedges and [list(t) for t in hyperedges]
    results = cal_high_order_correlations_rs(
        detection_events, hyperedges_pass, num_threads, max_iters
    )
    return CorrelationResult({res[0]: res[1] for res in results})
