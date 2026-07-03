from verl.utils.metric.utils import Metric
from verl.workers.engine_workers import _normalize_gathered_metric_values


def test_normalize_gathered_metric_values_keeps_scalar_lists():
    assert _normalize_gathered_metric_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_normalize_gathered_metric_values_flattens_nested_lists():
    assert _normalize_gathered_metric_values([[1.0], [2.0, 3.0]]) == [1.0, 2.0, 3.0]


def test_normalize_gathered_metric_values_aggregates_metric_lists():
    assert _normalize_gathered_metric_values(
        [
            Metric(aggregation="mean", value=1.0),
            Metric(aggregation="mean", value=3.0),
        ]
    ) == 2.0
