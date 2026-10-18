"""Side-by-side comparison of two runs of an experiment."""
from .versioning import format_value, metric_label

# Test metrics where a higher value is better. Train metrics get no better/worse colouring: lower
# train accuracy with higher test accuracy means less overfitting, which is good.
HIGHER_IS_BETTER = ('accuracy', 'precision', 'recall', 'f1')


def test_metric_higher_is_better(name):
    return not name.startswith('train_') and name.rsplit('_', 1)[-1] in HIGHER_IS_BETTER


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_values(values_a, values_b, label=str, higher_is_better=lambda name: False):
    """Rows of name, both values, delta (b - a, if numeric), whether they differ and better/worse."""
    rows = []
    for name in dict.fromkeys([*values_a, *values_b]):
        a, b = values_a.get(name), values_b.get(name)
        delta = b - a if is_number(a) and is_number(b) else None
        direction = ''
        if delta and higher_is_better(name):
            direction = 'up' if delta > 0 else 'down'
        rows.append({
            'name': label(name),
            'key': name,
            'a': format_value(a) if name in values_a else '–',
            'b': format_value(b) if name in values_b else '–',
            'delta': delta,
            'differs': (name in values_a) != (name in values_b) or a != b,
            'direction': direction,
        })
    return rows


def compare_runs(run_a, run_b):
    """{'params': rows, 'metrics': rows} comparing run_a (before) with run_b (after)."""
    return {
        'params': compare_values(run_a.params, run_b.params),
        'metrics': compare_values(
            run_a.metrics, run_b.metrics, label=metric_label,
            higher_is_better=test_metric_higher_is_better,
        ),
    }
