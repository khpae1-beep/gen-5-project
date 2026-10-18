"""Side-by-side comparison of two runs of an experiment."""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm

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


# --- "Explain the difference": an LLM explanation, fact-checked, with a code-written fallback.

MAX_EXPLANATION_LENGTH = 1200
NUMBER = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?')
PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent|percentage points?)', re.IGNORECASE)
ARROW_PAIR = re.compile(r'([^\s,;:→()]+)\s*→\s*([^\s,;:→()]+)')
FROM_TO_PAIR = re.compile(r'\bfrom\s+(None|-?\d+(?:\.\d+)?)\s+to\s+(None|-?\d+(?:\.\d+)?)\b', re.IGNORECASE)
CLAUSE_BREAK = re.compile(r'(?<=[.!?])\s+|;|,?\s+\b(?:but|while|whereas|however|although)\b', re.IGNORECASE)
METRIC_MENTION = re.compile(
    r'\b(?:(train(?:ing)?|test(?:ing)?)\s+)?(accuracy|precision|recall|f1)(?:[\s-]score)?\b', re.IGNORECASE
)
UP_WORDS = re.compile(r'\b(?:increas\w*|improv\w*|rose|rise[sn]?|rising|higher|better|gain\w*|grew|grow\w*|up)\b', re.IGNORECASE)
DOWN_WORDS = re.compile(r'\b(?:decreas\w*|drop\w*|fell|fall\w*|lower\w*|worse|declin\w*|down|reduc\w*)\b', re.IGNORECASE)
SAME_WORDS = re.compile(r'\b(?:unchanged|same|stayed|remained|steady|no change)\b', re.IGNORECASE)
SPLIT_WORD = re.compile(r'\b(train(?:ing)?|test(?:ing)?)\b', re.IGNORECASE)
GAP_WORDS = re.compile(r'\boverfit\w*|\bgap\b', re.IGNORECASE)
LESS_WORDS = re.compile(r'\b(?:less|narrow\w*|smaller|shrank|shrunk|reduc\w*|decreas\w*|lower)\b', re.IGNORECASE)
MORE_WORDS = re.compile(r'\b(?:more|wid\w*|larger|bigger|grew|increas\w*|higher)\b', re.IGNORECASE)

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    (
        'system',
        'You explain the difference between two versions of a machine learning model to a non-expert. '
        'Reply with 2 to 4 plain sentences: no lists, no headings, no quotes.',
    ),
    (
        'human',
        '{details}\n\n'
        'Explain what changed between v{a} and v{b} and how it affected performance, including '
        'overfitting (the gap between train and test accuracy). Use only the numbers given above, '
        'exactly as written, and do not say a value went up or down unless the data above shows it.',
    ),
])


class ExplanationCheckError(ValueError):
    """The LLM's explanation contradicts the real numbers."""


def accuracy_gap(run):
    train, test = run.metrics.get('train_accuracy'), run.metrics.get('test_accuracy')
    return train - test if is_number(train) and is_number(test) else None


def signed(value, places=3):
    text = format_value(round(value, places)) if isinstance(value, float) else str(value)
    return text if text.startswith('-') else f'+{text}'


def describe_comparison(run_a, run_b, comparison):
    """The facts the LLM gets, with numbers already formatted."""
    lines = [f'Comparing {run_a.experiment.name} v{run_a.version} (before) with v{run_b.version} (after).']
    param_changes = [row for row in comparison['params'] if row['differs']]
    lines.append('Parameter changes:')
    lines += [
        f'- {row["key"]}: {row["a"]} → {row["b"]}' + (f' ({signed(row["delta"])})' if row['delta'] is not None else '')
        for row in param_changes
    ] or ['- none']
    lines.append('Metrics (before → after):')
    for row in comparison['metrics']:
        if row['differs']:
            how = '' if row['delta'] is None else f' ({signed(row["delta"])}, {"higher" if row["delta"] > 0 else "lower"})'
            lines.append(f'- {row["name"]}: {row["a"]} → {row["b"]}{how}')
        else:
            lines.append(f'- {row["name"]}: {row["a"]} (unchanged)')
    gap_a, gap_b = accuracy_gap(run_a), accuracy_gap(run_b)
    if gap_a is not None and gap_b is not None:
        trend = 'smaller, less overfitting' if gap_b < gap_a else 'larger, more overfitting' if gap_b > gap_a else 'unchanged'
        lines.append(
            f'Train minus test accuracy gap: {format_value(gap_a)} → {format_value(gap_b)} '
            f'({signed(gap_b - gap_a)}, {trend}).'
        )
    return '\n'.join(lines)


def fallback_explanation(run_a, run_b, comparison):
    """A plain-code explanation of the difference."""
    a, b = f'v{run_a.version}', f'v{run_b.version}'
    if run_a.pk == run_b.pk:
        return f'{a} and {b} are the same run.'

    param_changes = [row for row in comparison['params'] if row['differs']]
    if param_changes:
        changes = [f'{row["key"]} from {row["a"]} to {row["b"]}' for row in param_changes]
        sentences = [f'{b} changes {", ".join(changes[:-1]) + " and " if len(changes) > 1 else ""}{changes[-1]} compared with {a}.']
    else:
        sentences = [f'{b} uses the same parameters as {a}.']

    metrics = [row for row in comparison['metrics'] if row['key'].startswith('test_')] or comparison['metrics']
    parts = []
    for row in metrics:
        if not row['differs']:
            parts.append(f'{row["name"]} stayed at {row["a"]}')
        elif row['delta'] is None:
            parts.append(f'{row["name"]} changed from {row["a"]} to {row["b"]}')
        else:
            verb = 'rose' if row['delta'] > 0 else 'fell'
            parts.append(f'{row["name"]} {verb} from {row["a"]} to {row["b"]} ({signed(row["delta"])})')
    if parts:
        sentences.append(parts[0][0].upper() + '; '.join(parts)[1:] + '.')

    gap_a, gap_b = accuracy_gap(run_a), accuracy_gap(run_b)
    if gap_a is not None and gap_b is not None:
        if gap_b < gap_a:
            sentences.append(
                f'The gap between train and test accuracy narrowed from {format_value(gap_a)} to '
                f'{format_value(gap_b)}, so {b} overfits less.'
            )
        elif gap_b > gap_a:
            sentences.append(
                f'The gap between train and test accuracy widened from {format_value(gap_a)} to '
                f'{format_value(gap_b)}, so {b} overfits more.'
            )
        else:
            sentences.append(f'The gap between train and test accuracy stayed at {format_value(gap_a)}.')
    return ' '.join(sentences)


def check_explanation(text, details, comparison, run_a, run_b):
    """Raise ExplanationCheckError if text states numbers, changes or directions not in the data."""
    allowed = {float(n) for n in NUMBER.findall(details)}
    allowed |= {abs(n) for n in allowed}
    allowed_percent = {round(abs(n) * 100, 1) for n in allowed if abs(n) <= 1}
    percents = {m.span() for m in PERCENT.finditer(text)}
    wrong = [m.group(1) + '%' for m in PERCENT.finditer(text) if round(float(m.group(1)), 1) not in allowed_percent]
    for match in NUMBER.finditer(text):
        inside_percent = any(start <= match.start() < end for start, end in percents)
        version_number = text[max(0, match.start() - 1):match.start()].lower() == 'v'
        if not inside_percent and not version_number and float(match.group()) not in allowed:
            wrong.append(match.group())
    if wrong:
        raise ExplanationCheckError(f'it mentions numbers that are not in the data: {", ".join(dict.fromkeys(wrong))}')

    real_pairs = set(ARROW_PAIR.findall(details))
    for old, new in ARROW_PAIR.findall(text) + FROM_TO_PAIR.findall(text):
        if (old, new) not in real_pairs and not any(
            is_float_pair(old, new, real_old, real_new) for real_old, real_new in real_pairs
        ):
            raise ExplanationCheckError(f'it describes a change {old}→{new} that is not in the data')

    deltas = {row['key']: (row['delta'], row['differs']) for row in comparison['metrics']}
    gap_a, gap_b = accuracy_gap(run_a), accuracy_gap(run_b)
    for clause in CLAUSE_BREAK.split(text):
        if not clause:
            continue
        if GAP_WORDS.search(clause):
            # "The gap between train and test accuracy decreased" is about the gap, checked below.
            check_gap_claim(clause, gap_a, gap_b)
            continue
        up, down, same = UP_WORDS.search(clause), DOWN_WORDS.search(clause), SAME_WORDS.search(clause)
        for mention in METRIC_MENTION.finditer(clause):
            prefix, name = mention.group(1) or '', mention.group(2)
            if not prefix:
                # "performance on the training data fell, with lower accuracy" is about train accuracy.
                earlier = SPLIT_WORD.findall(clause[:mention.start()])
                prefix = earlier[-1] if earlier else ''
            key = metric_key(prefix, name.lower(), deltas)
            if key is None:
                continue
            delta, differs = deltas[key]
            label = metric_label(key)
            if up and not down and not same and not (delta is not None and delta > 0):
                raise ExplanationCheckError(f'it says {label} went up, but it did not')
            if down and not up and not same and not (delta is not None and delta < 0):
                raise ExplanationCheckError(f'it says {label} went down, but it did not')
            if same and not up and not down and differs:
                raise ExplanationCheckError(f'it says {label} stayed the same, but it changed')


def check_gap_claim(clause, gap_a, gap_b):
    """A clause about overfitting or the train-test gap must match how the gap changed."""
    if gap_a is None or gap_b is None:
        return
    less, more = LESS_WORDS.search(clause), MORE_WORDS.search(clause)
    if less and not more and not gap_b < gap_a:
        raise ExplanationCheckError('it says overfitting decreased, but the train-test gap did not shrink')
    if more and not less and not gap_b > gap_a:
        raise ExplanationCheckError('it says overfitting increased, but the train-test gap did not grow')


def is_float_pair(old, new, real_old, real_new):
    try:
        return float(old) == float(real_old) and float(new) == float(real_new)
    except ValueError:
        return False


def metric_key(prefix, name, deltas):
    """The metric a mention like "test accuracy" or plain "accuracy" refers to, if it was compared."""
    split = 'train' if prefix.lower().startswith('train') else 'test' if prefix else None
    candidates = [f'{split}_{name}'] if split else [f'test_{name}', name]
    return next((key for key in candidates if key in deltas), None)


def explain_difference(run_a, run_b, llm=None):
    """Explain how run_b differs from run_a; returns (text, whether the LLM wrote it, why not).

    Uses get_llm(temperature=0) unless an llm is passed. The LLM's text is only used if it passes
    check_explanation(); otherwise, or if the LLM fails, fallback_explanation() is used.
    """
    comparison = compare_runs(run_a, run_b)
    if run_a.pk == run_b.pk:
        return fallback_explanation(run_a, run_b, comparison), False, 'both sides are the same run'

    details = describe_comparison(run_a, run_b, comparison)
    try:
        chain = EXPLAIN_PROMPT | (llm or get_llm(temperature=0)) | StrOutputParser()
        text = ' '.join(chain.invoke({'details': details, 'a': run_a.version, 'b': run_b.version}).split())
        if not text:
            raise ExplanationCheckError('the LLM returned an empty reply')
        if len(text) > MAX_EXPLANATION_LENGTH:
            raise ExplanationCheckError('the LLM reply is too long')
        check_explanation(text, details, comparison, run_a, run_b)
        return text, True, ''
    except ExplanationCheckError as exc:
        reason = f'the LLM explanation failed the fact check: {exc}'
    except Exception as exc:  # Any provider failure (e.g. Ollama not running) falls back to plain code.
        reason = f'the LLM was unavailable ({type(exc).__name__})'
    return fallback_explanation(run_a, run_b, comparison), False, reason
