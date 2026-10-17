import re

from core.llm import get_llm

from .metrics import anomaly_reasons, baseline_accuracy
from .models import MetricSnapshot

RECENT_WINDOWS = 10

# For the fact check of the LLM's summary.
CLAUSE_BREAK = re.compile(r'(?<=[.!?])\s+|;|,?\s+\b(?:but|while|whereas|however|although|unlike)\b', re.IGNORECASE)
ANOMALY_WORD = re.compile(r'anomal|flagged', re.IGNORECASE)
# "These are anomalies" refers to the windows named just before.
BACK_REFERENCE = re.compile(r'\b(?:this|these|that|those|they|both|it|which|each)\b', re.IGNORECASE)
NEGATION = re.compile(r"\b(?:not|no|never|none|neither|nor|without)\b|n't\b", re.IGNORECASE)
WINDOW_LIST = re.compile(
    r'\bwindows?\s+\d+(?:\s*(?:,|and|&|to|through|-|–)\s*(?:windows?\s+)?\d+)*', re.IGNORECASE
)
NUMBER = re.compile(r'\d+(?:\.\d+)?')
DECIMAL = re.compile(r'(?<![\d.])\d+\.\d+')
PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*%')
MILLISECONDS = re.compile(r'(\d+(?:\.\d+)?)\s*ms\b')


class FactCheckError(ValueError):
    """The LLM's summary contradicts the data it was given."""


def _format_accuracy(accuracy):
    return 'n/a (no actual values)' if accuracy is None else f'{accuracy:.2f}'


def build_prompt(model_name, windows, baseline):
    """windows is a list of (window number, snapshot, anomaly reasons), oldest first."""
    first, latest = windows[0][0], windows[-1][0]
    lines = [
        f'Window {number}{" (LATEST)" if number == latest else ""} ({snapshot.count} predictions): '
        f'accuracy {_format_accuracy(snapshot.accuracy)}, average latency {snapshot.avg_latency_ms:.1f} ms'
        + (f', ANOMALY: {"; ".join(reasons)}' if reasons else '')
        for number, snapshot, reasons in windows
    ]
    return (
        f'You are monitoring the machine learning model "{model_name}". Its expected accuracy (baseline) is '
        f'{baseline:.2f}. A window is anomalous when accuracy is more than 0.10 below the baseline or average '
        'latency is more than double the median of the earlier windows.\n\n'
        f'These are windows {first} to {latest}, listed in time order from oldest to newest. Higher window '
        f'numbers are more recent. Window {latest} is the latest (current) window; every other window is '
        'earlier.\n\n'
        + '\n'.join(lines)
        + '\n\nIn 3 to 5 plain-English sentences for a non-expert, explain how the model is performing, the '
        'anomalies marked ANOMALY above and what they might mean. Refer to windows by their numbers. When you '
        f'describe current or latest performance, use window {latest} only. Mention every window marked ANOMALY '
        'by its number together with the word "anomaly", and do not call any other window an anomaly; if none '
        'are marked, say there are no anomalies. Use the numbers exactly as given. Do not use bullet points or '
        'headings.'
    )


def latest_sentence(windows):
    """The code-written opening sentence about the latest window."""
    number, snapshot, reasons = windows[-1]
    flagged = f'flagged as an anomaly ({"; ".join(reasons)})' if reasons else 'not flagged as an anomaly'
    return (
        f'Window {number} is the latest: accuracy {_format_accuracy(snapshot.accuracy)}, '
        f'average latency {snapshot.avg_latency_ms:.1f} ms, {flagged}.'
    )


def window_numbers(text):
    """Window numbers mentioned in text, e.g. "windows 4 and 5" or "windows 4 to 6"."""
    numbers = set()
    for mention in WINDOW_LIST.finditer(text):
        tokens = re.findall(r'\d+|to|through|-|–', mention.group(), re.IGNORECASE)
        previous, in_range = None, False
        for token in tokens:
            if token.isdigit():
                current = int(token)
                numbers.update(range(previous, current + 1) if in_range and previous is not None else [current])
                previous, in_range = current, False
            else:
                in_range = True
    return numbers


def fact_check(text, windows, prompt, baseline):
    """Raise FactCheckError if text gets the anomalies or the numbers wrong.

    A clause that mentions windows together with "anomaly", "anomalous" or "flagged" claims those
    windows are anomalies, or with a negation ("not", "no", ...) that they are not. A clause like
    "These are anomalies" refers to the windows named just before it. Every number with a decimal
    point, % or ms must match the data sent in the prompt, or a window's gap from the baseline.
    The check errs towards rejecting, since the rule-based summary is always correct.
    """
    flagged = {number for number, _, reasons in windows if reasons}
    claimed, denied = set(), set()
    previous = set()
    for clause in CLAUSE_BREAK.split(text):
        if not clause:
            continue
        numbers = window_numbers(clause)
        if ANOMALY_WORD.search(clause):
            if not numbers and BACK_REFERENCE.search(clause):
                numbers = previous
            (denied if NEGATION.search(clause) else claimed).update(numbers)
        if numbers:
            previous = numbers

    def windows_text(numbers):
        numbers = [str(n) for n in sorted(numbers)]
        if len(numbers) == 1:
            return f'window {numbers[0]}'
        return f'windows {", ".join(numbers[:-1])} and {numbers[-1]}'

    def one_or_many(numbers, one, many):
        return one if len(numbers) == 1 else many

    if claimed - flagged:
        wrong = claimed - flagged
        raise FactCheckError(
            f'it calls {windows_text(wrong)} {one_or_many(wrong, "an anomaly", "anomalies")}, '
            f'but {one_or_many(wrong, "it was", "they were")} not flagged'
        )
    if denied & flagged:
        wrong = denied & flagged
        raise FactCheckError(
            f'it says flagged {windows_text(wrong)} {one_or_many(wrong, "is not an anomaly", "are not anomalies")}'
        )
    if flagged - claimed:
        missing = flagged - claimed
        raise FactCheckError(f'it does not mention flagged {windows_text(missing)} as anomalous')

    allowed = {float(n) for n in NUMBER.findall(prompt)}
    allowed |= {round(abs(baseline - s.accuracy), 2) for _, s, _ in windows if s.accuracy is not None}
    allowed_percent = {round(value * 100, 1) for value in allowed if value <= 1}
    wrong = [n for n in DECIMAL.findall(text) + MILLISECONDS.findall(text) if float(n) not in allowed]
    wrong += [f'{n}%' for n in PERCENT.findall(text) if round(float(n), 1) not in allowed_percent]
    if wrong:
        raise FactCheckError(f'it mentions numbers that do not match the data: {", ".join(dict.fromkeys(wrong))}')


def rule_based_summary(model_name, windows, baseline):
    snapshots = [snapshot for _, snapshot, _ in windows]
    total = sum(s.count for s in snapshots)
    labelled = [s.accuracy for s in snapshots if s.accuracy is not None]
    accuracy = f'{sum(labelled) / len(labelled):.2f}' if labelled else 'unknown'
    latency = sum(s.avg_latency_ms for s in snapshots) / len(snapshots)

    parts = [
        f'Over the last {len(snapshots)} windows ({total} predictions), {model_name} averaged accuracy '
        f'{accuracy} against a {baseline:.2f} baseline, with {latency:.1f} ms average latency.',
    ]
    anomalous = [(number, reasons) for number, _, reasons in windows if reasons]
    if anomalous:
        parts.append(
            f'{len(anomalous)} of these windows were anomalous: '
            + '; '.join(f'window {number}: {", ".join(reasons)}' for number, reasons in anomalous)
            + '.'
        )
    else:
        parts.append('No anomalies were detected.')
    return ' '.join(parts)


def _ask_llm(llm, prompt):
    response = llm.invoke(prompt)
    content = getattr(response, 'content', response)
    if not isinstance(content, str):
        content = ' '.join(part if isinstance(part, str) else part.get('text', '') for part in content)
    content = content.strip()
    if not content:
        raise ValueError('The LLM returned an empty response.')
    return content


def summarize_model(model_name, llm=None, recent=RECENT_WINDOWS):
    """Summarise the model's most recent snapshots and save the text on its latest snapshot.

    The summary starts with a code-written sentence about the latest window, followed by the
    LLM's explanation. Uses get_llm(temperature=0), for consistent summaries, unless an llm is
    passed. If the LLM call fails, or its text fails fact_check(), a rule-based summary is used
    instead and says why. Returns (latest snapshot, whether the LLM's text was used), or
    (None, False) when the model has no snapshots.
    """
    snapshots = list(MetricSnapshot.objects.filter(model_name=model_name).order_by('window_start', 'id'))
    if not snapshots:
        return None, False

    # Recompute the reasons, since the latest snapshot's summary is replaced below.
    baseline = baseline_accuracy()
    windows = [
        (i + 1, snapshot, anomaly_reasons(snapshot, [s.avg_latency_ms for s in snapshots[:i]], baseline))
        for i, snapshot in enumerate(snapshots)
    ][-recent:]

    prompt = build_prompt(model_name, windows, baseline)
    opening = latest_sentence(windows)
    try:
        explanation = _ask_llm(llm or get_llm(temperature=0), prompt)
        fact_check(explanation, windows, prompt, baseline)
        summary, used_llm = f'{opening} {explanation}', True
    except FactCheckError as exc:
        summary = (
            f'{opening} The LLM summary failed the fact check ({exc}), so this is a rule-based summary. '
            + rule_based_summary(model_name, windows, baseline)
        )
        used_llm = False
    except Exception as exc:  # Any provider failure (e.g. Ollama not running) falls back to the rules.
        summary = (
            f'{opening} The LLM was unavailable ({type(exc).__name__}), so this is a rule-based summary. '
            + rule_based_summary(model_name, windows, baseline)
        )
        used_llm = False

    latest = snapshots[-1]
    latest.summary = summary
    latest.save(update_fields=['summary'])
    return latest, used_llm
