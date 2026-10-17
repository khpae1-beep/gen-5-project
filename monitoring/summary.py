from core.llm import get_llm

from .metrics import anomaly_reasons, baseline_accuracy
from .models import MetricSnapshot

RECENT_WINDOWS = 10


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
        f'describe current or latest performance, use window {latest} only. Only describe the anomalies listed; '
        'if none are listed, say there are none. Use the numbers exactly as given. Do not use bullet points or '
        'headings.'
    )


def rule_based_summary(model_name, windows, baseline):
    snapshots = [snapshot for _, snapshot, _ in windows]
    latest_number, latest, _ = windows[-1]
    total = sum(s.count for s in snapshots)
    labelled = [s.accuracy for s in snapshots if s.accuracy is not None]
    accuracy = f'{sum(labelled) / len(labelled):.2f}' if labelled else 'unknown'
    latency = sum(s.avg_latency_ms for s in snapshots) / len(snapshots)

    parts = [
        f'Over the last {len(snapshots)} windows ({total} predictions), {model_name} averaged accuracy '
        f'{accuracy} against a {baseline:.2f} baseline, with {latency:.1f} ms average latency.',
        f'The latest window (window {latest_number}) had accuracy {_format_accuracy(latest.accuracy)} '
        f'and {latest.avg_latency_ms:.1f} ms latency.',
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

    Uses get_llm(temperature=0), for consistent summaries, unless an llm is passed. If the LLM call fails, saves a rule-based summary
    that says the LLM was unavailable. Returns (latest snapshot, whether the LLM was used),
    or (None, False) when the model has no snapshots.
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

    try:
        summary = _ask_llm(llm or get_llm(temperature=0), build_prompt(model_name, windows, baseline))
        used_llm = True
    except Exception as exc:  # Any provider failure (e.g. Ollama not running) falls back to the rules.
        summary = (
            f'The LLM was unavailable ({type(exc).__name__}), so this is a rule-based summary. '
            + rule_based_summary(model_name, windows, baseline)
        )
        used_llm = False

    latest = snapshots[-1]
    latest.summary = summary
    latest.save(update_fields=['summary'])
    return latest, used_llm
