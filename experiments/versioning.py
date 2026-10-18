"""Commit-style version notes for runs, e.g.
"v3: n_estimators 100→300, max_depth None→8; test accuracy 0.754→0.787".
"""
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm

MAX_NOTE_LENGTH = 300
NUMBER = re.compile(r'\d+(?:\.\d+)?')
CHANGE = re.compile(r'([^\s,;:→]+)\s*→\s*([^\s,;:→]+)')

NOTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        'system',
        'You write one-line, commit-style version notes for machine learning training runs. '
        'Reply with the note only: no quotes, no explanation, no extra lines.',
    ),
    (
        'human',
        'Write the version note for {experiment} v{version}.\n\n'
        '{details}\n\n'
        'Format: "v{version}: <parameter changes>; <metric changes>", using → between old and new '
        'values and the numbers exactly as written above. Example of the format only, from a different '
        'run: "v3: n_estimators 100→300, max_depth None→8; test accuracy 0.754→0.787". '
        'Keep it under 160 characters.',
    ),
])


def format_value(value):
    if isinstance(value, float):
        text = f'{value:.3f}'.rstrip('0')
        return text + '0' if text.endswith('.') else text
    return str(value)


def metric_label(name):
    return name.replace('_', ' ')


def headline_metrics(names):
    """Test metrics are what matter between versions; use them when there are any."""
    names = list(names)
    test_names = [name for name in names if name.startswith('test_')]
    return test_names or names


def fallback_note(run):
    """Write the note from run.changes in plain code."""
    if run.parent_id is None:
        params = ', '.join(f'{name}={format_value(value)}' for name, value in run.params.items())
        metrics = ', '.join(
            f'{metric_label(name)} {format_value(run.metrics[name])}' for name in headline_metrics(run.metrics)
        )
        return f'v{run.version}: first run' + (f' with {params}' if params else '') + (f'; {metrics}' if metrics else '')

    param_changes = run.changes.get('params', {})
    metric_changes = run.changes.get('metrics', {})
    params = ', '.join(
        f'{name} {format_value(change["old"])}→{format_value(change["new"])}' for name, change in param_changes.items()
    )
    metrics = ', '.join(
        f'{metric_label(name)} {format_value(metric_changes[name]["old"])}→{format_value(metric_changes[name]["new"])}'
        for name in headline_metrics(metric_changes)
    )
    return f'v{run.version}: {params or "no parameter changes"}; {metrics or "metrics unchanged"}'


def describe_run(run):
    """The changes the LLM gets, with numbers already formatted. Needs a parent run."""
    lines = [f'Changes from v{run.parent.version}:', 'Parameters:']
    param_changes = run.changes.get('params', {})
    lines += [
        f'- {name}: {format_value(change["old"])} → {format_value(change["new"])}'
        for name, change in param_changes.items()
    ] or ['- none']
    lines.append('Metrics:')
    metric_changes = run.changes.get('metrics', {})
    lines += [
        f'- {metric_label(name)}: {format_value(metric_changes[name]["old"])} → {format_value(metric_changes[name]["new"])}'
        for name in headline_metrics(metric_changes)
    ] or ['- unchanged']
    return '\n'.join(lines)


def llm_note(run, llm):
    details = describe_run(run)
    chain = NOTE_PROMPT | llm | StrOutputParser()
    reply = chain.invoke({'experiment': run.experiment.name, 'version': run.version, 'details': details})
    lines = [line.strip().strip('"\'`').strip() for line in reply.splitlines()]
    note = next((line for line in lines if line), '')
    if not note:
        raise ValueError('The LLM returned an empty note.')
    if len(note) > MAX_NOTE_LENGTH:
        raise ValueError('The LLM note is too long.')
    prefix = f'v{run.version}:'
    note = note if note.startswith(prefix) else f'{prefix} {note}'
    # Small models sometimes copy the example's values; every number must come from the run.
    invented = set(NUMBER.findall(note[len(prefix):])) - set(NUMBER.findall(details))
    if invented:
        raise ValueError(f'The LLM note has numbers not in the run: {sorted(invented)}')
    # ...and every old→new pair must be a real change, in the right direction.
    wrong = set(CHANGE.findall(note)) - set(CHANGE.findall(details))
    if wrong:
        raise ValueError(f'The LLM note has changes not in the run: {sorted(wrong)}')
    return note


def write_version_note(run, llm=None, use_llm=True):
    """Write and save run.version_note; return (note, whether the LLM wrote it).

    Uses get_llm(temperature=0) unless an llm is passed. The note is written from
    run.changes by fallback_note() instead when use_llm is False, when the LLM fails or invents
    numbers, or when there are no changes to describe (including a first run).
    """
    note, used_llm = None, False
    has_changes = any(run.changes.get('params', {}).values()) or any(run.changes.get('metrics', {}).values())
    if use_llm and has_changes:
        try:
            note, used_llm = llm_note(run, llm or get_llm(temperature=0)), True
        except Exception:  # Any provider failure (e.g. Ollama not running) falls back to plain code.
            note = None
    if note is None:
        note = fallback_note(run)
    run.version_note = note
    run.save(update_fields=['version_note'])
    return note, used_llm
