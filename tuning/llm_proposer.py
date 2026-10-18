"""Asking an LLM for the next params to try, with a random fallback.

The LLM sees the search space and every earlier trial's params and CV score (never test
accuracy), and replies with JSON: {"params": {...}, "reason": "one sentence"}. A reply that is
not valid JSON, has params outside the space or repeats an earlier trial is replaced by a
random proposal, and the reasoning says why. If only the reason is wrong about how the params
changed, the params are kept and the reason is replaced by a code-written description.
"""
import json
import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm

from .models import Trial
from .reason_check import ReasonCheckError, check_reason, describe_change
from .search_space import InvalidParams, describe_space, params_key, sample_unique_params, validate_params

MAX_REASON_LENGTH = 300
JSON_OBJECT = re.compile(r'\{.*\}', re.DOTALL)

PROPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        'system',
        'You tune the hyperparameters of a scikit-learn RandomForestClassifier that predicts heart disease. '
        'Reply with one JSON object and nothing else.',
    ),
    (
        'human',
        'Search space:\n{space}\n\n'
        'Goal: the highest mean {cv_folds}-fold cross-validation accuracy.\n\n'
        'Trials so far, in order:\n{history}\n\n'
        'Propose the params for the next trial. Every value must be inside the search space, use null for '
        'no limit, and the params must differ from every trial above. Use what the earlier scores suggest. '
        'Reply with JSON only, in this format, replacing each <...> with your value: '
        '{{"params": {{"n_estimators": <integer>, "max_depth": <integer or null>, "min_samples_split": <integer>, '
        '"min_samples_leaf": <integer>, "max_features": <"sqrt", "log2" or null>}}, '
        '"reason": "<one sentence explaining the choice>"}}',
    ),
])


@dataclass
class Proposal:
    params: dict
    reasoning: str
    proposed_by: str
    reason_check: str = ''  # A Trial.REASON_* value for LLM reasons.


class InvalidProposal(ValueError):
    """The LLM's reply cannot be used."""


def describe_history(history):
    """history is a list of dicts with number, params, cv_score and cv_std, oldest first."""
    if not history:
        return 'No trials yet.'
    lines = [
        f'Trial {t["number"]}: {json.dumps(t["params"])} → CV accuracy {t["cv_score"]:.4f} ± {t["cv_std"]:.4f}'
        for t in history
    ]
    best = max(history, key=lambda t: t['cv_score'])
    lines.append(f'Best so far: trial {best["number"]} with CV accuracy {best["cv_score"]:.4f}.')
    return '\n'.join(lines)


def parse_reply(reply, space):
    """(params, reason) from the LLM's reply; raises InvalidProposal or InvalidParams."""
    match = JSON_OBJECT.search(reply)  # Tolerates ```json fences and text around the object.
    if not match:
        raise InvalidProposal('the LLM reply had no JSON object')
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise InvalidProposal(f'the LLM reply was not valid JSON ({exc.msg})') from exc
    if not isinstance(data, dict):
        raise InvalidProposal('the LLM reply was not a JSON object')

    if isinstance(data.get('params'), dict):
        params = data['params']
    elif set(data) - {'reason'} == set(space):  # The params at the top level, next to "reason".
        params = {name: value for name, value in data.items() if name != 'reason'}
    else:
        raise InvalidProposal('the LLM reply had no "params" object')

    reason = data.get('reason')
    reason = ' '.join(reason.split()) if isinstance(reason, str) else ''
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[:MAX_REASON_LENGTH - 1].rstrip() + '…'
    return validate_params(params, space), reason


def llm_proposal(space, history, llm, cv_folds):
    """Ask the LLM; returns (params, checked reason) or raises InvalidProposal/InvalidParams."""
    inputs = {'space': describe_space(space), 'history': describe_history(history), 'cv_folds': cv_folds}
    reply = (PROPOSE_PROMPT | llm | StrOutputParser()).invoke(inputs)
    params, reason = parse_reply(reply, space)
    earlier = {params_key(t['params']): t['number'] for t in history}
    if params_key(params) in earlier:
        raise InvalidProposal(f'the LLM repeated the params of trial {earlier[params_key(params)]}')
    return (params, *checked_reason(reason, params, history[-1]['params'] if history else None, inputs))


def checked_reason(reason, params, previous, inputs, against='the previous trial'):
    """(reasoning, Trial.REASON_* status): the LLM's reason if it matches the change from previous,
    else a code-written description of the change."""
    description = describe_change(params, previous, against)
    if not reason:
        return f'{description} (The LLM gave no reason.)', Trial.REASON_MISSING
    shown = '\n'.join(str(value) for value in inputs.values())
    try:
        check_reason(reason, params, previous, shown)
    except ReasonCheckError as exc:
        return f"{description} The LLM's reason failed the fact check ({exc}).", Trial.REASON_FAILED
    return reason, Trial.REASON_PASSED


def propose_next(space, history, rng, llm=None, cv_folds=5):
    """The next trial's params from the LLM, or a random proposal if its reply cannot be used.

    Uses get_llm(temperature=0) unless an llm is passed. rng (a random.Random) is only used for
    the fallback, so a run that never falls back gives the same proposals for the same replies.
    """
    try:
        params, reason, status = llm_proposal(space, history, llm or get_llm(temperature=0), cv_folds)
        return Proposal(params, reason, Trial.PROPOSED_BY_LLM, status)
    except InvalidParams as exc:
        why = f"the LLM's params are invalid ({exc})"
    except InvalidProposal as exc:
        why = str(exc)
    except Exception as exc:  # Any provider failure (e.g. Ollama not running) falls back to random.
        why = f'the LLM was unavailable ({type(exc).__name__})'

    seen = {params_key(t['params']) for t in history}
    params = sample_unique_params(space, rng, seen)
    return Proposal(params, f'Random fallback: {why}.', Trial.PROPOSED_BY_FALLBACK)
