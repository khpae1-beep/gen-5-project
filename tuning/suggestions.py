"""Asking the LLM to suggest the next configurations to try, after a job has run.

Each suggestion must be inside the search space and not already tried, and its reason is
fact-checked against the best trial with reason_check. Unlike the proposer there is no random
fallback: if the LLM fails, there are simply no suggestions.
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_llm

from .llm_proposer import InvalidProposal, checked_reason, describe_history, proposal_from_dict
from .search_space import InvalidParams, describe_space, params_key

N_SUGGESTIONS = 3
JSON_VALUE = re.compile(r'[\[{].*[\]}]', re.DOTALL)

SUGGEST_PROMPT = ChatPromptTemplate.from_messages([
    (
        'system',
        'You tune the hyperparameters of a scikit-learn RandomForestClassifier that predicts heart disease. '
        'Reply with one JSON object and nothing else.',
    ),
    (
        'human',
        'Search space:\n{space}\n\n'
        'Goal: the highest mean {cv_folds}-fold cross-validation accuracy.\n\n'
        'All trials so far, in order:\n{history}\n\n'
        'Suggest {n} new configurations to try next that could beat the best trial (trial {best}). Every '
        'value must be inside the search space, use null for no limit, and each configuration must differ '
        'from every trial above and from the other suggestions. For each, give one sentence explaining how '
        'it differs from trial {best} and why. Reply with JSON only, in this format, replacing each <...>: '
        '{{"suggestions": [{{"params": {{"n_estimators": <integer>, "max_depth": <integer or null>, '
        '"min_samples_split": <integer>, "min_samples_leaf": <integer>, "max_features": <"sqrt", "log2" or '
        'null>}}, "reason": "<one sentence>"}}, ...]}}',
    ),
])


@dataclass
class Suggestion:
    params: dict
    reasoning: str
    reason_check: str  # A Trial.REASON_* value.
    llm_reason: str = ''  # The LLM's own reason, re-checked when the suggestion is tried.

    @property
    def params_json(self):
        return json.dumps(self.params)


@dataclass
class SuggestionResult:
    suggestions: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # Why each unusable suggestion was dropped.
    error: str = ''


def best_trial_of(trials):
    return max(trials, key=lambda t: (t.cv_score, -t.number))


def suggest_next(job, trials, llm=None):
    """Up to N_SUGGESTIONS new configurations for job, given its trials (oldest first).

    Uses get_llm(temperature=0) unless an llm is passed.
    """
    if not trials:
        return SuggestionResult(error='The job has no trials to learn from yet.')
    best = best_trial_of(trials)
    inputs = prompt_inputs(job, trials, best)
    try:
        reply = (SUGGEST_PROMPT | (llm or get_llm(temperature=0)) | StrOutputParser()).invoke(inputs)
    except Exception as exc:  # Any provider failure (e.g. Ollama not running).
        return SuggestionResult(error=f'The LLM was unavailable ({type(exc).__name__}).')

    items, error = parse_suggestions(reply)
    if error:
        return SuggestionResult(error=error)

    result = SuggestionResult()
    tried = {params_key(t.params): t.number for t in trials}
    suggested = set()
    for index, item in enumerate(items, start=1):
        if len(result.suggestions) == N_SUGGESTIONS:
            break
        try:
            params, reason = proposal_from_dict(item, job.search_space)
        except (InvalidParams, InvalidProposal) as exc:
            result.rejected.append(f'Suggestion {index} was dropped: {exc}.')
            continue
        key = params_key(params)
        if key in tried:
            result.rejected.append(f'Suggestion {index} was dropped: it repeats trial {tried[key]}.')
            continue
        if key in suggested:
            result.rejected.append(f'Suggestion {index} was dropped: it repeats an earlier suggestion.')
            continue
        suggested.add(key)
        reasoning, status = check_against_best(reason, params, best, inputs)
        result.suggestions.append(Suggestion(params, reasoning, status, reason))
    if not result.suggestions and not result.rejected:
        result.error = 'The LLM did not suggest any configurations.'
    return result


def prompt_inputs(job, trials, best):
    history = [{'number': t.number, 'params': t.params, 'cv_score': t.cv_score, 'cv_std': t.cv_std} for t in trials]
    return {
        'space': describe_space(job.search_space),
        'history': describe_history(history),
        'cv_folds': job.cv_folds,
        'n': N_SUGGESTIONS,
        'best': best.number,
    }


def recheck_reason(job, trials, params, llm_reason):
    """Check a suggestion's original reason again against the job's current best trial."""
    best = best_trial_of(trials)
    return check_against_best(llm_reason, params, best, prompt_inputs(job, trials, best))


def check_against_best(reason, params, best, inputs):
    """reason_check the reason against the best trial; returns (reasoning, Trial.REASON_* status)."""
    return checked_reason(reason, params, best.params, inputs, against=f'the best trial (trial {best.number})')


def parse_suggestions(reply):
    """(list of suggestion objects, error message) from the LLM's reply."""
    match = JSON_VALUE.search(reply)
    if not match:
        return [], 'The LLM reply had no JSON.'
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return [], f'The LLM reply was not valid JSON ({exc.msg}).'
    if isinstance(data, dict):
        data = data.get('suggestions')
    if not isinstance(data, list):
        return [], 'The LLM reply had no "suggestions" list.'
    return data, ''
