"""Asking the LLM to suggest the next configurations to try, after a job has run.

Each suggestion must be inside the search space and not already tried, and its reason is
fact-checked against the best trial with reason_check. There are always N_SUGGESTIONS: a slot
whose suggestion repeats a trial or an earlier suggestion, is invalid or is missing (including when
the LLM fails) is filled from the job's seeded random fallback stream and marked with why.
"""
import json
import re
from dataclasses import asdict, dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from django.utils import timezone

from core.llm import get_llm

from .engine import fallback_rng
from .llm_proposer import InvalidProposal, checked_reason, describe_history, proposal_from_dict
from .search_space import InvalidParams, describe_space, params_key, sample_unique_params

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


SOURCE_LLM = 'llm'
SOURCE_RANDOM = 'random'


@dataclass
class Suggestion:
    params: dict
    reasoning: str
    reason_check: str  # A Trial.REASON_* value; blank for random fills.
    llm_reason: str = ''  # The LLM's own reason, re-checked when the suggestion is tried.
    source: str = SOURCE_LLM  # SOURCE_RANDOM when the slot was filled at random.
    fill_reason: str = ''  # Why a random slot was filled, e.g. "LLM repeated trial 2".

    @property
    def label(self):
        return 'LLM' if self.source == SOURCE_LLM else f'random ({self.fill_reason})'


@dataclass
class SuggestionResult:
    suggestions: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # Why each unusable suggestion was dropped.
    error: str = ''

    def as_json(self):
        """The result as stored in TuningJob.suggestions."""
        return {
            'created_at': timezone.now().isoformat(),
            'configs': [{**asdict(s), 'label': s.label, 'tried_trial': None} for s in self.suggestions],
            'rejected': self.rejected,
            'error': self.error,
        }


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
    result = SuggestionResult()
    try:
        reply = (SUGGEST_PROMPT | (llm or get_llm(temperature=0)) | StrOutputParser()).invoke(inputs)
        items, error = parse_suggestions(reply)
    except Exception as exc:  # Any provider failure (e.g. Ollama not running).
        items, error = [], f'The LLM was unavailable ({type(exc).__name__}).'
    if error:
        result.error = f'{error} All {N_SUGGESTIONS} suggestions are random.'

    tried = {params_key(t.params): t.number for t in trials}
    suggested = {}  # params_key → slot number of the suggestions made so far.
    rng = fallback_rng(job.seed)

    def fill(why):
        params = sample_unique_params(job.search_space, rng, set(tried) | set(suggested))
        suggested[params_key(params)] = len(result.suggestions) + 1
        result.suggestions.append(Suggestion(
            params, f'Random fallback: the {why}.', '', source=SOURCE_RANDOM, fill_reason=why,
        ))

    for slot, item in enumerate(items[:N_SUGGESTIONS], start=1):
        try:
            params, reason = proposal_from_dict(item, job.search_space)
        except (InvalidParams, InvalidProposal) as exc:
            fill(f'LLM suggestion was invalid: {exc}')
            continue
        key = params_key(params)
        if key in tried:
            fill(f'LLM repeated trial {tried[key]}')
        elif key in suggested:
            fill(f'LLM repeated suggestion {suggested[key]}')
        else:
            suggested[key] = slot
            reasoning, status = check_against_best(reason, params, best, inputs)
            result.suggestions.append(Suggestion(params, reasoning, status, reason))

    count = len(items)
    while len(result.suggestions) < N_SUGGESTIONS:
        fill('LLM was unavailable' if error and 'unavailable' in error
             else 'LLM reply could not be used' if error
             else f'LLM gave only {count} suggestion{"" if count == 1 else "s"}')
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
