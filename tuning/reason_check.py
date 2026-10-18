"""Fact-checking the LLM's one-sentence reason against the actual change from the previous trial.

The reason is read left to right: a direction word ("increase", "fewer", "keep", ...) applies to
the parameters mentioned after it, until punctuation, a word like "try" or "while", or six words
pass. Each claim must match how the parameter changed from the previous trial. Every number must
come from the proposal, the previous trial or the data the LLM was shown, and "<param> to N" or
"from A to B" must give the real values. The check errs towards leniency on unclear wording, but
any claim it does read must be right.
"""
import re

PARAM_PATTERNS = {
    'n_estimators': r'n_estimators|number of (?:trees|estimators)|estimators',
    'max_depth': r'max_depth|max(?:imum)?\s+depth|tree depth|depth',
    'min_samples_split': r'min_samples_split|min(?:imum)?\s+samples?\s+(?:to\s+|per\s+)?split',
    'min_samples_leaf': r'min_samples_leaf|min(?:imum)?\s+samples?\s+(?:per\s+|in\s+a\s+)?leaf|leaf size',
    'max_features': r'max_features|max(?:imum)?\s+features',
}
UP = r'increas\w*|rais\w*|higher|larger|bigger|more|grow\w*|deeper'
DOWN = r'decreas\w*|lower\w*|reduc\w*|smaller|fewer|less|shallower|limit\w*'
SAME = r'keep\w*|kept|same|unchanged|retain\w*|maintain\w*|stay\w*'
STOP = r'[.,;:!?()]|\b(?:try|trying|set|setting|use|using|start\w*|while|but|which|whereas|because|since|no limit)\b'
MAX_DISTANCE = 6  # Words between a direction and the parameter it applies to.

TOKEN = re.compile(
    rf'(?P<stop>{STOP})'
    rf'|(?P<depth_trees>\b(?:deeper|shallower)\s+(?:decision\s+)?trees\b)'
    rf'|(?P<count_trees>\b(?:more|fewer|less)\s+(?:decision\s+)?trees\b)'
    + ''.join(rf'|(?P<p_{name}>\b(?:{pattern})\b)' for name, pattern in PARAM_PATTERNS.items())
    + rf'|(?P<up>\b(?:{UP})\b)|(?P<down>\b(?:{DOWN})\b)|(?P<same>\b(?:{SAME})\b)'
    r'|(?P<word>\b\w+\b)',
    re.IGNORECASE,
)
VALUE = r'''["']?(null|none|-?\d+(?:\.\d+)?|sqrt|log2)["']?'''
VALUE_CLAIM = re.compile(
    '(' + '|'.join(rf'(?P<v_{name}>\b(?:{pattern})\b)' for name, pattern in PARAM_PATTERNS.items()) + ')'
    + rf'(?:\s+(?:from|of)\s+{VALUE})?\s+to\s+{VALUE}',
    re.IGNORECASE,
)
NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?![\w.])')


class ReasonCheckError(ValueError):
    """The LLM's reason contradicts the actual change."""


def order_value(name, value):
    """Comparable value; for max_depth, None (no limit) is deeper than any number."""
    if value is None and name == 'max_depth':
        return float('inf')
    return value


def parse_value(text):
    text = text.strip('\'"').lower()
    if text in ('null', 'none'):
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number == int(number) else number


def format_value(value):
    return 'None' if value is None else str(value)


def direction_claims(reason):
    """[(param, 'up' | 'down' | 'same')] in the order they are made."""
    claims, direction, distance = [], None, 0
    for token in TOKEN.finditer(reason):
        kind = token.lastgroup
        if kind == 'stop':
            direction = None
        elif kind == 'depth_trees':  # "deeper trees" is about max_depth, not the number of trees.
            claims.append(('max_depth', 'up' if token.group().lower().startswith('deeper') else 'down'))
            direction = None
        elif kind == 'count_trees':
            claims.append(('n_estimators', 'up' if token.group().lower().startswith('more') else 'down'))
            direction = None
        elif kind in ('up', 'down', 'same'):
            direction, distance = kind, 0
        elif kind.startswith('p_'):
            if direction and distance <= MAX_DISTANCE:
                claims.append((kind[2:], direction))
            distance = 0
        elif direction:
            distance += 1
    return claims


def check_reason(reason, params, previous, shown_text):
    """Raise ReasonCheckError if reason misstates how params differ from previous (None for trial 1).

    shown_text is everything the LLM was shown (search space and history); its numbers may be cited.
    """
    for name, claim in direction_claims(reason):
        if previous is None:
            raise ReasonCheckError(f'it says {name} should {verb(claim)}, but there is no previous trial')
        old, new = previous[name], params[name]
        if claim == 'same':
            if old != new:
                raise ReasonCheckError(f'it says {name} stays the same, but it changes from {format_value(old)} to {format_value(new)}')
            continue
        if name == 'max_features' or old == new:
            if old == new:
                raise ReasonCheckError(f'it says {name} should {verb(claim)}, but it stays at {format_value(new)}')
            continue  # max_features has no order to check.
        actual = 'up' if order_value(name, new) > order_value(name, old) else 'down'
        if claim != actual:
            raise ReasonCheckError(
                f'it says {name} should {verb(claim)}, but it goes from {format_value(old)} to {format_value(new)}'
            )

    for match in VALUE_CLAIM.finditer(reason):
        name = next(group[2:] for group, value in match.groupdict().items() if group.startswith('v_') and value)
        old_text, new_text = match.group(match.lastindex - 1), match.group(match.lastindex)
        if parse_value(new_text) != params[name]:
            raise ReasonCheckError(f'it says {name} goes to {new_text}, but the proposal has {format_value(params[name])}')
        if old_text is not None and (previous is None or parse_value(old_text) != previous[name]):
            raise ReasonCheckError(f'it says {name} goes from {old_text}, but that is not the previous value')

    allowed = {float(n) for n in NUMBER.findall(shown_text)}
    for values in (params, previous or {}):
        allowed |= {float(v) for v in values.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    wrong = [n for n in NUMBER.findall(reason) if float(n) not in allowed]
    if wrong:
        raise ReasonCheckError(f'it mentions numbers that are not in the data: {", ".join(dict.fromkeys(wrong))}')


def verb(claim):
    return {'up': 'increase', 'down': 'decrease', 'same': 'stay the same'}[claim]


def describe_change(params, previous):
    """A code-written description of the proposal compared with the previous trial."""
    if previous is None:
        return 'First trial: ' + ', '.join(f'{name} {format_value(value)}' for name, value in params.items()) + '.'
    changed = [f'{name} {format_value(previous.get(name))}→{format_value(value)}'
               for name, value in params.items() if previous.get(name) != value]
    kept = [f'{name} {format_value(value)}' for name, value in params.items() if previous.get(name) == value]
    parts = []
    if changed:
        parts.append('Changed ' + ', '.join(changed))
    if kept:
        parts.append('kept ' + ', '.join(kept))
    text = '; '.join(parts)
    return text[0].upper() + text[1:] + ' compared with the previous trial.'
