"""The hyperparameter search space for the heart disease RandomForest, and random search over it.

A space is plain JSON, so it can be stored on a TuningJob and shown to the LLM as-is:
- {"type": "int", "low": a, "high": b} is any integer from a to b inclusive; with
  "allow_none": true, None is allowed too.
- {"type": "choice", "choices": [...]} is one of the listed values (None for JSON null).
"""
import json

HEART_DISEASE_RF_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'max_depth': {'type': 'int', 'low': 2, 'high': 20, 'allow_none': True},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
    'max_features': {'type': 'choice', 'choices': ['sqrt', 'log2', None]},
}

# How many random draws to try before giving up on finding params no earlier trial used.
MAX_UNIQUE_ATTEMPTS = 1000


class InvalidParams(ValueError):
    """Params outside the search space."""


def _none_like(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in ('none', 'null'))


def validate_params(params, space):
    """Return params normalised to the space (e.g. 300.0 → 300, "None" → None), or raise InvalidParams."""
    if not isinstance(params, dict):
        raise InvalidParams('params must be a JSON object')
    missing = [name for name in space if name not in params]
    extra = [name for name in params if name not in space]
    if missing:
        raise InvalidParams(f'missing {", ".join(missing)}')
    if extra:
        raise InvalidParams(f'unknown {", ".join(extra)}')

    clean = {}
    for name, spec in space.items():
        value = params[name]
        if spec['type'] == 'int':
            if _none_like(value):
                if not spec.get('allow_none'):
                    raise InvalidParams(f'{name} cannot be None')
                clean[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
                raise InvalidParams(f'{name} must be a whole number, got {value!r}')
            value = int(value)
            if not spec['low'] <= value <= spec['high']:
                raise InvalidParams(f'{name}={value} is outside {spec["low"]}–{spec["high"]}')
            clean[name] = value
        else:
            value = None if _none_like(value) else value
            if value not in spec['choices']:
                raise InvalidParams(f'{name}={value!r} is not one of {spec["choices"]}')
            clean[name] = value
    return clean


def params_key(params):
    """A hashable key for params, to spot repeated trials."""
    return json.dumps(params, sort_keys=True)


def sample_params(space, rng):
    """One uniformly random point of the space. rng is a random.Random."""
    params = {}
    for name, spec in space.items():
        if spec['type'] == 'int':
            options = list(range(spec['low'], spec['high'] + 1)) + ([None] if spec.get('allow_none') else [])
            params[name] = rng.choice(options)
        else:
            params[name] = rng.choice(spec['choices'])
    return params


def sample_unique_params(space, rng, seen_keys):
    """A random point no earlier trial used (seen_keys holds their params_key values)."""
    for _ in range(MAX_UNIQUE_ATTEMPTS):
        params = sample_params(space, rng)
        if params_key(params) not in seen_keys:
            return params
    raise RuntimeError('Could not find untried params; the search space may be exhausted.')


def describe_space(space):
    """The space in words, for the LLM prompt."""
    lines = []
    for name, spec in space.items():
        if spec['type'] == 'int':
            text = f'- {name}: an integer from {spec["low"]} to {spec["high"]}'
            lines.append(text + (', or null (no limit)' if spec.get('allow_none') else ''))
        else:
            lines.append(f'- {name}: one of {", ".join(json.dumps(c) for c in spec["choices"])}')
    return '\n'.join(lines)
