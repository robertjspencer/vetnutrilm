"""Strict output parsers, as used for the published results.

An output that is not exactly the expected JSON array is invalid. Invalid outputs are
reported separately and never silently counted as a correct negative.
"""
import json

from .prompts import MAX_NEW_TOKENS


def parse_binary(raw):
    """Return ([0] or [1], False) for a valid answer, or ([-1], True) if invalid."""
    try:
        value = json.loads(raw.strip())
        assert isinstance(value, list) and len(value) == 1
        assert type(value[0]) is int and value[0] in (0, 1)
        return value, False
    except (ValueError, AssertionError):
        return [-1], True


def parse_categories(raw):
    """Return (six 0/1 integers, False) for a valid answer, or ([0]*6, True) if invalid.

    The all-zero vector for invalid outputs is the prespecified scoring rule; the
    invalid flag must be reported alongside any metric.
    """
    try:
        value = json.loads(raw.strip())
        assert isinstance(value, list) and len(value) == 6
        assert all(type(v) is int and v in (0, 1) for v in value)
        return value, False
    except (ValueError, AssertionError):
        return [0] * 6, True


def is_truncated(generated_tokens):
    return generated_tokens >= MAX_NEW_TOKENS
