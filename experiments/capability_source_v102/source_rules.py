"""Conservative prefilter for future cohorts; V102 remains frozen and failed.

Reject the entire candidate sentence when any denial or uncertainty occurs.
This can discard a valid positive combined with an unrelated negative clause;
it deliberately does not attempt clinical scope resolution. Whole-record
adjudication is still required before freezing a future model cohort.
"""
import re


DENIAL_OR_UNCERTAINTY = re.compile(
    r'\b(?:no|not|without|absence|absent|negative|neither|nor|doubtful|'
    r'questionable|equivocal|uncertain|indeterminate|unclear|unconfirmed)\b|'
    r'\bfree\s+of\b|\brule[ -]?out\b', re.I)


def affirmative_candidate(finding, existing_positive):
    """Add a conservative semantic veto to existing label/box admission."""
    return bool(existing_positive(finding)) and not DENIAL_OR_UNCERTAINTY.search(finding['sentence_en'])
