"""The three operators, and the two metrics that tell them apart.

AUROC is threshold-free, so it has already divided out b -- the very thing these operators remove. Measuring
them by AUROC alone makes them look like they do nothing. Balanced accuracy at the natural zero is where
they show up, which is why both are here and why the analyses report both.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def marginalise(margins: dict, keys=None) -> np.ndarray:
    """Average the margin over presentations. Removes the variance of b; needs K readings and nothing else."""
    keys = list(margins) if keys is None else list(keys)
    return np.mean([np.asarray(margins[k], float) for k in keys], axis=0)


def centre(margins: dict, keys=None, stat=np.median) -> dict:
    """Subtract, per presentation, where the model puts its zero on this batch.

    The statistic is taken over the batch's own readings, so no label is touched. It removes the mean of b,
    which is what averaging cannot reach when b never changes sign -- a model answering Yes to everything
    stays at chance however many presentations it is given."""
    keys = list(margins) if keys is None else list(keys)
    return {k: np.asarray(margins[k], float) - stat(np.asarray(margins[k], float)) for k in keys}


def difference(margin, counterpart_margin) -> np.ndarray:
    """Subtract a matched input read under the same presentation. Removes b exactly, since it is the same
    constant in both terms, and leaves a score that is by construction a function of a named region."""
    return np.asarray(margin, float) - np.asarray(counterpart_margin, float)


def auroc(pos, neg) -> float:
    """Rank form, not the double loop: the pairwise version is O(n*m) in Python and the analyses call this
    tens of thousands of times over a few hundred items each, which turns a second into several minutes."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def balanced_accuracy(margin, label) -> float:
    """Decide by the sign. This is the axis the operators act on."""
    margin, label = np.asarray(margin, float), np.asarray(label, bool)
    return (float((margin[label] > 0).mean()) + float((margin[~label] <= 0).mean())) / 2


def skew(margin, label) -> float:
    """How lopsided the two error rates are -- what the comparison operator is worth, measured."""
    margin, label = np.asarray(margin, float), np.asarray(label, bool)
    return abs(float((margin[label] > 0).mean()) - float((margin[~label] <= 0).mean()))
