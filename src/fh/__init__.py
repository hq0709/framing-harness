"""A medical VLM's answer carries an offset that depends on how the question was presented.

    m(x, q) = b(presentation, q) + e(x, q)

Three operators remove b, ordered by what they need:

    marginalise   average the margin over K presentations of the same evidence   (needs nothing)
    centre        subtract that presentation's batch median                      (needs a batch, costs nothing)
    difference    subtract a matched input read under the same presentation      (needs a counterpart)
"""
from fh.config import COCO, DATA, MSD, RESULTS, need, out          # noqa: F401
from fh.operators import auroc, balanced_accuracy, centre, marginalise  # noqa: F401
from fh.presentations import PRESENTATIONS, present                # noqa: F401
from fh.readout import load_any, read_yesno, render                # noqa: F401

__all__ = ["COCO", "DATA", "MSD", "RESULTS", "need", "out", "auroc", "balanced_accuracy",
           "centre", "marginalise", "PRESENTATIONS", "present", "load_any", "read_yesno", "render"]
