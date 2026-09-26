"""
Turning results/ into the numbers the analysis notebook prints.

    registry         arm vocabulary, loading, shared row helpers
    tables           headline rates, integrity checks, paired comparisons
    confidence       caving against the model's own turn-1 confidence
    degeneracy       checks that the tuned model is still a model
    examples         matched before/after cases
    training_curves  per-step metrics logged during training

plots.py draws these; it is imported separately so matplotlib stays optional.
"""

from episteme import config

from episteme.analysis.registry import (ARMS, BY_LABEL, CAVED, CHANGED,
                                        CORRECTED, LABELS, PRESSURE, answered,
                                        buckets, key, load, load_all,
                                        load_meta, rate)
from episteme.analysis.tables import (by_pressure, by_source, by_subject,
                                      condition_drift, corrected_bounds,
                                      headline, mcnemar, missingness, paired,
                                      pressure_within_arm, provenance,
                                      reference_policies, stars)
from episteme.analysis.confidence import (CONF_BINS, confidence_auc,
                                          confidence_bins, confidence_fit,
                                          confidence_frame, confidence_gate)
from episteme.analysis.degeneracy import (CAPITULATION, DEFERENCE, degeneracy)
from episteme.analysis.examples import find_cases, matched, show_case
from episteme.analysis.training_curves import (curve_comparison,
                                               curve_endpoints, curves,
                                               load_curve)
