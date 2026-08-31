"""The two classifiers behind every decoding table.

Two models, not eight: one linear, one nonlinear, so a table answers "is there
a linear boundary?" and "does flexibility buy anything?" without inviting a
pick-the-best-of-eight selection effect.

Both weight the classes equally (``class_weight="balanced"``): the strategy
split inside a maze can be as lopsided as 12/50, and an unweighted fit
saturates to the majority class -- every prediction becomes a constant and the
balanced accuracy collapses to chance no matter what the features hold.

Hyperparameters are fixed, not tuned. There is no inner search.

**Logistic (L2)** -- a linear weighted sum of features squashed through a
sigmoid, maximising label likelihood with a quadratic penalty on the weights.
Standardisation matters for it (the blocks mix milliseconds, {0,1} flags and
proportions), so it runs behind a `StandardScaler`.

**Random forest** -- 300 deep decision trees on bootstrap resamples, majority
vote. Captures interactions and non-monotone effects; grown deep, it fits the
training rows almost perfectly, so its train column is the clearest overfitting
readout in the tables. Trees split on thresholds, so it skips the scaler.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0


def make_models():
    """Ordered ``name -> pipeline factory``. Factories keep folds independent."""
    return {
        "Logistic (L2)": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": lambda: Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }
