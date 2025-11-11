# src/experiments.py
EXPERIMENTS = [
    dict(
        name="logreg_basic_C1",
        model_type="logreg",
        feature_set="basic",
        C=1.0,
        seed=42,
    ),
    dict(
        name="logreg_adv_C1",
        model_type="logreg",
        feature_set="advanced",
        C=1.0,
        seed=42,
    ),
    dict(
        name="logreg_adv_C0.1",
        model_type="logreg",
        feature_set="advanced",
        C=0.1,
        seed=42,
    ),
]
