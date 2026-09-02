"""Which registered alias should the filtered DeepScaleR parquet declare?

DeepScaleR is boxed-answer maths and we are running DAPO, so `math_dapo` is the
a-priori right choice. Verify it actually scores our format before committing,
including the cases that matter: a correct boxed answer, a wrong one, and a
truncated completion with no box at all.
"""
from verl.utils.reward_score import default_compute_score as dcs

CORRECT = "Let me think. The total is 42.\n\nThe answer is $\\boxed{42}$."
WRONG = "Let me think. The total is 41.\n\nThe answer is $\\boxed{41}$."
NOBOX = "Let me think about this. The total works out to roughly forty-two but"
GT = "42"

for alias in ("math_dapo", "math", "math_dapo_reasoning"):
    print(f"--- {alias} ---")
    for label, sol in (("correct boxed", CORRECT), ("wrong boxed", WRONG),
                       ("truncated, no box", NOBOX)):
        try:
            r = dcs(alias, sol, GT)
            print(f"    {label:<20} -> {r!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {label:<20} -> RAISED {type(exc).__name__}: {str(exc)[:120]}")
    print()
