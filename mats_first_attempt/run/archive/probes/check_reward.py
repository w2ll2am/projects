"""Is verl's DAPO reward function registered for the DeepScaleR data source?

If it is not, every reward is 0 and the run looks like a learning-rate problem.
This is the assumption most likely to waste an 11-hour training run, and it is
checkable in seconds without a GPU or the filtered dataset.
"""
import inspect
import re

src = ""
try:
    from verl.utils.reward_score import default_compute_score as dcs
    src = inspect.getsource(dcs)
    print("imported verl.utils.reward_score.default_compute_score OK")
except Exception as exc:  # noqa: BLE001
    print("import default_compute_score FAILED:", type(exc).__name__, exc)

# Also pull the whole module, since the dispatch table often lives beside it.
mod_src = ""
try:
    import verl.utils.reward_score as rs
    mod_src = inspect.getsource(rs)
    print("module file:", rs.__file__)
except Exception as exc:  # noqa: BLE001
    print("import module FAILED:", type(exc).__name__, exc)

blob = src + "\n" + mod_src
names = sorted(set(re.findall(r"""['"]([A-Za-z0-9_./-]{3,})['"]""", blob)))
print("\nquoted identifiers in the dispatch source (%d):" % len(names))
for n in names:
    print("   ", n)

print()
for probe in ("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler", "DeepScaleR",
              "math", "math_dapo", "aime"):
    hit = probe in names
    loose = probe.lower() in blob.lower()
    print(f"  {probe!r:48} exact={hit}  appears_anywhere={loose}")

# What does it actually do with an unknown data_source?
print("\n--- behaviour on an unknown data_source ---")
try:
    r = dcs("agentica-org/DeepScaleR-Preview-Dataset",
            "The answer is \\boxed{42}", "42")
    print("  returned:", r)
except Exception as exc:  # noqa: BLE001
    print("  RAISED:", type(exc).__name__, str(exc)[:300])
    print("  (raising is the SAFE failure - a silent 0.0 would look like an LR problem)")
