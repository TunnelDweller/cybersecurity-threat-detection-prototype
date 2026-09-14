

import os
import numpy as np
import joblib
from scipy.stats import wilcoxon

RESULTS_DIR = "./results"


def load(name):
    return joblib.load(os.path.join(RESULTS_DIR, name))


def report_pairwise(name_a, vals_a, name_b, vals_b, metric, n_used):
    vals_a = np.array(vals_a[:n_used])
    vals_b = np.array(vals_b[:n_used])
    diff = vals_a - vals_b

    if np.all(diff == 0):
        print(f"  {name_a} vs {name_b} on {metric} (n={n_used} folds):")
        print(f"    All {n_used} paired differences are exactly zero -- "
              f"Wilcoxon cannot be computed (no variation to test).")
        return

    try:
        stat, p = wilcoxon(vals_a, vals_b)
    except ValueError as e:
        print(f"  {name_a} vs {name_b} on {metric}: could not compute ({e})")
        return

    mean_diff = np.mean(diff)
    sig = "significant (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"

    print(f"  {name_a} vs {name_b} on {metric} (n={n_used} folds):")
    print(f"    Mean difference ({name_a} - {name_b}): {mean_diff:+.4f}")
    print(f"    Wilcoxon statistic: {stat:.4f}, p-value: {p:.4f} -> {sig}")


def main():
    print("Loading fold-level results for all three models...\n")
    rf_res  = load("rf_results.pkl")
    nn_res  = load("nn_results.pkl")
    dqn_res = load("dqn_results.pkl")

    models = {"RF": rf_res, "MLP": nn_res, "DQN": dqn_res}
    n_folds = {name: res["n_folds"] for name, res in models.items()}

    print("=" * 70)
    print("  SUMMARY TABLE — Mean ± Std across folds")
    print("=" * 70)
    print(f"{'Model':<8} {'Folds':<7} {'Accuracy':<18} {'F1-Score':<18} {'Train Time (s)':<18}")
    print("-" * 70)
    for name, res in models.items():
        acc = res["summary"]["accuracy"]
        f1  = res["summary"]["f1"]
        tt  = res["summary"]["train_time"]
        print(f"{name:<8} {res['n_folds']:<7} {acc['mean']:.4f} ± {acc['std']:.4f}   "
              f"{f1['mean']:.4f} ± {f1['std']:.4f}   "
              f"{tt['mean']:.2f} ± {tt['std']:.2f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  PAIRWISE STATISTICAL COMPARISONS (Wilcoxon signed-rank test)")
    print("=" * 70)
    print("\nNOTE: RF and MLP have 10 paired folds each. DQN has 6 (a subset")
    print("of the same underlying splits, for practical runtime reasons --")
    print("see train_dqn.py). Comparisons involving DQN therefore use only")
    print("the first 6 folds of RF/MLP's results, matched to DQN's 6.\n")

    dqn_n = n_folds["DQN"]
    pairs = [
        ("RF", "MLP", n_folds["RF"]), 
        ("RF", "DQN", dqn_n),            
        ("MLP", "DQN", dqn_n),           
    ]

    for metric in ["accuracy", "f1"]:
        print(f"\n--- Metric: {metric} ---")
        for a, b, n_use in pairs:
            report_pairwise(
                a, models[a]["summary"][metric]["all"],
                b, models[b]["summary"][metric]["all"],
                metric, n_use
            )

    print("\nInterpretation note for the thesis:")
    print("A p-value below 0.05 indicates the observed difference between")
    print("two models across the compared folds is unlikely to be due to")
    print("chance alone. A p-value at or above 0.05 means the difference")
    print("could plausibly be due to random variation between folds, and")
    print("should NOT be reported as a confirmed difference between models.")

    output = {"summary": {name: res["summary"] for name, res in models.items()},
              "n_folds": n_folds}
    joblib.dump(output, os.path.join(RESULTS_DIR, "comparison_summary.pkl"))
    print(f"\nSaved combined summary to '{RESULTS_DIR}/comparison_summary.pkl'")


if __name__ == "__main__":
    main()