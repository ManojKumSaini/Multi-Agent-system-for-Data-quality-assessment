
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy.stats import norm
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
import warnings
import os


# ============================================================
# DISTRIBUTION FUNCTIONS (from Guide)
# ============================================================

def exp_decay(x, lambd):
    return np.exp(-lambd * x)


def weibull_decay(x, lambd, alpha):
    return np.exp(-lambd * (x ** alpha))


def exp_weibull_decay(x, lambd, alpha, beta):
    return 1 - (1 - np.exp(-lambd * x ** alpha)) ** beta


def piecewise_exp_decay(x, lambda1, lambda2, b1):
    x = np.array(x, dtype=float)
    y = np.zeros_like(x)
    mask1 = x < b1
    mask2 = x >= b1
    y[mask1] = np.exp(-lambda1 * x[mask1])
    y[mask2] = np.exp(-lambda1 * b1) * np.exp(-lambda2 * (x[mask2] - b1))
    return y


def logistic_decay(x, lambd, alpha):
    return 1 / (1 + (x / lambd) ** alpha)


def lognormal_decay(x, sigma, mu):
    x = np.maximum(x, 1e-8)
    return 1 - norm.cdf((np.log(x) - mu) / sigma)


def gamma_decay(x, lambd, beta):
    return 1 - gammainc(beta, lambd * x)


def generalized_gamma_decline(x, lambd, beta, p):
    x = np.maximum(x, 1e-8)
    return 1 - gammainc(beta / p, (x / lambd) ** p)


def burr_xii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return (1 + x ** c) ** (-k)


def burr_iii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return 1 - (1 + x ** (-c)) ** (-k)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_piecewise_bounds(t_max):
    return ([0, 0, 1e-3], [np.inf, np.inf, t_max - 1e-3])


# Distribution registry
distributions = [
    {"name": "Exponential", "func": exp_decay, "bounds": (0, np.inf), "n_params": 1},
    {"name": "Weibull", "func": weibull_decay, "bounds": (0, np.inf), "n_params": 2},
    {"name": "Exp_Weibull", "func": exp_weibull_decay, "bounds": (0, np.inf), "n_params": 3},
    {"name": "Piecewise_Exp", "func": piecewise_exp_decay, "bounds": "dynamic", "n_params": 3},
    {"name": "Logistic", "func": logistic_decay, "bounds": (0, np.inf), "n_params": 2},
    {"name": "LogNormal", "func": lognormal_decay, "bounds": (0, np.inf), "n_params": 2},
    {"name": "Gamma", "func": gamma_decay, "bounds": (0, np.inf), "n_params": 2},
    {"name": "Gen_Gamma", "func": generalized_gamma_decline, "bounds": (0, np.inf), "n_params": 3},
    {"name": "Burr_XII", "func": burr_xii_decay, "bounds": (0, np.inf), "n_params": 2},
    {"name": "Burr_III", "func": burr_iii_decay, "bounds": (0, np.inf), "n_params": 2},
]


# ============================================================
# CORE PROCESSING FUNCTION
# ============================================================

def process_topic(topic_df, topic_id):
    """Process a single topic: data prep, exhaustive search, rank, output."""

    topic_df = topic_df.copy()
    topic_name = f"topic_{topic_id}"

    # ----------------------------------------------------------
    # Step 1: Data Preparation
    # ----------------------------------------------------------

    # Fix NaN labels where entity_similarity_score == 1.0
    mask = topic_df["final_label"].isna() & (topic_df["entity_similarity_score"] == 1.0)
    topic_df.loc[mask, "final_label"] = False

    # Keep only: (False AND entity_similarity==1.0) OR (True)
    topic_df = topic_df[
        ((topic_df["final_label"] == False) & (topic_df["entity_similarity_score"] == 1.0))
        | (topic_df["final_label"] == True)
    ].copy()

    # Convert to int
    topic_df["final_label"] = topic_df["final_label"].astype(int)

    # Find max date_difference among positive labels
    max_true_date = topic_df.loc[topic_df["final_label"] == 1, "date_difference"].max()
    if pd.isna(max_true_date):
        print(f"  [SKIP] Topic {topic_id}: No positive labels found.")
        return None

    # Truncate time window
    topic_df = topic_df[topic_df["date_difference"] <= max_true_date].copy()

    # Print stats
    row_count = len(topic_df)
    count_1 = int(topic_df["final_label"].sum())
    count_0 = row_count - count_1
    print(f"  Topic {topic_id} | Rows: {row_count} | Matches(1): {count_1} | Non-matches(0): {count_0}")

    # Skip if insufficient data
    if row_count < 50:
        print(f"  [SKIP] Topic {topic_id}: Fewer than 50 rows after preparation.")
        return None

    # ----------------------------------------------------------
    # Step 2: Exhaustive Search
    # ----------------------------------------------------------

    results = []
    t_max = topic_df["date_difference"].max()

    for bin_size in range(1, 200):
        try:
            # Quantile-based binning
            topic_df["time_bin"] = pd.qcut(
                topic_df["date_difference"], q=bin_size, duplicates="drop"
            )
            topic_df["mid_bin"] = topic_df["time_bin"].apply(
                lambda interval: int((interval.left + interval.right) / 2)
            )

            # Train/test split: 20% of min bin count, stratified per bin
            bin_counts = topic_df["mid_bin"].value_counts()
            n_samples = int(0.2 * bin_counts.min())
            if n_samples < 1:
                continue

            # Stratified sampling per bin
            test_df = topic_df.groupby("mid_bin", group_keys=False).apply(
                lambda group: group.sample(n=min(n_samples, len(group) - 1), random_state=42)
            )
            train_df = topic_df.drop(test_df.index)

            # Skip if test has only one class
            if test_df["final_label"].nunique() < 2:
                continue

            # Fit each distribution
            train_x = train_df["mid_bin"].values.astype(float)
            train_y = train_df["final_label"].values.astype(float)
            test_x = test_df["mid_bin"].values.astype(float)
            test_y = test_df["final_label"].values.astype(float)

            for dist in distributions:
                try:
                    # Get bounds
                    if dist["bounds"] == "dynamic":
                        bounds = get_piecewise_bounds(t_max)
                    else:
                        bounds = dist["bounds"]

                    # Fit on train
                    params, _ = curve_fit(
                        dist["func"],
                        train_x,
                        train_y,
                        bounds=bounds,
                        maxfev=10000,
                    )

                    # Predict on test
                    predictions = dist["func"](test_x, *params)
                    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

                    # Evaluate
                    log_loss_val = log_loss(test_y, predictions)
                    auc_val = roc_auc_score(test_y, predictions)
                    n_params = dist["n_params"]
                    aic_val = 2 * n_params + 2 * log_loss_val
                    bic_val = np.log(len(train_df)) * n_params + 2 * log_loss_val

                    results.append({
                        "distribution": dist["name"],
                        "bin_size": bin_size,
                        "log_loss": log_loss_val,
                        "auc": auc_val,
                        "aic": aic_val,
                        "bic": bic_val,
                        "params": params.tolist(),
                    })

                except (RuntimeError, ValueError, RuntimeWarning):
                    continue

        except Exception:
            continue

    # ----------------------------------------------------------
    # Step 3: Rank Results
    # ----------------------------------------------------------

    if not results:
        print(f"  [SKIP] Topic {topic_id}: No valid models found.")
        return None

    results_df = pd.DataFrame(results)
    results_df["log_loss_rank"] = results_df["log_loss"].rank(method="min")
    results_df["auc_rank"] = results_df["auc"].rank(method="min", ascending=False)
    results_df["aic_rank"] = results_df["aic"].rank(method="min")
    results_df["bic_rank"] = results_df["bic"].rank(method="min")
    results_df["avg_rank"] = (
        results_df["log_loss_rank"]
        + results_df["auc_rank"]
        + results_df["aic_rank"]
        + results_df["bic_rank"]
    ) / 4
    results_df = results_df.sort_values("avg_rank").reset_index(drop=True)

    # Save full results
    results_df.to_csv(f"results_{topic_name}.csv", index=False)

    # ----------------------------------------------------------
    # Step 4: Output — Best Model
    # ----------------------------------------------------------

    best = results_df.iloc[0]
    print(
        f"  BEST → Distribution: {best['distribution']} | "
        f"Bins: {best['bin_size']} | "
        f"Log-Loss: {best['log_loss']:.4f} | "
        f"AUC: {best['auc']:.4f} | "
        f"Params: {best['params']}"
    )

    # Plot: empirical vs fitted curve
    try:
        # Re-bin with best bin_size for plotting
        topic_df["time_bin"] = pd.qcut(
            topic_df["date_difference"], q=int(best["bin_size"]), duplicates="drop"
        )
        topic_df["mid_bin"] = topic_df["time_bin"].apply(
            lambda interval: int((interval.left + interval.right) / 2)
        )

        # Empirical probabilities (full data, not just test)
        empirical = topic_df.groupby("mid_bin", as_index=False)["final_label"].mean()
        x_plot = np.sort(empirical["mid_bin"].values)
        y_empirical = empirical.set_index("mid_bin").loc[x_plot, "final_label"].values

        # Fitted curve
        best_func = next(d["func"] for d in distributions if d["name"] == best["distribution"])
        x_smooth = np.linspace(x_plot.min(), x_plot.max(), 200)
        y_fitted = best_func(x_smooth, *best["params"])

        plt.figure(figsize=(10, 6))
        plt.scatter(x_plot, y_empirical, color="steelblue", label="Empirical", zorder=3)
        plt.plot(x_smooth, y_fitted, color="red", linewidth=2, label=f"Fitted: {best['distribution']}")
        plt.title(f"Topic {topic_id}: {best['distribution']} (bins={int(best['bin_size'])})")
        plt.xlabel("Time (days)")
        plt.ylabel("P(matching)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_{topic_name}.png", dpi=150)
        plt.close()
        print(f"  Plot saved: plot_{topic_name}.png")
    except Exception as e:
        print(f"  [WARNING] Plot failed for topic {topic_id}: {e}")

    return {
        "topic": topic_id,
        "distribution": best["distribution"],
        "bin_size": int(best["bin_size"]),
        "params": best["params"],
        "log_loss": best["log_loss"],
        "auc": best["auc"],
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    topic_ids = [0, 1, 2, 3, 4, 5, 6, 7, 15, 17]
    summary = []

    print("=" * 60)
    print("PHASE 5: Survival Analysis & Currency Modelling")
    print("=" * 60)

    for topic_id in topic_ids:
        filepath = f"topic_{topic_id}/topic_labeled_updated.csv"

        if not os.path.exists(filepath):
            print(f"\n[SKIP] File not found: {filepath}")
            continue

        print(f"\n{'─' * 40}")
        print(f"Processing Topic {topic_id}...")
        print(f"{'─' * 40}")

        df = pd.read_csv(filepath)
        result = process_topic(df, topic_id)

        if result is not None:
            summary.append(result)

    # ----------------------------------------------------------
    # Step 5: Summary
    # ----------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("SUMMARY: Best Model Per Topic")
    print(f"{'=' * 60}")

    if summary:
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv("currency_model_summary.csv", index=False)
        print("\nSaved: currency_model_summary.csv")
    else:
        print("No valid models found for any topic.")

    print("\n# STATUS: Phase 5 - Survival Analysis & Currency Modelling - COMPLETE")


if __name__ == "__main__":
    main()