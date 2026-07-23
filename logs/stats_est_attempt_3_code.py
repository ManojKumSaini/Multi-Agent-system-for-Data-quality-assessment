import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy.stats import norm
import importlib

import warnings


def binary_log_loss(y_true, y_prob, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def binary_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score requires both classes present")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def get_matplotlib_pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception:
        return None


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


def get_piecewise_bounds(t_min, t_max):
    return ([0, 0, t_min + 1e-3], [np.inf, np.inf, t_max - 1e-3])


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


def process_topic(topic_df, topic_name):
    topic_df = topic_df.copy()

    mask = topic_df["final_label"].isna() & (topic_df["entity_similarity_score"] == 1.0)
    topic_df.loc[mask, "final_label"] = False

    topic_df = topic_df[
        ((topic_df["final_label"] == False) & (topic_df["entity_similarity_score"] == 1.0))
        | (topic_df["final_label"] == True)
    ].copy()

    topic_df["final_label"] = topic_df["final_label"].astype(int)
    max_true_date = topic_df.loc[topic_df["final_label"] == 1, "date_difference"].max()
    if pd.isna(max_true_date):
        print(f"Skipping topic '{topic_name}' because it has no positive labels.")
        return None

    topic_df = topic_df[topic_df["date_difference"] <= max_true_date].copy()
    row_count = len(topic_df)
    count_1 = int(topic_df["final_label"].sum())
    count_0 = int(row_count - count_1)
    print(f"Topic: {topic_name} | Rows: {row_count} | 1s: {count_1} | 0s: {count_0}")

    if row_count < 50:
        print(f"Skipping topic '{topic_name}' due to insufficient data.")
        return None

    results = []
    t_min = topic_df["date_difference"].min()
    t_max = topic_df["date_difference"].max()

    for bin_size in range(1, 200):
        try:
            time_bin = pd.qcut(topic_df["date_difference"], q=bin_size, duplicates="drop")
            topic_df["time_bin"] = time_bin
            topic_df["mid_bin"] = topic_df["time_bin"].apply(lambda interval: int((interval.left + interval.right) / 2))

            bin_counts = topic_df["time_bin"].value_counts()
            n_samples = int(0.2 * bin_counts.min())
            if n_samples < 1:
                continue

            test_parts = []
            train_parts = []
            for bin_interval in topic_df["time_bin"].cat.categories:
                bin_df = topic_df[topic_df["time_bin"] == bin_interval]
                if len(bin_df) < 2:
                    continue
                sample_n = min(n_samples, len(bin_df) - 1)
                test_part = bin_df.sample(n=sample_n, random_state=42)
                train_part = bin_df.drop(test_part.index)
                test_parts.append(test_part)
                train_parts.append(train_part)

            if not test_parts or not train_parts:
                continue

            test_df = pd.concat(test_parts, axis=0)
            train_df = pd.concat(train_parts, axis=0)

            if len(test_df["final_label"].unique()) == 1:
                continue

            for dist in distributions:
                try:
                    if dist["bounds"] == "dynamic":
                        bounds = get_piecewise_bounds(t_min, t_max)
                    else:
                        bounds = dist["bounds"]

                    train_x = train_df["mid_bin"].values
                    train_y = train_df["final_label"].values
                    test_x = test_df["mid_bin"].values
                    test_y = test_df["final_label"].values

                    params, _ = curve_fit(
                        dist["func"],
                        train_x,
                        train_y,
                        bounds=bounds,
                        maxfev=10000,
                    )

                    predictions = dist["func"](test_x, *params)
                    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

                    log_loss_val = binary_log_loss(test_y, predictions)
                    auc_val = binary_roc_auc_score(test_y, predictions)
                    n_params = dist["n_params"]
                    aic_val = 2 * n_params + 2 * log_loss_val
                    bic_val = np.log(len(train_df)) * n_params + 2 * log_loss_val

                    results.append(
                        {
                            "distribution": dist["name"],
                            "bin_size": bin_size,
                            "log_loss": log_loss_val,
                            "auc": auc_val,
                            "aic": aic_val,
                            "bic": bic_val,
                            "params": params.tolist(),
                        }
                    )
                except (RuntimeError, ValueError, RuntimeWarning):
                    continue
        except Exception as exc:
            warnings.warn(f"Error processing bin_size={bin_size} for topic '{topic_name}': {exc}")
            continue

    if not results:
        print(f"Topic: {topic_name} - No valid models found.")
        return None

    results_df = pd.DataFrame(results)
    results_df["log_loss_rank"] = results_df["log_loss"].rank(method="min")
    results_df["auc_rank"] = results_df["auc"].rank(method="min", ascending=False)
    results_df["aic_rank"] = results_df["aic"].rank(method="min")
    results_df["bic_rank"] = results_df["bic"].rank(method="min")
    results_df["avg_rank"] = (
        results_df["log_loss_rank"] + results_df["auc_rank"] + results_df["aic_rank"] + results_df["bic_rank"]
    ) / 4
    results_df = results_df.sort_values("avg_rank").reset_index(drop=True)
    best_model = results_df.iloc[0]

    print(
        f"Topic: {topic_name}, Best Distribution: {best_model['distribution']}, "
        f"Log-Loss: {best_model['log_loss']:.4f}, AUC: {best_model['auc']:.4f}"
    )

    results_df.to_csv(f"results_{topic_name}.csv", index=False)

    plt = get_matplotlib_pyplot()
    if plt is not None:
        try:
            empirical = test_df.groupby("mid_bin", as_index=False)["final_label"].mean()
            x_vals = empirical["mid_bin"].values
            y_empirical = empirical["final_label"].values
            best_func = next(d["func"] for d in distributions if d["name"] == best_model["distribution"])
            y_vals = best_func(x_vals, *best_model["params"])

            plt.figure(figsize=(10, 6))
            plt.scatter(x_vals, y_empirical, label="Empirical")
            plt.plot(x_vals, y_vals, label="Fitted Curve")
            plt.title(f"Topic {topic_name}: {best_model['distribution']} (bins={best_model['bin_size']})")
            plt.xlabel("Time (days)")
            plt.ylabel("P(matching)")
            plt.legend()
            plt.savefig(f"plot_{topic_name}.png")
            plt.close()
        except Exception:
            pass

    return {
        "topic": topic_name,
        "distribution": best_model["distribution"],
        "params": best_model["params"],
        "log_loss": best_model["log_loss"],
        "auc": best_model["auc"],
    }


def main():
    df = pd.read_csv("topic_labeled_updated.csv")
    if "topic" not in df.columns:
        df["topic"] = "all"

    summary = []
    for topic in df["topic"].unique():
        topic_df = df[df["topic"] == topic].copy()
        best_model = process_topic(topic_df, topic)
        if best_model is not None:
            summary.append(
                {
                    "Topic": topic,
                    "Best Distribution": best_model["distribution"],
                    "Params": best_model["params"],
                    "Log-Loss": best_model["log_loss"],
                    "AUC": best_model["auc"],
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("currency_model_summary.csv", index=False)
    print("Processing complete. Results saved to currency_model_summary.csv")
    print("# STATUS: Phase 5 - Survival Analysis & Currency Modelling - Task Completed. Ready for Next Step.")


if __name__ == "__main__":
    main()
