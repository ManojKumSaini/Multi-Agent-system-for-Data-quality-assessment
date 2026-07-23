import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy.stats import norm
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt

# Define distribution functions
def exp_decay(x, lambd):
    return np.exp(-lambd * x)

def weibull_decay(x, lambd, alpha):
    return np.exp(-lambd * (x**alpha))

def exp_weibull_decay(x, lambd, alpha, beta):
    return 1 - (1 - np.exp(-lambd * x**alpha))**beta

def piecewise_exp_decay(x, lambda1, lambda2, b1):
    x = np.array(x, dtype=float)
    y = np.zeros_like(x)
    mask1 = x < b1
    mask2 = x >= b1
    y[mask1] = np.exp(-lambda1 * x[mask1])
    y[mask2] = np.exp(-lambda1 * b1) * np.exp(-lambda2 * (x[mask2] - b1))
    return y

def logistic_decay(x, lambd, alpha):
    return 1 / (1 + (x / lambd)**alpha)

def lognormal_decay(x, sigma, mu):
    x = np.maximum(x, 1e-8)
    return 1 - norm.cdf((np.log(x) - mu) / sigma)

def gamma_decay(x, lambd, beta):
    return 1 - gammainc(beta, lambd * x)

def generalized_gamma_decline(x, lambd, beta, p):
    x = np.maximum(x, 1e-8)
    return 1 - gammainc(beta / p, (x / lambd)**p)

def burr_xii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return (1 + x**c)**(-k)

def burr_iii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return 1 - (1 + x**(-c))**(-k)

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

def get_piecewise_bounds(t_min, t_max):
    return ([0, 0, t_min + 1e-3], [np.inf, np.inf, t_max - 1e-3])

def process_topic(topic_df, topic):
    # Step 1: Data Preparation
    # Set NaN final_label to 0 where entity_similarity_score == 1.0
    mask = (topic_df['final_label'].isna()) & (topic_df['entity_similarity_score'] == 1.0)
    topic_df.loc[mask, 'final_label'] = 0
    
    # Filter rows
    condition = ((topic_df['final_label'] == 0) & (topic_df['entity_similarity_score'] == 1.0)) | (topic_df['final_label'] == 1)
    topic_df = topic_df[condition]
    
    # Convert final_label to int
    topic_df['final_label'] = topic_df['final_label'].astype(int)
    
    # Find max date_difference where final_label == 1
    max_date = topic_df[topic_df['final_label'] == 1]['date_difference'].max()
    topic_df = topic_df[topic_df['date_difference'] <= max_date]
    
    # Check row count
    if len(topic_df) < 50:
        print(f"Topic: {topic} - Less than 50 rows, skipping.")
        return
    
    print(f"Topic: {topic}, Rows: {len(topic_df)}, 1s: {sum(topic_df['final_label'])}, 0s: {len(topic_df) - sum(topic_df['final_label'])}")
    
    # Step 2: Exhaustive Search
    results = []
    t_min = topic_df['date_difference'].min()
    t_max = topic_df['date_difference'].max()
    
    for bin_size in range(1, 200):
        try:
            time_bin = pd.qcut(topic_df['date_difference'], q=bin_size, duplicates='drop')
            topic_df['time_bin'] = time_bin
            unique_bins = topic_df['time_bin'].unique()
            
            # Calculate mid_bin
            mid_bin = []
            for bin in unique_bins:
                subset = topic_df[topic_df['time_bin'] == bin]
                min_date = subset['date_difference'].min()
                max_date = subset['date_difference'].max()
                mid = (min_date + max_date) / 2
                mid_bin.append(mid)
            mid_bin = np.array(mid_bin)
            
            # Train/test split
            n_samples = int(0.2 * min(topic_df['time_bin'].value_counts()))
            if n_samples < 1:
                continue
            
            test_samples = []
            train_samples = []
            for bin in unique_bins:
                subset = topic_df[topic_df['time_bin'] == bin]
                n = len(subset)
                if n == 0:
                    continue
                n_samples_bin = min(n_samples, n)
                test_subset = subset.sample(n=n_samples_bin, random_state=42)
                test_samples.append(test_subset)
                train_subset = subset.drop(test_subset.index)
                train_samples.append(train_subset)
            
            test_df = pd.concat(test_samples)
            train_df = pd.concat(train_samples)
            
            # Check if test has only one class
            if len(test_df['final_label'].unique()) == 1:
                continue
            
            # Process each distribution
            for dist in distributions:
                func = dist['func']
                n_params = dist['n_params']
                bounds = dist['bounds']
                
                if bounds == "dynamic":
                    bounds = get_piecewise_bounds(t_min, t_max)
                
                try:
                    params, _ = curve_fit(func, train_df['mid_bin'], train_df['final_label'], bounds=bounds, maxfev=10000)
                    p = func(test_df['mid_bin'], *params)
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    
                    log_loss_val = log_loss(test_df['final_label'], p)
                    auc_val = roc_auc_score(test_df['final_label'], p)
                    aic_val = 2 * n_params + 2 * log_loss_val
                    bic_val = np.log(len(train_df)) * n_params + 2 * log_loss_val
                    
                    results.append({
                        'distribution': dist['name'],
                        'bin_size': bin_size,
                        'log_loss': log_loss_val,
                        'auc': auc_val,
                        'aic': aic_val,
                        'bic': bic_val,
                        'params': params
                    })
                except (RuntimeError, ValueError, RuntimeWarning):
                    continue
        except Exception as e:
            continue
    
    # Step 3: Rank Results
    if not results:
        print(f"Topic: {topic} - No valid models found.")
        return
    
    results_df = pd.DataFrame(results)
    results_df['log_loss_rank'] = results_df['log_loss'].rank(method='min')
    results_df['auc_rank'] = results_df['auc'].rank(method='min', ascending=False)
    results_df['avg_rank'] = (results_df['log_loss_rank'] + results_df['auc_rank']) / 2
    best_model = results_df.sort_values('avg_rank').iloc[0]
    
    # Step 4: Output
    best_model_row = {
        'topic': topic,
        'distribution': best_model['distribution'],
        'bin_size': best_model['bin_size'],
        'params': best_model['params'],
        'log_loss': best_model['log_loss'],
        'auc': best_model['auc']
    }
    
    # Save to results file
    results_df.to_csv(f'results_{topic}.csv', index=False)
    
    # Plotting
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(test_df['mid_bin'], test_df['final_label'].mean(), label='Empirical')
        x_vals = np.linspace(t_min, t_max, 100)
        y_vals = func(x_vals, *best_model['params'])
        plt.plot(x_vals, y_vals, label='Fitted')
        plt.title(f"Topic {topic}: {best_model['distribution']} (bins={best_model['bin_size']})")
        plt.xlabel("Time (days)")
        plt.ylabel("P(matching)")
        plt.legend()
        plt.savefig(f'plot_{topic}.png')
        plt.close()
    except ImportError:
        pass
    
    return best_model_row

# Main execution
df = pd.read_csv('topic_labeled_updated.csv')
topics = df['topic'].unique() if 'topic' in df.columns else ['all']
results_summary = []

for topic in topics:
    if topic == 'all':
        topic_df = df.copy()
    else:
        topic_df = df[df['topic'] == topic]
    
    best_model = process_topic(topic_df, topic)
    if best_model:
        results_summary.append(best_model)

# Save summary
results_summary_df = pd.DataFrame(results_summary)
results_summary_df.to_csv('currency_model_summary.csv', index=False)

STATUS: Phase 5 - Survival Analysis & Currency Modelling - Task Completed. Ready for Next Step.