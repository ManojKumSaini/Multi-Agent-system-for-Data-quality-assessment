import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammainc
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score

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

# Define registry
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

def process_topic(topic_df):
    # Step 1: Data Preparation
    topic_df['final_label'] = topic_df['final_label'].fillna(False)
    topic_df = topic_df[( (topic_df['final_label'] == False) & (topic_df['entity_similarity_score'] == 1.0) ) | (topic_df['final_label'] == True)]
    topic_df['final_label'] = topic_df['final_label'].astype(int)
    
    max_date = topic_df[topic_df['final_label'] == 1]['date_difference'].max()
    if pd.isna(max_date):
        print(f"Topic {topic} has no positive labels. Skipping.")
        return None
    
    topic_df = topic_df[topic_df['date_difference'] <= max_date]
    row_count = len(topic_df)
    pos_count = topic_df[topic_df['final_label'] == 1].shape[0]
    neg_count = row_count - pos_count
    
    if row_count < 50:
        print(f"Topic {topic} has fewer than 50 rows. Skipping.")
        return None
    
    print(f"Topic {topic}: Rows={row_count}, Pos={pos_count}, Neg={neg_count}")
    
    # Step 2: Exhaustive Search
    results = []
    for bin_size in range(1, 200):
        try:
            time_bin = pd.qcut(topic_df['date_difference'], q=bin_size, duplicates='drop')
            topic_df['time_bin'] = time_bin
            mid_bin = []
            for bin in time_bin.cat.categories:
                subset = topic_df[topic_df['time_bin'] == bin]
                min_date = subset['date_difference'].min()
                max_date = subset['date_difference'].max()
                mid = (min_date + max_date) / 2
                mid_bin.append(mid)
            mid_bin = np.array(mid_bin, dtype=int)  # Convert to int as required
            
            # Train/test split
            bin_counts = topic_df['time_bin'].value_counts()
            n_samples = int(0.2 * bin_counts.min())
            if n_samples < 1:
                continue
            
            test_rows = []
            train_rows = []
            for bin in time_bin.cat.categories:
                bin_df = topic_df[topic_df['time_bin'] == bin]
                test = bin_df.sample(n_samples, random_state=42)
                train = bin_df.drop(test.index)
                test_rows.append(test)
                train_rows.append(train)
            
            test_df = pd.concat(test_rows)
            train_df = pd.concat(train_rows)
            
            if len(test_df[test_df['final_label'] == 1]) == 0 or len(test_df[test_df['final_label'] == 0]) == 0:
                continue
            
            # Fit distributions
            for dist in distributions:
                if dist['bounds'] == 'dynamic':
                    t_min = topic_df['date_difference'].min()
                    t_max = topic_df['date_difference'].max()
                    bounds = get_piecewise_bounds(t_min, t_max)
                else:
                    bounds = dist['bounds']
                
                try:
                    params, _ = curve_fit(dist['func'], train_df['date_difference'], train_df['final_label'], bounds=bounds, maxfev=10000)
                    p = dist['func'](test_df['date_difference'], *params)
                    p = np.clip(p, 1e-15, 1-1e-15)
                    
                    log_loss_val = log_loss(test_df['final_label'], p)
                    auc_val = roc_auc_score(test_df['final_label'], p)
                    aic_val = 2*dist['n_params'] + 2*log_loss_val
                    bic_val = np.log(len(train_df))*dist['n_params'] + 2*log_loss_val
                    
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
        except:
            continue
    
    if not results:
        print(f"Topic {topic} has no valid models. Skipping.")
        return None
    
    # Step 3: Rank Results
    results_df = pd.DataFrame(results)
    results_df['log_loss_rank'] = results_df['log_loss'].rank(method='min')
    results_df['auc_rank'] = results_df['auc'].rank(method='min', ascending=False)
    results_df['aic_rank'] = results_df['aic'].rank(method='min')
    results_df['bic_rank'] = results_df['bic'].rank(method='min')
    results_df['avg_rank'] = (results_df['log_loss_rank'] + results_df['auc_rank'] + results_df['aic_rank'] + results_df['bic_rank']) / 4
    best_model = results_df.sort_values('avg_rank').iloc[0]
    
    # Step 4: Output
    print(f"Topic {topic}: Best Model - {best_model['distribution']} (bin_size={best_model['bin_size']})")
    print(f"Params: {best_model['params']}, Log-Loss: {best_model['log_loss']}, AUC: {best_model['auc']}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mid_bin, topic_df['final_label'].mean(), label='Empirical')
    x_vals = np.linspace(0, topic_df['date_difference'].max(), 100)
    y_vals = best_model['func'](x_vals, *best_model['params'])
    plt.plot(x_vals, y_vals, label='Fitted')
    plt.title(f"Topic {topic}: {best_model['distribution']} (bins={best_model['bin_size']})")
    plt.xlabel("Time (days)")
    plt.ylabel("P(matching)")
    plt.legend()
    plt.savefig(f"results_{topic}.png")
    plt.close()
    
    # Save per-topic CSV
    results_df.to_csv(f'results_{topic}.csv', index=False)
    
    # Step 5: Summary
    return {
        'topic': topic,
        'distribution': best_model['distribution'],
        'params': best_model['params'],
        'log_loss': best_model['log_loss'],
        'auc': best_model['auc']
    }

def main():
    data = pd.read_csv('topic_labeled_updated.csv')
    topics = data['topic'].unique()
    results_summary = []
    
    for topic in topics:
        topic_df = data[data['topic'] == topic].copy()
        if 'topic' not in data.columns:
            topic_df = data.copy()
        result = process_topic(topic_df)
        if result:
            results_summary.append(result)
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('currency_model_summary.csv', index=False)
    print("Processing complete. Results saved to currency_model_summary.csv")

if __name__ == "__main__":
    main()