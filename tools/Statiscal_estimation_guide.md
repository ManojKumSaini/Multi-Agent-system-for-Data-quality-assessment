
# Phase 5 Guide: Distribution Functions Reference

This is a code-only reference. Copy these functions directly into your implementation.

```python
import numpy as np
from scipy.special import gammainc
from scipy.stats import norm

# ============================================================
# 1. Exponential | params: 1 | bounds: (0, np.inf)
# ============================================================
def exp_decay(x, lambd):
    return np.exp(-lambd * x)

# ============================================================
# 2. Weibull | params: 2 | bounds: (0, np.inf)
# ============================================================
def weibull_decay(x, lambd, alpha):
    return np.exp(-lambd * (x**alpha))

# ============================================================
# 3. Exponentiated Weibull | params: 3 | bounds: (0, np.inf)
# ============================================================
def exp_weibull_decay(x, lambd, alpha, beta):
    return 1 - (1 - np.exp(-lambd * x**alpha))**beta

# ============================================================
# 4. Piecewise Exponential | params: 3 | bounds: DYNAMIC
# ============================================================
def piecewise_exp_decay(x, lambda1, lambda2, b1):
    x = np.array(x, dtype=float)
    y = np.zeros_like(x)
    mask1 = x < b1
    mask2 = x >= b1
    y[mask1] = np.exp(-lambda1 * x[mask1])
    y[mask2] = np.exp(-lambda1 * b1) * np.exp(-lambda2 * (x[mask2] - b1))
    return y
# bounds must be generated dynamically from data:
# get_piecewise_bounds(t_min, t_max) -> ([0, 0, t_min+1e-3], [np.inf, np.inf, t_max-1e-3])

# ============================================================
# 5. Logistic (Log-Logistic) | params: 2 | bounds: (0, np.inf)
# ============================================================
def logistic_decay(x, lambd, alpha):
    return 1 / (1 + (x / lambd)**alpha)

# ============================================================
# 6. Log-Normal | params: 2 | bounds: (0, np.inf)
# ============================================================
def lognormal_decay(x, sigma, mu):
    x = np.maximum(x, 1e-8)
    return 1 - norm.cdf((np.log(x) - mu) / sigma)

# ============================================================
# 7. Gamma | params: 2 | bounds: (0, np.inf)
# ============================================================
def gamma_decay(x, lambd, beta):
    return 1 - gammainc(beta, lambd * x)

# ============================================================
# 8. Generalised Gamma | params: 3 | bounds: (0, np.inf)
# ============================================================
def generalized_gamma_decline(x, lambd, beta, p):
    x = np.maximum(x, 1e-8)
    return 1 - gammainc(beta / p, (x / lambd)**p)

# ============================================================
# 9. Burr XII | params: 2 | bounds: (0, np.inf)
# ============================================================
def burr_xii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return (1 + x**c)**(-k)

# ============================================================
# 10. Burr III | params: 2 | bounds: (0, np.inf)
# ============================================================
def burr_iii_decay(x, c, k):
    x = np.maximum(x, 1e-6)
    return 1 - (1 + x**(-c))**(-k)
```

## Registry (for looping)

```python
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
```

## Piecewise Bounds Helper

```python
def get_piecewise_bounds(t_min, t_max):
    return ([0, 0, t_min + 1e-3], [np.inf, np.inf, t_max - 1e-3])
```
