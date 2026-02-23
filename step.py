''''
MIT License
Copyright (c) 2025 Hernán R. Sánchez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
from scipy.stats import norm, poisson
from scipy.special import hyp1f1, gammaln, erfcx
import matplotlib.pyplot as plt
from time import perf_counter
from math import comb



def _poch_scalar(a, k):
    if k == 0:
        return 1.0
    return float(np.exp(gammaln(a + k) - gammaln(a)))

def hyp1f1_scaled_pos_asymp(a, b, Z, K=6):
    Z = np.asarray(Z, dtype=np.float64)
    if np.any(Z <= 0):
        raise ValueError("hyp1f1_scaled_pos_asymp requires Z>0.")

    log_pref = (gammaln(b) - gammaln(a)) + (a - b) * np.log(Z)

    S = np.ones_like(Z)
    for k in range(1, K):
        ck = _poch_scalar(b - a, k) * _poch_scalar(1.0 - a, k) / float(np.exp(gammaln(k + 1.0)))
        S = S + ck / (Z ** k)

    return np.exp(log_pref) * S

def _signed_logsumexp_two(loga, signa, logb, signb):

    a_zero = ~np.isfinite(loga)
    b_zero = ~np.isfinite(logb)

    sign = np.where(a_zero, signb, signa)
    logabs = np.where(a_zero, logb, loga)

    both = (~a_zero) & (~b_zero)
    if not np.any(both):
        return sign, logabs

    la = loga[both]; lb = logb[both]
    sa = signa[both]; sb = signb[both]

    swap = lb > la
    la2 = np.where(swap, lb, la)
    lb2 = np.where(swap, la, lb)
    sa2 = np.where(swap, sb, sa)
    sb2 = np.where(swap, sa, sb)

    r = np.exp(lb2 - la2)
    inner = sa2 + sb2 * r

    s = np.sign(inner)
    with np.errstate(divide="ignore", invalid="ignore"):
        l = la2 + np.log(np.abs(inner))

    sign_out = sign.copy()
    logabs_out = logabs.copy()
    sign_out[both] = s
    logabs_out[both] = l
    return sign_out, logabs_out



def peak(t_vals, mu_G, sigma_G, beta_s, Lambda_s, l_bounds=None, tol=1e-12,
                  a_switch=6.0, Z_switch=40.0, asymp_K=6):
    """
    Original 1F1 formulation, but numerically stabilized on the left flank.

    Parameter mapping to the theoretical stochastic-diffusive model (Sec. 4.3):
    ---------------------------------------------------------------------------
     mu_G = (1 + mu_f) * (x / v)
     sigma_G^2 = (1 + mu_f)^2 * ( 2*D_eff*x / v^3 + sigma_0^2 / v^2 ) + sigma_f^2 * (x / v)
     beta_s   : Release rate of the slow-kinetic mechanism
     Lambda_s : Expected number of slow-kinetic retention events per particle

     x        : column length
     v        : linear mobile phase velocity
     D_eff    : effective axial dispersion coefficient
     sigma_0^2: injected plug variance
     mu_f     : expected fast retention factor contribution = lambda_f / beta_f
     sigma_f^2: variance-rate prefactor for fast kinetics = 2*lambda_f / beta_f^2
    """
    t = np.atleast_1d(t_vals).astype(np.float64)

    if sigma_G <= 0:
        raise ValueError("sigma_G must be positive.")
    if Lambda_s < 0:
        raise ValueError("Lambda_s cannot be negative.")
    if beta_s < 0:
        raise ValueError("beta_s cannot be negative.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    if Lambda_s == 0 or beta_s == 0:
        return norm.pdf(t, loc=mu_G, scale=sigma_G)

    total_pdf = np.zeros_like(t)

    # --- Poisson truncation ---
    if l_bounds is None:
        l_max = int(poisson.isf(tol, Lambda_s))
        l_min = int(poisson.ppf(tol, Lambda_s))
    else:
        l_min, l_max = l_bounds
        if l_min < 0:
            raise ValueError("l_min must be non-negative.")
        if l_max < l_min:
            raise ValueError("l_max must be >= l_min.")

    l_range = np.arange(l_min, l_max + 1, dtype=int)
    if l_range.size == 0:
        return norm.pdf(t, loc=mu_G, scale=sigma_G)

    log_weights = poisson.logpmf(l_range, Lambda_s)

    # --- precompute ---
    sigma2 = sigma_G**2
    log_sigma = np.log(sigma_G)
    log_beta = np.log(beta_s)
    log_2 = np.log(2.0)
    kinetic_shift = beta_s * sigma2

    diff_shifted = t - mu_G - kinetic_shift
    Z = (diff_shifted**2) / (2.0 * sigma2)

    # left-flank indicator via a=(mu+beta*sigma^2-t)/(sigma*sqrt2)
    a = (mu_G + kinetic_shift - t) / (sigma_G * np.sqrt(2.0))
    left_deep = (a > a_switch) & (Z > Z_switch)

    log_stable_exp = -beta_s * (t - mu_G) + 0.5 * (beta_s * sigma_G)**2

    for l, log_w in zip(l_range, log_weights):
        l = int(l)

        if l == 0:
            total_pdf += np.exp(log_w) * norm.pdf(t, loc=mu_G, scale=sigma_G)
            continue

        log_base_coeff = (-0.5 * l * log_2) + (l * log_beta) + ((l - 2) * log_sigma)

        lg_l_div_2 = gammaln(l / 2.0)
        lg_l_plus_1_div_2 = gammaln((l + 1) / 2.0)

        log_c1 = log_sigma - 0.5 * log_2 - lg_l_plus_1_div_2
        log_c2 = -lg_l_div_2

        M1 = np.empty_like(t)
        M2 = np.empty_like(t)

        # default branch: your Kummer(-Z)
        idx = ~left_deep
        if np.any(idx):
            Zm = -Z[idx]
            M1[idx] = hyp1f1(0.5 - l / 2.0, 0.5, Zm)
            M2[idx] = hyp1f1(1.0 - l / 2.0, 1.5, Zm)

        # deep-left branch: asymptotic exp(-Z)*1F1(original,+Z)
        if np.any(left_deep):
            Zp = Z[left_deep]
            # term1 original parameters: a1=l/2, b1=0.5
            M1[left_deep] = hyp1f1_scaled_pos_asymp(l / 2.0, 0.5, Zp, K=asymp_K)
            # term2 original: a2=(l+1)/2, b2=1.5
            M2[left_deep] = hyp1f1_scaled_pos_asymp((l + 1) / 2.0, 1.5, Zp, K=asymp_K)

        # common log prefactor
        log_common = log_w + log_base_coeff + log_stable_exp

        # signed-log evaluation of val1 + val2
        with np.errstate(divide="ignore", invalid="ignore"):
            sign1 = np.sign(M1)
            logabs1 = log_c1 + np.log(np.abs(M1))

            sign2 = np.sign(diff_shifted) * np.sign(M2)
            logabs2 = log_c2 + np.log(np.abs(diff_shifted)) + np.log(np.abs(M2))

        logabs1 = np.where((sign1 == 0) | (~np.isfinite(logabs1)), -np.inf, logabs1)
        logabs2 = np.where((sign2 == 0) | (~np.isfinite(logabs2)), -np.inf, logabs2)

        s, logabs_sum = _signed_logsumexp_two(logabs1, sign1, logabs2, sign2)

        total_pdf += s * np.exp(log_common + logabs_sum)

    total_pdf = np.where(np.isfinite(total_pdf), total_pdf, 0.0)
    return np.clip(total_pdf, 0.0, np.inf)
