''''
MIT License
Copyright (c) 2025 Hernán R. Sánchez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import numpy as np
from scipy.stats import norm, poisson
from scipy.special import hyp1f1, gammaln

def peak(
    t_vals, mu_G, sigma_G, beta_s, Lambda_s, l_bounds=None, tol=1e-12
):
    """
    Computes the chromatographic peak probability density function (PDF) based on
    the Stochastic-Diffusive model described by Sánchez, H. R.

    Reference: https://doi.org/10.48550/arXiv.2511.15088 (Eq. 24)
    If you use this software, please cite the associated article:
       Sánchez, H. R. (2025). Chromatographic Peak Shape from a Stochastic Model.
       arXiv preprint arXiv:2511.15088 (or the subsequent journal publication if available).
    Parameters
    ----------
    t_vals : np.ndarray
        Time vector.
    mu_G : float
        Mean of the Gaussian component.
    sigma_G : float
        Standard deviation of the Gaussian component (sigma_G > 0).
    beta_s : float
        Kinetic release rate (beta_s >= 0).
    Lambda_s : float
        Expected number of slow retention events (Lambda_s >= 0).
    l_bounds : tuple(int, int), optional
        Manual summation bounds (l_min, l_max).
    tol : float, optional
        Tolerance for Poisson series truncation (default 1e-12).

    Returns
    -------
    np.ndarray
        PDF values.
    """
    # --- 1. Parameter Validation ---
    t = np.atleast_1d(t_vals).astype(np.float64)

    if sigma_G <= 0:
        raise ValueError("sigma_G must be positive.")
    if Lambda_s < 0:
        raise ValueError("Lambda_s cannot be negative.")
    if beta_s < 0:
        raise ValueError("beta_s cannot be negative.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    # Degenerate case: No slow retention (Lambda=0) or instantaneous release (beta=0).
    if Lambda_s == 0 or beta_s == 0:
        return norm.pdf(t, loc=mu_G, scale=sigma_G)

    total_pdf = np.zeros_like(t)

    # --- 2. Poisson Summation Range ---
    if l_bounds is None:
        # Upper Bound: P(N > l_max) <= tol
        l_max = int(poisson.isf(tol, Lambda_s))
        # Lower Bound: P(N <= l_min) <= tol
        l_min = int(poisson.ppf(tol, Lambda_s))
    else:
        l_min, l_max = l_bounds
        if l_min < 0:
            raise ValueError(f"l_min must be non-negative (got {l_min}).")
        if l_max < l_min:
            raise ValueError(f"l_bounds must satisfy l_max >= l_min >=0 (got l_min={l_min}, l_max={l_max}).")

    l_range = np.arange(l_min, l_max + 1, dtype=int)

    if l_range.size == 0:
        return norm.pdf(t, loc=mu_G, scale=sigma_G)

    log_weights = poisson.logpmf(l_range, Lambda_s)

    # --- 3. Pre-computations ---
    sigma2 = sigma_G**2
    log_sigma = np.log(sigma_G)
    log_beta = np.log(beta_s)
    log_2 = np.log(2.0)
    kinetic_shift = beta_s * sigma2

    # Geometric argument Z (always non-negative)
    diff_shifted = t - mu_G - kinetic_shift
    Z = (diff_shifted**2) / (2.0 * sigma2)

    # --- STABILITY KERNEL (Kummer Transformation) ---
    # Cmbine the Gaussian decay exp(-(t-mu)^2 / 2sigma^2) with the exp(Z)
    # from Kummer's identity analytically.
    # Result: exp( -beta_s*(t - mu_G) + 0.5*(beta_s*sigma_G)^2 )
    # This prevents intermediate overflow for large t.
    log_stable_exp = -beta_s * (t - mu_G) + 0.5 * (beta_s * sigma_G)**2

    # --- 4. Series Evaluation ---
    for l, log_w in zip(l_range, log_weights):
        l = int(l)

        # Base Case: l=0 (Pure Gaussian)
        if l == 0:
            log_pdf_0 = log_w + norm.logpdf(t, loc=mu_G, scale=sigma_G)
            total_pdf += np.exp(log_pdf_0)
            continue

        # General Case: l > 0
        # Log-space coefficients (Prefactor of Eq. 38)
        log_base_coeff = (-0.5 * l * log_2) + (l * log_beta) + ((l - 2) * log_sigma)

        lg_l_div_2 = gammaln(l / 2.0)
        lg_l_plus_1_div_2 = gammaln((l + 1) / 2.0)

        # --- Hypergeometric Terms using Kummer Identity ---
        # Identity: 1F1(a, b, Z) = exp(Z) * 1F1(b-a, b, -Z)
        # The exp(Z) is absorbed into 'log_stable_exp' above.
        # We perform the evaluation at -Z (negative argument), which is more stable.

        # Term 1: 1F1(l/2, 0.5, Z) -> becomes -> 1F1(0.5 - l/2, 0.5, -Z)
        log_c1 = log_sigma - 0.5 * log_2 - lg_l_plus_1_div_2
        M1_kummer = hyp1f1(0.5 - l / 2.0, 0.5, -Z)

        # Term 2: 1F1((l+1)/2, 1.5, Z) -> becomes -> 1F1(1.5 - (l+1)/2, 1.5, -Z)
        # Simplify params: 1.5 - (l+1)/2 = 1.0 - l/2
        log_c2_mag = -lg_l_div_2
        M2_kummer = hyp1f1(1.0 - l / 2.0, 1.5, -Z)

        # --- Assembly ---
        # We use log_stable_exp here instead of the standard Gaussian decay
        log_common = log_w + log_base_coeff + log_stable_exp

        # Linear reconstruction
        val1 = np.exp(log_c1) * M1_kummer
        val2 = np.exp(log_c2_mag) * diff_shifted * M2_kummer

        total_pdf += np.exp(log_common) * (val1 + val2)

    return total_pdf
