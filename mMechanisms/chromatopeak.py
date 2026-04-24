import ctypes as ct
import pathlib
import platform

import numpy as np
from numpy.ctypeslib import ndpointer

__all__ = ["ChromatoPeak"]


def _library_name():
    system = platform.system()
    if system == "Darwin":
        return "libchromatopeak.dylib"
    if system == "Windows":
        return "chromatopeak.dll"
    return "libchromatopeak.so"


def _find_library():
    name = _library_name()
    candidates = [
        pathlib.Path(__file__).resolve().parent / name,
        pathlib.Path.cwd() / name,
        pathlib.Path(name),
    ]
    for path in candidates:
        try:
            return ct.CDLL(str(path))
        except OSError:
            continue
    raise ImportError(
        f"Fortran library {name!r} not found. Compile chromatopeak_core.f90 first."
    )


_lib = _find_library()
_c_int = ct.c_int
_c_double = ct.c_double
_dbl_ptr = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

_lib.peak_pdf.restype = None
_lib.peak_pdf.argtypes = [_dbl_ptr, _c_int, _dbl_ptr, _c_int, _c_int, _c_int, _dbl_ptr]

_lib.peak_pdf_jac.restype = None
_lib.peak_pdf_jac.argtypes = [_dbl_ptr, _c_int, _dbl_ptr, _c_int, _c_int, _c_int, _dbl_ptr, _dbl_ptr]

_lib.peak_pdf_jac_fitvars.restype = None
_lib.peak_pdf_jac_fitvars.argtypes = [_dbl_ptr, _c_int, _dbl_ptr, _c_int, _c_int, _c_int, _dbl_ptr, _dbl_ptr]

_lib.required_L_fitvars.restype = None
_lib.required_L_fitvars.argtypes = [_dbl_ptr, _c_int, _c_int, _c_double, _c_int, ct.POINTER(_c_int), ct.POINTER(_c_double)]


def _softplus(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    high = x > 30.0
    low = x < -30.0
    mid = ~(high | low)
    out[high] = x[high]
    out[low] = np.exp(x[low])
    out[mid] = np.log1p(np.exp(x[mid]))
    return out


def _softplus_inv(y):
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-15)
    out = np.empty_like(y)
    high = y > 30.0
    out[high] = y[high]
    out[~high] = np.log(np.expm1(y[~high]))
    return out


def _order_components(Lambda, theta):
    Lambda = np.atleast_1d(np.asarray(Lambda, dtype=np.float64))
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if Lambda.shape != theta.shape:
        raise ValueError("Lambda and theta must have the same length M")
    order = np.argsort(theta)
    return Lambda[order], theta[order]


def _pack_params(mu_G, sigma_G, Lambda, theta):
    Lambda, theta = _order_components(Lambda, theta)
    if sigma_G <= 0.0:
        raise ValueError("sigma_G must be > 0")
    if np.any(Lambda < 0.0):
        raise ValueError("all Lambda_j must be >= 0")
    if np.any(theta <= 0.0):
        raise ValueError("all theta_j must be > 0")
    return np.concatenate([[mu_G, sigma_G], Lambda, theta]).astype(np.float64)


def _physical_to_xi(mu_G, sigma_G, Lambda, theta):
    Lambda, theta = _order_components(Lambda, theta)
    if sigma_G <= 0.0:
        raise ValueError("sigma_G must be > 0")
    if np.any(Lambda < 0.0):
        raise ValueError("all Lambda_j must be >= 0")
    if np.any(theta <= 0.0):
        raise ValueError("all theta_j must be > 0")

    M = len(Lambda)
    xi = np.empty(2 + 2 * M, dtype=np.float64)
    xi[0] = float(mu_G)
    xi[1] = float(_softplus_inv([sigma_G])[0])
    xi[2:2 + M] = _softplus_inv(np.maximum(Lambda, 1e-15))
    xi[2 + M] = float(_softplus_inv([theta[0]])[0])
    for j in range(1, M):
        xi[2 + M + j] = float(_softplus_inv([theta[j] - theta[j - 1]])[0])
    return xi


def _xi_to_physical(xi, M):
    xi = np.asarray(xi, dtype=np.float64)
    mu_G = float(xi[0])
    sigma_G = float(_softplus([xi[1]])[0])
    Lambda = _softplus(xi[2:2 + M])
    theta = np.cumsum(_softplus(xi[2 + M:2 + 2 * M]))
    return mu_G, sigma_G, Lambda, theta


def _full_x_from_physical(p0, M):
    p0 = np.asarray(p0, dtype=np.float64)
    if len(p0) != 3 + 2 * M:
        raise ValueError(f"p0 must have length {3 + 2 * M}")
    A0 = max(float(p0[0]), 0.0)
    xi0 = _physical_to_xi(p0[1], p0[2], p0[3:3 + M], p0[3 + M:3 + 2 * M])
    return np.concatenate([[A0], xi0])


def _physical_from_full_x(x, M):
    x = np.asarray(x, dtype=np.float64)
    popt = np.empty(3 + 2 * M, dtype=np.float64)
    popt[0] = x[0]
    popt[1], popt[2], popt[3:3 + M], popt[3 + M:3 + 2 * M] = _xi_to_physical(x[1:], M)
    return popt


def _reshape_fortran_jacobian(raw, nrow, ncol):
    return np.ndarray(shape=(ncol, nrow), dtype=np.float64, buffer=raw, order="F").T.copy()


def _peak_pdf(t, params, M, L):
    t = np.ascontiguousarray(t, dtype=np.float64)
    params = np.ascontiguousarray(params, dtype=np.float64)
    f_out = np.empty(len(t), dtype=np.float64)
    _lib.peak_pdf(t, len(t), params, len(params), M, L, f_out)
    return f_out


def _peak_pdf_jac(t, params, M, L):
    t = np.ascontiguousarray(t, dtype=np.float64)
    params = np.ascontiguousarray(params, dtype=np.float64)
    nt = len(t)
    npar = len(params)
    f_out = np.empty(nt, dtype=np.float64)
    raw = np.empty(nt * npar, dtype=np.float64)
    _lib.peak_pdf_jac(t, nt, params, npar, M, L, f_out, raw)
    return f_out, _reshape_fortran_jacobian(raw, nt, npar)


def _peak_pdf_jac_fitvars(t, xi, M, L):
    t = np.ascontiguousarray(t, dtype=np.float64)
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    nt = len(t)
    nxi = len(xi)
    f_out = np.empty(nt, dtype=np.float64)
    raw = np.empty(nt * nxi, dtype=np.float64)
    _lib.peak_pdf_jac_fitvars(t, nt, xi, nxi, M, L, f_out, raw)
    return f_out, _reshape_fortran_jacobian(raw, nt, nxi)


def _required_L_fitvars(xi, M, eps_L, L_cap):
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    L_req = _c_int(0)
    deficit = _c_double(0.0)
    _lib.required_L_fitvars(xi, len(xi), M, float(eps_L), int(L_cap), ct.byref(L_req), ct.byref(deficit))
    return L_req.value, deficit.value


def _fit_stats(y_obs, y_fit, npar):
    residuals = np.asarray(y_fit - y_obs, dtype=np.float64)
    n = len(residuals)
    sse = float(np.dot(residuals, residuals))
    rmse = float(np.sqrt(sse / max(n, 1)))
    stderr = float(np.sqrt(sse / max(n - npar, 1)))
    return {"sse": sse, "rmse": rmse, "stderr": stderr, "residuals": residuals}


def _format_fit_params_vector(p):
    p = np.asarray(p, dtype=np.float64)
    M = (len(p) - 3) // 2
    Lambda = np.array2string(p[3:3 + M], precision=6, separator=", ")
    theta = np.array2string(p[3 + M:3 + 2 * M], precision=6, separator=", ")
    return f"A={p[0]:.6e}, mu_G={p[1]:.6e}, sigma_G={p[2]:.6e}, Lambda={Lambda}, theta={theta}"


def _half_height_widths(t, y):
    idx_peak = int(np.argmax(y))
    y_peak = float(y[idx_peak])
    t_peak = float(t[idx_peak])
    if y_peak <= 0.0:
        return 0.0, 0.0, t_peak
    half = 0.5 * y_peak

    left = t[0]
    for i in range(idx_peak, 0, -1):
        if y[i - 1] <= half <= y[i]:
            denom = y[i] - y[i - 1]
            frac = 0.0 if abs(denom) < 1e-15 else (half - y[i - 1]) / denom
            left = t[i - 1] + frac * (t[i] - t[i - 1])
            break

    right = t[-1]
    for i in range(idx_peak, len(t) - 1):
        if y[i] >= half >= y[i + 1]:
            denom = y[i + 1] - y[i]
            frac = 0.0 if abs(denom) < 1e-15 else (half - y[i]) / denom
            right = t[i] + frac * (t[i + 1] - t[i])
            break

    return max(t_peak - left, 0.0), max(right - t_peak, 0.0), t_peak


def _auto_seed_M1(t_data, y_data):
    t = np.asarray(t_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)
    w = np.clip(y, 0.0, None)
    area = float(np.trapz(w, t))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("The experimental signal must contain positive area for auto-initialization.")

    left_hw, right_hw, t_peak = _half_height_widths(t, w)
    mean_obs = float(np.trapz(t * w, t) / area)
    span = max(float(t[-1] - t[0]), 1e-12)
    sigma0 = max(left_hw / np.sqrt(2.0 * np.log(2.0)), 0.01 * span, 1e-6)
    tail_shift = max(mean_obs - t_peak, 0.0)
    tail_excess = max(right_hw - left_hw, 0.0)
    theta1 = max(tail_shift, 0.5 * tail_excess, 0.25 * sigma0, 1e-6)
    lam_from_shift = tail_shift / theta1 if theta1 > 0.0 else 1.0
    lam_from_width = right_hw / max(left_hw, 1e-12) - 1.0
    Lambda1 = float(np.clip(max(lam_from_shift, lam_from_width, 0.5), 0.1, 10.0))
    mu0 = float(t_peak - 0.5 * min(tail_shift, sigma0))
    return np.array([area, mu0, sigma0, Lambda1, theta1], dtype=np.float64)


def _ordered_concat_fit_params(A, mu_G, sigma_G, Lambda, theta):
    Lambda = np.asarray(Lambda, dtype=np.float64).copy()
    theta = np.asarray(theta, dtype=np.float64).copy()
    order = np.argsort(theta)
    Lambda = Lambda[order]
    theta = theta[order]
    for j in range(1, len(theta)):
        if theta[j] <= theta[j - 1]:
            theta[j] = theta[j - 1] + 1e-12
    return np.concatenate([[A, mu_G, sigma_G], Lambda, theta]).astype(np.float64)


def _append_weak_component(prev_popt, frac=0.05, C=1.10, side="up"):
    prev_popt = np.asarray(prev_popt, dtype=np.float64)
    M_prev = (len(prev_popt) - 3) // 2
    A = float(prev_popt[0])
    mu_G = float(prev_popt[1])
    sigma_G = float(prev_popt[2])
    Lambda = np.asarray(prev_popt[3:3 + M_prev], dtype=np.float64).copy()
    theta = np.asarray(prev_popt[3 + M_prev:3 + 2 * M_prev], dtype=np.float64).copy()
    order = np.argsort(theta)
    Lambda = Lambda[order]
    theta = theta[order]

    idx_target = int(np.argmax(Lambda * theta))
    lam0 = float(Lambda[idx_target])
    th0 = float(theta[idx_target])
    th_new = th0 / C if side == "down" else C * th0
    lam_new = frac * lam0
    lam_old = lam0 - lam_new * (th_new / th0)
    if lam_old <= 1e-12:
        raise ValueError("weak appended component produced a non-positive old lambda")

    Lambda[idx_target] = lam_old
    Lambda_new = np.concatenate([Lambda, [lam_new]])
    theta_new = np.concatenate([theta, [th_new]])
    return _ordered_concat_fit_params(A, mu_G, sigma_G, Lambda_new, theta_new)


def _assisted_x_bounds(prev_popt, candidate_p0, t_data):
    prev_popt = np.asarray(prev_popt, dtype=np.float64)
    candidate_p0 = np.asarray(candidate_p0, dtype=np.float64)
    m = (len(candidate_p0) - 3) // 2
    A_prev = max(float(prev_popt[0]), 1e-12)
    mu_prev = float(prev_popt[1])
    sigma_prev = max(float(prev_popt[2]), 1e-12)
    M_prev = (len(prev_popt) - 3) // 2
    Lambda_prev = np.asarray(prev_popt[3:3 + M_prev], dtype=np.float64)
    theta_prev = np.asarray(prev_popt[3 + M_prev:3 + 2 * M_prev], dtype=np.float64)
    span = max(float(np.max(t_data) - np.min(t_data)), 1e-12)
    theta_cap = max(5.0 * span, 3.0 * float(np.max(theta_prev)), 10.0 * sigma_prev)
    lambda_cap = max(10.0, 10.0 * float(np.sum(Lambda_prev)))

    lb = np.full(3 + 2 * m, -np.inf, dtype=np.float64)
    ub = np.full(3 + 2 * m, np.inf, dtype=np.float64)
    lb[0] = 0.25 * A_prev
    ub[0] = 4.0 * A_prev
    lb[1] = mu_prev - 5.0 * sigma_prev
    ub[1] = mu_prev + 5.0 * sigma_prev
    lb[2] = 0.5 * sigma_prev
    ub[2] = 2.0 * sigma_prev
    lb[3:3 + m] = 0.0
    ub[3:3 + m] = lambda_cap
    lb[3 + m:3 + 2 * m] = 1e-12
    ub[3 + m:3 + 2 * m] = theta_cap

    x_lb = _full_x_from_physical(lb, m)
    x_ub = _full_x_from_physical(ub, m)
    gap_ub = max(theta_cap / m, 1e-12)
    z_gap_ub = float(_softplus_inv([gap_ub])[0])

    x_lb[0] = lb[0]
    x_ub[0] = ub[0]
    x_lb[1] = lb[1]
    x_ub[1] = ub[1]
    x_lb[2] = float(_softplus_inv([lb[2]])[0])
    x_ub[2] = float(_softplus_inv([ub[2]])[0])
    for j in range(m):
        x_lb[3 + j] = float(_softplus_inv([max(lb[3 + j], 1e-15)])[0])
        x_ub[3 + j] = float(_softplus_inv([ub[3 + j]])[0])
        x_lb[3 + m + j] = float(_softplus_inv([1e-12])[0])
        x_ub[3 + m + j] = z_gap_ub

    x0 = _full_x_from_physical(candidate_p0, m)
    x0 = np.minimum(np.maximum(x0, x_lb), x_ub)
    return x_lb, x_ub, _physical_from_full_x(x0, m)


def _fit_subset(model, t_data, y_data, p0, free_mask, eps_L, L_cap, L_min, max_stages, x_bounds=None, **kwargs):
    from scipy.optimize import least_squares

    M = model.M
    t_data = np.ascontiguousarray(t_data, dtype=np.float64)
    y_data = np.asarray(y_data, dtype=np.float64)
    x0_full = _full_x_from_physical(p0, M)
    free_mask = np.asarray(free_mask, dtype=bool)
    if free_mask.shape != x0_full.shape:
        raise ValueError("free_mask has incorrect shape")

    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        popt = _physical_from_full_x(x0_full, M)
        y_fit = popt[0] * _peak_pdf(t_data, popt[1:], M, model.L)
        info = {"L_used": model.L, "L_required": None, "deficit": None, "n_stages": 0, "y_fit": y_fit}
        info.update(_fit_stats(y_data, y_fit, len(popt)))
        return popt, info

    if x_bounds is None:
        lb_full = np.full(len(x0_full), -np.inf, dtype=np.float64)
        ub_full = np.full(len(x0_full), np.inf, dtype=np.float64)
        lb_full[0] = 0.0
    else:
        lb_full = np.asarray(x_bounds[0], dtype=np.float64).copy()
        ub_full = np.asarray(x_bounds[1], dtype=np.float64).copy()
        if lb_full.shape != x0_full.shape or ub_full.shape != x0_full.shape:
            raise ValueError("x_bounds has incorrect shape")

    x0_full = np.minimum(np.maximum(x0_full, lb_full), ub_full)
    L_req, deficit = _required_L_fitvars(x0_full[1:], M, eps_L, L_cap)
    L_stage = max(int(L_min), int(np.ceil(1.2 * L_req)), L_req + 5)

    kwargs = dict(kwargs)
    kwargs.setdefault("method", "trf")
    kwargs.setdefault("x_scale", "jac")

    result = None
    x_template = x0_full.copy()
    z0 = x_template[free_idx].copy()
    n_stages = 0

    for _ in range(max_stages):
        n_stages += 1
        cache = {"xi": None, "L": None, "f": None, "Jxi": None}

        def unpack(z):
            x = x_template.copy()
            x[free_idx] = z
            return x

        def eval_cached(x, L):
            xi = np.ascontiguousarray(x[1:], dtype=np.float64)
            if cache["xi"] is not None and cache["L"] == L and np.array_equal(cache["xi"], xi):
                return cache["f"], cache["Jxi"]
            f, Jxi = _peak_pdf_jac_fitvars(t_data, xi, M, L)
            cache["xi"] = xi.copy()
            cache["L"] = L
            cache["f"] = f
            cache["Jxi"] = Jxi
            return f, Jxi

        def residuals(z):
            x = unpack(z)
            f, _ = eval_cached(x, L_stage)
            return x[0] * f - y_data

        def jacobian(z):
            x = unpack(z)
            f, Jxi = eval_cached(x, L_stage)
            Jfull = np.empty((len(t_data), len(x)), dtype=np.float64)
            Jfull[:, 0] = f
            Jfull[:, 1:] = x[0] * Jxi
            return Jfull[:, free_idx]

        result = least_squares(
            residuals,
            z0,
            jac=jacobian,
            bounds=(lb_full[free_idx], ub_full[free_idx]),
            **kwargs,
        )
        z0 = result.x.copy()
        x_template[free_idx] = z0

        L_req, deficit = _required_L_fitvars(x_template[1:], M, eps_L, L_cap)
        if L_stage >= L_req:
            break
        if L_req >= L_cap:
            p_bad = _physical_from_full_x(x_template, M)
            raise RuntimeError(
                f"L_cap={L_cap} was reached before meeting eps_L={eps_L:g}. "
                f"Final deficit={deficit:.3e}. Parameters: {_format_fit_params_vector(p_bad)}"
            )
        L_stage = max(int(np.ceil(1.2 * L_req)), L_stage + 10)

    if L_stage < L_req:
        p_bad = _physical_from_full_x(x_template, M)
        raise RuntimeError(
            f"max_stages={max_stages} was exhausted before reaching the required truncation. "
            f"L_used={L_stage}, L_required={L_req}, deficit={deficit:.3e}. "
            f"Parameters: {_format_fit_params_vector(p_bad)}"
        )

    popt = _physical_from_full_x(x_template, M)
    y_fit = popt[0] * _peak_pdf(t_data, popt[1:], M, L_stage)
    info = {
        "L_used": L_stage,
        "L_required": L_req,
        "deficit": deficit,
        "n_stages": n_stages,
        "y_fit": y_fit,
        "opt_result": result,
    }
    info.update(_fit_stats(y_data, y_fit, len(popt)))
    return popt, info


class ChromatoPeak:
    """Python interface to the Fortran chromatographic peak backend.

    Fit parameters are ordered as
    [A, mu_G, sigma_G, Lambda_1, ..., Lambda_M, theta_1, ..., theta_M].
    Components are internally ordered by increasing theta.
    """

    def __init__(self, M=1, L=30):
        if M < 1:
            raise ValueError("M must be >= 1")
        if L < 1:
            raise ValueError("L must be >= 1")
        self.M = int(M)
        self.L = int(L)
        self.npar_pdf = 2 + 2 * self.M
        self.npar_fit = 3 + 2 * self.M

    def pdf(self, t, mu_G, sigma_G, Lambda, theta):
        params = _pack_params(mu_G, sigma_G, Lambda, theta)
        if len(params) != self.npar_pdf:
            raise ValueError(f"Lambda and theta must have length {self.M}")
        return _peak_pdf(t, params, self.M, self.L)

    def pdf_jac(self, t, mu_G, sigma_G, Lambda, theta):
        params = _pack_params(mu_G, sigma_G, Lambda, theta)
        if len(params) != self.npar_pdf:
            raise ValueError(f"Lambda and theta must have length {self.M}")
        return _peak_pdf_jac(t, params, self.M, self.L)

    def required_L(self, mu_G, sigma_G, Lambda, theta, eps_L=1e-8, L_cap=2000):
        xi = _physical_to_xi(mu_G, sigma_G, Lambda, theta)
        return _required_L_fitvars(xi, self.M, eps_L, L_cap)

    def fit(self, t_data, y_data, p0, eps_L=1e-8, L_cap=2000, L_min=15, max_stages=3, **kwargs):
        from scipy.optimize import least_squares

        if max_stages < 1:
            raise ValueError("max_stages must be >= 1")

        t_data = np.ascontiguousarray(t_data, dtype=np.float64)
        y_data = np.asarray(y_data, dtype=np.float64)
        if t_data.ndim != 1 or y_data.ndim != 1 or len(t_data) != len(y_data):
            raise ValueError("t_data and y_data must be one-dimensional arrays with the same length")

        x0 = _full_x_from_physical(p0, self.M)
        L_req, deficit = _required_L_fitvars(x0[1:], self.M, eps_L, L_cap)
        L_stage = max(int(L_min), int(np.ceil(1.2 * L_req)), L_req + 5)

        kwargs = dict(kwargs)
        kwargs.setdefault("method", "trf")
        kwargs.setdefault("x_scale", "jac")

        lower = np.full(len(x0), -np.inf, dtype=np.float64)
        upper = np.full(len(x0), np.inf, dtype=np.float64)
        lower[0] = 0.0

        result = None
        n_stages = 0
        for _ in range(max_stages):
            n_stages += 1
            cache = {"xi": None, "L": None, "f": None, "J": None}

            def eval_cached(xi, L):
                xi = np.ascontiguousarray(xi, dtype=np.float64)
                if cache["xi"] is not None and cache["L"] == L and np.array_equal(cache["xi"], xi):
                    return cache["f"], cache["J"]
                f, J = _peak_pdf_jac_fitvars(t_data, xi, self.M, L)
                cache["xi"] = xi.copy()
                cache["L"] = L
                cache["f"] = f
                cache["J"] = J
                return f, J

            def residuals(x):
                f, _ = eval_cached(x[1:], L_stage)
                return x[0] * f - y_data

            def jacobian(x):
                f, Jxi = eval_cached(x[1:], L_stage)
                J = np.empty((len(t_data), len(x)), dtype=np.float64)
                J[:, 0] = f
                J[:, 1:] = x[0] * Jxi
                return J

            result = least_squares(residuals, x0, jac=jacobian, bounds=(lower, upper), **kwargs)
            x0 = result.x.copy()

            L_req, deficit = _required_L_fitvars(x0[1:], self.M, eps_L, L_cap)
            if L_stage >= L_req:
                break
            if L_req >= L_cap:
                p_bad = _physical_from_full_x(x0, self.M)
                raise RuntimeError(
                    f"L_cap={L_cap} was reached before meeting eps_L={eps_L:g}. "
                    f"Final deficit={deficit:.3e}. Parameters: {_format_fit_params_vector(p_bad)}"
                )
            L_stage = max(int(np.ceil(1.2 * L_req)), L_stage + 10)

        if L_stage < L_req:
            p_bad = _physical_from_full_x(x0, self.M)
            raise RuntimeError(
                f"max_stages={max_stages} was exhausted before reaching the required truncation. "
                f"L_used={L_stage}, L_required={L_req}, deficit={deficit:.3e}. "
                f"Parameters: {_format_fit_params_vector(p_bad)}"
            )

        popt = _physical_from_full_x(x0, self.M)
        y_fit = popt[0] * _peak_pdf(t_data, popt[1:], self.M, L_stage)
        info = {
            "L_used": L_stage,
            "L_required": L_req,
            "deficit": deficit,
            "n_stages": n_stages,
            "y_fit": y_fit,
            "opt_result": result,
        }
        info.update(_fit_stats(y_data, y_fit, len(popt)))
        return popt, info

    def fit_progressive(
        self,
        t_data,
        y_data,
        eps_L=1e-8,
        L_cap=2000,
        L_min=15,
        max_stages=3,
        verbose=False,
        stop_on_failure=False,
        **kwargs,
    ):
        t_data = np.ascontiguousarray(t_data, dtype=np.float64)
        y_data = np.asarray(y_data, dtype=np.float64)
        if t_data.ndim != 1 or y_data.ndim != 1 or len(t_data) != len(y_data):
            raise ValueError("t_data and y_data must be one-dimensional arrays with the same length")

        history = []
        failures = []
        p0 = _auto_seed_M1(t_data, y_data)
        guided_specs = [
            {"frac": 0.05, "C": 1.10, "side": "up"},
            {"frac": 0.10, "C": 1.20, "side": "up"},
            {"frac": 0.10, "C": 1.10, "side": "down"},
        ]

        for m in range(1, self.M + 1):
            model_m = ChromatoPeak(M=m, L=self.L)

            if m == 1:
                try:
                    popt, info = model_m.fit(
                        t_data,
                        y_data,
                        p0,
                        eps_L=eps_L,
                        L_cap=L_cap,
                        L_min=L_min,
                        max_stages=max_stages,
                        **kwargs,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Progressive fitting failed at M=1. {exc}") from exc
            else:
                prev_popt = history[-1]["popt"]
                attempts = []
                popt = None
                info = None

                for spec in guided_specs:
                    try:
                        p0_new = _append_weak_component(prev_popt, **spec)
                        x_lb, x_ub, p0_new = _assisted_x_bounds(prev_popt, p0_new, t_data)
                        x_bounds = (x_lb, x_ub)

                        free_phase1 = np.zeros(3 + 2 * m, dtype=bool)
                        free_phase1[3:3 + m] = True
                        p1, _ = _fit_subset(
                            model_m,
                            t_data,
                            y_data,
                            p0_new,
                            free_phase1,
                            eps_L,
                            L_cap,
                            L_min,
                            max_stages,
                            x_bounds=x_bounds,
                            **kwargs,
                        )

                        free_phase2 = np.zeros(3 + 2 * m, dtype=bool)
                        free_phase2[3:3 + 2 * m] = True
                        p2, _ = _fit_subset(
                            model_m,
                            t_data,
                            y_data,
                            p1,
                            free_phase2,
                            eps_L,
                            L_cap,
                            L_min,
                            max_stages,
                            x_bounds=x_bounds,
                            **kwargs,
                        )

                        free_phase3 = np.ones(3 + 2 * m, dtype=bool)
                        popt, info = _fit_subset(
                            model_m,
                            t_data,
                            y_data,
                            p2,
                            free_phase3,
                            eps_L,
                            L_cap,
                            L_min,
                            max_stages,
                            x_bounds=x_bounds,
                            **kwargs,
                        )
                        info["guided_seed"] = p0_new
                        info["guided_spec"] = dict(spec)
                        break
                    except Exception as exc:
                        attempts.append(
                            {
                                "guided_spec": dict(spec),
                                "candidate_p0": p0_new if "p0_new" in locals() else None,
                                "message": str(exc),
                            }
                        )

                if popt is None:
                    msg = f"Progressive fit failed at M={m}. Guided continuation failed."
                    failures.append({"M": m, "errors": attempts})
                    if stop_on_failure:
                        detail = [msg]
                        for attempt in attempts:
                            spec = attempt["guided_spec"]
                            detail.append(f"frac={spec['frac']:.3f}, C={spec['C']:.3f}, side={spec['side']}")
                            if attempt["candidate_p0"] is not None:
                                detail.append(_format_fit_params_vector(attempt["candidate_p0"]))
                            detail.append(attempt["message"])
                        raise RuntimeError("\n".join(detail))
                    if verbose:
                        print(msg)
                    break

            entry = {
                "M": m,
                "popt": popt,
                "info": info,
                "y_fit": info["y_fit"],
                "sse": info["sse"],
                "rmse": info["rmse"],
                "stderr": info["stderr"],
                "L_used": info["L_used"],
                "L_required": info["L_required"],
                "deficit": info["deficit"],
            }
            history.append(entry)

            if verbose:
                Lambda = popt[3:3 + m]
                theta = popt[3 + m:3 + 2 * m]
                print(
                    f"M={m}: RMSE={entry['rmse']:.6e}, L={entry['L_used']}, "
                    f"deficit={entry['deficit']:.3e}"
                )
                print(f"  A={popt[0]:.6e}, mu_G={popt[1]:.6e}, sigma_G={popt[2]:.6e}")
                print(f"  Lambda={np.array2string(Lambda, precision=6, separator=', ')}")
                print(f"  theta ={np.array2string(theta, precision=6, separator=', ')}")

        if not history:
            raise RuntimeError("Progressive fitting failed already at M=1.")
        return history[-1]["popt"], {
            "history": history,
            "failures": failures,
            "M_reached": history[-1]["M"],
        }
