"""Fit one chromatographic peak from a two-column file.

Usage:
    python example_minimal.py sample.csv --M 3
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from chromatopeak import ChromatoPeak


def load_two_columns(filename, time_scale=1.0):
    path = pathlib.Path(filename)
    try:
        data = np.loadtxt(path, delimiter=",", comments="#")
    except ValueError:
        data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("input file must contain at least two columns: time and signal")
    t = np.asarray(data[:, 0], dtype=np.float64) * float(time_scale)
    y = np.asarray(data[:, 1], dtype=np.float64)
    order = np.argsort(t)
    return t[order], y[order]


def crop_peak(t_full, y_full, threshold_frac=1e-3, subtract_baseline=True):
    t_full = np.asarray(t_full, dtype=np.float64)
    y_full = np.asarray(y_full, dtype=np.float64)
    if subtract_baseline:
        y_full = y_full - np.min(y_full)
    ymax = float(np.max(y_full))
    if ymax <= 0.0:
        raise ValueError("signal has no positive values after preprocessing")

    idx_max = int(np.argmax(y_full))
    threshold = float(threshold_frac) * ymax
    try:
        idx_start = int(np.where(y_full > threshold)[0][0])
        idx_end_rel = int(np.where(y_full[idx_max:] < threshold)[0][0])
        idx_end = idx_max + idx_end_rel
    except IndexError:
        idx_start = 0
        idx_end = len(y_full) - 1

    t = t_full[idx_start:idx_end + 1]
    y = y_full[idx_start:idx_end + 1]
    return t, 100.0 * y / ymax, t_full, 100.0 * y_full / ymax, idx_start, idx_end


def print_fit_result(entry):
    m = entry["M"]
    popt = entry["popt"]
    Lambda = popt[3:3 + m]
    theta = popt[3 + m:3 + 2 * m]

    print(f"\nModel, M={m}")
    print("-" * 72)
    print(f"A        = {popt[0]: .8e}")
    print(f"mu_G     = {popt[1]: .8e}")
    print(f"sigma_G  = {popt[2]: .8e}")
    for j, value in enumerate(Lambda, start=1):
        print(f"Lambda_{j} = {value: .8e}")
    for j, value in enumerate(theta, start=1):
        print(f"theta_{j}  = {value: .8e}")
    print(f"SSE={entry['sse']:.8e}   RMSE={entry['rmse']:.8e}   StdErr={entry['stderr']:.8e}")
    print(f"L_used={entry['L_used']}   L_required={entry['L_required']}   deficit={entry['deficit']:.3e}")


def build_curves(history, t_fit, y_data):
    curves = {}
    residuals = {}
    for entry in history:
        m = entry["M"]
        popt = entry["popt"]
        model_m = ChromatoPeak(M=m, L=int(entry["L_used"]))
        Lambda = popt[3:3 + m]
        theta = popt[3 + m:3 + 2 * m]
        curves[f"M={m}"] = popt[0] * model_m.pdf(t_fit, popt[1], popt[2], Lambda, theta)
        residuals[f"M={m}"] = entry["y_fit"] - y_data
    return curves, residuals


def save_curves_csv(filename, t_fit, curves, t_data, y_data, residuals):
    n = max(len(t_fit), len(t_data))

    def pad(values):
        values = np.asarray(values, dtype=np.float64)
        out = np.full(n, np.nan, dtype=np.float64)
        out[:len(values)] = values
        return out

    columns = [("t_fit", pad(t_fit))]
    for name, values in curves.items():
        columns.append(("y_" + name.replace("=", ""), pad(values)))
    columns.append(("t_data", pad(t_data)))
    columns.append(("y_data", pad(y_data)))
    for name, values in residuals.items():
        columns.append(("resid_" + name.replace("=", ""), pad(values)))

    header = ",".join(name for name, _ in columns)
    matrix = np.column_stack([values for _, values in columns])
    np.savetxt(filename, matrix, delimiter=",", header=header, comments="")


def plot_fit(t_data, y_data, t_fit, curves, residuals, figure_file=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(t_data, y_data, "x", markersize=4, alpha=0.6, label="data")
    for name, values in curves.items():
        axes[0].plot(t_fit, values, linewidth=2, label=name)
    axes[0].set_ylabel("signal")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, values in residuals.items():
        axes[1].plot(t_data, values, "o", markersize=2, linewidth=0.7, label=name)
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("residual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if figure_file is not None:
        fig.savefig(figure_file, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {figure_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fit one chromatographic peak with ChromatoPeak.")
    parser.add_argument("data_file", help="CSV/txt file with two columns: time and signal")
    parser.add_argument("--M", type=int, default=3, help="maximum number of components")
    parser.add_argument("--eps-L", type=float, default=1e-4, help="truncation tolerance for L")
    parser.add_argument("--threshold-frac", type=float, default=1e-3, help="fraction of peak height used for cropping")
    parser.add_argument("--no-baseline", action="store_true", help="do not subtract the minimum signal")
    parser.add_argument("--time-scale", type=float, default=1.0, help="factor applied to the time column before fitting")
    parser.add_argument("--L0", type=int, default=30, help="initial L stored in the model object")
    parser.add_argument("--L-cap", type=int, default=2000, help="maximum allowed L")
    parser.add_argument("--L-min", type=int, default=15, help="minimum allowed L during fitting")
    parser.add_argument("--max-stages", type=int, default=3, help="maximum L-refinement stages")
    parser.add_argument("--max-nfev", type=int, default=40000, help="maximum least-squares evaluations")
    parser.add_argument("--ftol", type=float, default=1e-12)
    parser.add_argument("--xtol", type=float, default=1e-12)
    parser.add_argument("--gtol", type=float, default=1e-12)
    parser.add_argument("--save-figure", default=None, help="optional output image file")
    parser.add_argument("--save-csv", default=None, help="optional output CSV with curves and residuals")
    parser.add_argument("--no-plot", action="store_true", help="do not open the matplotlib window")
    args = parser.parse_args()

    t_full, y_full = load_two_columns(args.data_file, time_scale=args.time_scale)
    t, y, _, _, idx_start, idx_end = crop_peak(
        t_full, y_full,
        threshold_frac=args.threshold_frac,
        subtract_baseline=not args.no_baseline,
    )

    print(f"Loaded file: {args.data_file}")
    print(f"Total points: {len(t_full)}")
    print(f"Fitted points: {len(t)}   crop indices: [{idx_start}, {idx_end}]")
    print(f"t range used: {t[0]:.8e} to {t[-1]:.8e}")

    model = ChromatoPeak(M=args.M, L=args.L0)
    _, progressive = model.fit_progressive(
        t, y,
        eps_L=args.eps_L,
        L_cap=args.L_cap,
        L_min=args.L_min,
        max_stages=args.max_stages,
        verbose=False,
        ftol=args.ftol,
        xtol=args.xtol,
        gtol=args.gtol,
        max_nfev=args.max_nfev,
    )

    history = progressive["history"]
    for entry in history:
        print_fit_result(entry)

    if progressive["failures"]:
        print("\nProgressive fit did not reach the requested M.")
        for failure in progressive["failures"]:
            print(f"  Failed at M={failure['M']}")

    print("\nSummary")
    print("-" * 72)
    print(f"{'Model':<8s} {'SSE':>14s} {'RMSE':>14s} {'StdErr':>14s} {'L':>8s}")
    for entry in history:
        print(f"M={entry['M']:<6d} {entry['sse']:14.6e} {entry['rmse']:14.6e} {entry['stderr']:14.6e} {entry['L_used']:8d}")

    t_fit = np.linspace(t[0], t[-1], 1200)
    curves, residuals = build_curves(history, t_fit, y)

    if args.save_csv is not None:
        save_curves_csv(args.save_csv, t_fit, curves, t, y, residuals)
        print(f"CSV saved to: {args.save_csv}")

    if not args.no_plot:
        plot_fit(t, y, t_fit, curves, residuals, figure_file=args.save_figure)


if __name__ == "__main__":
    main()
