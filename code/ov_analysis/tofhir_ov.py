#!/usr/bin/env python
"""OV vs IV correlation study — local rewrite of old/tofhir_ov.py.

Reproduces the plots in old/plot_9506_9513/ using:
  - TRAY3_RU0/OV_TRAY3_RU0/iminuit_values_9506..9513.json  (8-run LYSO OV scan)
  - TRAY3_RU0/IV_TRAY3_RU0/vbr_dm.json                     (single IV scan run)
  - TRAY3_RU0/T3R0_struct.json                             (nominal Hamamatsu VBR)

Usage:
    python ov_analysis/tofhir_ov.py
"""

import json
import math
import re
import sys
from pathlib import Path

import iminuit
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from iminuit.cost import LeastSquares
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import use_cms_style, cms_label, COLOR_A, COLOR_B, COLOR_DATA, COLOR_DIFF, COLOR_BAND, CMAP_ORDINAL

REPO_ROOT = Path(__file__).resolve().parent.parent
OV_DIR = REPO_ROOT / "TRAY3_RU0" / "OV_TRAY3_RU0"
IV_FILE = REPO_ROOT / "TRAY3_RU0" / "IV_TRAY3_RU0" / "vbr_dm.json"
STRUCT_FILE = REPO_ROOT / "TRAY3_RU0" / "T3R0_struct.json"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# The 8 OV-scan runs, in acquisition order. Each run's json reports the same
# nominal ov/att_gain (3.0), but the true overvoltage steps down by 0.1 V per
# run within the scan (run 9513 is the highest OV, 9506 the lowest) - see
# old/tofhir_ov.py:read_runs, which derives it the same way from run order.
OV_RUNS = [9506, 9507, 9508, 9509, 9510, 9511, 9512, 9513]
OV_VALUES = [3.0 - 0.1 * i for i in reversed(range(len(OV_RUNS)))]  # 2.3 .. 3.0

VALID_PEAK_RANGE = (100, 600)  # ADC range considered a valid LYSO photopeak fit
CHANNEL_TO_ARRAY = {"CH_1": "A", "CH_2": "B"}
BAR_ORDER = [f"BAR_{i}" for i in range(16)]

# DM_11 is excluded from the OV-vs-IV correlation on independent evidence (not because it's
# inconvenient): its array B channel fails to fit on every single bar, and its array A is the
# only >4 sigma VB_shift outlier in the whole tray (-0.85 V vs +0.45..+0.62 V on every other DM).
EXCLUDED_DMS = {"11"}


def func_fit(ov, N_coef, VB_shift):
    return N_coef * 0.638 * (1 - np.exp(-0.651 * (ov - VB_shift))) * (7.044e04 + 2.895e05 * (ov - VB_shift))


def weighted_average(values, errors):
    values, errors = np.asarray(values), np.asarray(errors)
    if len(values) == 0 or np.any(errors == 0):
        return np.nan
    weights = 1 / errors**2
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_average_error(errors):
    errors = np.asarray(errors)
    if len(errors) == 0 or np.any(errors == 0):
        return np.nan
    return float(1 / np.sqrt(np.sum(1 / errors**2)))


def load_ov_scan():
    """Load the 8 LYSO OV-scan runs into data[dm][chip][bar][channel] = {'value': [...8], 'error': [...8]}."""
    data = {}
    for run in OV_RUNS:
        run_data = json.load(open(OV_DIR / f"iminuit_values_{run}.json"))["iminuit"]
        for dm, chips in run_data.items():
            for chip, bars in chips.items():
                for bar, channels in bars.items():
                    for channel, params in channels.items():
                        if channel == "BAR":
                            continue
                        mu = params["mu"]
                        entry = (data.setdefault(dm, {}).setdefault(chip, {})
                                 .setdefault(bar, {}).setdefault(channel, {"value": [], "error": []}))
                        entry["value"].append(mu["value"])
                        entry["error"].append(mu["error"])
    return data


def valid_point_mask(value, error):
    """True for points usable in the VB_shift fit: in-range and with a finite, nonzero error.

    A handful of upstream photopeak fits report error == 0 or NaN, which would give the point
    infinite weight in the LeastSquares cost (Migrad fails outright rather than just
    down-weighting a noisy point) - shared by fit_vb_shifts and the two illustration plots so
    a point silently dropped from the fit doesn't also appear, unexplained, as a plain data
    point in a plot that's supposed to show what the fit actually saw.
    """
    return (value >= VALID_PEAK_RANGE[0]) & (value <= VALID_PEAK_RANGE[1]) & ~np.isnan(error) & (error > 0)


def fit_vb_shifts(data):
    """Fit func_fit(ov, N_coef, VB_shift) per dm/chip/bar/channel against the photopeak vs OV points."""
    fits = {}
    ov = np.array(OV_VALUES)
    for dm, chips in data.items():
        for chip, bars in chips.items():
            for bar, channels in bars.items():
                for channel, series in channels.items():
                    value = np.array(series["value"])
                    error = np.array(series["error"])
                    mask = valid_point_mask(value, error)

                    result = {"N_coef": np.nan, "VB_shift": np.nan, "VB_shift_error": np.nan, "chi2_ndof": np.nan}
                    if mask.sum() >= 2:
                        cost = LeastSquares(ov[mask], value[mask], error[mask], func_fit)
                        m = iminuit.Minuit(cost, N_coef=0.001, VB_shift=0.0)
                        m.migrad()
                        m = iminuit.Minuit(cost, N_coef=m.values["N_coef"], VB_shift=m.values["VB_shift"])
                        m.migrad()
                        # The double migrad() call only re-polishes the same minimum found on the
                        # first pass - it is not a substitute for actually checking convergence.
                        # Without this, a fit Migrad never moved on (e.g. an infinite-weight point,
                        # or a genuinely unconstrained one) silently reports its initial guess as
                        # "the answer", which is only caught downstream by luck (NaN errors, the
                        # |VB_shift|<=1 sanity cut) rather than by design.
                        ndof = mask.sum() - 2
                        if m.valid:
                            result = {
                                "N_coef": float(m.values["N_coef"]),
                                "VB_shift": float(m.values["VB_shift"]),
                                "VB_shift_error": float(m.errors["VB_shift"]),
                                "chi2_ndof": float(m.fval / ndof) if ndof > 0 else np.nan,
                            }
                    (fits.setdefault(dm, {}).setdefault(chip, {})
                         .setdefault(bar, {}))[channel] = result
    return fits


def plot_fit_grids(data, fits):
    """Per DM/chip, a 4x4 small-multiples grid (one cell per bar) of photopeak [ADC] vs OV with the fit curve.

    Replaces one file per bar (384 total) with one grid per DM/chip (24 total), so all 16
    bars of a chip can be scanned and compared at a glance instead of opened one by one.
    """
    ov = np.array(OV_VALUES)
    ov_curve = np.linspace(ov.min(), ov.max(), 50)

    for dm, chips in data.items():
        dm_path = PLOTS_DIR / dm
        dm_path.mkdir(parents=True, exist_ok=True)
        for chip, bars in chips.items():
            fig, axs = plt.subplots(4, 4, figsize=(22, 22), sharex=True, sharey=True)
            fig.suptitle(f"SiPM array {dm} {chip}", fontsize=34, y=0.995)

            for idx, bar in enumerate(BAR_ORDER):
                ax = axs.flat[idx]
                ax.axhspan(*VALID_PEAK_RANGE, color=COLOR_BAND, alpha=0.15, zorder=0)
                for channel, color in [("CH_1", COLOR_A), ("CH_2", COLOR_B)]:
                    value = np.array(bars[bar][channel]["value"])
                    error = np.array(bars[bar][channel]["error"])
                    fit = fits[dm][chip][bar][channel]

                    ax.errorbar(ov, value, error, fmt="o", color=color, ecolor=color,
                                markersize=5, linewidth=1.3, linestyle="none", zorder=2)
                    if not np.isnan(fit["VB_shift"]):
                        ax.plot(ov_curve, func_fit(ov_curve, fit["N_coef"], fit["VB_shift"]),
                                color=color, linewidth=1.8, alpha=0.85, zorder=1)

                ax.text(0.05, 0.93, bar.replace("BAR_", "#"), transform=ax.transAxes, fontsize=18,
                        ha="left", va="top", fontweight="bold")
                ax.tick_params(axis="both", labelsize=15)
                ax.grid(True, alpha=0.25)

            # A handful of upstream photopeak fits report wildly wrong mu/error (e.g. 1e6+ ADC) -
            # plotted raw, a single such point stretches the shared y-axis until every real point
            # (always within VALID_PEAK_RANGE) is squashed to an invisible line at the bottom.
            axs.flat[0].set_ylim(VALID_PEAK_RANGE[0] - 50, VALID_PEAK_RANGE[1] + 50)

            for ax in axs[-1, :]:
                ax.set_xlabel(r"$V_{OV}$ [V]", fontsize=19)
            for ax in axs[:, 0]:
                ax.set_ylabel("peak [ADC]", fontsize=19)

            handles = [
                plt.Line2D([0], [0], color=COLOR_A, marker="o", linestyle="-", label="CH_1 / array A"),
                plt.Line2D([0], [0], color=COLOR_B, marker="o", linestyle="-", label="CH_2 / array B"),
            ]
            fig.legend(handles=handles, loc="upper right", fontsize=20, frameon=True, bbox_to_anchor=(0.995, 0.975))
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(dm_path / f"{dm}_{chip}_fits.png", dpi=140, bbox_inches="tight",
                        facecolor="white", pad_inches=0.1)
            plt.close(fig)


# Example bar/channel used by plot_example_fit/plot_model_curve to illustrate the method
# (slide-9-style "how does this work" figures): the cleanest converged CH_2/array-B fit in
# the tray, restricted to bars where all 8 OV points pass valid_point_mask (no excluded
# point to explain on a plot whose only job is to show how the fit works), picked over the
# DM_4/CHIP_1/BAR_1/CH_2 example used in the old thesis slide (chi2/ndof ~ not checked
# there). CH_2 (blue, COLOR_B) chosen for these two standalone/pedagogical plots.
EXAMPLE_DM, EXAMPLE_CHIP, EXAMPLE_BAR, EXAMPLE_CHANNEL = "DM_3", "CHIP_0", "BAR_15", "CH_2"


def plot_example_fit(data, fits):
    """Single-panel 'how the VB_shift fit works' figure: one bar/channel's photopeak [ADC] vs OV,
    data points plus the func_fit curve over the measured range, VB_shift called out explicitly.

    This is the CMS-style replacement for the single, unstyled example plot that used to
    illustrate the method on the conference-talk slide (plot_fit_grids shows the same fit,
    but as one of 16 small cells per DM/chip - too small to read on its own as a methods slide).
    """
    ov = np.array(OV_VALUES)
    series = data[EXAMPLE_DM][EXAMPLE_CHIP][EXAMPLE_BAR][EXAMPLE_CHANNEL]
    fit = fits[EXAMPLE_DM][EXAMPLE_CHIP][EXAMPLE_BAR][EXAMPLE_CHANNEL]
    value, error = np.array(series["value"]), np.array(series["error"])
    color = COLOR_A if EXAMPLE_CHANNEL == "CH_1" else COLOR_B
    array = CHANNEL_TO_ARRAY[EXAMPLE_CHANNEL]
    valid = valid_point_mask(value, error)

    ov_curve = np.linspace(ov.min(), ov.max(), 100)
    fit_curve = func_fit(ov_curve, fit["N_coef"], fit["VB_shift"])

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(ov_curve, fit_curve, color="0.5", linewidth=2.5, zorder=1,
            label=rf"fit: $\Delta V_{{BR}}$ = {fit['VB_shift']:.2f} ± {fit['VB_shift_error']:.2f} V")
    ax.errorbar(ov[valid], value[valid], error[valid], fmt="o", color=color, ecolor=color, markersize=11,
                linestyle="none", label=f"data: {EXAMPLE_BAR.replace('_', ' ')}, array {array}", zorder=2)
    if (~valid).any():
        # Drawn hollow, same convention as build_vb_shift_summary's rejected points - this
        # point had a NaN/zero error upstream and was never part of the fit, so plotting it
        # solid (as if it just missed the line) would misrepresent what the fit actually used.
        ax.scatter(ov[~valid], value[~valid], facecolors="none", edgecolors=color, marker="o",
                   s=130, linewidths=1.8, zorder=2, label="excluded (invalid error)")

    # Default 5%-margin autoscale leaves the highest point almost touching the top spine,
    # crowding the CMS label sitting just above it - extra headroom on top.
    ax.margins(y=0.12)
    ax.set_xlabel(r"$V_{OV}$ [V]", fontsize=30, labelpad=10)
    ax.set_ylabel("peak [ADC]", fontsize=30, labelpad=18)
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=19, frameon=True, loc="lower right")
    # Default loc (0, above the axes) hardcodes ypad=0 internally (mplhep's exp_text) - it's
    # drawn flush against the top spine by design and that offset can't be overridden via
    # kwargs. loc=2 (top-left corner, inside the axes, same convention already used in
    # build_vb_shift_summary) gives it real padding instead - the y-margin added above makes
    # room for it there without colliding with the data.
    cms_label(ax, loc=2)
    fig.savefig(PLOTS_DIR / "OV_scan_example_fit.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)


def plot_model_curve(data, fits):
    """Illustrates *why* the fit is done over such a narrow OV window: the full func_fit model
    from below breakdown up to a bit past the measured range, with the actual measured data
    points overlaid at their true position on the curve (the same points shown zoomed-in in
    plot_example_fit).
    """
    fit = fits[EXAMPLE_DM][EXAMPLE_CHIP][EXAMPLE_BAR][EXAMPLE_CHANNEL]
    color = COLOR_A if EXAMPLE_CHANNEL == "CH_1" else COLOR_B

    # 3.3 V - a short step past OV_VALUES' 3.0 V ceiling - is enough to show the curve keep
    # going roughly linearly beyond the measured window without the long, empty, uninformative
    # stretch that a wider range (4.0, 5.0 V) added past the actually-measured points.
    ov_max = 3.3
    vb = fit["VB_shift"]
    ov_curve = np.linspace(vb, ov_max, 300)
    model_curve = func_fit(ov_curve, fit["N_coef"], vb)

    ov = np.array(OV_VALUES)
    series = data[EXAMPLE_DM][EXAMPLE_CHIP][EXAMPLE_BAR][EXAMPLE_CHANNEL]
    value, error = np.array(series["value"]), np.array(series["error"])
    valid = valid_point_mask(value, error)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(ov_curve, model_curve, color="0.5", linewidth=2.5, zorder=1)
    ax.errorbar(ov[valid], value[valid], error[valid], fmt="o", color=color, ecolor=color, markersize=9,
                linestyle="none", zorder=2, label="measured data")
    if (~valid).any():
        ax.scatter(ov[~valid], value[~valid], facecolors="none", edgecolors=color, marker="o",
                   s=100, linewidths=1.8, zorder=2, label="excluded (invalid error)")
    ax.plot(vb, 0, "o", color="0.2", markersize=10, zorder=3)
    ax.annotate(rf"$\Delta V_{{BR}}$ = {vb:.2f} V", (vb, 0), textcoords="offset points", xytext=(20, 25),
                fontsize=19, color="0.2", zorder=3,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    # The formula used to live in the legend, but at this length it kept colliding with
    # whichever corner the legend/annotation/cms_label needed - a plain top-right text (the
    # one corner the curve, data, and VB_shift annotation never reach) has no such conflict.
    ax.text(0.97, 0.95, r"$E_{ADC}(V_{OV}) \propto E_{dep} \cdot Gain(V_{OV}-\Delta V_{BR}) \cdot PDE(V_{OV}-\Delta V_{BR})$",
            transform=ax.transAxes, fontsize=15, ha="right", va="top", color="0.2")

    # Extra headroom on top (vs. just the curve's own max) so the CMS label sitting just
    # above the axes isn't crowded against the top spine / legend box below it.
    ax.set_xlim(vb - 0.3, ov_max)
    ax.set_ylim(min(0, model_curve.min()) - 20, model_curve.max() * 1.15)
    ax.set_xlabel(r"$V_{OV}$ [V]", fontsize=30, labelpad=10)
    ax.set_ylabel("peak [ADC]", fontsize=30, labelpad=18)
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, alpha=0.3)
    # lower-right is empty (the VB_shift annotation sits at the bottom-left, near x = vb) -
    # upper-left is now reserved for cms_label(loc=2), which needs that corner to itself.
    ax.legend(fontsize=17, frameon=True, loc="lower right")
    cms_label(ax, loc=2)
    fig.savefig(PLOTS_DIR / "OV_scan_model_curve.png", dpi=160, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)


def plot_histogram(data, xlabel, path, log_x=False, color=COLOR_DATA, title=None, extra_stats=None):
    """A single-series QA histogram: thin step outline, stats in a text box (not a legend abuse).

    extra_stats, if given, is appended as an extra line in the stats box (e.g. a relative/
    normalized version of mean+std, alongside the always-shown absolute mean+std).
    """
    data = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 9))
    n_bins = math.ceil(np.sqrt(len(data))) * 3
    if log_x:
        # A quantity like chi2/ndof can span orders of magnitude with a long tail -
        # linear bins bury the bulk of the distribution in the first bar.
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), n_bins + 1)
        ax.set_xscale("log")
    else:
        bins = n_bins
    ax.hist(data, bins=bins, histtype="stepfilled", facecolor=color, alpha=0.25, edgecolor=color, linewidth=2.2)
    mean, std = np.mean(data), np.std(data)
    ax.axvline(mean, color="0.2", linestyle="--", linewidth=2)
    if title:
        ax.set_title(title, fontsize=34, pad=18)
    ax.set_xlabel(xlabel, fontsize=34, labelpad=20)
    ax.set_ylabel("count", fontsize=34, labelpad=18)
    ax.tick_params(axis="both", labelsize=26)
    stats_text = f"mean = {mean:.2f}\nstd = {std:.2f}"
    if extra_stats:
        stats_text += f"\n{extra_stats}"
    # Below the (loc=2, top-left, 2-line) "CMS Private Work" label rather than top-right,
    # so both text blocks read as one left-aligned column instead of competing corners.
    ax.text(0.03, 0.80, stats_text, transform=ax.transAxes, fontsize=26,
            ha="left", va="top", bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    # loc=2 (inside top-left, same fix applied to the other headline plots) - default loc=0
    # sits flush against the top spine with no real gap above it.
    cms_label(ax, loc=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def build_vb_shift_summary(fits):
    """Per DM/chip, a dot plot of the per-bar VB_shift (16 bars, both channels dodged left/right).

    Plots DM_X/DM_X_CHIP_Y.png and returns vbr_corr[dm][chip][array] = {'value', 'error'},
    the weighted-mean VB_shift used later to correct the nominal Hamamatsu VBR. Bars that
    converge but fail the |VB_shift|<=1 sanity cut are drawn as hollow markers at their
    actual (out-of-range) value, instead of just vanishing from the plot.
    """
    vbr_corr = {}
    rejected_pct = []
    x_idx = np.arange(16)
    dodge = 0.15

    for dm, chips in fits.items():
        dm_index = str(int(re.findall(r"\d+", dm)[0]))
        dm_path = PLOTS_DIR / dm
        dm_path.mkdir(parents=True, exist_ok=True)
        for chip, bars in chips.items():
            chip_index = re.findall(r"\d+", chip)[0]

            fig, ax = plt.subplots(figsize=(14, 8))
            stats_lines = []
            for channel, color, dx, label in [("CH_1", COLOR_A, -dodge, "CH_1 / array A"),
                                               ("CH_2", COLOR_B, +dodge, "CH_2 / array B")]:
                array = CHANNEL_TO_ARRAY[channel]
                values = np.array([bars[b][channel]["VB_shift"] for b in BAR_ORDER])
                errors = np.array([bars[b][channel]["VB_shift_error"] for b in BAR_ORDER])

                finite = ~np.isnan(values)
                accept = finite & (np.abs(values) <= 1) & ~np.isnan(errors) & (errors > 0)
                reject = finite & ~accept
                perc_loss = (1 - accept.sum() / len(values)) * 100
                rejected_pct.append(perc_loss)

                mean = weighted_average(values[accept], errors[accept])
                err = weighted_average_error(errors[accept])
                std = np.std(values[accept]) if accept.any() else np.nan
                vbr_corr.setdefault(dm_index, {}).setdefault(chip_index, {})[array] = {"value": mean, "error": err}

                ax.errorbar(x_idx[accept] + dx, values[accept], yerr=errors[accept], fmt="o", color=color,
                            ecolor=color, markersize=8, linestyle="none", label=label, zorder=3)
                if reject.any():
                    ax.scatter(x_idx[reject] + dx, values[reject], facecolors="none", edgecolors=color,
                               marker="o", s=60, linewidths=1.5, zorder=3)
                ax.axhspan(mean - std, mean + std, color=color, alpha=0.12, zorder=1)
                ax.axhline(mean, color=color, linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)
                mean_str = f"{mean:.2f} V" if not np.isnan(mean) else "N/A (all bars rejected)"
                stats_lines.append(f"{label}: mean = {mean_str}, rejected = {perc_loss:.0f}%")

            ax.set_xticks(x_idx)
            ax.set_xticklabels([str(i) for i in range(16)])
            ax.set_xlim(-0.6, 15.6)
            ax.set_xlabel("bar index", fontsize=30, labelpad=16)
            ax.set_ylabel(r"$\Delta V_{BR}$ [V]", fontsize=30, labelpad=16)
            ax.set_title(f"SiPM array {dm} {chip}", fontsize=32, pad=18)
            ax.tick_params(axis="both", labelsize=24)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=22, frameon=True, loc="best")
            ax.text(0.01, 0.02, "\n".join(stats_lines), transform=ax.transAxes, fontsize=19,
                    va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
            # loc=2 draws "CMS Private Work" inside the top-left corner instead of in the strip
            # above the axes, which would otherwise collide with the plot title on this figure.
            cms_label(ax, loc=2)
            fig.savefig(dm_path / f"{dm}_{chip}.png", dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)

    plot_histogram(rejected_pct, "Percentage [%]", PLOTS_DIR / "vb_shift_data_rejected.png")
    return vbr_corr


def _draw_correlation_scatter(rows, xlabel, path, q_val, q_err):
    """OV-scan vs IV-scan breakdown-voltage estimate, one point per physical sensor module
    (24 = 12 DM x 2 chip - see correlate_with_iv for why array A/B are averaged together
    first). Colored by an ordinal index instead of a categorical array A/B split - matching
    how this was colored in the original thesis plot. Meant to be narrated live on a
    conference slide: no per-point labels, no note box, only the fit offset is called out.
    """
    x_arr = np.array([r["x"] for r in rows])
    y_arr = np.array([r["y"] for r in rows])
    color_idx = np.array([r["color_idx"] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 11))
    # x and y are offset from each other by ~q (that's the whole point of the plot), so a
    # single shared center would center their *average* while leaving each one individually
    # off-center. Centering each axis on its own data - with the same half_span for both, so
    # equal-aspect still gives a square box - centers both at once.
    half_span = max(x_arr.max() - x_arr.min(), y_arr.max() - y_arr.min()) / 2 * 1.3
    x_center = (x_arr.min() + x_arr.max()) / 2
    y_center = (y_arr.min() + y_arr.max()) / 2
    x_lo, x_hi = x_center - half_span, x_center + half_span
    y_lo, y_hi = y_center - half_span, y_center + half_span
    ax.plot([x_lo, x_hi], [x_lo + q_val, x_hi + q_val], color="0.15", linewidth=2.5, zorder=1,
            label=rf"fit: $\Delta V_{{BR}}$ = {q_val:.2f} $\pm$ {q_err:.2f} V")

    ax.errorbar(x_arr, y_arr, xerr=[r["err_x"] for r in rows], fmt="none",
                ecolor="0.6", elinewidth=1.2, alpha=0.7, zorder=2)
    # A discrete 24-color map with a boundary at every integer (drawedges=True) reads as "24
    # distinct arrays" rather than a continuous quantity, matching what the color means here.
    boundaries = np.arange(-0.5, 24.5, 1)
    cmap = plt.get_cmap("plasma", 24)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    sc = ax.scatter(x_arr, y_arr, c=color_idx, cmap=cmap, norm=norm,
                     s=120, edgecolors="0.2", linewidths=0.6, zorder=3)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=30, labelpad=18)
    ax.set_ylabel(r"$V_{BR}$ (IV scan) [V]", fontsize=30, labelpad=18)
    ax.tick_params(axis="both", labelsize=24)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=19, frameon=True, loc="lower right")
    cms_label(ax, loc=2)

    # A plain fig.colorbar(sc, ax=ax) is only as tall as the drawing area *inside* the axes
    # spines; adding a title above it then overshoots the axes' own top edge. Building the
    # colorbar axes with make_axes_locatable ties its height to the main axes' box exactly,
    # then shrinking it ~8% from the top leaves headroom for the "SiPM Array" title to sit
    # inside that same span instead of sticking out above it.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.92])
    cbar = fig.colorbar(sc, cax=cax, boundaries=boundaries, ticks=range(0, 24, 2), drawedges=True)
    cbar.ax.set_title("DM CHIP", fontsize=20, pad=10)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def correlate_with_iv(vbr_corr):
    """Compare the LYSO OV-scan VBR (nominal Hamamatsu + measured VB_shift) to the measured IV-scan VBR.

    One point per physical sensor module (24 = 12 DM x 2 chip): array A and array B are
    averaged together first, since they're the two ends of the *same* physical module (same
    barcode) - T3R0_struct.json's nominal voltages agree to <=0.01 V between A/B within a
    chip, but differ by up to ~0.6 V between chip 0 and chip 1 of the same DM (different
    barcodes), confirming chip = physical module and A/B = redundant readout of it, not two
    separate arrays. x = OV-scan estimate, y = IV-scan measurement.

    The slope is fixed to m=1 (perfect proportional agreement assumed) and only the offset q
    is fit - equivalent to the mean of the residuals, so q and its standard error match the
    diff histogram's mean/std exactly, just shown as a line on the scatter instead of a bin.

    A free-slope fit was tried and dropped: with x spanning only ~1.3 V, the slope is very
    weakly constrained (m = 0.76 +/- 0.12 either way), and weighting by the reported x errors
    is unreliable since plot_chi2_summary shows those errors are underestimated - a free,
    weighted fit swung between m=0.95 and m=1.13 depending on the weighting scheme alone.
    """
    vbr_iv = json.load(open(IV_FILE))
    vbr_iv = dict(sorted(vbr_iv.items(), key=lambda kv: int(kv[0])))
    struct = json.load(open(STRUCT_FILE))
    nominal = {str(dm_entry["position"]): dm_entry["voltages"] for dm_entry in struct["DMs"]}

    rows = []
    for dm in vbr_iv:
        dm_idx = int(dm)
        for chip in sorted(vbr_iv[dm], key=int):
            chip_idx = int(chip)
            shift_a, shift_b = vbr_corr[dm][chip]["A"], vbr_corr[dm][chip]["B"]
            shift_value = weighted_average([shift_a["value"], shift_b["value"]],
                                            [shift_a["error"], shift_b["error"]])
            shift_error = weighted_average_error([shift_a["error"], shift_b["error"]])
            nominal_avg = np.mean([nominal[dm][chip]["A"], nominal[dm][chip]["B"]])
            iv_avg = np.mean([vbr_iv[dm][chip]["A"], vbr_iv[dm][chip]["B"]])
            rows.append({"dm": dm, "chip": chip, "x": nominal_avg + shift_value,
                         "err_x": shift_error, "y": iv_avg, "color_idx": dm_idx * 2 + chip_idx})

    n_total = len(rows)
    missing = [r for r in rows if np.isnan(r["x"])]
    excluded_known = [r for r in rows if r["dm"] in EXCLUDED_DMS and not np.isnan(r["x"])]
    rows = [r for r in rows if not np.isnan(r["x"]) and r["dm"] not in EXCLUDED_DMS]
    dropped = missing + excluded_known
    if dropped:
        dms = sorted({r["dm"] for r in dropped}, key=int)
        print(f"  excluded {len(dropped)}/{n_total} points: DM_{', DM_'.join(dms)} "
              f"(known anomalous channel or missing fit - see EXCLUDED_DMS)")

    x_arr = np.array([r["x"] for r in rows])
    y_arr = np.array([r["y"] for r in rows])
    resid = x_arr - y_arr  # VB_OV - VB_IV
    q_val = -np.mean(resid)  # y = x + q  =>  q = mean(y - x)
    q_err = np.std(resid, ddof=1) / np.sqrt(len(resid))
    print(f"  fixed-slope fit y = x + q (m=1):  q = {q_val:.3f} +/- {q_err:.3f} V  (0 = perfect agreement); n={len(rows)}")

    # Quantitative goodness-of-method figures (as in the thesis): both as a fraction of the
    # tray's mean breakdown voltage, rather than absolute numbers in V that say nothing on
    # their own about whether ~0.47 V is a big or small disagreement.
    #   - bias = mean(resid)/mean(VB_IV): the systematic offset between the two methods.
    #   - spread = std(resid)/mean(VB_IV): how much a *single* array's OV-vs-IV agreement
    #     typically varies around that bias - the actual per-point reliability of the method,
    #     which the bias's own standard error (a property of the mean, shrinking with more
    #     points) does not capture.
    mean_vb_iv = np.mean(y_arr)
    n = len(resid)
    bias_val = q_val / mean_vb_iv * 100
    bias_err = q_err / mean_vb_iv * 100
    spread_val = np.std(resid, ddof=1) / mean_vb_iv * 100
    # Standard error of a sample std dev (assuming ~normal residuals): s / sqrt(2(n-1)).
    spread_err = spread_val / np.sqrt(2 * (n - 1))
    print(f"  relative bias:   diff/mean = {bias_val:.2f}% +/- {bias_err:.2f}%  "
          f"(mean VB_IV = {mean_vb_iv:.2f} V)")
    print(f"  relative spread: std/mean  = {spread_val:.2f}% +/- {spread_err:.2f}%")

    _draw_correlation_scatter(rows, r"$V_{BR}$ (OV scan) [V]", PLOTS_DIR / "OV_vs_IV_correlation.png",
                               q_val, q_err)

    plot_histogram(resid, r"$V_{BR}$ (OV scan) $-$ $V_{BR}$ (IV scan) [V]",
                    PLOTS_DIR / "OV_vs_IV_diff_hist.png", color=COLOR_DIFF,
                    title=r"$V_{BR}$ difference",
                    extra_stats=(f"bias = {bias_val:.2f}% $\\pm$ {bias_err:.2f}%\n"
                                 f"spread = {spread_val:.2f}% $\\pm$ {spread_err:.2f}%"))


def plot_chi2_summary(fits):
    """Diagnostic: distribution of chi2/ndof across all valid, converged bar/channel fits.

    A fit landing inside the |VB_shift|<=1 sanity cut is not necessarily a *good* fit - a
    point with a tiny but nonzero error that disagrees with the rest of the trend forces
    the fit to run through it, which shows up here as a large chi2/ndof rather than as an
    invalid or out-of-range VB_shift.
    """
    chi2_values, n_invalid = [], 0
    for chips in fits.values():
        for bars in chips.values():
            for channels in bars.values():
                for fit in channels.values():
                    if np.isnan(fit["VB_shift"]):
                        n_invalid += 1
                    elif not np.isnan(fit["chi2_ndof"]):
                        chi2_values.append(fit["chi2_ndof"])
    n_total = sum(len(channels) for chips in fits.values() for bars in chips.values() for channels in bars.values())
    print(f"  {n_invalid}/{n_total} fits did not converge (Migrad invalid or <2 usable points)")
    print(f"  chi2/ndof over the {len(chi2_values)} converged fits: "
          f"mean={np.mean(chi2_values):.2f}, median={np.median(chi2_values):.2f}, max={np.max(chi2_values):.2f}")
    plot_histogram(chi2_values, r"$\chi^2$/ndof", PLOTS_DIR / "chi2_ndof_distribution.png", log_x=True)


def main():
    use_cms_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading OV scan...")
    data = load_ov_scan()
    print("Fitting VB_shift per DM/chip/bar/channel...")
    fits = fit_vb_shifts(data)
    plot_chi2_summary(fits)
    print("Plotting per-DM/chip fit grids (24 files)...")
    plot_fit_grids(data, fits)
    print("Plotting single-bar example fit + model curve...")
    plot_example_fit(data, fits)
    plot_model_curve(data, fits)
    print("Building VB_shift summary plots...")
    vbr_corr = build_vb_shift_summary(fits)
    print("Correlating with IV scan...")
    correlate_with_iv(vbr_corr)
    print(f"Done. Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
