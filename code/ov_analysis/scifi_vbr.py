#!/usr/bin/env python
"""CMS-styled copy of the Mu3e SciFi breakdown-voltage (Vbr) figure.

Reproduces sequencer_scripts/scifi_scripts/tests/output_plots/test_plot_vbr.png
(the "MOD_20 [TEST], simulated data" IV + log-derivative panel) using the same
CMS/mplhep look as the rest of ov_analysis (plot_style.use_cms_style / cms_label),
with the two panels laid out side by side instead of stacked and full axis
labels + Vbr/Ibr/mu annotations.

The IV curve is simulated (as in the original test plot); the log-derivative and
the Gaussian breakdown fit are computed with the exact algorithm from
scifi_scripts/calibrations/vbr_find.py, inlined below.

Usage:
    python ov_analysis/scifi_vbr.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import use_cms_style, cms_label, COLOR_A, COLOR_B

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

IV_START, IV_END = 54.0, 58.0
MODULE = "MOD_20"


# --- vbr_find.py algorithm (inlined verbatim) -------------------------------

def compute_log_derivative(voltage, current):
    deriv_voltage, derivative = [], []
    for k in range(1, len(voltage)):
        i0, i1 = current[k - 1], current[k]
        if i0 <= 0 or i1 <= 0:
            continue
        dv = voltage[k] - voltage[k - 1]
        if dv == 0:
            continue
        d = (math.log10(i1) - math.log10(i0)) / dv
        deriv_voltage.append((voltage[k] + voltage[k - 1]) / 2)
        derivative.append(d)
    return deriv_voltage, derivative


def _gaussian(v, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((v - mu) / sigma) ** 2)


def find_breakdown_voltage(voltage, current, fit_window_v=0.5):
    deriv_voltage, derivative = compute_log_derivative(voltage, current)
    peak_idx = max(range(len(derivative)), key=lambda k: derivative[k])
    v_br_raw = deriv_voltage[peak_idx]
    peak_height = derivative[peak_idx]
    closest_idx = min(range(len(voltage)), key=lambda k: abs(voltage[k] - v_br_raw))
    i_br = current[closest_idx]

    window = [(v, d) for v, d in zip(deriv_voltage, derivative) if abs(v - v_br_raw) <= fit_window_v]
    window_v = np.array([v for v, _ in window])
    window_d = np.array([d for _, d in window])
    popt, _ = curve_fit(_gaussian, window_v, window_d,
                        p0=[peak_height, v_br_raw, fit_window_v / 2])
    fit_amplitude, v_br_fit, fit_sigma = popt
    return {
        "deriv_voltage": np.array(deriv_voltage),
        "derivative": np.array(derivative),
        "v_br_raw": float(v_br_raw),
        "i_br": float(i_br),
        "v_br_fit": float(v_br_fit),
        "fit_sigma": float(abs(fit_sigma)),
        "fit_amplitude": float(fit_amplitude),
    }


# --- simulated IV curve (stand-in for the real ScanVbr output) --------------

def simulate_iv():
    """Soft-plateau diode IV: gentle sub-breakdown leakage, sharp turn-on near
    ~55.85 V, ~1.5 mA plateau, with a few percent of measurement noise."""
    rng = np.random.default_rng(20)
    v = np.arange(IV_START, IV_END + 1e-9, 0.05)
    v_turn, width, plateau = 55.83, 0.062, 1.52
    leakage = 0.020 + 0.012 * (v - IV_START) / (IV_END - IV_START)
    ideal = leakage + plateau / (1.0 + np.exp(-(v - v_turn) / width))
    current = ideal * rng.normal(1.0, 0.012, size=v.size)
    return v.tolist(), current.tolist()


def main():
    use_cms_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    voltage, current = simulate_iv()
    vbr = find_breakdown_voltage(voltage, current)
    v = np.array(voltage)
    i = np.array(current)

    v_fit = np.linspace(vbr["v_br_fit"] - 3 * vbr["fit_sigma"],
                        vbr["v_br_fit"] + 3 * vbr["fit_sigma"], 200)
    gauss = _gaussian(v_fit, vbr["fit_amplitude"], vbr["v_br_fit"], vbr["fit_sigma"])

    fig, (ax_iv, ax_deriv) = plt.subplots(1, 2, figsize=(20, 8))

    # -- left: I vs V (log) -------------------------------------------------
    ax_iv.plot(v, i, color=COLOR_B, linewidth=2.4, zorder=2)
    ax_iv.axvline(vbr["v_br_raw"], color=COLOR_A, linestyle=":", linewidth=2.2, alpha=0.8, zorder=3)
    ax_iv.set_yscale("log")
    ax_iv.set_ylim(1e-3, i.max() * 2)
    ax_iv.set_xlim(IV_START, IV_END)
    ax_iv.set_xlabel("V [V]", fontsize=30, labelpad=10)
    ax_iv.set_ylabel("I [mA]", fontsize=30, labelpad=14)
    ax_iv.tick_params(axis="both", labelsize=24)
    ax_iv.grid(True, which="both", alpha=0.3)
    cms_label(ax_iv, loc=2)

    # -- right: dlogI/dV vs V --------------------------------------------------
    ax_deriv.axvline(vbr["v_br_raw"], color=COLOR_A, linestyle=":", linewidth=2.2, alpha=0.8, zorder=1)
    ax_deriv.plot(vbr["deriv_voltage"], vbr["derivative"], color=COLOR_B, linewidth=2.2, zorder=2,
                  label=(rf"Derivative,  $V_{{br}}$ = {vbr['v_br_raw']:.2f} V,  "
                         rf"$I_{{br}}$ = {vbr['i_br']:.2f} mA"))
    ax_deriv.plot(v_fit, gauss, color=COLOR_A, linestyle="--", linewidth=2.6, zorder=3,
                  label=rf"Gaussian fit:  $\mu$ = {vbr['v_br_fit']:.2f} V")
    ax_deriv.set_xlim(IV_START, IV_END)
    ax_deriv.set_xlabel("V [V]", fontsize=30, labelpad=10)
    ax_deriv.set_ylabel("dlogI/dV [mA/V]", fontsize=30, labelpad=14)
    ax_deriv.tick_params(axis="both", labelsize=24)
    ax_deriv.grid(True, alpha=0.3)
    ax_deriv.legend(fontsize=18, frameon=True, loc="upper left")

    fig.tight_layout()
    out = PLOTS_DIR / "scifi_vbr_example.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"v_br_raw = {vbr['v_br_raw']:.3f} V, i_br = {vbr['i_br']:.3f} mA, "
          f"mu = {vbr['v_br_fit']:.3f} V, sigma = {vbr['fit_sigma']:.3f} V")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
