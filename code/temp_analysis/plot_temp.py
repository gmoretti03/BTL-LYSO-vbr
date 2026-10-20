#!/usr/bin/env python
"""Temperature uniformity study — local rewrite of old/plot_temp.py (multi-run / all_runs mode).

Reproduces old/iv_scan/runs_36_67/ using the local per-run IV-scan temperature logs in
old/iv_scan/run_36/ ... run_67/, dropping the mtd_daq/load_general_config/NAS dependencies.
Unrelated to the OV-vs-IV study in ov_analysis/ (different data, different question).

Usage:
    python temp_analysis/plot_temp.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import use_cms_style, cms_label, PETROFF

REPO_ROOT = Path(__file__).resolve().parent.parent
IV_SCAN_DIR = REPO_ROOT / "old" / "iv_scan"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Runs 51 and 60 are missing iv_scans.json/vbr_dm.json (aborted runs) - excluded here as in
# the original old/plot_temp.py:read_runs.
RUN_RANGE = [r for r in range(36, 68) if r not in (51, 60)]

# 8 DMs instrumented with temperature sensors (of the 12 on the tray) plus the readout board
# modules - each DM key has 2 sensors ('0','1'), CCBOARD has 3, PCCA/PCCB have 2.
MODULES = ["DM00", "DM01", "DM02", "DM03", "DM08", "DM09", "DM10", "DM11", "CCBOARD", "PCCA", "PCCB"]

# DM01 sensor '1' reads a constant 150.0 C on every run and every DAC cycle checked - a
# fault/disconnected-sensor placeholder value, not a real reading (its delta is always
# exactly 0, which would otherwise misleadingly read as "perfectly stable").
FAULTY_SENSORS = {("DM01", "1")}

# Which of the 5 cooling-fan configurations each run was taken under. The original
# old/plot_temp.py:runs_hist_iv_scan listed run 61 twice (in both "off" and "upupleft") and
# never listed run 62 at all - an evident 61/62 typo, fixed here to actually use 62.
FAN_CONFIGS = {
    "off": [36, 37, 38, 58, 59, 61],
    "upleftright": [39, 40, 41, 52, 53, 54],
    "upupup": [42, 43, 44, 55, 56, 57],
    "upupleft": [45, 46, 47, 62, 63, 64],
    "upupright": [48, 49, 50, 65, 66, 67],
}


def read_run_deltas(run):
    """For one IV-scan run, return {module: {sensor: last_cycle - first_cycle}}."""
    run_dir = IV_SCAN_DIR / f"run_{run}"
    cycles = sorted(int(re.search(r"\d+", p.stem).group()) for p in run_dir.glob("temps_dac*.json"))
    first = json.load(open(run_dir / f"temps_dac{cycles[0]}.json"))
    last = json.load(open(run_dir / f"temps_dac{cycles[-1]}.json"))
    return {module: {sensor: last[module][sensor] - first[module][sensor] for sensor in first[module]}
            for module in MODULES}


def load_all_runs():
    return {run: read_run_deltas(run) for run in RUN_RANGE}


def _module_grid(n_modules, figsize):
    """3x4 small-multiples grid: one panel per module, with one spare (11 modules, 12 cells)."""
    fig, axs = plt.subplots(3, 4, figsize=figsize)
    for ax in axs.flat[n_modules:]:
        ax.axis("off")
    return fig, axs


def _plot_module_lines(x_values, series_by_module, xlabel, title, output_path):
    """Shared small-multiples line plot: one panel per module.

    series_by_module[module] = {sensor_label: [y-values matching x_values]}, already filtered
    for FAULTY_SENSORS by the caller. Color means "which sensor slot" consistently across
    every panel (never more than 3), never "which module" - that's what the panel is for.
    """
    fig, axs = _module_grid(len(MODULES), figsize=(24, 14))
    for idx, module in enumerate(MODULES):
        ax = axs.flat[idx]
        for i, (sensor, values) in enumerate(series_by_module[module].items()):
            ax.plot(x_values, values, color=PETROFF[i], marker="o", markersize=4, linewidth=1.3,
                    label=f"sensor {sensor}")
        ax.axhline(0, color="0.3", linewidth=1)
        ax.set_title(module, fontsize=20, pad=8)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, frameon=True)

    fig.text(0.02, 0.5, "temperature increase [°C]", va="center", rotation="vertical", fontsize=22)
    fig.suptitle(title, fontsize=30, y=1.0)
    cms_label(axs.flat[0], loc=2)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def plot_temps_vs_run(deltas):
    """Small multiples: per-module temperature increase (first-vs-last IV-scan cycle) across runs.

    Replaces old/plot_temp.py:plot_runs's single plot with 22 line series sharing 11 ad-hoc
    colors (2+ sensors x 11 modules) - color there carried no information once you already
    have 11 different markers/labels to track.
    """
    runs = sorted(deltas.keys())
    series_by_module = {
        module: {sensor: [deltas[run][module][sensor] for run in runs]
                 for sensor in deltas[runs[0]][module] if (module, sensor) not in FAULTY_SENSORS}
        for module in MODULES
    }
    _plot_module_lines(runs, series_by_module, "run number",
                        "Temperature increase per IV-scan run", PLOTS_DIR / "temps_vs_run.png")


def plot_temps_by_fan_config(deltas):
    """Small multiples: mean +/- std temperature increase per module, grouped by fan config.

    Replaces old/plot_temp.py:runs_hist_iv_scan's 5x11 grid of tiny (N=6) histograms with
    mean/std stuffed into a text box - a histogram is the wrong form for 6 points, and the
    text box duplicated what a proper error bar already shows. One point + error bar per
    config, one panel per module, is direct and comparable at a glance.
    """
    configs = list(FAN_CONFIGS.keys())
    x_idx = np.arange(len(configs))
    fig, axs = _module_grid(len(MODULES), figsize=(24, 14))

    for idx, module in enumerate(MODULES):
        ax = axs.flat[idx]
        sensors = [s for s in deltas[RUN_RANGE[0]][module] if (module, s) not in FAULTY_SENSORS]
        for i, sensor in enumerate(sensors):
            means, stds = [], []
            for cfg in configs:
                values = [deltas[run][module][sensor] for run in FAN_CONFIGS[cfg] if run in deltas]
                means.append(np.mean(values))
                stds.append(np.std(values))
            ax.errorbar(x_idx + i * 0.08, means, yerr=stds, fmt="o", color=PETROFF[i],
                        markersize=7, capsize=3, linestyle="none", label=f"sensor {sensor}")
        ax.set_xticks(x_idx)
        ax.set_xticklabels(configs, rotation=30, fontsize=11, ha="right")
        ax.set_title(module, fontsize=20, pad=8)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=10, frameon=True)

    fig.text(0.02, 0.5, "temperature increase [°C]", va="center", rotation="vertical", fontsize=22)
    fig.suptitle("Temperature increase by cooling-fan configuration", fontsize=30, y=1.0)
    cms_label(axs.flat[0], loc=2)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(PLOTS_DIR / "temps_by_fan_config.png", dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


# Optional single-run mode (old/plot_temp.py:plot_cycles/schematic_CC) - applicable to the new
# single IV run (TRAY3_RU0/IV_TRAY3_RU0/) or to any individual old/iv_scan/run_N/, since both
# share the same temps_dac*.json structure. Unlike the multi-run mode above, "cycle" here means
# one DAC step *within* a single run, not one run among many.

def _dac_cycles(run_dir):
    return sorted(int(re.search(r"\d+", p.stem).group()) for p in run_dir.glob("temps_dac*.json"))


def plot_cycles(run_dir, title, output_path):
    """Small multiples: per-module temperature increase across DAC cycles within one run,
    each cycle measured relative to that run's first cycle. Single-run counterpart of
    plot_temps_vs_run, sharing the same panel-per-module design.
    """
    cycles = _dac_cycles(run_dir)
    first = json.load(open(run_dir / f"temps_dac{cycles[0]}.json"))
    snapshots = {c: json.load(open(run_dir / f"temps_dac{c}.json")) for c in cycles}
    series_by_module = {
        module: {sensor: [snapshots[c][module][sensor] - first[module][sensor] for c in cycles]
                 for sensor in first[module] if (module, sensor) not in FAULTY_SENSORS}
        for module in MODULES
    }
    _plot_module_lines(cycles, series_by_module, "DAC cycle", title, output_path)


def plot_schematic(run_dir, title, output_path):
    """Board-schematic heatmap: first-vs-last-cycle temperature change laid out by each sensor's
    physical position on the tray.

    A single diverging colormap centered at 0 covers the whole board. The original
    old/plot_temp.py:schematic_CC used a second, sequential (viridis) colormap just for the
    CC/board row despite it being the same kind of signed delta as the SiPM rows around it -
    an unjustified inconsistency (and a second color scale to read), dropped here.
    """
    cycles = _dac_cycles(run_dir)
    first = json.load(open(run_dir / f"temps_dac{cycles[0]}.json"))
    last = json.load(open(run_dir / f"temps_dac{cycles[-1]}.json"))

    rows, cols = 5, 12
    matrix = np.full((rows, cols), np.nan)
    labels = np.full((rows, cols), "", dtype=object)

    for j in range(2):
        matrix[2][j] = last["PCCB"][f"TEMP{j + 1}"] - first["PCCB"][f"TEMP{j + 1}"]
        labels[2][j] = f"PCCB:TEMP{j + 1}"
        matrix[2][j + 10] = last["PCCA"][f"TEMP{j + 1}"] - first["PCCA"][f"TEMP{j + 1}"]
        labels[2][j + 10] = f"PCCA:TEMP{j + 1}"
    for i in range(cols - 4):
        k = 1 if i <= 2 else (2 if i < 5 else 3)
        matrix[2][i + 2] = last["CCBOARD"][f"TEMP{k}"] - first["CCBOARD"][f"TEMP{k}"]
        labels[2][i + 2] = f"CCBOARD:TEMP{k}"

    # (row, col, module, sensor key, sensor name) - physical layout of the 8 instrumented DMs
    # around the CC board, ported as-is from old/plot_temp.py:schematic_CC (correction=0 case,
    # the only one ever used for iv_scan-style directories).
    sensor_data = [
        (1, 2, "DM00", "0", "SCA-A"), (0, 3, "DM00", "1", "SCA-A"),
        (4, 2, "DM08", "0", "SCA-A"), (3, 3, "DM08", "1", "SCA-A"),
        (1, 4, "DM01", "1", "SCA-A"), (0, 5, "DM01", "0", "SCA-A"),
        (4, 4, "DM09", "1", "lpGBT-L0"), (3, 5, "DM09", "0", "lpGBT-L0"),
        (1, 6, "DM02", "0", "SCA-B"), (0, 7, "DM02", "1", "SCA-B"),
        (4, 6, "DM10", "0", "SCA-B"), (3, 7, "DM10", "1", "SCA-B"),
        (1, 8, "DM03", "1", "SCA-B"), (0, 9, "DM03", "0", "SCA-B"),
        (4, 8, "DM11", "1", "lpGBT-L1"), (3, 9, "DM11", "0", "lpGBT-L1"),
    ]
    for r, c, module, sensor, sensor_name in sensor_data:
        labels[r][c] = f"{module}:{sensor_name}-TEMP{int(sensor) + 1}"
        if (module, sensor) in FAULTY_SENSORS:
            continue  # leave as nan - faulty sensor, no data
        matrix[r][c] = last[module][sensor] - first[module][sensor]

    fig, ax = plt.subplots(figsize=(20, 10))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.matshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("temperature increase [°C]", fontsize=18)

    for r in range(rows):
        for c in range(cols):
            if labels[r][c]:
                ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=9,
                        rotation=90, color="black")

    ax.axhline(1.5, color="black", linewidth=2)
    ax.axhline(2.5, color="black", linewidth=2)
    ax.axvline(1.5, color="black", linewidth=2)
    ax.axvline(9.5, color="black", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=26, pad=20)
    cms_label(ax, loc=2)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def main():
    use_cms_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {len(RUN_RANGE)} IV-scan runs ({RUN_RANGE[0]}-{RUN_RANGE[-1]})...")
    deltas = load_all_runs()
    print("Plotting temperature vs run...")
    plot_temps_vs_run(deltas)
    print("Plotting temperature by fan configuration...")
    plot_temps_by_fan_config(deltas)

    single_run_dir = REPO_ROOT / "TRAY3_RU0" / "IV_TRAY3_RU0"
    print(f"Plotting single-run cycles for {single_run_dir.name}...")
    plot_cycles(single_run_dir, "SiPM array TRAY3_RU0 - temperature per DAC cycle",
                PLOTS_DIR / "TRAY3_RU0_cycles.png")
    print(f"Plotting board schematic for {single_run_dir.name}...")
    plot_schematic(single_run_dir, "SiPM array TRAY3_RU0 - board temperature change",
                   PLOTS_DIR / "TRAY3_RU0_schematic.png")

    print(f"Done. Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
