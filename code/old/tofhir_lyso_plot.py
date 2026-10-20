#!/usr/bin/env python
import json

import awkward as ak
import iminuit
import iminuit.cost
import matplotlib as mpl
import numba
import pandas as pd
from scipy.signal import find_peaks

mpl.use("Agg")
import concurrent.futures
import os
from math import ceil
from sys import argv

import hist
import matplotlib.pyplot as plt
import numpy as np
import uproot
from mtd_daq.configurator.configurator import get_mtd_daq_path, load_general_config
from mtd_daq.configurator.mappings import bars_to_channelIDs

n_bins = 256
min_energy = 0
max_energy = 1024
min_ti = 0.0



@numba.njit
def model(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_SM_map(structure_file):
    with open(structure_file) as file:
        ru = json.load(file)

    sms = {}
    for dm in ru["DMs"]:
        sms[int(dm["position"])] = {}
        for chip in dm["sm_barcodes"]:
            sms[int(dm["position"])][int(chip)] = int(dm["sm_barcodes"][chip])
    return sms


def add_dict(d1, d2):
    if isinstance(d1, dict):
        d = {}
        common_keys = set(list(d1.keys())).intersection(list(d2.keys()))
        for key in common_keys:
            d[key] = add_dict(d1[key], d2[key])
        for key in d1:
            if key in common_keys:
                continue
            
            d[key] = d1[key]
        for key in d2:
            if key in common_keys:
                continue
            d[key] = d2[key]

        return d
    elif isinstance(d1, ak.highlevel.Array):
        return ak.concatenate([d1, d2])
    elif isinstance(d1, np.ndarray) and len(d1.shape) != 0:
        print("Debug np", d1, d2)
        return np.concatenate([d1, d2])
    elif isinstance(d1, set):
        return d1.union(d2)
    else:
        return d1 + d2

def task_fun(input_path, output_path, run, dm, chip, SMs_map):
    try:
        f = uproot.open(f"{input_path}/run_DM{dm:02}_chip{chip}.root")
    except Exception as _:
        return {bar: 0 for bar in range(16)}, {}, 0

    tot_events = f["events"].num_entries
    chunksize = 2_000_000

    nchunks = ceil(tot_events / chunksize)
    print("Nchunks", nchunks)

    histos = []

    for bar in range(16):
        histos.append(
            {
                "mask": {
                    "DM_pos": dm,
                    "chipID": chip,
                    "ch1": bars_to_channelIDs[bar][0],
                    "ch2": bars_to_channelIDs[bar][1],
                },
                "histos": {
                    "ch1": hist.Hist(hist.axis.Regular(n_bins, min_energy, max_energy)),
                    "ch2": hist.Hist(hist.axis.Regular(n_bins, min_energy, max_energy)),
                    "bar": hist.Hist(hist.axis.Regular(n_bins, min_energy, max_energy)),
                    # correlation between two channels of the bar
                    "corr": hist.Hist(
                        hist.axis.Regular(n_bins, min_energy, max_energy, name="ch1"),
                        hist.axis.Regular(n_bins, min_energy, max_energy, name="ch2"),
                    ),
                },
                "bar": bar,
            }
        )

    for i in range(nchunks):
        print("Processing chunk", i)
        start = i * chunksize
        stop = min((i + 1) * chunksize, tot_events)
        branches = ["energy", "time", "timeEndQ", "channelID"]
        events = f["events"].arrays(
            branches, entry_start=start, entry_stop=stop, library="ak"
        )

        events["ti"] = events.timeEndQ - events.time / 6250.0
        events = events[events.ti >= min_ti]
        events = events[(events.energy >= min_energy) & (events.energy <= max_energy)]

        for histo in histos:
            ch1 = histo["mask"]["ch1"]
            ch2 = histo["mask"]["ch2"]

            a = ak.drop_none(
                ak.pad_none(
                    events[(events.channelID == ch1)].energy,
                    1,
                    clip=True,
                )[:, 0],
                axis=0,
            )
            histo["histos"]["ch1"].fill(a)

            a = ak.drop_none(
                ak.pad_none(
                    events[(events.channelID == ch2)].energy,
                    1,
                    clip=True,
                )[:, 0],
                axis=0,
            )
            histo["histos"]["ch2"].fill(a)

            a = 0.5 * ak.drop_none(
                ak.pad_none(
                    events[(events.channelID == ch1)].energy,
                    1,
                    clip=True,
                )[:, 0]
                + ak.pad_none(
                    events[(events.channelID == ch2)].energy,
                    1,
                    clip=True,
                )[:, 0],
                axis=0,
            )
            histo["histos"]["bar"].fill(a)

            ch1_v = ak.pad_none(
                events[(events.channelID == ch1)].energy,
                1,
                clip=True,
            )[:, 0]

            ch2_v = ak.pad_none(
                events[(events.channelID == ch2)].energy,
                1,
                clip=True,
            )[:, 0]
            mask = ~ak.is_none(ch1_v + ch2_v)
            histo["histos"]["corr"].fill(ch1=ch1_v[mask], ch2=ch2_v[mask])
    
    output_iminuit = {}
    counts_per_bar = {}
    d = {}
    total_peaks = []
    peak_bar = 0
    peak_bar_sigma = 0
    for histo in histos:
        bar = histo["bar"]
        dm_pos = histo["mask"]["DM_pos"]
        chip_id = histo["mask"]["chipID"]   

        ch1 = histo["mask"]["ch1"]
        ch2 = histo["mask"]["ch2"]        
        edges = histo["histos"]["ch1"].axes[0].edges
        plt.clf()
        
        for h in ["ch1", "ch2", "bar"]:
            # does not include underflow/overflow
            integral = np.sum(histo["histos"][h].values(False))
            if h == "bar":
                kwargs = {"linewidth": 2, "color": "black"}
            elif h == "ch1":
                kwargs = {"linestyle": "--", "color": "red"}
            else:
                kwargs = {"linestyle": "--", "color": "blue"}

            plt.stairs(
                histo["histos"][h].values(), edges, label=f"{h}={integral}", **kwargs
            )
            if h == "bar":
                # fit
                h = histo["histos"][h]
                _peaks, props = find_peaks(
                    h.values(),
                    width=5,
                    height=0.01,
                )
                df = {
                    "peaks": h.axes[0].centers[_peaks],
                    "prom": props["prominences"],
                    "widths": props["widths"],
                }
                df = pd.DataFrame(df)
                df = df[df["prom"] > 100]
                df = df.sort_values("peaks")

                peaks = df["peaks"].to_numpy()
                if len(peaks) == 0:
                    peak = 250
                else:
                    peak = peaks[-1]
                peak_idx = np.argmin(np.abs(h.axes[0].centers - peak))

                # get info from the histogram
                x = h.axes[0].centers
                y = h.values()
                yerr = np.sqrt(h.variances())

                # Limit the fit around the peak
                mask_fit = (x > peak * 0.8) & (x < peak * 1.2)

                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()

                # Redo the fit using the estimated mu and sigma to limit the range
                _, mu, sigma = [m1.values[k] for k in m1.pos2var]
                mask_fit = (x > mu - 0.8 * sigma) & (x < mu + 0.8 * sigma)
                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()
                plt.plot(
                    x[mask_fit],
                    model(x[mask_fit], *[m1.values[k] for k in m1.pos2var]),
                    label=(
                        "$\\mu$={:.02f} $\\pm$ {:.02f}\n"
                        "$\\sigma$={:.02f} $\\pm$ {:.02f}"
                    ).format(
                        m1.values["mu"],
                        m1.errors["mu"],
                        m1.values["sigma"],
                        m1.errors["sigma"],
                    ),
                    color="orange",
                    linewidth=2,
                )
                plt.plot(peak, y[peak_idx], "ro")
                total_peaks.append(
                    pd.DataFrame(
                        {
                            "DM": [dm],
                            "chip": [chip],
                            "SM_id": [SMs_map[dm][chip]],
                            "bar": [bar],
                            "peak": [m1.values["mu"]],
                            "peak_err": [m1.errors["mu"]],
                            "sigma": [m1.values["sigma"]],
                            "sigma_err": [m1.errors["sigma"]],
                        }
                    )
                )
                peak_bar = m1.values["mu"]
                peak_bar_sigma =  m1.values["sigma"]
                output_iminuit.setdefault(f'CHIP_{chip_id}', {}).setdefault(f'BAR_{bar}', {}).setdefault(f'BAR', {})
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['mu'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['sigma'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['mu']['value'] = m1.values["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['mu']['error'] = m1.errors["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['sigma']['value'] = m1.values["sigma"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'BAR']['sigma']['error'] = m1.errors["sigma"]
            if h == "ch1":
                # fit
                h = histo["histos"][h]
                x = h.axes[0].centers
                #global
                '''
                _peaks, props = find_peaks(
                    h.values(),
                    width=3,
                    height=0.01,
                )
                
                df = {
                    "peaks": h.axes[0].centers[_peaks],
                    "prom": props["prominences"],
                    "widths": props["widths"],
                }
                df = pd.DataFrame(df)
                df = df[df["prom"] > 100]
                df = df.sort_values("peaks")

                peaks = df["peaks"].to_numpy()
                
                if len(peaks) == 0:
                    peak = peak_bar
                else:
                    peak = peaks[-1]
                '''
                #local
                msk_start = int(peak_bar - peak_bar_sigma)
                msk_end = int(peak_bar + peak_bar_sigma)
                mask_h = (x >  msk_start) & (x < msk_end)
                masked_h = h.values()[mask_h]
                if len(masked_h) != 0:
                   index_max_masked_h = np.argmax(masked_h)
                   peak = x[index_max_masked_h] + msk_start
                else:
                   peak = peak_bar
                 
                # get info from the histogram
                y = h.values()
                yerr = np.sqrt(h.variances())
                # Limit the fit around the peak
                mask_fit = (x > peak * 0.8) & (x < peak * 1.2)

                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()
                # Redo the fit using the estimated mu and sigma to limit the range
                _, mu, sigma = [m1.values[k] for k in m1.pos2var]
                mask_fit = (x > mu - 0.8 * sigma) & (x < mu + 0.8 * sigma)
                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()
                output_iminuit.setdefault(f'CHIP_{chip_id}', {}).setdefault(f'BAR_{bar}', {}).setdefault(f'CH_1', {})
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['sigma'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu']['value'] = m1.values["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu']['error'] = m1.errors["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['sigma']['value'] = m1.values["sigma"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['sigma']['error'] = m1.errors["sigma"]
            elif h == "ch2":
                # fit
                h = histo["histos"][h]
                x = h.axes[0].centers
                '''
                #global
                _peaks, props = find_peaks(
                    h.values(),
                    width= 3,
                    height=0.01,
                )
                
                df = {
                    "peaks": h.axes[0].centers[_peaks],
                    "prom": props["prominences"],
                    "widths": props["widths"],
                }
                df = pd.DataFrame(df)
                df = df[df["prom"] > 100]
                df = df.sort_values("peaks")

                peaks = df["peaks"].to_numpy()
                
                if len(peaks) == 0:
                    peak = peak_bar
                else:
                    peak = peaks[-1]
                '''
                #local
                msk_start = int(peak_bar - peak_bar_sigma)
                msk_end = int(peak_bar + peak_bar_sigma)
                mask_h = (x >  msk_start) & (x < msk_end)
                masked_h = h.values()[mask_h]
                if len(masked_h) != 0:
                   index_max_masked_h = np.argmax(masked_h)
                   peak = x[index_max_masked_h] + msk_start
                else:
                   peak = peak_bar
                # get info from the histogram
                y = h.values()
                yerr = np.sqrt(h.variances())
                # Limit the fit around the peak
                mask_fit = (x > peak * 0.8) & (x < peak * 1.2)

                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()

                # Redo the fit using the estimated mu and sigma to limit the range
                _, mu, sigma = [m1.values[k] for k in m1.pos2var]
                mask_fit = (x > mu - 0.8 * sigma) & (x < mu + 0.8 * sigma)
                c = iminuit.cost.LeastSquares(
                    x[mask_fit], y[mask_fit], yerr[mask_fit], model
                )
                m1 = iminuit.Minuit(c, A=200, mu=peak, sigma=40)
                m1.migrad()
                output_iminuit.setdefault(f'CHIP_{chip_id}', {}).setdefault(f'BAR_{bar}', {}).setdefault(f'CH_2', {})
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['sigma'] = {}
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu']['value'] = m1.values["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu']['error'] = m1.errors["mu"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['sigma']['value'] = m1.values["sigma"]
                output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['sigma']['error'] = m1.errors["sigma"]
           
        
        plt.title(
            f"Energy spectrum Run={run} DM_pos={dm_pos} chip_id={chip_id} bar={bar}"
        )
        plt.axvline(output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu']['value'],
                    label=(
                        "$\\mu$={:.02f} $\\pm$ {:.02f}\n"
                        "$\\sigma$={:.02f} $\\pm$ {:.02f}"
                    ).format(
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu']['value'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['mu']['error'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['sigma']['value'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_1']['sigma']['error'],
                    ),
                    color='red', linestyle='-', linewidth=1)
        plt.axvline(output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu']['value'],
                    label=(
                        "$\\mu$={:.02f} $\\pm$ {:.02f}\n"
                        "$\\sigma$={:.02f} $\\pm$ {:.02f}"
                    ).format(
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu']['value'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['mu']['error'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['sigma']['value'],
                        output_iminuit[f'CHIP_{chip_id}'][f'BAR_{bar}'][f'CH_2']['sigma']['error'],
                    ),
                    color='blue', linestyle='-', linewidth=1)
        plt.legend()
        vals = (histo["histos"]["bar"].values())[: int(0.8 * n_bins)]
        ymin, ymax = np.min(vals), np.max(vals)

        delta = ymax - ymin
        plt.ylim(max(ymin - 0.3 * delta, 0), ymax + 0.3 * delta)
        plt.xlim(min_energy, max_energy)

        # plt.xlim(0, 1024)
        # plt.ylim(0, 2000)

        counts_per_bar[bar] = np.sum(vals)
        plt.ylabel("Counts", loc="top")
        plt.xlabel("Energy [ADC]", loc="right")
        plt.savefig(
            f"{output_path}/spectrum_DM{dm_pos:02}_chip{chip_id}_bar{bar:02}.png",
            dpi=150,
        )
        d[f"h_DM{dm_pos:02}_chip{chip_id}_bar{bar:02}"] = histo["histos"]["bar"]
        d[f"h_DM{dm_pos:02}_chip{chip_id}_ch{ch1:02}"] = histo["histos"]["ch1"]
        d[f"h_DM{dm_pos:02}_chip{chip_id}_ch{ch2:02}"] = histo["histos"]["ch2"]

        # Bar correlation of channels
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.tight_layout(pad=-0.5)
        histo["histos"]["corr"].plot2d(ax=ax, norm=mpl.colors.LogNorm(1, 2000))

        plt.title(
            f"Energy Correlation Run={run} DM_pos={dm_pos} chip_id={chip_id} bar={bar}"
        )

        os.makedirs(f"{output_path}/bar_correlation/", exist_ok=True)
        plt.savefig(
            f"{output_path}/bar_correlation/bar_correlation_DM{dm_pos:02}_chip{chip_id}_bar{bar:02}.png",
            # f"test.png",
            dpi=150,
            facecolor="white",
            pad_inches=0.1,
            bbox_inches="tight",
        )
        plt.close()
        
    
    f.close()
    total_peaks = pd.concat(total_peaks)
    return counts_per_bar, d, total_peaks, output_iminuit


def plot_bar_map(data, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))  # 3x4 grid of subplots

    colormap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(cmap=colormap)

    vals = []
    for dm in range(12):
        for chip in range(2):
            for bar in range(16):
                vals.append(data[dm][chip][bar])

    vmin, vmax = min(vals), max(vals)
    vmin = 0
    sm.set_clim(vmin, vmax)
    norm = plt.Normalize(vmin, vmax)
    plt.colorbar(sm, ax=ax, orientation="vertical", label="Counts")

    cols = 4
    rows = 3

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-0.5, cols - 0.5)
    dy = -0.1 + 0.01
    ax.set_ylim(-0.5 + dy, rows - 0.5 + dy)

    switch_chip = False
    for i in range(rows):
        for j in range(cols):
            dm_index = i * 4 + j  # Determine DM index (0 to 11)

            ax.add_patch(
                plt.Rectangle(
                    (j - 0.45, i - 0.4),
                    0.9,
                    0.8,
                    color="red",
                    alpha=0.4,
                    linewidth=0.0,
                )
            )

            ax.text(
                j,
                i - 0.45,
                f"DM{dm_index:02}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

            chips = [0, 1]
            if switch_chip:
                chips = [1, 0]
                switch_chip = False
            else:
                switch_chip = True
            print(dm_index, chips)

            ax.text(
                j - 0.3,
                i - 0.45,
                f"C-{chips[0]}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
            ax.text(
                j + 0.3,
                i - 0.45,
                f"C-{chips[1]}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )

            for chip_id in range(2):
                chip_x = j - 0.45 + 0.47 * chip_id % 2
                chip_y = i - 0.4

                sm_dx = 0.43

                ax.add_patch(
                    plt.Rectangle(
                        (chip_x, chip_y),
                        sm_dx,
                        0.8,
                        color="lightblue",
                        linewidth=0.0,
                        alpha=0.3,
                    )
                )
                # 16 inside

                bar_order = list(range(16))[::-1]
                if chip_id == 1:
                    bar_order = bar_order[::-1]

                # left = data[dm_index][chips[chip_id]][bar_order[0]]
                # right = data[dm_index][chips[chip_id]][bar_order[-1]]
                # print(left, right)

                for i_bar, bar_idx in enumerate(bar_order):
                    value = data[dm_index][chips[chip_id]][bar_idx]
                    color = colormap(norm(value))  # Map value to color

                    dx = sm_dx / 16
                    dy = 0.8
                    ax.add_patch(
                        plt.Rectangle(
                            (chip_x + dx * i_bar, chip_y),
                            dx,
                            dy,
                            color=color,
                            linewidth=0.0,
                        )
                    )
                    ax.text(
                        chip_x + dx * i_bar + dx / 3,
                        chip_y + dy / 2,
                        f"{value}",
                        # ha="center",
                        # va="center",
                        fontsize=4,
                        color="black",
                        rotation=90,
                    )

    plt.gca().invert_yaxis()
    plt.savefig(f"{output_path}/bar_map.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    mtd_daq_path = get_mtd_daq_path()
    if len(argv) != 2:
        print("Usage: plot_peak.py run_number")

    run = int(argv[1])

    # get the run folder
    parent_runs_folder = load_general_config()["tofhir_runs_folder"]
    run_folder = f"{parent_runs_folder}/run_{run}"

    # load the general config of the run
    config = load_general_config(f"{run_folder}/general.cfg")

    key = "tofhir"
    runs_folder = config[f"{key}_runs_folder"]
    input_path = f"{runs_folder}/run_{run}"

    output_folder = config[f"{key}_plots_folder"]
    output_path = f"{output_folder}/plots_run_{run}"

    os.makedirs(output_path, exist_ok=True)

    structure_file = f"{run_folder}/structure.json"
    SMs_map = get_SM_map(structure_file)
   	
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        
        dm_chips = []
        tasks = []
        for dm in range(12):
            for chip in range(2):
                tasks.append(
                    executor.submit(
                        task_fun, input_path, output_path, run, dm, chip, SMs_map
                    ),
                )
                
                dm_chips.append((dm, chip))
        fits = {
        	f'DM_{dm}': {}
         	for dm in range(12)
        }
        bars_map = {
            dm: {chip: {bar: 0 for bar in range(16)} for chip in range(2)}
            for dm in range(12)
        }
        
        concurrent.futures.wait(tasks)
        d = {}
        total_peaks = []
        for dm_chip, task in zip(dm_chips, tasks):
            dm, chip = dm_chip
            if len(task.result()) == 4:
            	bars, partial_d, partial_peaks, fit = task.result()
            	fits[f'DM_{dm}'].update(fit)
            else:
            	bars, partial_d, partial_peaks = task.result()
            
           
            for i in range(16):
                bars_map[dm][chip][i] = int(bars[i])
            d = add_dict(d, partial_d)

            if not isinstance(partial_peaks, int):
                total_peaks.append(partial_peaks)
    
       
    
    
    
    
    directory_fits = f"/nas-cms/DATA/runs/tofhir/run_{run}"
    with open(f"{directory_fits}/general.cfg", 'r') as file:
       for line in file:
          if 'ov =' in line:
             ov = float(line.split('=')[1].strip())
    with open(f"{directory_fits}/config.ini", 'r') as file:
       for line in file:
          if 'channel.cfg_a2_attenuator_gain = ' in line:
             att_gain = float(line.split('=')[1].strip())
    fits_out = {
       'ov': ov,
       'att_gain': att_gain,
       'iminuit': fits
    }
    path_fits = os.path.join(directory_fits, "iminuit_values.json")
    if not os.path.exists(directory_fits):
       os.makedirs(directory_fits)
    with open(path_fits, "w") as file:
       json.dump(fits_out, file, indent=4)
  
    total_peaks = pd.concat(total_peaks)
    total_peaks.to_parquet(f"{output_path}/peaks.pq")

    total_peaks = pd.read_parquet(f"{output_path}/peaks.pq")

    for dm in range(12):
        for chip in range(2):
            mask = (total_peaks["DM"] == dm) & (total_peaks["chip"] == chip)
            if len(total_peaks[mask]) == 0:
                continue
            x = total_peaks[mask]["bar"]
            y = total_peaks[mask]["peak"]
            yerr = total_peaks[mask]["peak_err"]
            plt.clf()
            plt.errorbar(x, y, yerr=yerr, color="black", fmt="o--")
            plt.title(f"Energy peaks Run={run} DM_pos={dm} chip_id={chip}")
            plt.ylabel("Energy [ADC]", loc="top")
            plt.xlabel("Bar ID", loc="right")
            plt.ylim(0, 700)
            plt.savefig(
                f"{output_path}/peaks_DM{dm:02}_chip{chip}.png",
                dpi=150,
            )

    with open(f"{output_path}/counts.json", "w") as file:
        json.dump(bars_map, file)

    with open(f"{output_path}/counts.json", "r") as file:
        bars_map = json.load(file)

    def decode(o):
        if isinstance(o, dict):
            d = {}
            for k in o:
                d[int(k)] = decode(o[k])
        else:
            return int(o)
        return d

    bars_map = decode(bars_map)
    print(bars_map)


    plot_bar_map(bars_map, output_path)

    # output_root_file.Close()
    output_root_file = uproot.recreate(f"{output_path}/histos.root")
    for key in d:
        output_root_file[key] = d[key]
    output_root_file.close()
