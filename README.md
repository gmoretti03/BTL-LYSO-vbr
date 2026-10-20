# BTL-LYSO-vbr

**Exploiting the Intrinsic Radioactivity of LYSO Crystals for the Measurement of the
SiPM Breakdown Voltage for the BTL Detector**

Bachelor thesis in Physics, University of Milano-Bicocca, a.y. 2024/2025.
Author: Giacomo Moretti. Supervisor: Prof. Federico De Guio.

## Context

The Barrel Timing Layer (BTL) of the CMS MIP Timing Detector reads LYSO:Ce crystal
bars with SiPMs. SiPMs must be operated at a known overvoltage above their breakdown
voltage (VBR), which drifts with temperature and radiation damage. This work measures
VBR *in situ* from the natural radioactivity of the crystals (176Lu): the LYSO
photopeak is fitted as a function of the applied voltage (OV scan) and the resulting
VBR is compared with a dedicated IV scan on the same tray.

**Main result:** the OV-scan VBR is systematically ~0.47 V higher than the IV-scan
one, uniformly across 11 of 12 detector modules, while nominal Hamamatsu values agree
with the IV scan to ~0.03 V. The origin of the offset is not resolved.

## Contents

| Path | Content |
|------|---------|
| `thesis/thesis.pdf` | Full thesis (English) |
| `thesis/summary_it.pdf` | Short Italian version |
| `slides/thesis_defense.pdf` | Thesis defense slides |
| `slides/SIF_conference_talk.pdf` | Talk at the SIF national conference |
| `code/ov_analysis/` | OV vs IV analysis |
| `code/temp_analysis/` | Temperature uniformity study |
| `code/plot_style.py` | Shared CMS plot style |
| `code/old/` | Original scripts for the CMS lab machines |

## Code and data

The scripts need Python 3 with `numpy`, `matplotlib` and `iminuit`. The input data
(LYSO fit results, IV scans, temperatures) come from the CMS BTL lab and are **not
included**, so the scripts are provided for reference and will not run as-is.
The scripts in `code/old/` additionally depend on the CMS-internal `mtd_daq` package.

## License

Code: MIT (see `LICENSE`). Thesis and slides: all rights reserved by the author.
