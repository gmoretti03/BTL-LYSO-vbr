#!/usr/bin/env python

from fileinput import filename
import json

import iminuit
import iminuit.cost
from iminuit import Minuit
from iminuit.cost import LeastSquares
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
import numpy as np
import copy
from sys import argv
import sys
import os
import re
from mtd_daq.configurator.configurator import load_general_config
from matplotlib.ticker import AutoMinorLocator
import time

def progress_bar(total, current):
    progress = int(current + 1 / total * 12)
    bar = '[' + '#' * progress + '-' * (12 - progress) + ']'
    print(f'\r{bar} DM_{current}/{total} Done', end='', flush=True)


def weighted_average(values, errors):
    values = np.array(values)
    errors = np.array(errors)

    if len(values) == 0 or np.any(errors == 0):
        return np.nan

    weights = 1 / errors**2
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return np.nan

    average = np.sum(values * weights) / sum_weights
    return float(average)

def weighted_average_error(errors):
    errors = np.array(errors)

    if len(errors) == 0 or np.any(errors == 0):
        return np.nan

    weights = 1 / errors**2
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return np.nan

    return float(1 / np.sqrt(sum_weights))



def func_fit(ov, N_coef, VB_shift):
	return N_coef*0.638*(1 - np.exp(-0.651*(ov-VB_shift)))*(7.044E04 + 2.895E05*(ov-VB_shift))
	
def linear_fit(x, q):
	x = np.array(x)
	return 1.0 * x + q

def odr_cost_function_linear(x, y, xerr, yerr):
	'''Cost function for orthogonal distance regression with y = a*x + b'''
	def chi2(q):
		result = 0.0
		for xi, yi, xi_err, yi_err in zip(x, y, xerr, yerr):
			model_y = 1.0*xi + q
			dydx = 1.0  # derivative of y = a*x + b w.r.t x is just a
			y_var = yi_err**2 + (dydx * xi_err)**2
			result += ((yi - model_y) ** 2) / y_var
		return result
	return chi2	


def read_runs():
	all_index_paras = {}
	all_iminuit_values = {}
	all_iminuit_values.setdefault('att_gain', {})
	#tray
	#'''
	index_runs = np.arange(first_run, last_run+1, 1)
	#'''
	#no tray
	'''
	index_runs_raw = np.arange(first_run, last_run+1, 1)
	mask = ((index_runs_raw <= 1739) | (index_runs_raw >= 2043) & (index_runs_raw <= 2050) | (index_runs_raw >= 6204) & (index_runs_raw <= 6240) | (index_runs_raw >= 7424))
	index_runs = index_runs_raw[mask]
	'''
	print(f'runs\n{index_runs}')
	i = 7	
	for run in index_runs:
		path = f"{path_runs}run_{run}/"
		iminuit_values = json.load(open(f"{path}iminuit_values.json"))
		#tray
		#'''
		all_index_paras.setdefault(iminuit_values['att_gain'], {}).setdefault(iminuit_values['ov']-(i*0.1), [run])
		i -= 1
		if i == -1:
			i = 7
		#'''
		#no tray
		'''
		if run <= 2050:
			all_index_paras.setdefault(iminuit_values['att_gain'], {}).setdefault(iminuit_values['ov']-(i*0.1), [run])
			i -= 1
			if i == -1:
				i = 7 
		else:
			all_index_paras[iminuit_values['att_gain']][iminuit_values['ov']].append(run)
		'''
	all_index_paras = {k: {subk: all_index_paras[k][subk] for subk in sorted(all_index_paras[k].keys())} for k in sorted(all_index_paras.keys())}
	for att_gain in (list(all_index_paras.keys())):
		all_iminuit_values['att_gain'][att_gain] = {}
		all_iminuit_values['att_gain'][att_gain]['ov'] = []
		for ov in (list(all_index_paras[att_gain].keys())):
			index = [int(x) for x in all_index_paras[att_gain][ov]]
			for num_int in index:
				path = f"{path_runs}run_{num_int}/"
				iminuit_values = json.load(open(f"{path}iminuit_values.json"))
				all_iminuit_values['att_gain'][att_gain]['ov'].append(ov)
				if ov == min(all_index_paras[att_gain]):
					for arg in list(iminuit_values.keys()):
						if arg == 'iminuit':
							all_iminuit_values['att_gain'][att_gain][arg] = {}
							for dm in list(iminuit_values[arg].keys()):
								all_iminuit_values['att_gain'][att_gain][arg][dm] = {}
								for chip in (list(iminuit_values[arg][dm].keys())):
									all_iminuit_values['att_gain'][att_gain][arg][dm][chip] = {}
									for bar in (list(iminuit_values[arg][dm][chip].keys())):
										all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar]  = {}
										for channel in (list(iminuit_values[arg][dm][chip][bar].keys())):
											if channel != 'BAR':
												all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel]  = {}
												for paras in (list(iminuit_values[arg][dm][chip][bar][channel].keys())):
													if paras != 'sigma':
														all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel][paras] = {}
														for vals in (list(iminuit_values[arg][dm][chip][bar][channel][paras].keys())):
															all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel][paras][vals] = []
				for arg in list(iminuit_values.keys()):
					if arg == 'iminuit':
						for dm in list(iminuit_values[arg].keys()):
							for chip in (list(iminuit_values[arg][dm].keys())):
								for bar in (list(iminuit_values[arg][dm][chip].keys())):
									for channel in (list(all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar].keys())):
										for paras in (list(all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel].keys())):
											for vals in (list(all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel][paras].keys())):
												all_iminuit_values['att_gain'][att_gain][arg][dm][chip][bar][channel][paras][vals].append(iminuit_values[arg][dm][chip][bar][channel][paras][vals])
	return(all_iminuit_values)

def error_correction(raw_data):
	for att_gain in (list(raw_data['att_gain'].keys())):
		att_gain_path = f'{study_path}/att_gain_{att_gain}'
		os.makedirs(att_gain_path, exist_ok=True)
		ov_path = f'{att_gain_path}/ov_3.0'
		os.makedirs(ov_path, exist_ok=True)
		print(f'\nerr_corr attg_{att_gain}')
		for arg in list(raw_data['att_gain'][att_gain].keys()):
			if arg == 'iminuit':
				pb = 0
				total_loss = []
				for dm in list(raw_data['att_gain'][att_gain][arg].keys()):
					dm_loss = []
					dm_path = f'{ov_path}/{dm}'
					os.makedirs(dm_path, exist_ok=True)
					for chip in (list(raw_data['att_gain'][att_gain][arg][dm].keys())):
						for bar in (list(raw_data['att_gain'][att_gain][arg][dm][chip].keys())):
							peaks = {}
							for channel in (list(raw_data['att_gain'][att_gain][arg][dm][chip][bar].keys())):
								peaks.setdefault(channel, {})
								for paras in (list(raw_data['att_gain'][att_gain][arg][dm][chip][bar][channel].keys())):
									value = raw_data['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['value']
									error = raw_data['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['error']
										
									value_mean = value[8:] 
									error_mean = error[8:]
									value = value[:7]
									error = error[:7]
											
									value_mean = np.array(value_mean)
									error_mean = np.array(error_mean)
											
									mask_value = (~np.isnan(value_mean)) & (value_mean >= 100) & (value_mean <= 600)
									mask_error = (~np.isnan(error_mean)) & (error_mean > 0)
									mask = mask_value & mask_error
									peaks[channel].setdefault('raw_peaks', [value_mean])
									peaks[channel].setdefault('valid_peaks', [value_mean[mask]])
									peaks[channel].setdefault('valid_peaks_error', [error_mean[mask]])
									perc_loss = (1 - (len(value_mean[mask])/len(value_mean)))*100
									dm_loss.append(perc_loss)
									total_loss.append(perc_loss)
									peaks[channel].setdefault('perc_loss', [perc_loss])
									if len(value_mean[mask]) == 0:
										correction = 0
									else:
										correction = np.std(value_mean[mask])
									value.append(weighted_average(value_mean[mask], error_mean[mask]))
									error.append(weighted_average_error(error_mean[mask]))
									raw_data['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['value'] = value
									for i in range(len(value) - 1):
										old_error = error[i]
										new_error = np.sqrt(correction**2 + old_error**2)
										error[i] = new_error
									raw_data['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['error'] = error
							#comment to avoid plot
							'''				
							fig, axs = plt.subplots(2, 1, figsize=(20, 20))
							plt.subplots_adjust(hspace=0.1)
							fig.suptitle(f"SiPM array {dm}", fontsize=25)
							fig.subplots_adjust(top=0.95)
							for ch in list(peaks.keys()):
								if ch == 'CH_1':
									i = 0
									c = 'red'
									val = 'CH_1_peaks'
									raw_vect = peaks[ch]['raw_peaks'][0]
									valid_vect = peaks[ch]['valid_peaks'][0]
									valid_err_vect = peaks[ch]['valid_peaks_error'][0]
									loss = peaks[ch]['perc_loss'][0]	
								elif ch == 'CH_2':
									i = 1
									c = 'blue'
									val = 'CH_2_peaks'
									raw_vect = peaks[ch]['raw_peaks'][0]
									valid_vect = peaks[ch]['valid_peaks'][0]
									valid_err_vect = peaks[ch]['valid_peaks_error'][0]
									loss = peaks[ch]['perc_loss'][0]
								ax = axs[i]
								#ax.set_title(f'{val}', fontsize=35, loc='right', pad=15)
								ax.text(0.95, 0.95, f'{channel}', transform=ax.transAxes, fontsize=20, ha='left', va='top', bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=1))
								ax.hist(raw_vect, bins=math.ceil(3*np.sqrt(len(raw_vect))), color=c, alpha=0.7)
								mean = weighted_average(valid_vect, valid_err_vect)
								if len(valid_vect) == 0:
									std = np.nan
								else:
									std = np.std(valid_vect)
								ax.axvline(mean, color='black', linestyle='--', linewidth=3, label=f'mean = {mean:.2f}')
								ax.axvspan(mean - std, mean + std, color='gray', alpha=0.3, label=f'±1 std = {std:.2f}')
								ax.legend(title=f'rejected data = {perc_loss:.2f}%', frameon=True, fontsize=25,  title_fontsize=25)
								if i != 0:
									ax.set_xlabel('ADC', fontsize=25, labelpad = 15)
								ax.tick_params(axis='both', labelsize=25)
							plt.savefig(
								f"{dm_path}/{dm}_{chip}_{bar}.png",
								dpi=180,
								bbox_inches="tight",
								facecolor="white",
								pad_inches=0.1,
							)
							plt.close()
							'''	
					plot_histogram(dm_loss)
					plt.savefig(
						f"{dm_path}/{dm}_data_rejected.png",
						dpi=180,
						bbox_inches="tight",
						facecolor="white",
						pad_inches=0.1,
					)
					plt.close()			
					progress_bar(11, pb)
					pb += 1
				plot_histogram(total_loss)
				plt.savefig(
					f"{ov_path}/ov_3.0_data_rejected.png",
					dpi=180,
					bbox_inches="tight",
					facecolor="white",
					pad_inches=0.1,
				)
				plt.close()
	return raw_data
	

def plot_histogram(data):
    fig, ax = plt.subplots(figsize=(12, 8))

    bins = math.ceil(np.sqrt(len(data)))*3
    ax.hist(data, bins=bins, color='orange', alpha=0.7)

    mean = np.mean(data)
    std = np.std(data)

    ax.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'mean = {mean:.2f}')
    ax.axvspan(mean - std, mean + std, color='grey', alpha=0.3, label=f'±1 std = {std:.2f}')
    ax.set_xlabel('Percentage [%]', fontsize=35, labelpad=25)
    ax.legend(frameon=True, fontsize=30)
    ax.tick_params(axis='both', labelsize=28)
    plt.tight_layout()
																					

def check_vect(vect_x, vect_y, err_vect_y):
	'''
	max_distance = 100
	distances = [abs(vect_y[i+1] - vect_y[i]) for i in range(len(vect_y) - 1)]
	'''
	vect_x_filtered = []
	vect_y_filtered = []
	err_vect_y_filtered = []
	for x, y, e in zip(vect_x[0:], vect_y[0:], err_vect_y[0:]):
		if 100 <= y <= 600 and ~np.isnan(e): 
			vect_x_filtered.append(x)
			vect_y_filtered.append(y)
			err_vect_y_filtered.append(e)
	return vect_x_filtered, vect_y_filtered, err_vect_y_filtered
	

def fit_iminuit(imi_fits, path):
	new_folder = "parameter_study"
	full_path = os.path.join(path, new_folder)
	os.makedirs(full_path, exist_ok=True)
	filename = "VB_shifts_per_const.txt"
	values = np.arange(0.5, 1.75, 0.25)
	print(values)
	with open(filename, 'w') as out_file:
		out_file.write("Parameters | " + " ".join(f"{v:.3f}" for v in values) + "\n")
		for att_gain in (list(imi_fits['att_gain'].keys())):
			for arg in list(imi_fits['att_gain'][att_gain].keys()):
				if arg == 'iminuit':
					for dm in list(imi_fits['att_gain'][att_gain][arg].keys()):
						for chip in (list(imi_fits['att_gain'][att_gain][arg][dm].keys())):
							for bar in (list(imi_fits['att_gain'][att_gain][arg][dm][chip].keys())):
								for channel in (list(imi_fits['att_gain'][att_gain][arg][dm][chip][bar].keys())):
									for paras in (list(imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel].keys())):
										ov = imi_fits['att_gain'][att_gain]['ov']
										E = imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['value']
										err = imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['error']
										ov, E, err = check_vect(ov, E, err)
										base_consts = {
											"c1": 0.638,
											"c2": 0.651,
											"q": 7.044E04,
											"m": 2.895E05,
										}
										
										const_names = ["c1", "c2", "q", "m"]
										if dm == "DM_3" and chip == "CHIP_0" and bar == "BAR_1" and channel in ["CH_1", "CH_2"]:
											results = {c: [] for c in const_names}
											plt.figure(figsize=(8, 5))
											for val in values:
												for const_to_modify in const_names:
													consts = base_consts.copy()
													consts[const_to_modify] *= val
													'''
													if const_to_modify == 'c1':
														consts['c2'] *= val
														consts['q'] *= val
														consts['m'] *= val
													'''

													def func_fit_val(ov, N_coef, VB_shift):
														delta = np.array(ov) - VB_shift
														return N_coef * consts["c1"] * (1 - np.exp(-consts["c2"] * delta)) * (consts["q"] + consts["m"] * delta)

													c = LeastSquares(ov, E, err, func_fit_val)
													m1 = iminuit.Minuit(c, N_coef=0.001, VB_shift=0.0)
													m1.migrad()
													
													N_coef, VB_shift = [m1.values[k] for k in m1.pos2var]
													c = LeastSquares(ov, E, err, func_fit_val)
													m1 = iminuit.Minuit(c, N_coef=N_coef, VB_shift=VB_shift)
													m1.migrad()
													
													E_model = func_fit_val(ov, N_coef, VB_shift)
													if(const_to_modify == "c2"):
														vb_val = m1.values["VB_shift"]
														vb_err = m1.errors["VB_shift"]
														perc = int(val * 100)
														plt.plot(ov, E_model, linestyle='--', linewidth=2, label=f"{perc}% | VB_shift = {vb_val:.3f} ± {vb_err:.3f}")

													results[const_to_modify].append(m1.values['VB_shift'])

											plt.errorbar(ov, E, yerr=err, fmt='o', color='black', label='Data', markersize=8)

											# Setup grafico
											plt.title(f"{dm} {chip} {bar} {channel}", fontsize=20)
											plt.xlabel("OV [V]", fontsize=20)
											plt.ylabel("peak [ADC]", fontsize=20)
											plt.legend(fontsize=12)
											plt.grid(True)
											plt.xticks(fontsize=20)
											plt.yticks(fontsize=20)
											plt.tight_layout()

											# Salva la figura
											plt.savefig(
												f"{	full_path}/{dm}_{chip}_{bar}_{channel}.png",
												dpi=180,
												bbox_inches="tight",
												facecolor="white",
												pad_inches=0.1,
											)
											plt.close()
													
											prefix = f"{dm} {chip} {bar} {channel}"
											for const, vb_list in results.items():
												vb_str = " ".join(f"{v:.6f}" for v in vb_list)
												print(f"{prefix} | {const} | {vb_str}")
												out_file.write(f"{prefix} | {const} | {vb_str}\n")
										c = iminuit.cost.LeastSquares(ov, E, err, func_fit)
										m1 = iminuit.Minuit(c, N_coef= 0.001, VB_shift=0)
										m1.migrad()
										N_coef, VB_shift = [m1.values[k] for k in m1.pos2var]
										c = iminuit.cost.LeastSquares(ov, E, err, func_fit)
										m1 = iminuit.Minuit(c, N_coef, VB_shift)
										m1.migrad()
										del imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]
										for k in m1.pos2var:
											imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel].setdefault(f'{k}', {})
											imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel][f'{k}']['value'] = m1.values[k]
											imi_fits['att_gain'][att_gain][arg][dm][chip][bar][channel][f'{k}']['error'] = m1.errors[k]
	print('\nfit done')

def plot_runs(imi_vals, fit_vals):

	vbr_corr = {}
	
	for att_gain in (list(imi_vals['att_gain'].keys())):
		vbr_corr.setdefault(att_gain, {})
		att_gain_path = f'{study_path}/att_gain_{att_gain}'
		for arg in list(imi_vals['att_gain'][att_gain].keys()):
			if arg == 'iminuit':
				print(f'\nplot attg_{att_gain}')
				d = 0
				pb = 0
				total_loss = []
				for dm in list(imi_vals['att_gain'][att_gain][arg].keys()):
					dm_path = f'{att_gain_path}/{dm}'
					os.makedirs(dm_path, exist_ok=True)
					for chip in (list(imi_vals['att_gain'][att_gain][arg][dm].keys())):
						vb_shift = {}
						for bar in (list(imi_vals['att_gain'][att_gain][arg][dm][chip].keys())):
							#comment to avoid plot
							'''
							fig, axs = plt.subplots(2, 1, figsize=(20, 20))
							plt.subplots_adjust(hspace=0.1)
							fig.suptitle(f"{dm}_{chip}_{bar}", fontsize=35)
							fig.subplots_adjust(top=0.95)
							'''
							for channel in (list(imi_vals['att_gain'][att_gain][arg][dm][chip][bar].keys())):
								vb_shift.setdefault(channel, {})
								vb_shift[channel].setdefault('values', [])
								vb_shift[channel].setdefault('errors', [])
								for paras in (list(imi_vals['att_gain'][att_gain][arg][dm][chip][bar][channel].keys())):
									x = imi_vals['att_gain'][att_gain]['ov'][:8]
									y = imi_vals['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['value']
									err = imi_vals['att_gain'][att_gain][arg][dm][chip][bar][channel][paras]['error']
											
									if channel == 'CH_1':
										i = 0
										c = 'red'
										vect = vb_shift[channel]['values']
										err_vect = vb_shift[channel]['errors']
									elif channel == 'CH_2':
										i = 1
										c = 'blue'
										vect = vb_shift[channel]['values']
										err_vect = vb_shift[channel]['errors']
												
									N_coef_val = fit_vals['att_gain'][att_gain][arg][dm][chip][bar][channel]['N_coef']['value']
									VB_shift_val = fit_vals['att_gain'][att_gain][arg][dm][chip][bar][channel]['VB_shift']['value']
									VB_shift_err_val = fit_vals['att_gain'][att_gain][arg][dm][chip][bar][channel]['VB_shift']['error']
													
									vect.append(VB_shift_val)
									err_vect.append(VB_shift_err_val)
										
									#comment to avoid plot
									'''
									ax = axs[i]
									ax.set_ylabel('peak [ADC]', fontsize=35, labelpad=25)
									if i == 1:
										ax.set_xlabel('OV [V]', fontsize=35, labelpad=20)
									#ax.set_title(f'{channel}', fontsize=35, loc='right', pad=20)
									ax.text(0.95, 0.95, f'{channel}', transform=ax.transAxes, fontsize=30, ha='left', va='top', bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=1))
									ax.errorbar(x, y, err, fmt='',color=c, ecolor=c, linestyle = '--', linewidth=4, marker='o', markersize=12)
									ov = np.linspace(min(x), max(x), 100)
									ax.plot(ov, func_fit(ov, N_coef_val,VB_shift_val), color='black', linewidth=3, label=f"VB_shift = {VB_shift_val:.2f} ± {VB_shift_err_val:.2f}")
									ax.axhspan(100, 600, xmin=0, xmax=1, color='gray', alpha=0.3)	
									ax.tick_params(axis='both', labelsize=35)
									ax.legend(fontsize=35)
									ax.grid(True)
							plt.savefig(
									f"{dm_path}/{dm}_{chip}_{bar}.png",
									dpi=180,
									bbox_inches="tight",
									facecolor="white",
									pad_inches=0.1,
							)
							plt.close()
						'''
						fig, axs = plt.subplots(2, 1, figsize=(20, 20))
						plt.subplots_adjust(hspace=0.1)
						fig.suptitle(f"SiPM array {dm}", fontsize=35)
						fig.subplots_adjust(top=0.95)
						for ch in list(vb_shift.keys()):
							if ch == 'CH_1':
								i = 0
								c = 'red'
								letter = 'A'
								val = 'CH1_VB_shift'
								vect = vb_shift[ch]['values']
								err_vect = vb_shift[ch]['errors']
							elif ch == 'CH_2':
								i = 1
								c = 'blue'
								letter = 'B'
								val = 'CH2_VB_shift'
								vect = vb_shift[ch]['values']
								err_vect = vb_shift[ch]['errors']
							ax = axs[i]
							ax.text(0.95, 0.95, f'{val}', transform=ax.transAxes, fontsize=30, ha='right', va='top', bbox=dict(boxstyle='square', facecolor='white', edgecolor='none', alpha=1))
							#ax.set_title(f'{val}', fontsize=30, loc='right', pad=15)
							ax.hist(vect, bins=math.ceil(3*np.sqrt(len(vect))), color=c, alpha=0.7)
							
							vect = np.array(vect)
							err_vect = np.array(err_vect)
							mask_value = (~np.isnan(vect)) & (np.abs(vect) <= 1) ###sicuri di questa condizione?
							mask_error = (~np.isnan(err_vect)) & (err_vect > 0)
							mask = mask_value & mask_error
							perc_loss = (1 - (len(vect[mask])/len(vect)))*100
							total_loss.append(perc_loss)
							if perc_loss == 100:
								print(f'{dm} {chip} no data available')
							mean = weighted_average(vect[mask], err_vect[mask])
							err = weighted_average_error(err_vect[mask])
							std = np.std(vect[mask])
							num_chip = re.findall(r'\d+', chip)[0]
							vbr_corr[att_gain].setdefault(f'{d}', {}).setdefault(num_chip, {}).setdefault(letter, {})
							vbr_corr[att_gain][f'{d}'][num_chip][letter]['value'] = float(mean)
							vbr_corr[att_gain][f'{d}'][num_chip][letter]['error'] = float(err)
							ax.axvline(mean, color='black', linestyle='--', linewidth=2, label=f'mean = {mean:.2f}')
							ax.axvspan(mean - std, mean + std, color='gray', alpha=0.3, label=f'±1 std = {std:.2f}')
							ax.legend(title=f'rejected data = {perc_loss:.2f}%', frameon=True, fontsize=35,  title_fontsize=35)
							if i != 0:
								ax.set_xlabel('VB shift [V]', fontsize=35, labelpad = 25)
							ax.tick_params(axis='both', labelsize=35)
						plt.savefig(
							f"{dm_path}/{dm}_{chip}.png",
							dpi=180,
							bbox_inches="tight",
							facecolor="white",
							pad_inches=0.1,
						)
						plt.close()
					d = d + 1
					progress_bar(11, pb)
					pb += 1
				plot_histogram(total_loss)
				plt.savefig(
					f"{att_gain_path}/attg_{att_gain}_data_rejected.png",
					dpi=180,
					bbox_inches="tight",
					facecolor="white",
					pad_inches=0.1,
				)
				plt.close()
	return vbr_corr
	

def correlation(vbr_shift):
	#no tray
	#iv = [39, 40, 41, 52, 53, 54]
	#tray
	#iv = [140, 141, 142, 143, 144, 145, 146, 147]
	iv = [150, 151, 152, 153, 154, 155, 156, 157]
	vbr_iv = {}
	vbr_hama = {}
	for atg in list(vbr_shift.keys()):
		vbr_iv.setdefault(atg, {})
		vbr_hama.setdefault(atg, {})
		for iv_r in iv:
			x = []
			y = []
			err_x = [] 
			err_y = [0] * len(iv)
			vbr_iv[atg].setdefault(iv_r, {})
			vbr_hama[atg].setdefault(iv_r, {})
			with open(f"/nas-cms/DATA/runs/iv_scan/run_{iv_r}/vbr_dm.json") as file:
				vbr_iv[atg][iv_r] = json.load(file)
			vbr_iv[atg][iv_r] = dict(sorted(vbr_iv[atg][iv_r].items(), key=lambda x: int(x[0])))
			for dm in list(vbr_iv[atg][iv_r].keys()):
				for chip in list(vbr_iv[atg][iv_r][dm].keys()):
					for array in list(vbr_iv[atg][iv_r][dm][chip].keys()):
						num = vbr_iv[atg][iv_r][dm][chip][array]
						y.append(num)
			atg_path = f'{study_path}/att_gain_{atg}'
			
			vbr_hama[atg][iv_r] = copy.deepcopy(vbr_iv[atg][iv_r])
			with open(f'/home/cmsdaq/DAQ/mtd_daq/data/structures/TRAY3_RU0.json') as file: #TRAY3_RU0 - new_structure_complete_3
				vbr_structure = json.load(file)
			for key in list(vbr_structure.keys()):
				if key == "DMs":
					for index in range(len(vbr_structure[key])):
						dm = vbr_structure[key][index]['position']
						for chip in list(vbr_structure[key][index]['voltages'].keys()):
							for array in list(vbr_structure[key][index]['voltages'][chip].keys()):
								#no tray
								'''
								if dm != 1:
									vbr_hama[atg][iv_r][str(dm)][chip][array] = vbr_structure[key][index]['voltages'][chip][array]
								'''
								#tray
								vbr_hama[atg][iv_r][str(dm)][chip][array] = vbr_structure[key][index]['voltages'][chip][array]
			vbr_hama[atg][iv_r] = dict(sorted(vbr_hama[atg][iv_r].items(), key=lambda x: int(x[0])))
			for dm in list(vbr_hama[atg][iv_r].keys()):
				for chip in list(vbr_hama[atg][iv_r][dm].keys()):
					for array in list(vbr_hama[atg][iv_r][dm][chip].keys()):
						num = vbr_hama[atg][iv_r][dm][chip][array]
						vbr_hama[atg][iv_r][dm][chip][array] = {
							'value': num + vbr_shift[atg][dm][chip][array]['value'], #correction applied
							'error': vbr_shift[atg][dm][chip][array]['error']
						}
						x.append(vbr_hama[atg][iv_r][dm][chip][array]['value'])
						err_x.append(vbr_hama[atg][iv_r][dm][chip][array]['error'])		
			fig, ax = plot_corr(x, y, err_x , err_y, 'HPK')	
			#ax.set_title(f'VBR Pull Distribution', fontsize=35, pad=15)
			ax.set_xlabel('OV scan [V]', fontsize=45, labelpad=25)
			ax.set_ylabel(f'IV scan [V]', fontsize=45, labelpad=25)
			ax.tick_params(axis='both', labelsize=45)
			ax.legend(frameon=True, fontsize=40)		
			plt.savefig(
				f"{atg_path}/attg_{atg}_HPK_IV_{iv_r}.png",
				dpi=160,
				bbox_inches="tight",
				facecolor="white",
				pad_inches=0.1,
			)
			plt.close()
			
			x_array = np.array(x)
			y_array = np.array(y)
			err_x_array = np.array(err_x) 
			mask = ~np.isnan(x_array) & ~np.isnan(y_array)
						
			# Applichiamo la maschera
			x_array = x_array[mask]
			y_array = y_array[mask]
			err_x_array = err_x_array[mask]

			'''
			print(f'\nRUN: {iv_r}\n')
			# Fit: x = a * y + b, con errore su x
			least_squares = LeastSquares(y_array, x_array, err_x_array, lambda y, a, b: a * y + b)
			m = Minuit(least_squares, a=1.0, b=0.5)
			m.migrad()

			# Chi² e ndof
			chi2 = m.fval
			ndof = len(x_array) - m.nfit
			print("== Chi² del fit x = a*y + b ==")
			print(f"Chi² = {chi2:.2f}")
			print(f"Chi² / ndof = {chi2 / ndof:.2f}")

			# Residui normalizzati: (x - x_fit) / err_x
			x_fit = m.values['a'] * y_array + m.values['b']
			residuals_norm = (x_array - x_fit) / err_x_array

			print("\n== Residui normalizzati (x - fit) / err_x ==")
			print(f"Media: {np.mean(residuals_norm):.4f}")
			print(f"Std:   {np.std(residuals_norm):.4f}")
			'''
			
			#(x - y) / media(x, y)
			
			'''
			mean_xy = 0.5 * (x_array + y_array)
			ratio_mean_xy = np.abs((x_array - y_array)) / mean_xy
			mean_ratio_mean_xy = np.mean(ratio_mean_xy)
			std_ratio_mean_xy = np.std(ratio_mean_xy)
			'''
			ratio_mean_xy = np.abs((x_array - y_array)) - 0.50  #- vbr_shift[atg][dm][chip][array]['value']
			mean_ratio_mean_xy = np.mean(ratio_mean_xy)
			std_ratio_mean_xy = np.std(ratio_mean_xy)


			#(x - y) / err_x
			#ratio = (x_array-y_array)/err_x_array
			
			fig, ax = plt.subplots(figsize=(12, 8))
			bins = math.ceil(np.sqrt(len(ratio_mean_xy)))*3
			ax.set_title(f'VB Difference', fontsize=35, pad=15)
			ax.hist(ratio_mean_xy, bins=bins, color='green', alpha=0.7)
			ax.axvline(mean_ratio_mean_xy, color='blue', linestyle='--', linewidth=2, label=f'Mean = {mean_ratio_mean_xy:.3f}')
			ax.set_xlabel('(VB_OV - VB_IV)', fontsize=35, labelpad=25)
			ax.tick_params(axis='both', labelsize=28)
			ax.legend(frameon=True, fontsize= 30)
			ax.text(0.95, 0.95, f'Std = {std_ratio_mean_xy:.3f} V',
        		horizontalalignment='right',
        		verticalalignment='top',
        		transform=ax.transAxes,
        		fontsize=30,
        		bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
			plt.tight_layout()
			plt.savefig(
    			f"{atg_path}/attg_{atg}_diff_hist_{iv_r}.png",
    			dpi=160,
    			bbox_inches="tight",
    			facecolor="white",
    			pad_inches=0.1,
			)
			plt.close()
	#print(vbr_iv)
	print('\nplot HPK done')
	#no tray
	'''
	reprod_LYSO_path = f'{study_path}/corr_LYSO'
	os.makedirs(reprod_LYSO_path, exist_ok=True)
	atg_index = list(vbr_shift.keys())
	atg_index_copy = atg_index.copy()
	for iv_index in iv:
		for i in atg_index:
			for j in atg_index_copy:
				if j > i:
					x = []
					y = []
					err_x = []
					err_y = []
					for dm in list(vbr_hama[i][iv_index].keys()):
						for chip in list(vbr_hama[i][iv_index][dm].keys()):
							for array in list(vbr_hama[i][iv_index][dm][chip].keys()):
								x.append(vbr_hama[i][iv_index][dm][chip][array]['value'])
								err_x.append(vbr_hama[i][iv_index][dm][chip][array]['error'])
								y.append(vbr_hama[j][iv_index][dm][chip][array]['value'])
								err_y.append(vbr_hama[j][iv_index][dm][chip][array]['error'])
					#manca plot di correlazione
					fig, ax = plot_corr(x, y, err_x , err_y, 'LYSO')	
					#ax.set_title(f'VBR attgain_{atg}', fontsize=35, pad=15)
					ax.set_xlabel(f'VBR at attgain_{i} [V]', fontsize=35, labelpad=25)
					ax.set_ylabel(f'VBR at attgain_{j} [V]', fontsize=35, labelpad=25)
					ax.tick_params(axis='both', labelsize=35)
					ax.legend(frameon=True, fontsize=35)		
					plt.savefig(
						f"{reprod_LYSO_path}/LYSO_{iv}_attg_{i}_{j}.png",
						dpi=160,
						bbox_inches="tight",
						facecolor="white",
						pad_inches=0.1,
					)
					plt.close()
					#histogram RMS
					x_array = np.array(x)
					y_array = np.array(y)
					err_x_array = np.array(err_x)
					err_y_array = np.array(err_y) 
					mask = ~np.isnan(x_array) & ~np.isnan(y_array)
								
					# Applichiamo la maschera
					x_array = x_array[mask]
					y_array = y_array[mask]
					err_x_array = err_x_array[mask]
					err_y_array = err_y_array[mask]
							
					# Calcoliamo la differenza
					diff = np.abs(x_array - y_array)
					mean = (x_array + y_array) /2
					ratio_err = diff /  mean
					mean_ratio_err = np.mean(ratio_err)
					std_ratio_err = np.std(ratio_err)

					fig, ax = plt.subplots(figsize=(12, 8))
					bins = math.ceil(np.sqrt(len(diff)))*3
					ax.set_title(f'Relative Difference Histogram', fontsize=35, pad=15)
					ax.hist(ratio_err, bins=bins, color='green', alpha=0.7)
					ax.axvline(mean_ratio_err, color='blue', linestyle='--', linewidth=2, label=f'Mean = {mean_ratio_err:.3f}')
					ax.tick_params(axis='both', labelsize=35)
					ax.legend(frameon=True, fontsize= 35)
					ax.text(0.95, 0.95, f'Std = {std_ratio_err:.3f} V',
						horizontalalignment='right',
						verticalalignment='top',
						transform=ax.transAxes,
						fontsize=30,
						bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
					
					plt.tight_layout()
					plt.savefig(
						f"{reprod_LYSO_path}/LYSO_{iv_index}_attg_{i}_{j}_diff_hist.png",
						dpi=160,
						bbox_inches="tight",
						facecolor="white",
						pad_inches=0.1,
					)
					plt.close()
	print('\nplot LYSO done')
	'''
	
	iv_copy = iv.copy()
	for atg_index in list(vbr_shift.keys()):
		atg_index_path = f'{study_path}/att_gain_{atg_index}'
		reprod_IV_path = f'{atg_index_path}/corr_IV'
		os.makedirs(reprod_IV_path, exist_ok=True)
		for i in iv:
			for j in iv_copy:
				if j > i:
					x = []
					y = []
					err_x = []
					err_y = []
					for dm in list(vbr_hama[atg_index][i].keys()):
						for chip in list(vbr_hama[atg_index][i][dm].keys()):
							for array in list(vbr_hama[atg_index][i][dm][chip].keys()):
								x.append(vbr_hama[atg_index][i][dm][chip][array]['value'])
								err_x.append(vbr_hama[atg_index][i][dm][chip][array]['error'])
								y.append(vbr_hama[atg_index][j][dm][chip][array]['value'])
								err_y.append(vbr_hama[atg_index][j][dm][chip][array]['error'])
					fig, ax = plot_corr(x, y, err_x, err_y, 'IV')	
					ax.set_title(f'VBR attgain_{atg_index}', fontsize=25, pad=15)
					ax.set_xlabel(f'IV_{i} [V]', fontsize=20, labelpad=15)
					ax.set_ylabel(f'IV_{j} [V]', fontsize=20, labelpad=15)
					plt.savefig(
						f"{reprod_IV_path}/attg_{atg_index}_IV_{i}_{j}.png",
						dpi=160,
						bbox_inches="tight",
						facecolor="white",
						pad_inches=0.1,
					)
					plt.close()
	print('\nplot IV done')



def plot_corr(x, y, err_x, err_y, type_plot):
	fig, ax = plt.subplots(1, 1, figsize=(20, 20))
	cost = odr_cost_function_linear(x, y, err_x, err_y)
	if type_plot == 'HPK':
		num_pairs = len(x) // 2
		colors = plt.cm.rainbow(np.linspace(0, 1, num_pairs))
		for i in range(num_pairs):
			x1, x2 = x[2*i], x[2*i + 1]
			y1, y2 = y[2*i], y[2*i + 1]
			color = colors[i]
			ax.scatter([x1, x2], [y1, y2], color=color,  s=100)
			ax.plot([x1, x2], [y1, y2], color=color , linewidth=3, linestyle='--')
			
		sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=0, vmax=num_pairs-1))
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax)

		cbar.set_ticks(np.arange(0, num_pairs, 1))  
		cbar.set_ticklabels([str(i) for i in range(num_pairs)])
		cbar.ax.tick_params(labelsize=40)

		cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(4))
		cbar.ax.tick_params(which='minor', length=4, color='gray')
		color_fit = 'black'
	else:
		ax.errorbar(x, y, xerr=err_x, yerr=err_y, linestyle='none', ecolor='blue', elinewidth=0.5, capsize=0, marker='o', markersize=8, color='blue', alpha=0.7)
		color_fit = 'orange'
	c = cost
	m1 = iminuit.Minuit(c, q=0.0)
	m1.migrad()
	q = [m1.values[k] for k in m1.pos2var]
	c = cost
	m1 = iminuit.Minuit(c, q)
	m1.migrad()
	q = m1.values[m1.pos2var]
	err_q = m1.errors[m1.pos2var]
	ax.plot(x, linear_fit(x, q), color=color_fit, linewidth=3, label=f"y = 1.00 x + q\nq: {q[0]:.2f} ± {err_q[0]:.2f}")
	ax.tick_params(axis='both', which='major', labelsize=20) 
	ax.grid(True)
	ax.legend(fontsize=15)
	return fig, ax



if __name__ == "__main__":
	if len(sys.argv) < 3:
		if len(sys.argv) < 2:
			print("Error: first run number is missing")
		if len(sys.argv) < 3:
			print("Error: last rub number is missing")
		sys.exit(1)
	
	first_run = int(argv[1])
	last_run = int(argv[2])
	
	config = load_general_config()
	directory = "tofhir"
	path_runs = f"/nas-cms/DATA/runs/{directory}/"
	parent_plots_path = load_general_config()[f"{directory}_plots_folder"]
	plots_path = f"{parent_plots_path}/ov_studies"
	os.makedirs(plots_path, exist_ok=True)
	study_path = f'{plots_path}/new_runs_{first_run}_{last_run}_5'
	os.makedirs(study_path, exist_ok=True)
	#no tray
	'''
	runs_red = read_runs()
	new_runs_red = error_correction(runs_red)
	'''
	#tray
	#'''
	new_runs_red = read_runs()
	#'''
	runs_red_fit = copy.deepcopy(new_runs_red)
	fit_iminuit(runs_red_fit, study_path)
	vb = plot_runs(new_runs_red, runs_red_fit)
	correlation(vb)


	
	
	
	
	
	
	
	
	
	




