#!/usr/local/bin/python3

import mne
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

# Generate a json.product to display messages on Brainlife UI
dict_json_product = {'brainlife': []}

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# Read the raw file
data_file = (config.pop('fif'))
raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

# Read the calibration files
if 'cross_talk_correction' in config.keys():
    cross_talk_file = config.pop('cross_talk_correction')
else:
    cross_talk_file = None

if 'calibration' in config.keys():
    calibration_file = config.pop('calibration')
else:
    calibration_file = None

# Head pos file
if 'head_position' in config.keys():
    head_pos_file = config.pop('head_position')
    if head_pos_file is not None:  # when App is run locally and "head_position": null in config.json
        head_pos_file = mne.chpi.read_head_pos(head_pos_file)
else:
    head_pos_file = None

# Warning if h_freq is None
h_freq = config['param_h_freq']
if h_freq is None:
    UserWarning_message = f'No low-pass filter will be applied to the data. ' \
                          'Make sure line noise and cHPI artifacts were removed before finding ' \
                          'bad channels.'
    warnings.warn(UserWarning_message)
    dict_json_product['brainlife'].append({'type': 'warning', 'msg': UserWarning_message})

# Find bad channels
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
    raw_check, cross_talk=cross_talk_file, calibration=calibration_file, head_pos=head_pos_file,
    return_scores=True, h_freq=h_freq,
    limit=config['param_limit'],
    duration=config['param_duration'],
    min_count=config['param_min_count'],
    int_order=config['param_int_order'],
    ext_order=config['param_ext_order'],
    coord_frame=config['param_coord_frame'],
    regularize=config['param_regularize'],
    ignore_ref=config['param_ignore_ref'],
    bad_condition=config['param_bad_condition'],
    skip_by_annotation=config['param_skip_by_annotation'],
    mag_scale=config['param_mag_scale']
)

del raw_check

# Add bad channels in raw.info
bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads

# Save file
raw.save("out_dir_bad_channels/bad_channels-raw.fif", overwrite=True)

# Generate a report

# Create report
report = mne.Report(title='Results identification of bad channels', verbose=True)

# Plot relevant figures

# Scores for automated noisy channels detection
# Only select the data for gradiometer channels
ch_type = 'grad'
ch_subset = auto_scores['ch_types'] == ch_type
ch_names = auto_scores['ch_names'][ch_subset]
scores = auto_scores['scores_noisy'][ch_subset]
limits = auto_scores['limits_noisy'][ch_subset]
bins = auto_scores['bins']  # The windows that were evaluated
# We will label each segment by its start and stop time, with up to 3
# digits before and 3 digits after the decimal place (1 ms precision).
bin_labels = [f'{start:3.3f} – {stop:3.3f}'
              for start, stop in bins]

# Store the data in a Pandas DataFrame. The seaborn heatmap function
# we will call below will then be able to automatically assign the correct
# labels to all axes.
data_to_plot = pd.DataFrame(data=scores,
                            columns=pd.Index(bin_labels, name='Time (s)'),
                            index=pd.Index(ch_names, name='Channel'))

# Plot the "raw" scores
fig_noisy, ax = plt.subplots(1, 2, figsize=(18, 16))
sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
            ax=ax[0])
[ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
    for x in range(1, len(bins))]
ax[0].set_title('All Scores', fontweight='bold')

# Adjust the color range to highlight segments that exceeded the limit
sns.heatmap(data=data_to_plot,
            vmin=np.nanmin(limits),  # bads in input data have NaN limits
            cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
[ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
    for x in range(1, len(bins))]
ax[1].set_title('Scores > Limit', fontweight='bold')

# Add figures to report
report.add_figs_to_section(fig_noisy, captions=f'Automated noisy channel detection: {ch_type}',
                           comments=f'Noisy channels detected: {auto_noisy_chs}')

# Scores for automated flat channels detection
# Only select the data for gradiometer channels
scores = auto_scores['scores_flat'][ch_subset]
limits = auto_scores['limits_flat'][ch_subset]

# Store the data in a Pandas DataFrame
data_to_plot = pd.DataFrame(data=scores,
                            columns=pd.Index(bin_labels, name='Time (s)'),
                            index=pd.Index(ch_names, name='Channel'))

# Plot the "raw" scores
fig_flat, ax = plt.subplots(1, 2, figsize=(18, 16))
sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
            ax=ax[0])
[ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
    for x in range(1, len(bins))]
ax[0].set_title('All Scores', fontweight='bold')

# Adjust the color range to highlight segments that are below the limit
sns.heatmap(data=data_to_plot,
            vmax=np.nanmax(limits),  # bads in input data have NaN limits
            cmap='Reds_r', cbar_kws=dict(label='Score'), ax=ax[1])
[ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
    for x in range(1, len(bins))]
ax[1].set_title('Scores < Limit', fontweight='bold')

# Add figures to report
report.add_figs_to_section(fig_flat, captions=f'Automated flat channel detection: {ch_type}',
                           comments=f'Flat channels detected: {auto_flat_chs}')

# If they exist, plot bad channels in time and frequency domains
# Noisy channels
if auto_noisy_chs:

    # Select random grad channels to plot including the noisy ones
    ch_to_plot = random.sample(ch_names.tolist(), 49)
    ch_to_plot += auto_noisy_chs
    raw_ch_to_plot = raw.copy()

    # Plot channels in time domain
    raw_ch_to_plot.pick_channels(ch_to_plot)
    fig_raw_noisy_channels = raw_ch_to_plot.plot(duration=20, n_channels=50, butterfly=False,
                                                 show_scrollbars=False)
    del raw_ch_to_plot

    # Plot psd of all grad channels including the noisy one
    fig_raw_psd_all = mne.viz.plot_raw_psd(raw, picks='grad')

    # Plot psd of all grad channels without the noisy one
    # Select all grad channels and exclude the noisy ones
    raw_clean = raw.copy()
    raw_clean.pick_types(meg='grad', exclude='bads')
    fig_raw_psd_clean = mne.viz.plot_raw_psd(raw_clean, picks='grad')
    del raw_clean

    # Add figures to report
    report.add_figs_to_section(fig_raw_noisy_channels, captions=f'Grad MEG signals including automated '
                                                                f'detected noisy channels',
                               comments='The noisy channels are in gray.')
    captions_fig_raw_psd_all = f'Power spectral density of grad MEG signals including the automated ' \
                               f'detected noisy channels'
    captions_fig_raw_psd_clean = f'Power spectral density of grad MEG signals without the automated ' \
                                 f'detected noisy channels'
    report.add_figs_to_section(figs=[fig_raw_psd_all, fig_raw_psd_clean],
                               captions=[captions_fig_raw_psd_all, captions_fig_raw_psd_clean])

# Flat channels
if auto_flat_chs:

    # Select random grad channels to plot including the flat ones
    ch_to_plot = random.sample(ch_names.tolist(), 49)
    ch_to_plot += auto_flat_chs
    raw_ch_to_plot = raw.copy()

    # Plot channels in time domain
    raw_ch_to_plot.pick_channels(ch_to_plot)
    fig_raw_flat_channels = raw_ch_to_plot.plot(duration=20, n_channels=50, butterfly=False,
                                                show_scrollbars=False)
    del raw_ch_to_plot

    # Plot psd of all grad channels including the flat ones
    fig_raw_psd_all = mne.viz.plot_raw_psd(raw, picks='grad')

    # Plot psd of all grad channels without the flat ones
    # Select all grad channels excluding the noisy ones
    raw_clean = raw.copy()
    raw_clean.pick_types(meg='grad', exclude='bads')
    fig_raw_psd_clean = mne.viz.plot_raw_psd(raw_clean, picks='grad')
    del raw_clean

    # Add figures to report
    report.add_figs_to_section(fig_raw_flat_channels, captions=f'Grad MEG signals including automated '
                                                               f'detected noisy channels',
                               comments='The noisy channels are in gray.')
    captions_fig_raw_psd_all = f'Power spectral density of grad MEG signals including the automated ' \
                               f'detected noisy channels'
    captions_fig_raw_psd_clean = f'Power spectral density of grad MEG signals without the automated ' \
                                 f'detected noisy channels'
    report.add_figs_to_section(figs=[fig_raw_psd_all, fig_raw_psd_clean],
                               captions=[captions_fig_raw_psd_all, captions_fig_raw_psd_clean])


# Save report
report.save('out_dir_report/report_bad_channels.html', overwrite=True)

# Success message in product.json
dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Bad channels were successfully detected.'})

# Save the dict_json_product in a json file
with open('product.json', 'w') as outfile:
    json.dump(dict_json_product, outfile)
