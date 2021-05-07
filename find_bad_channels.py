#!/usr/local/bin/python3

import mne
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import shutil
from mne_bids import BIDSPath, write_raw_bids


def find_bad_channels(raw, cross_talk_file, calibration_file, head_pos_file, param_h_freq, param_origin,  
                      param_return_scores, param_limit, param_duration, param_min_count,
                      param_int_order, param_ext_order, param_coord_frame,
                      param_regularize, param_ignore_ref, param_bad_condition, 
                      param_skip_by_annotation, param_mag_scale, param_extended_proj):
    """Detect bad channels and save file with bad channels marked as bad in info.

    Parameters
    ----------
    raw: instance of mne.io.Raw
        Data to be filtered.
    cross_talk_file: str or None
        Path to the FIF file with cross-talk correction information.
    calibration_file: str or None
        Path to the '.dat' file with fine calibration coefficients. This file is machine/site-specific.
    head_pos_file: array or None
        If array, movement compensation will be performed.
    param_h_freq: float or None
        The cutoff frequency (in Hz) of the low-pass filter that will be applied before processing the data. 
        This defaults to 40., which should provide similar results to MaxFilter. 
    param_origin: str or array_like, shape (3,)
        Origin of internal and external multipolar moment space in meters. The default is 'auto', which means
        (0., 0., 0.) when coord_frame='meg', and a head-digitization-based origin fit using fit_sphere_to_headshape()
        when coord_frame='head'.
    param_return_scores: bool
        If True, return a dictionary with scoring information for each evaluated segment of the data. Default is True.
    param_limit: float
        Detection limit for noisy segments. Smaller values will find more bad channels at increased
        risk of including good ones. Default is 7.
    param_duration: float
        Duration of the segments into which to slice the data for processing, in seconds. Default is 5.
    param_min_count: int
        Minimum number of times a channel must show up as bad in a chunk. Default is 5.
    param_int_order: int
        Order of internal component of spherical expansion. Default is 8.
    param_ext_order: int
        Order of external component of spherical expansion. Default is 3.
    param_coord_frame: str
        The coordinate frame that the origin is specified in, either 'meg' or 'head' (default).
    param_regularize: str or None
        Basis regularization type, must be “in” (default) or None.
    param_ignore_ref: bool
        If True, do not include reference channels in compensation. Default is False.
    param_bad_condition: str
        How to deal with ill-conditioned SSS matrices. Can be “error” (default), “warning”, “info”, or “ignore”.
    param_skip_by_annotation: str or list of str
        If a string (or list of str), any annotation segment that begins with the given string will not be included in
        filtering, and segments on either side of the given excluded annotated segment will be filtered separately.
        Default is ['edge', 'bad_acq_skip']. 
    param_mag_scale: float or str
        The magenetometer scale-factor used to bring the magnetometers to approximately the same order of magnitude as
        the gradiometers (default 100.), as they have different units (T vs T/m). Can be "auto."
    param_extended_proj: list
        The empty-room projection vectors used to extend the external SSS basis (i.e., use eSSS). Default is an empty list.

    Returns
    -------
    raw: instance of mne.io.Raw
        The raw data with bad channels marked as "bad" in info.
    noisy_chs: list
        List of bad MEG channels that were automatically detected as being noisy among the good MEG channels.
    flat_chs: list
        List of MEG channels that were detected as being flat in at least min_count segments.
    scores: dict
        A dictionary with information produced by the scoring algorithms.
    """

    # Check if Maxwell Filter was already applied on the data
    if raw.info['proc_history']:
        sss_info = raw.info['proc_history'][0]['max_info']['sss_info']
        tsss_info = raw.info['proc_history'][0]['max_info']['max_st']
        if bool(sss_info) or bool(tsss_info) is True:
            value_error_message = f'You cannot use Maxwell filtering to detect bad channels if data have been already ' \
                                  f'processed with Maxwell filtering.'
            # Raise exception
            raise ValueError(value_error_message)

    # Find bad channels
    raw_check = raw.copy()
    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(raw_check,
                                                                                             cross_talk=cross_talk_file,
                                                                                             calibration=calibration_file,
                                                                                             head_pos=head_pos_file,
                                                                                             h_freq=param_h_freq,
                                                                                             origin=param_origin,
                                                                                             return_scores=param_return_scores,                                                                                             
                                                                                             limit=param_limit,
                                                                                             duration=param_duration,
                                                                                             min_count=param_min_count,
                                                                                             int_order=param_int_order,
                                                                                             ext_order=param_ext_order,
                                                                                             coord_frame=param_coord_frame,
                                                                                             regularize=param_regularize,
                                                                                             ignore_ref=param_ignore_ref,
                                                                                             bad_condition=param_bad_condition,
                                                                                             skip_by_annotation=param_skip_by_annotation,
                                                                                             mag_scale=param_mag_scale,
                                                                                             extended_proj=param_extended_proj)
    del raw_check

    # Add bad channels in raw.info
    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads

    return raw, auto_noisy_chs, auto_flat_chs, auto_scores


def _generate_report(raw_before_preprocessing, raw_after_preprocessing, auto_scores, auto_noisy_chs, auto_flat_chs,
                     data_file_before, report_cross_talk_file, report_calibration_file, report_head_pos_file,
                     param_h_freq, param_origin,  
                     param_return_scores, param_limit, param_duration, param_min_count,
                     param_int_order, param_ext_order, param_coord_frame,
                     param_regularize, param_ignore_ref, param_bad_condition, 
                     param_skip_by_annotation, param_mag_scale, param_extended_proj):
    # Generate a report

    # Create instance of mne.Report # 
    report = mne.Report(title='Results identification of bad channels', verbose=True)

    ## Give some info about the file before preprocessing ## 
    bad_channels = raw_before_preprocessing.info['bads']
    sampling_frequency = raw_before_preprocessing.info['sfreq']
    highpass = raw_before_preprocessing.info['highpass']
    lowpass = raw_before_preprocessing.info['lowpass']

    # Put this info in html format # 
    # Info on data
    html_text_info = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Input file: {data_file_before}</td>
            </tr>
            <tr>
                <td>Bad channels before automated detection: {bad_channels}</td>
            </tr>
            <tr>
                <td>Sampling frequency: {sampling_frequency}Hz</td>
            </tr>
            <tr>
                <td>Highpass: {highpass}Hz</td>
            </tr>
            <tr>
                <td>Lowpass: {lowpass}Hz</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_info, captions='MEG recording features', section='Data info', replace=False)

   
    ## Plot diagnostic figures ##

    # Scores for automated noisy channels detection #
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
                               comments=f'Noisy channels detected (grad and mag): {auto_noisy_chs}',
                               section='Diagnostic figures')

    # Scores for automated flat channels detection #
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
                               comments=f'Flat channels detected (grad and mag): {auto_flat_chs}',
                               section='Diagnostic figures')

    
    ## Plot PSD ##

    # Select only meg signals #
    raw_before_preprocessing.pick_types(meg=True)
    raw_after_preprocessing.pick_types(meg=True)

    ## Plot PSD before and after flat channels detection ##

    # Select good channels 
    channels = raw_after_preprocessing.info['ch_names']

    # Define list of good and flat channels
    good_channels_and_flats = auto_flat_chs + channels
    raw_flat_channels_before_preprocessing = raw_before_preprocessing.copy()
    raw_flat_channels_before_preprocessing = raw_flat_channels_before_preprocessing.pick(picks=good_channels_and_flats)

    # Plot PSD of gradiometers #

    # Select only gradiometers for data before preprocessing
    raw_select_grad_before_preprocessing = raw_flat_channels_before_preprocessing.copy()
    raw_grad_before_preprocessing = raw_select_grad_before_preprocessing.pick(picks='grad')
    grad_channels = raw_grad_before_preprocessing.info['ch_names'] 

    # Select only gradiometers for data after preprocessing
    raw_select_grad_after_preprocessing = raw_after_preprocessing.copy()
    raw_grad_after_preprocessing = raw_select_grad_after_preprocessing.pick(picks='grad')

    # Plot PSD for grad + flat grad
    fig_raw_psd_all_before_grad = mne.viz.plot_raw_psd(raw_grad_before_preprocessing, picks=grad_channels)

    # Add figures to report
    captions_fig_raw_psd_all_before_grad = f'Power spectral density of MEG signals including the automated ' \
                                           f'detected flat channels (Gradiometers)'
    report.add_figs_to_section(figs=fig_raw_psd_all_before_grad,
                               captions=captions_fig_raw_psd_all_before_grad,
                               comments='Noisy channels are not included',
                               section='Power Spectral Density for Gradiometers')


    ## Plot PSD before and after noisy channels detection ##

    # Define list of good and flat channels
    good_channels_and_noisy = auto_noisy_chs + channels
    raw_noisy_channels_before_preprocessing = raw_before_preprocessing.copy()
    raw_noisy_channels_before_preprocessing = raw_noisy_channels_before_preprocessing.pick(picks=good_channels_and_noisy)

    # Plot PSD of gradiometers #

    # Select only gradiometers for data before preprocessing
    raw_select_grad_before_preprocessing = raw_noisy_channels_before_preprocessing.copy()
    raw_grad_before_preprocessing = raw_select_grad_before_preprocessing.pick(picks='grad')
    grad_channels = raw_grad_before_preprocessing.info['ch_names'] 

    # Select only gradiometers for data after preprocessing
    raw_grad_after_preprocessing = raw_select_grad_after_preprocessing.pick(picks='grad')

    # Plot PSD for grad + flat grad
    fig_raw_psd_all_before_grad = mne.viz.plot_raw_psd(raw_grad_before_preprocessing, picks=grad_channels)

    # Add figures to report
    captions_fig_raw_psd_all_before_grad = f'Power spectral density of MEG signals including the automated ' \
                                           f'detected noisy channels (Gradiometers)'
    report.add_figs_to_section(figs=fig_raw_psd_all_before_grad,
                               captions=captions_fig_raw_psd_all_before_grad,
                               comments='Flat channels are not included',
                               section='Power Spectral Density for Gradiometers')

    # Plot PSD of grad excluding flat grads # 
    fig_raw_psd_all_after_grad = mne.viz.plot_raw_psd(raw_grad_after_preprocessing, picks='meg')
    captions_fig_raw_psd_all_after_grad = f'Power spectral density of MEG signals without the automated ' \
                                          f'detected noisy and flat channels (Gradiometers)'
    report.add_figs_to_section(figs=fig_raw_psd_all_after_grad,
                               captions=captions_fig_raw_psd_all_after_grad,
                               section='Power Spectral Density for Gradiometers')

    ## Plot PSD before and after flat channels detection ##

    # Plot PSD of magnetometers #

    # Select only magnetometers for data before preprocessing
    raw_select_mag_before_preprocessing = raw_flat_channels_before_preprocessing.copy()
    raw_mag_before_preprocessing = raw_select_mag_before_preprocessing.pick(picks='mag')
    mag_channels = raw_mag_before_preprocessing.info['ch_names'] 

    # Select only gradiometers for data after preprocessing
    raw_select_mag_after_preprocessing = raw_after_preprocessing.copy()
    raw_mag_after_preprocessing = raw_select_mag_after_preprocessing.pick(picks='mag')

    # Plot PSD for mag + flat grad
    fig_raw_psd_all_before_mag = mne.viz.plot_raw_psd(raw_mag_before_preprocessing, picks=mag_channels)

    # Add figures to report
    captions_fig_raw_psd_all_before_mag = f'Power spectral density of MEG signals including the automated ' \
                                          f'detected flat channels (Magnetometers)'
    report.add_figs_to_section(figs=fig_raw_psd_all_before_mag,
                               captions=captions_fig_raw_psd_all_before_mag,
                               comments='Noisy channels are not included',
                               section='Power Spectral Density for Magnetometers')


    ## Plot PSD before and after noisy channels detection ##

    # Plot PSD of magnetometers #

    # Select only magnetometers for data before preprocessing
    raw_select_mag_before_preprocessing = raw_noisy_channels_before_preprocessing.copy()
    raw_mag_before_preprocessing = raw_select_mag_before_preprocessing.pick(picks='mag')
    mag_channels = raw_mag_before_preprocessing.info['ch_names'] 

    # Select only magnetometers for data after preprocessing
    raw_mag_after_preprocessing = raw_select_mag_after_preprocessing.pick(picks='mag')

    # Plot PSD for mag + noisy mag
    fig_raw_psd_all_before_mag = mne.viz.plot_raw_psd(raw_mag_before_preprocessing, picks=mag_channels)

    # Add figures to report
    captions_fig_raw_psd_all_before_grad = f'Power spectral density of MEG signals including the automated ' \
                                           f'detected noisy channels (Magnetometers)'

    report.add_figs_to_section(figs=fig_raw_psd_all_before_mag,
                               captions=captions_fig_raw_psd_all_before_mag,
                               comments='Flat channels are not included',
                               section='Power Spectral Density for Magnetometers')


    # Plot PSD of mag excluding noisy mag # 
    fig_raw_psd_all_after_mag = mne.viz.plot_raw_psd(raw_mag_after_preprocessing, picks='meg')
    captions_fig_raw_psd_all_after_mag = f'Power spectral density of MEGsignals without the automated ' \
                                          f'detected noisy and flat channels (Magnetometers)'

    report.add_figs_to_section(figs=fig_raw_psd_all_after_mag,
                               captions=captions_fig_raw_psd_all_after_mag,
                               section='Power Spectral Density for Magnetometers')

    # Delete useless copies
    del raw_select_grad_before_preprocessing
    del raw_select_grad_after_preprocessing
    del raw_grad_before_preprocessing 
    del raw_grad_after_preprocessing 

    del raw_select_mag_before_preprocessing
    del raw_select_mag_after_preprocessing
    del raw_mag_before_preprocessing 
    del raw_mag_after_preprocessing 


    ## If they exist, plot bad channels in time domain ##
    # Noisy channels
    if auto_noisy_chs:

        # Select random grad channels to plot including the noisy ones
        ch_to_plot = random.sample(ch_names.tolist(), 49)
        ch_to_plot += auto_noisy_chs
        raw_ch_to_plot = raw_after_preprocessing.copy()

        # Plot channels in time domain
        raw_ch_to_plot.pick_channels(ch_to_plot)
        fig_raw_noisy_channels = raw_ch_to_plot.plot(duration=20, n_channels=50, scalings='auto', butterfly=False,
                                                     show_scrollbars=False)
        del raw_ch_to_plot

        report.add_figs_to_section(fig_raw_noisy_channels, captions=f'MEG signals including automated '
                                                                    f'detected noisy channels',
                                   comments='The noisy channels are in gray.', section='Time domain')

    # Flat channels
    if auto_flat_chs:
        # Select random grad channels to plot including the flat ones
        ch_to_plot = random.sample(ch_names.tolist(), 49)
        ch_to_plot += auto_flat_chs
        raw_ch_to_plot = raw_after_preprocessing.copy()

        # Plot channels in time domain
        raw_ch_to_plot.pick_channels(ch_to_plot)
        fig_raw_flat_channels = raw_ch_to_plot.plot(duration=20, n_channels=50, scalings='auto', butterfly=False,
                                                    show_scrollbars=False)
        del raw_ch_to_plot

        report.add_figs_to_section(fig_raw_flat_channels, captions=f'MEG signals including automated '
                                                                   f'detected flat channels',
                                   comments='The flat channels are in gray.', section='Time domain')


    ## Values of the parameters of the App ## 
    mne_version = mne.__version__

    # Put this info in html format # 
    html_text_parameters = f"""<html>

    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Cross-talk file: {report_cross_talk_file}</td>
            </tr>
            <tr>
                <td>Calibration file: {report_calibration_file}</td>
            </tr>
            <tr>
                <td>Headshape file: {report_head_pos_file}</td>
            </tr>
            <tr>
                <td>Origin: {param_origin}</td>
            </tr>
            <tr>
                <td>Limit: {param_limit} noisy segments</td>
            </tr>
            <tr>
                <td>Duration: {param_duration}s</td>
            </tr>
            <tr>
                <td>Min count: {param_min_count} times</td>
            </tr>
            <tr>
                <td>Order of internal component of sherical expansion: {param_int_order}</td>
            </tr>
            <tr>
                <td>Order of external component of sherical expansion: {param_ext_order}</td>
            </tr>
            <tr>
                <td>Coordinate frame: {param_coord_frame}</td>
            </tr>
            <tr>
                <td>Regularize: {param_regularize}</td>
            </tr>
            <tr>
                <td>Ignore reference channel: {param_ignore_ref}</td>
            </tr>
            <tr>
                <td>Bad condition: {param_bad_condition}</td>
            </tr>
            <tr>
                <td>Magnetomer scale-factor: {param_mag_scale}</td>
            </tr>
            <tr>
                <td>Skip by annotation: {param_skip_by_annotation}</td>
            </tr>
            <tr>
                <td>Cutoff frequency of the low-pass filter: {param_h_freq}Hz</td>
            </tr>
            <tr>
                <td>Empty-room projection vectors: {param_extended_proj}</td>
            </tr>
            <tr>
                <td>MNE version used: {mne_version}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_parameters, captions='Values of the parameters of the App', 
                                section='Parameters of the App', replace=False)


    # Save report
    report.save('out_dir_report/report_bad_channels.html', overwrite=True)


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the raw file
    data_file = (config.pop('fif'))
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

    
    ## Read the optional files ##

    # Read the crosstalk file
    cross_talk_file = config.pop('crosstalk')
    if os.path.exists(cross_talk_file) is False:
        cross_talk_file = None
        report_cross_talk_file = 'No cross-talk file provided'
    else:
        shutil.copy2(cross_talk_file, 'out_dir_bad_channels/crosstalk_meg.fif')  # required to run a pipeline on BL
        report_cross_talk_file = 'Cross-talk file provided'

    # Read the calibration file
    calibration_file = config.pop('calibration')
    if os.path.exists(calibration_file) is False:
        calibration_file = None
        report_calibration_file = 'No calibration file provided'
    else:
        shutil.copy2(calibration_file, 'out_dir_bad_channels/calibration_meg.dat')  # required to run a pipeline on BL
        report_calibration_file = 'Calibration file provided'

    # Read the destination file
    destination_file = config.pop('destination')
    if os.path.exists(destination_file) is True:
        shutil.copy2(destination_file, 'out_dir_bad_channels/destination.fif')  # required to run a pipeline on BL

    # Read head pos file
    head_pos = config.pop('headshape')
    if os.path.exists(head_pos) is True:
        head_pos_file = mne.chpi.read_head_pos(head_pos)
        shutil.copy2(head_pos, 'out_dir_bad_channels/headshape.pos')  # required to run a pipeline on BL
        report_head_pos_file = 'Headshape file provided'
    else:
        head_pos_file = None
        report_head_pos_file = 'No headshape file provided'

    # Read events file 
    events_file = config.pop('events')
    if os.path.exists(events_file) is True:
        shutil.copy2(events_file, 'out_dir_bad_channels/events.tsv')  # required to run a pipeline on BL


    # Convert all "" into None when the App runs on BL
    tmp = dict((k, None) for k, v in config.items() if v == "")
    config.update(tmp)

    # Check if param_extended_proj parameter is an empty list string
    if config['param_extended_proj'] == '[]':
        config['param_extended_proj'] = [] # required to run a pipeline on BL

    # Display a warning if h_freq is None
    if config['param_h_freq'] is None:
        user_warning_message = f'No low-pass filter will be applied to the data. ' \
                               f'Make sure line noise and cHPI artifacts were removed before finding ' \
                               f'bad channels.'
        warnings.warn(user_warning_message)
        dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message})

    # Check if config['param_return_scores'] is True   
    if config['param_return_scores'] is not True:
        value_error_message = f'param_return_scores must be True.'
        raise ValueError(value_error_message) 

    
    ## Convert parameters ##   

    # Deal with param_origin parameter #
    # Convert origin parameter into array when the app is run locally
    if isinstance(config['param_origin'], list):
       config['param_origin'] = np.array(config['param_origin'])

    # Convert origin parameter into array when the app is run on BL
    if isinstance(config['param_origin'], str) and config['param_origin'] != "auto":
       param_origin = list(map(float, config['param_origin'].split(', ')))
       config['param_origin'] = np.array(param_origin)

    # Raise an error if param origin is not an array of shape 3
    if config['param_origin'] != "auto" and config['param_origin'].shape[0] != 3:
        value_error_message = f"Origin parameter must contain three elements."
        raise ValueError(value_error_message)

    # Deal with param_mag_scale parameter #
    # Convert param_mag_scale into a float when not "auto" when the app runs on BL
    if isinstance(config['param_mag_scale'], str) and config['param_mag_scale'] != "auto":
        config['param_mag_scale'] = float(config['param_mag_scale'])

    # Deal with skip_by_annotation parameter #
    # Convert param_mag_scale into a list of strings when the app runs on BL
    skip_by_an = config['param_skip_by_annotation']
    if skip_by_an == "[]":
        skip_by_an = []
    elif isinstance(skip_by_an, str) and skip_by_an.find("[") != -1 and skip_by_an != "[]": 
        skip_by_an = skip_by_an.replace('[', '')
        skip_by_an = skip_by_an.replace(']', '')
        skip_by_an = list(map(str, skip_by_an.split(', ')))         
    config['param_skip_by_annotation'] = skip_by_an 

    
    ## Define kwargs ##

    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  


    # Apply find bad channels     
    # raw_copy = raw.copy()
    # raw_bad_channels, auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels(raw_copy, cross_talk_file,
    #                                                                                  calibration_file,
    #                                                                                  head_pos_file, 
    #                                                                                  **kwargs)
    # del raw_copy


    ## Create channels.tsv ##

    # Create a BIDSPath
    bids_path = BIDSPath(subject='subject',
                         session=None,
                         task='task',
                         run='01',
                         acquisition=None,
                         processing=None,
                         recording=None,
                         space=None,
                         suffix=None,
                         datatype='meg',
                         root='bids')

    # Write BIDS to create channels.tsv BIDS compliant
    write_raw_bids(raw, bids_path, overwrite=True)

    # Extract channels.tsv from bids path
    channels_tsv = 'bids/sub-test/meg/sub-test_task-test_run-01_channels.tsv'

    # Read it as a dataframe
    df_channels = pd.read_csv(channels_tsv, sep='\t')

    # Update df_channels with bad channels
    # bads = raw_bad_channels.info['bads']
    bads = ['MEG0113', 'MEG0112']
    
    # for bad in bads:
    #     index_bad_channel = df_channels[df_channels['name'] == bad].index
    #     df_channels.loc[index_bad_channel, 'status'] = 'bad'

    # index_bad_channels = [df_channels[df_channels['name'] == bad].index for bads in bads]
    # [df_channels.loc[index, 'status'] = 'bad' for index_bad_channels]
    # print(df_channels)


    # for bad in bads:
    #     if df_channels[df_channels['name'] == 'MEG0113']:
    #         print('test')
    #         # df_channels[df_channels['name'] == bad]['status'] = 'bad'


    shutil.copy2(channels_tsv, 'out_dir_bad_channels/channels.tsv')




    # Write a success message in product.json
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'Bad channels were successfully detected.'})

    # Write an info message in product.json
    dict_json_product['brainlife'].append({'type': 'info', 'msg': f'This algorithm is not fully reliable. '
                                                                  f"Don't hesitate to check all of the "
                                                                  f"signals visually "
                                                                  f"before performing an another preprocessing step."})

    # Generate report
    # _generate_report(raw, raw_bad_channels, auto_scores, auto_noisy_chs, auto_flat_chs, data_file, 
    #                  report_cross_talk_file, report_calibration_file, report_head_pos_file, **kwargs)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
