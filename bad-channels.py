#!/usr/local/bin/python3

import mne
import json
import warnings

# Print mne version
print(mne.__version__)

# Generate a json.product to display messages on Brainlife UI
dict_json_product = {'brainlife': []}

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# Read the raw file
data_file = str(config.pop('input_raw'))
raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

# Read the calibration files
cross_talk_file = config.pop('input_cross_talk')
if cross_talk_file is not None:
    cross_talk_file = str(cross_talk_file)

calibration_file = config.pop('input_calibration')
if calibration_file is not None:
    calibration_file = str(calibration_file)

# Head pos file
head_pos_file = config.pop('head_pos')
if head_pos_file is not None:
    head_pos_file = mne.chpi.read_head_pos(str(head_pos_file))

# Warning if h_freq is None
h_freq = config['params_find_bad_channels_maxwell']['h_freq']
if h_freq is None:
    UserWarning_message = f'No low-pass filter will be applied to the data. ' \
                  'Make sure line noise and cHPI artifacts were removed before finding ' \
                  'bad channels.'
    warnings.warn(UserWarning_message)
    dict_json_product['brainlife'].append({'type': 'warning', 'msg': UserWarning_message})

# Find bad channels
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, scores = mne.preprocessing.find_bad_channels_maxwell(
    raw_check, cross_talk=cross_talk_file, calibration=calibration_file, head_pos=head_pos_file,
    return_scores=True, **config['params_find_bad_channels_maxwell'])

del raw_check

bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads

# Save file
raw.save(raw.filenames[0].replace('-raw.fif', '_%s.fif' % config['output_tag']), **config['params_save'])

# Save the dict_json_product in a json file
with open('product.json', 'w') as outfile:
    json.dump(dict_json_product, outfile)
