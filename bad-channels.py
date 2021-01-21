import mne
import json

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# Read the file
data_file = str(config.pop('input_raw'))
raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)
cross_talk_file = str(config.pop('input_cross_talk'))
calibration_file = str(config.pop('input_calibration'))

# Find bad channels
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
    raw_check, cross_talk=cross_talk_file, calibration=calibration_file, **config['params'])

bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads

# Save file
raw.save(raw.filenames[0].replace('-raw.fif', '_%s.fif' % config['output_tag']), overwrite=True)