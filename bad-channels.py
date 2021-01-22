import mne
import json


# Test version
print(mne.__version__)

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

# Find bad channels
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(
    raw_check, cross_talk=cross_talk_file, calibration=calibration_file, **config['params'])

bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
raw.info['bads'] = bads

# Save file
raw.save(raw.filenames[0].replace('-raw.fif', '_%s.fif' % config['output_tag']), overwrite=True)