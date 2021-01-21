# app-bad-channels

This is a draft of a future Brainlife App that detects bad channels in MEG recordings using [`mne.preprocessing.find_bad_channels_maxwell`](https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html#mne.preprocessing.maxwell_filter).

# app-bad-channels-documentation

1) Detect bad channels thanks to SSS without removing external components
2) Input files are:
    * a MEG file in `.fif` format,
    * an optional fine calibration file in `.dat`,
    * an optional crosstalk compensation file in `.fif`
3) Ouput file is a `.fif` MEG file with bad channels marked as "bad" in its `mne.info`

### Authors
- [Aurore Bussalb](aurore.bussalb@icm-institute.org)

### Contributors
- [Aurore Bussalb](aurore.bussalb@icm-institute.org)
- [Maximilien Chaumon](maximilien.chaumon@icm-institute.org)

### Citations
1. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Running the App 

### On Brainlife.io

This App has not yet been registered in Brainlife.io.

### Running Locally (on your machine)

1. git clone this repo
2. Inside the cloned directory, create `config.json` with something like the following content with paths to your input files (see `config.json.example`)

```json
{
  "input": "rest1-raw.fif"
}
```

3. Launch the App by executing `main`

```bash
./main
```

## Output

The output file is a MEG file in `.fif` format with its channels detected as "bad" by this App are marked as "bad" in `mne.info`.
