# Poetry Analysis

## Overview

**Poetry Analysis** is a research project aimed at studying the **neuro-aesthetic mechanisms underlying poetry perception** using EEG data.

The goal of this project is to analyze how the human brain differentiates **poetic texts** from **structurally similar non-poetic control texts**. Using machine learning and signal processing techniques, the project aims to classify brain responses elicited during poetry reading and explore neural signatures associated with aesthetic and creative judgments.

This work contributes to the emerging field of **neuroaesthetics and cognitive neuroscience**.

---

## Research Objective

The main objective of this project is to:

* Analyze EEG responses during poetry reading
* Investigate neural differences between **poetry and control text**
* Explore brain dynamics associated with **aesthetic perception and creativity**
* Build **machine learning models** to classify neural responses

---

## Dataset

This project uses the **Poetry Assessment EEG Dataset**.

The dataset contains **64-channel EEG recordings** and behavioral responses collected while participants read and evaluated short poems.

### Dataset characteristics

* **Participants:** 51
* **EEG channels:** 64
* **Sampling rate:** 512 Hz
* **Stimuli:** 210 short texts

  * 70 Haiku
  * 70 Senryu
  * 70 Control texts (non-poetic)

Participants rated each stimulus on five dimensions:

* Aesthetic appeal
* Vivid imagery
* Emotional impact
* Originality
* Creativity

The dataset also includes:

* Resting-state EEG
* Behavioral ratings
* Psychometric questionnaires
* Stimulus metadata

These data enable detailed analysis of the neural mechanisms involved in **poetic perception and creative cognition**.

---

## Dataset Source

The dataset originates from the following publication:

**Chaudhuri, S. & Bhattacharya, J. (2025)**
*An EEG Dataset on Aesthetic and Creative Judgments of Brief Structured Poetry*
Scientific Data (Nature Portfolio)

DOI:
https://doi.org/10.1038/s41597-025-06189-w

OpenNeuro dataset:

Soma Chaudhuri and Joydeep Bhattacharya (2025). Poetry Assessment EEG Dataset 1. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds006648.v1.0.0

Soma Chaudhuri and Joydeep Bhattacharya (2025). Poetry Assessment EEG Dataset 2. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds006647.v1.0.1

Please cite the original dataset if you use this repository.

---

## Experimental Design

Participants performed the following tasks during EEG recording:

1. Pre-experiment questionnaires
2. Resting-state EEG recording
3. Reading and contemplation of poems
4. Subjective evaluation of each poem
5. Post-experiment resting-state EEG

Each stimulus trial consisted of:

* fixation cross
* reading phase
* contemplation phase
* rating phase

This design enables linking **neural responses with subjective aesthetic judgments**.

---

## Project Structure

```
.
├── data
│   ├── features
│   │   └── sub-021_bpfeatures.parquet
│   ├── intrmd_data
│   │   ├── epochs
│   │   ├── filtered
│   │   ├── ica_signal
│   │   ├── labeled
│   │   └── raw_label
│   └── raw_data
│       ├── ds006647-download
│       └── ds006648-download
├── experimental_rubbish
│   ├── preprocessing_eeg.py
│   ├── rawdata_anlys.py
│   ├── v00_preprocess_eeg_fixed.py
│   └── v00_preprocess_eeg.py
├── logs
│   ├── fetrs
│   │   └── bandpower.log
│   └── preprs
│       ├── sub-021_log.log
│       └── sub-026_log.log
├── README.md
├── results
│   └── reports
│       └── raw_eeg_quality_report.csv
├── scripts
│   ├── foo
│   │   ├── add_label.py
│   │   ├── bi_ml_pipeline.py
│   │   ├── descriptive_stats.py
│   │   ├── features_extraction.py
│   │   ├── fooo.py
│   │   ├── preprocess.py
│   │   ├── rawdata_anlys.ipynb
│   │   ├── roi_features.py
│   │   ├── train_cnnlstm.py
│   │   ├── train_eegnet.py
│   │   └── visualise.ipynb
│   └── main
│       ├── __pycache__
│       │   └── run_preprocessing.cpython-314.pyc
│       ├── run_dl.py
│       ├── run_features.py
│       ├── run_full_pipeline.py
│       ├── run_labeling.py
│       ├── run_ml.py
│       └── run_preprocessing.py
├── setup.py
├── src
│   ├── poetryeeg_anlys
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-314.pyc
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   ├── constants.py
│   │   │   ├── paths.py
│   │   │   └── settings.py
│   │   ├── dl
│   │   │   ├── __init__.py
│   │   │   ├── cnnlstm.py
│   │   │   ├── eegnet.py
│   │   │   └── trainer.py
│   │   ├── features
│   │   │   ├── __init__.py
│   │   │   ├── bandpower.py
│   │   │   ├── descriptive.py
│   │   │   ├── feature_utils.py
│   │   │   └── roi_bandpower.py
│   │   ├── labeling
│   │   │   ├── __init__.py
│   │   │   ├── add_labels.py
│   │   │   └── behavior.py
│   │   ├── ml
│   │   │   ├── __init__.py
│   │   │   ├── evaluation.py
│   │   │   ├── feature_selection.py
│   │   │   ├── models.py
│   │   │   └── pipeline.py
│   │   ├── pipelines
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   ├── feature_pipeline.py
│   │   │   ├── full_pipeline.py
│   │   │   ├── ml_pipeline.py
│   │   │   └── preprocessing_pipeline.py
│   │   ├── preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   ├── epoching.py
│   │   │   ├── filtering.py
│   │   │   ├── ica.py
│   │   │   ├── preprocess.py
│   │   │   └── quality.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   ├── helpers.py
│   │   │   ├── io.py
│   │   │   ├── logger.py
│   │   │   └── validation.py
│   │   └── visualization
│   │       ├── __init__.py
│   │       ├── eeg_viz.py
│   │       └── plots.py
│   └── poetryeeg_anlys.egg-info
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
└── tests
    └── test_imports.py
```

### Folder Description

**data/**

* Raw and intermediate EEG datasets.

**scripts/**

* Analysis scripts and notebooks for signal inspection and experimentation.

**src/**

* Core Python package for the project.

**results/**

* Generated reports and analysis outputs.

---

## Current Development Stage

This repository currently focuses on:

* EEG data exploration
* Raw signal evaluation
* Data quality assessment

Future stages will include:

* EEG preprocessing
* Feature extraction
* Spectral analysis
* Machine learning classification
* Brain network analysis

---

## Planned Analysis Pipeline

The project will progressively implement:

1. Raw EEG inspection
2. Signal preprocessing
3. Artifact removal
4. Epoch extraction
5. Feature engineering
6. Machine learning classification
7. Visualization and interpretation

---

## Technologies

The project is implemented in **Python 3.9.6** and relies on scientific computing libraries such as:

* NumPy
* Pandas
* Matplotlib
* MNE
* SciPy
* Scikit-learn

Additional libraries may be added as the project evolves.

---

## Target Users

This repository is intended for:

* Computational neuroscience researchers
* Neuroaesthetics researchers
* Machine learning researchers working with EEG
* Students learning EEG analysis

---

## Citation

If you use this repository or dataset in your research, please cite:

Chaudhuri, S., & Bhattacharya, J. (2025).
An EEG Dataset on Aesthetic and Creative Judgments of Brief Structured Poetry.
Scientific Data.

DOI: https://doi.org/10.1038/s41597-025-06189-w

---

## License

License information will be added later.

---

## Author

This project is being developed as part of research in **computational neuroscience and neuroaesthetics**.
