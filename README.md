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

https://openneuro.org/datasets/ds006648

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
│   ├── intrmd_data
│   └── raw_data
│       └── ds006648-download
│
├── results
│   └── reports
│
├── scripts
│   ├── foo
│   │   ├── 01_raw_eeg_evaluation.ipynb
│   │   ├── rawdata_anlys.ipynb
│   │   └── rawdata_anlys.py
│   └── main
│
├── src
│   └── poetryeeg_anlys
│       ├── __init__.py
│       └── config.py
│
└── setup.py
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
