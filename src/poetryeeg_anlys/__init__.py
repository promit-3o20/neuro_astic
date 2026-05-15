"""
poetryeeg_anlys
===============
Modular EEG analysis package for the Poetry–EEG study (BIDS dataset ds006648).

Sub-packages
------------
config          Centralised paths, settings, and constants.
preprocessing   Raw data loading, filtering, ICA, and epoching.
labeling        Behavioral alignment and label encoding.
features        Band-power and ROI feature extraction.
ml              Classical ML pipelines, models, and evaluation.
dl              Deep-learning models (EEGNet, CNN-LSTM) and trainer.
utils           Logger, I/O helpers, and validation utilities.
pipelines       Orchestrator pipelines (preprocessing → ML → DL).
visualization   EEG and result plotting utilities.
"""

__version__ = "0.1.0"
__author__ = "Pramit Biswas"
