import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_raw():
    """
    This function read the raw EEGLAB/edf etc. raw signals using mne library and save as a dataframe for the preprocessing

    Parameters: raw_sig, input_dir
    Returns: sig_df
    """
    pass

def save_to_fif():
    """
    It saves the preprocessed signal into .fif(Functional Imaging File) file for fast access.

    Parameters: cln_sig, output_dir
    Returns: saved_files
    """
    pass

def filters():
    """
    This function apply the bandpass and notch filters from mne library

    Parameters: signal, notch_freq, bp_high_freq, bp_low_freq
    Returns: filtered_sig
    """
    pass

def visuals():
    pass

def sig_info():
    pass 

def logs():
    pass
