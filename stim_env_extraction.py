#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:04:33 2025

Import wav_file master look up sheet
For each block, 
    for each trial
        based on condition
        import wavfile of target and background sound
        do gammatone filter then hilbert transform, add up env across freq bands
        concatenate the lists, separately for target and backgrond
        
Save out 8 blocks of envelopes


@author: zhaotc
"""
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import gammatone, hilbert,filtfilt
import json


def get_env (fs, audio_signal):
    # this function takes the fs, and audio data, first put it through a gammatone filter bank
    # then for each bank, hilbert transform is applied
    # then env from all bands added together
    # then band-pass env and then downsample?
    # Define parameters, based on Viswanathan et al, 2021 JASA
    num_filters = 10
    low_freq = 100
    high_freq = 8500

    # Generate ERB-spaced center frequencies
    # (This would typically involve a function to calculate ERB-scale frequencies)
    center_frequencies = np.logspace(np.log10(low_freq), np.log10(high_freq), num_filters)

    # Create and apply gammatone filters
    filtered_signals = []
    envs = []
    for cf in center_frequencies:
        b, a = gammatone(cf, ftype='fir', fs=float(fs)) # Get filter coefficients
        filtered_signal = filtfilt(b, a, audio_signal) # Apply filter changed from lfilter to filtfilt 10/14/2025 suggested by Yi
        # hibert transform
        env = np.abs(hilbert(filtered_signal))
        filtered_signals.append(filtered_signal)
        envs.append(env)

    envelope = np.mean(np.asarray(envs), axis = 0)
    
    return envelope



#%% add a column of stimulus duration for each trial and then analyze stim duration by conditions
root_path = '/mnt/storage/RASP/stimuli/'
csvFile = pd.read_csv(root_path + 'Wav_Files_by_Block_zc.csv')

#%% Extract the envelope by blocks
for i in np.arange(1,csvFile['Block Number'].max(axis = 0)+1):

    block = csvFile[csvFile['Block Number'] == i]
    block = block.reset_index()
    envs_ta = list(np.arange(len(block)))    
    envs_bg = list(np.arange(len(block)))   
    
    
    for index in block.index:
        
        if block.loc[index, 'Condition'] == 'TA':
            fs_t, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + 'alt_' + block.loc[index,'Target Wav'] + '.wav') 
            fs_b, bg_audio = wavfile.read(root_path + 'ScaledSpeakers/' + block.loc[index,'Background Wav'] + '.wav') 
        elif block.loc[index, 'Condition'] == 'BA':
            fs_t, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + block.loc[index,'Target Wav'] + '.wav') 
            fs_b, bg_audio = wavfile.read(root_path + 'ScaledSpeakers/' + 'alt_' + block.loc[index,'Background Wav'] + '.wav') 
        elif block.loc[index,'Condition'] == 'CT':
            fs_t, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + block.loc[index,'Target Wav'] + '.wav') 
            fs_b, bg_audio = wavfile.read(root_path + 'ScaledSpeakers/' + block.loc[index,'Background Wav'] + '.wav') 
        else: print("something is wrong!")
        
        #tar_env = get_env(fs_t,target_audio)
        #bg_env = get_env(fs_b,bg_audio)
        
        tar_env = np.abs(hilbert(target_audio))  # broadband env 10/14/25 per Yi suggestion
        bg_env = np.abs(hilbert(bg_audio))
        envs_ta[index] = tar_env 
        envs_bg[index] = bg_env    
        
    envs_ta = [arr.tolist() for arr in envs_ta] # change into list in list for json saving
    envs_bg = [arr.tolist() for arr in envs_bg]
    # write the lists into txt files
    file_target = root_path + 'env_bb_target_block' + str(i) + '.txt'
    file_background = root_path + 'env_bb_background_block' + str(i) + '.txt'
    
    with open(file_target, "w") as f:
        json.dump(envs_ta, f)
        
    with open(file_background, "w") as f:
        json.dump(envs_bg, f)

