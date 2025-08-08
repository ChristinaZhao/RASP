#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:12:08 2025

RASP EEG analysis

For each participant
    figure out the sequence of blocks run
    create an array of the 300 trials in sequence of blocks played with conditions tagged
    load env in that order
    load preprocessed file
    epoch, get rid of practice trials
    
    for the 300 trials
        calculate a metric/metrics between env and epoch data
        
    aggregate by condition
    plot


@author: zhaotc
"""

import mne
import json
import mne_connectivity
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# group data

participants = {'rasp_02': [4, 3, 6, 7, 8, 5, 2, 1], 
                'rasp_04': [6, 7, 2, 8, 4, 3, 1, 5],
                'rasp_09': [3, 6, 5, 7, 4, 2, 1, 8],
                'rasp_13': [2, 3, 6, 8, 1, 7, 5, 4],
                'rasp_23': [6, 7, 2, 5, 8, 3, 4, 1],
                'rasp_26': [6, 8, 4, 1, 5, 3, 7, 2],
                'rasp_28': [7, 2, 6, 4, 8, 5, 3, 1],
                'rasp_29': [5, 4, 1, 6, 3, 8, 7, 2],
                'rasp_31': [3, 2, 7, 5, 6, 1, 8, 4],
                'rasp_37': [3, 2, 8, 6, 5, 4, 1, 7],
                'rasp_38': [8, 7, 5, 1, 3, 2, 4, 6],
                'rasp_39': [1, 3, 8, 4, 5, 2, 6, 7],
                'rasp_40': [8, 6, 5, 1, 4, 7, 3, 2], 
                'rasp_46': [6, 4, 3, 5, 2, 7, 1, 8], 
                'rasp_50': [4, 8, 3, 7, 1, 6, 2, 5], 
                'rasp_57': [2, 5, 7, 8, 3, 4, 6, 1], 
                'rasp_59': [7, 2, 3, 5, 8, 4, 1, 6], 
                'rasp_61': [5, 3, 4, 7, 2, 6, 1, 8], 
                'rasp_62': [2, 7, 1, 3, 6, 5, 8, 4], 
                'rasp_67': [1, 7, 4, 8, 5, 2, 3, 6]}


#%%
root_path = '/mnt/storage/RASP/'


participant = 'rasp_02'
#%% there should be a look up table for block seq by participant ID

block_seq = participants[participant] # sequence of blocks played for this participant

#%% handle condition sequence for the participant
csvFile = pd.read_csv(root_path + '/stimuli/Wav_Files_by_Block_zc.csv')

condition_array = []

for i in block_seq:
    block = csvFile[csvFile['Block Number'] == i]
    condition_array.append(block.loc[:,'Condition'])

flat_condition = [item for sublist in condition_array for item in sublist]

#%% load env files in the block seq order, should be 300 trials
envs_ta = []
envs_bg = []

for i in block_seq:
    file_target = root_path + '/stimuli/env_target_block' + str(i) + '.txt'
    file_background = root_path + '/stimuli/env_background_block' + str(i) + '.txt'
    
    with open(file_target, "r") as file:
        env_ta = json.load(file)

    with open(file_background, "r") as file:
        env_bg = json.load(file)

    envs_ta.append(env_ta)
    envs_bg.append(env_bg)
    
envs_ta = [item for sublist in envs_ta for item in sublist]
envs_bg = [item for sublist in envs_bg for item in sublist]
#%% load preprocessed EEG file
file_eeg = root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_ssp_ica_preprocessed.fif'
eeg = mne.io.Raw(file_eeg,preload = True)

#bandpass filter between 1-10hz
eeg.filter(l_freq=1,h_freq=10,method='iir',iir_params=dict(order=4,ftype='butter'))

#%% handling events
events,event_id = mne.events_from_annotations(eeg)
## only take the last 300 epochs and get rid of the practice trials
#events_select = events[events[:,2]>4] ## for general checking, verifying events
events_8_select = events[events[:,2]==event_id['8']][-300:,:] 
#%% only for checking
#print(event_id)
# events_8 = events[events[:,2]==event_id['8']]
# block_event = events_select[events_select[:,2] != event_id['8']]
# block_event = block_event[block_event[:,2] != event_id['4']]
#%% Epoch
# visualize the epochs and figure out which ones would be dropped
epochs = mne.Epochs(eeg, events = events_8_select, tmin=0.0, tmax=4.0, baseline=None, reject=dict(eeg = 100e-6))
evoked = epochs.average()
evoked.plot()

epochs.drop_bad()
drops = epochs.drop_log
print('percent trials dropped:', epochs.drop_log_stats())

# select the epochs for analysis without actually dropping trials
epochs_select = mne.Epochs(eeg, events = events_8_select, tmin=0.0, tmax=4.0, baseline=None, reject= None)

epochs_select_resample = mne.Epochs.resample(epochs_select.load_data(), sfreq = 250)
epochs_select_resample.pick_types(eeg = True) # getting rid of eog, ecg etc. 

# pick specific channels
#epochs_select_resample.pick_cnannels(['Cz','FCz','CPz'])
epochs_select_data = epochs_select_resample.get_data()

evoked_select_resample = epochs_select_resample.average()
evoked_select_resample.plot()
#%% calculate connectivity measures
envs = envs_ta
connectivity_methods = ['coh', 'plv','wpli']
# Provide the freq points
freqs = np.linspace(3, 10, 8)
cohs = []
plvs = []
wplis = []
xcorrs = np.zeros((len(envs),int(57)))
xinds = np.zeros((len(envs),int(57)))

for i in range(len(envs)):
    env = np.asarray(envs[i])
    #filter
    b, a = butter(N = 4, Wn = 10, btype = 'lowpass', fs = 44100)
    filtered_env = filtfilt(b, a, env)
    #resample
    num = round(len(env)*250/44100)
    filtered_env_resample = resample(filtered_env,num)
    filtered_env_resample_reshape = filtered_env_resample.reshape(1, 1, num)
    
    epoch = epochs_select_data[i,:,0:num].reshape(1,len(epochs_select_data[i,:,0:num]),num) # cut the epoch to the length of the actual stimulus
    
    print(i)
    
    data = np.concat((epoch, filtered_env_resample_reshape), axis = 1) # env is the last channel
    
    # need to scale the data because the envelope is on a much more different scale than EEG
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[0,:,:].T).T
    scaled_data = scaled_data.reshape(1, scaled_data.shape[0],scaled_data.shape[1])
    
    names = epochs_select_resample.info['ch_names'] + ['envelope']
    
    #con = mne_connectivity.spectral_connectivity_time(data, freqs = freqs, fmin=1, fmax = 10, sfreq = 250, mode = 'multitaper', method = connectivity_methods)
    
    con = mne_connectivity.spectral_connectivity_time(scaled_data, freqs = freqs, fmax = 10, sfreq = 250, mode = 'multitaper', method = connectivity_methods)
    
    coh = np.squeeze(con[0].get_data(output = 'dense')[:,-1,:,:].mean(axis=2)[:,0:-1])
    cohs.append(coh)
    
    plv = np.squeeze(con[1].get_data(output = 'dense')[:,-1,:,:].mean(axis=2)[:,0:-1])
    plvs.append(plv)
        
    wpli = np.squeeze(con[2].get_data(output = 'dense')[:,-1,:,:].mean(axis=2)[:,0:-1])
    wplis.append(wpli)

    for j in range(epoch.shape[1]):
        xcorr = np.correlate(filtered_env_resample, epoch[0,j,:], mode = 'full')
        max_corr = np.max(abs(xcorr))
        max_index = np.argmax(abs(xcorr))
        xcorrs[i,j] = max_corr
        xinds [i,j] = max_index

cohs = np.array(cohs)
plvs = np.array(plvs)
wplis = np.array(wplis)

#%% drop rows in results where trials would be rejected

for d in range(len(drops)):
    if len(drops[d])>0:
        cohs[d,:] = np.nan
        plvs[d,:] = np.nan
        wplis[d,:] = np.nan
        xcorrs[d,:] = np.nan

#%%
cohs_results = pd.DataFrame(cohs,columns = epochs_select_resample.info['ch_names'])
cohs_results['Condition'] = flat_condition
cohs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_cohs_ta.csv')

plvs_results = pd.DataFrame(plvs,columns = epochs_select_resample.info['ch_names'])
plvs_results['Condition'] = flat_condition
plvs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_plvs_ta.csv')

wplis_results = pd.DataFrame(wplis,columns = epochs_select_resample.info['ch_names'])
wplis_results['Condition'] = flat_condition
wplis_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_wplis_ta.csv')

xcorrs_results=pd.DataFrame (xcorrs, columns = epochs_select_resample.info['ch_names'])    
xcorrs_results['Condition'] = flat_condition
xcorrs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_xcorrs_ta.csv')
#%% sort by conditions
grouped_results = wplis_results.groupby('Condition').agg('mean')