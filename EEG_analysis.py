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

participant = 'rasp_62'
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
    file_target = root_path + '/stimuli/env_gamma_target_block' + str(i) + '.txt'
    file_background = root_path + '/stimuli/env_gamma_background_block' + str(i) + '.txt'
    
    with open(file_target, "r") as file:
        env_ta = json.load(file)

    with open(file_background, "r") as file:
        env_bg = json.load(file)

    envs_ta.append(env_ta)
    envs_bg.append(env_bg)
    
envs_ta = [item for sublist in envs_ta for item in sublist]
envs_bg = [item for sublist in envs_bg for item in sublist]
#%% load preprocessed EEG file
file_eeg = root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_ssp_ica_intrabad_preprocessed.fif'
eeg = mne.io.Raw(file_eeg,preload = True)

#bandpass filter between 1-10hz
eeg.filter(l_freq = None, h_freq=40,method='iir',iir_params=dict(order=4,ftype='butter')) #took out l_freq = 1 per Yi' suggestion 10/14/25
#%% handling events
events,event_id = mne.events_from_annotations(eeg)
## only take the last 300 epochs and get rid of the practice trials
#events_select = events[events[:,2]>4] ## for general checking, verifying events
events_8_select = events[events[:,2]==event_id['8']][-300:,:]
#mne.write_events(root_path + '/data/RASP_' + participant[5:7] +'/rasp_' + participant[5:7] +'_select_eve.fif', events_8_select, overwrite=True)
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
#%% calculate connectivity measures for ta stream
envs = envs_ta
connectivity_methods = ['coh', 'plv','wpli']
# Provide the freq points
freqs = np.linspace(3.5, 9, 12) # 3.5 - 9Hz selected based on Van Den Bosch (2020)
cohs = [] 
plvs = []
wplis = []
xcorrs = np.zeros((len(envs),epochs_select_data.shape[1]))
xinds = np.zeros((len(envs),epochs_select_data.shape[1]))

for i in range(len(envs)):
    env = np.asarray(envs[i])
    #lowpass filter
    b, a = butter(N = 4, Wn = 40, btype = 'lowpass', fs = 44100)
    filtered_env = filtfilt(b, a, env)
    #resample
    num = round(len(env)*250/44100)
    filtered_env_resample = resample(filtered_env,num)
    filtered_env_resample_reshape = filtered_env_resample.reshape(1, 1, num)
    
    #epoch = epochs_select_data[i,:,0:num].reshape(1,len(epochs_select_data[i,:,0:num]),num) # cut the epoch to the length of the actual stimulus
    
    # instead of cutting the epoch, we are zeropadding env to epoch length (4s)
    # changed 10/14 per Yi's suggestion
    
    epoch = epochs_select_data[i,:,:].reshape(1,epochs_select_data.shape[1],epochs_select_data.shape[2])
    
    pad_size = epoch.shape[2] - num
    
    filtered_env_resample_reshape_zp = np.pad(filtered_env_resample_reshape, ((0, 0), (0, 0), (0, pad_size)), mode='constant') # pad the envelope
    
    print(i)
    
    data = np.concat((epoch, filtered_env_resample_reshape_zp), axis = 1) # env is the last channel
    
    # need to scale the data because the envelope is on a much more different scale than EEG
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[0,:,:].T).T
    scaled_data = scaled_data.reshape(1, scaled_data.shape[0],scaled_data.shape[1])
    
    names = epochs_select_resample.info['ch_names'] + ['envelope']
    
    #con = mne_connectivity.spectral_connectivity_time(data, freqs = freqs, fmin=1, fmax = 10, sfreq = 250, mode = 'multitaper', method = connectivity_methods)
    
    ## more read on the doc
    con = mne_connectivity.spectral_connectivity_time(scaled_data, freqs = freqs, fmax = 10, sfreq = 250, mode = 'multitaper', n_cycles = 5, method = connectivity_methods, n_jobs = 5)
    
    
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
cohs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_cohs_ta_new.csv')

plvs_results = pd.DataFrame(plvs,columns = epochs_select_resample.info['ch_names'])
plvs_results['Condition'] = flat_condition
plvs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_plvs_ta_new.csv')

wplis_results = pd.DataFrame(wplis,columns = epochs_select_resample.info['ch_names'])
wplis_results['Condition'] = flat_condition
wplis_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_wplis_ta_new.csv')

xcorrs_results=pd.DataFrame (xcorrs, columns = epochs_select_resample.info['ch_names'])    
xcorrs_results['Condition'] = flat_condition
xcorrs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_xcorrs_ta_new.csv')
#%% calculate connectivity measures for bg stream
envs = envs_bg
connectivity_methods = ['coh', 'plv','wpli']
# Provide the freq points
freqs = np.linspace(3.5, 9, 12)  # 3.5 - 9Hz selected based on Van Den Bosch (2020)
cohs = [] 
plvs = []
wplis = []
xcorrs = np.zeros((len(envs),epochs_select_data.shape[1]))
xinds = np.zeros((len(envs),epochs_select_data.shape[1]))

for i in range(len(envs)):
    env = np.asarray(envs[i])
    #filter
    b, a = butter(N = 4, Wn = 40, btype = 'lowpass', fs = 44100)
    filtered_env = filtfilt(b, a, env)
    #resample
    num = round(len(env)*250/44100)
    filtered_env_resample = resample(filtered_env,num)
    filtered_env_resample_reshape = filtered_env_resample.reshape(1, 1, num)
    
    #epoch = epochs_select_data[i,:,0:num].reshape(1,len(epochs_select_data[i,:,0:num]),num) # cut the epoch to the length of the actual stimulus
    
    # instead of cutting the epoch, we are zeropadding env to epoch length (4s)
    # changed 10/14 per Yi's suggestion
    
    epoch = epochs_select_data[i,:,:].reshape(1,epochs_select_data.shape[1],epochs_select_data.shape[2])
    
    pad_size = epoch.shape[2] - num
    
    filtered_env_resample_reshape_zp = np.pad(filtered_env_resample_reshape, ((0, 0), (0, 0), (0, pad_size)), mode='constant') # pad the envelope
    
    print(i)
    
    data = np.concat((epoch, filtered_env_resample_reshape_zp), axis = 1) # env is the last channel
    
    # need to scale the data because the envelope is on a much more different scale than EEG
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[0,:,:].T).T
    scaled_data = scaled_data.reshape(1, scaled_data.shape[0],scaled_data.shape[1])
    
    names = epochs_select_resample.info['ch_names'] + ['envelope']
    
    #con = mne_connectivity.spectral_connectivity_time(data, freqs = freqs, fmin=1, fmax = 10, sfreq = 250, mode = 'multitaper', method = connectivity_methods)
    
    ## more read on the doc
    con = mne_connectivity.spectral_connectivity_time(scaled_data, freqs = freqs, fmax = 10, sfreq = 250, mode = 'multitaper', n_cycles = 5, method = connectivity_methods, n_jobs = 5) 
    
    
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
cohs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_cohs_bg_new.csv')

plvs_results = pd.DataFrame(plvs,columns = epochs_select_resample.info['ch_names'])
plvs_results['Condition'] = flat_condition
plvs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_plvs_bg_new.csv')

wplis_results = pd.DataFrame(wplis,columns = epochs_select_resample.info['ch_names'])
wplis_results['Condition'] = flat_condition
wplis_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_wplis_bg_new.csv')

xcorrs_results=pd.DataFrame (xcorrs, columns = epochs_select_resample.info['ch_names'])    
xcorrs_results['Condition'] = flat_condition
xcorrs_results.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_xcorrs_bg_new.csv')
#%% sort by conditions
#grouped_results = wplis_results.groupby('Condition').agg('mean')
# #%% do mTRF data prep
# import eelbrain

# events_eel = eelbrain.load.mne.events(raw = eeg, events = root_path + '/data/RASP_' + participant[5:7] +'/rasp_' + participant[5:7] +'_select_eve.fif')


# envelopes_2 =[]

# for envs in [envs_ta, envs_bg]:

#     envelopes = []
#     for i in range(len(envs)):
#         env = np.asarray(envs[i])
#         #filter
#         b, a = butter(N = 4, Wn = 10, btype = 'lowpass', fs = 44100)
#         filtered_env = filtfilt(b, a, env)
#         #resample
#         num = round(len(env)*1000/44100)
#         filtered_env_resample = resample(filtered_env,num)
#         filtered_env_resample_reshape = filtered_env_resample.reshape(1, 1, num)
        
#         time = eelbrain.UTS(tmin = 0, tstep = 1/1000, nsamples = num)
#         envelope = eelbrain.NDVar(filtered_env_resample,time)
        
#         envelopes.append(envelope)
    
#     envelopes_2.append(envelopes)

# events_eel['envelopes_ta'] = envelopes_2[0]
# events_eel['envelopes_bg'] = envelopes_2[1]

# # Extract the stimulus duration (in seconds) from the envelopes
# events_eel['duration_ta'] = eelbrain.Var([envelope.time.tstop for envelope in events_eel['envelopes_ta']])

# events_eel['duration_bg'] = eelbrain.Var([envelope.time.tstop for envelope in events_eel['envelopes_bg']])

# events_eel['eeg_ta'] = eelbrain.load.mne.variable_length_epochs(events_eel, tmin = 0, tstop='duration_ta')
# events_eel['eeg_bg'] = eelbrain.load.mne.variable_length_epochs(events_eel, tmin = 0, tstop='duration_bg')

# #%% drop the bad trials
# events_eel['condition'] = flat_condition
# keep_trials = [i for i, val in enumerate(drops) if val == ()]
# events_eel_rejected = events_eel[keep_trials]
# #%% split the data into 3 by condition
# BA_ind = [i for i, x in enumerate(events_eel_rejected['condition']) if x=='BA']
# CT_ind = [i for i, x in enumerate(events_eel_rejected['condition']) if x=='CT']
# TA_ind = [i for i, x in enumerate(events_eel_rejected['condition']) if x=='TA']

# BA_events = events_eel[BA_ind,:]
# CT_events = events_eel[CT_ind,:]
# TA_events = events_eel[TA_ind,:]
# #%% actual mTRF
# # separate mTRF for 3 conditions and separte for target vs. background
# labels = ['BA','CT','TA']
# conds = [BA_events, CT_events, TA_events]
# streams = ['ta','bg']

# rs = []
# pe = []

# for i,cond in enumerate(conds):
#     for s in streams:
#         decoder = eelbrain.boosting(y='envelopes_'+s, x='eeg_'+s, tstart = -0.1, tstop = 0.5, delta = 0.05, test = 1, partitions= 5, data = cond, partition_results = True)
#         rs.append(decoder.r)
#         pe.append(decoder.proportion_explained)
#         # for j in range(len(cond['duration_bg'])):
#         #     eeg = cond[j, 'eeg_'+s] / decoder.x_scale
        
#         #     # Predict the envelope by convolving the decoder with the EEG
#         #     y_pred = eelbrain.convolve(decoder.h, eeg, name='predicted envelope')
            
#         #     y = cond[i, 'envelopes_'+s]
#         #     y = y - decoder.y_mean
#         #     y /= decoder.y_scale / y_pred.std()
#         #     y.name = 'envelope'
        
#         #     r = eelbrain.correlation_coefficient(y, y_pred)
#         #     rs.append(r)
#         # rs = np.asarray(rs)
#         # rs.save(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_mtrf_' + labels(i) + '_' + s +'.csv')
#             #p = eelbrain.plot.UTS([[y_pred, y]], w=10, h=3, title=f"{r=}")

# mtrf = pd.DataFrame()
# mtrf['r'] = rs
# mtrf['pe'] = pe
# mtrf ['condition'] = [item for item in labels for _ in range(2)]
# mtrf ['stream'] = streams*3

# mtrf.to_csv(root_path + '/data/RASP_' + participant[5:7] + '/'+ participant + '_mtrf.csv')
# # #%%
# # fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
# # p = eelbrain.plot.UTS(events_eel[1, 'envelopes_ta'], axes = axes[0])
# # p = eelbrain.plot.Butterfly(events_eel[1, 'eeg_ta'], axes = axes[1])