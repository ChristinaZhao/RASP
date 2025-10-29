#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:03:30 2023

Preprocess EEG recorded in Curry EEG system. 
1. visualize raw data (slow)
2. extract events 
3. ssp cleaning
4. re-reference
5. filtering
5. epochs 
@author: tzcheng
 
@author: tzcheng
"""

import mne 
import matplotlib.pyplot as plt
import numpy as np
import json
#from scipy.spatial.transform import Rotation as R
# import curryreader as cr

#%% bad channels
participants = {'02': ['Fpz'], 
                '04': ['Fp2'],
                '09': [],
                '13': ['Fpz'],
                '23': [],
                '26': [],
                '28': ['Fp2','O1'],
                '29': ['CPz','Fpz','Fp1'],
                '31': [],
                '37': ['T7','F8'],
                '38': [],
                '39': ['C1'],
                '40': [], 
                '46': [], 
                '50': ['AF3'], 
                '57': [], 
                '59': [], 
                '61': ['F6','PO8'], 
                '62': ['PO7','PO8','O2'], 
                '67': []}
#%%
root_path = '/mnt/storage/RASP/'

participant = '62'

bad_channels = participants[participant] ## not sure from looking at runsheet
## load the data with read_raw_curry()
raw = mne.io.read_raw_curry(root_path + '/data/RASP_' + participant +'/Acquisition_rasp_' + participant +'.cdt', preload=True)

## read in montage that was output from EEGLAB reading in pilot_6 file
montage = mne.channels.read_custom_montage(root_path + 'CHN_EEGLAB_channel_location_updated.loc')

## drop ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2'] that are not in montage
#raw.drop_channels(['Trigger','F11','F12','FT11','FT12','Cb1', 'Cb2'] + bad_channels)
raw.drop_channels(['Trigger','F11','F12','FT11','FT12','Cb1', 'Cb2']) # changed 9/8/25 CZ

## rotate the layout 180 degrees (don't need it if set custom_montage)
# r = R.from_euler('z', 180, degrees=True)
# pos = np.zeros([64,3])
# pos_180 = np.zeros([64,3])
# for nch in np.arange(0,64,1):
#     tmp_pos = raw0.info['chs'][nch]['loc'][:3]
#     pos[nch,:] = tmp_pos
#     pos_180[nch,:] = r.apply(tmp_pos)
#     raw0.info['chs'][nch]['loc'][:3] = r.apply(tmp_pos)

chan_types_dict = {'HEOG':'eog', 'VEOG':'eog', 'EMG1':'ecg'}
raw.set_channel_types(chan_types_dict)
    
raw.set_montage(montage)

# added 9/8/2025 to interpolate bad channels instead of dropping them, this way everyone has 58 channels in the end.
raw.info['bads'] = bad_channels
raw.interpolate_bads() 
#raw.plot_sensors(show_names = True,sphere="eeglab")
#raw0.plot()

# Plot each channel individually
# raw.plot(n_channels=len(raw.ch_names), scalings='auto', title='EEG Data')
#%% Get events
# events,event_id = mne.events_from_annotations(raw)
# mne.write_events(root_path + '/data/RASP_' + participant +'/rasp_' + participant +'_eve.fif', events, overwrite=True)
#%% Referencing to the average of Mastoids
raw.set_eeg_reference(ref_channels=['M1','M2'],ch_type='eeg', projection=False)
#%% Do some artifact removal (SSP, but it doesn't work well with EOG)
## Repairing with SSP

ecg_evoked = mne.preprocessing.create_ecg_epochs(raw,ch_name = 'EMG1').average()
ecg_evoked.apply_baseline((None,None))
ecg_evoked.plot_joint()

ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='EMG1', n_jobs = 5)
mne.viz.plot_projs_topomap(ecg_projs,info=raw.info)
fig = mne.viz.plot_projs_joint(ecg_projs, ecg_evoked, picks_trace="O1")
fig.suptitle("ECG projectors")

#%
# eog_evoked = mne.preprocessing.create_eog_epochs(raw,ch_name=['VEOG','HEOG','Fp1','Fp2']).average()
# eog_evoked.apply_baseline((None,None))
# eog_evoked.plot_joint()pick = list(set(eog_indices + muscle_indices))

# eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['VEOG','HEOG'], n_jobs = 5)
# mne.viz.plot_projs_topomap(eog_projs,info=raw.info)
# fig = mne.viz.plot_projs_joint(eog_projs, eog_evoked, picks_trace ="Fz")
# fig.suptitle("EOG projectors")

#%% select the appropriate projector, if first component is >80%, then select just first component
raw.add_proj(ecg_projs)
# raw.add_proj(eog_projs)

raw.apply_proj()

#%%#%% Do ICA for EOG and maybe some muscle 
# ## Repairing with ICA
raw_ica = raw.copy()
raw_ica.filter(l_freq = 1, h_freq = None)
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(raw_ica)

ica.plot_sources(raw_ica)

# auto detect ecg, eog and muscle

#ecg_indices, ecg_scores = ica.find_bads_ecg(raw_ica,ch_name='EMG1', threshold="auto", method = 'correlation', measure = 'zscore')

eog_indices, eog_scores = ica.find_bads_eog(raw_ica, measure = 'zscore')

#% if too many epochs are rejected
## then try ICA reject muscle too
muscle_indices, muscle_scores = ica.find_bads_muscle(raw_ica, sphere = 'eeglab')

## visualize
pick = list(set(eog_indices + muscle_indices))
ica.plot_properties(raw_ica,picks = pick, log_scale = True)

ica.plot_scores(eog_scores, exclude = eog_indices)
ica.plot_components(eog_indices)
ica.plot_scores(muscle_scores, exclude = muscle_indices)
ica.plot_components(muscle_indices)
#%%
values_to_remove = input('Enter components that do not make sense:')
if len(values_to_remove) > 0:
    values_to_remove = list(map(float, values_to_remove.split(','))) # inspect all the automatically selected component and get rid of any that doesn't make sense

    pick = [int(item) for item in pick if item not in values_to_remove]

else:
    pick = [int(item) for item in pick]
    
print(pick) # Output: [1, 3, 4, 6]

file_name_pick = root_path + '/data/RASP_' + participant +'/rasp_' + participant +'_ssp_ica_reject.json'

with open(file_name_pick, 'w') as file:
    json.dump(pick, file, indent=4)

ica.plot_overlay(raw, exclude=pick, picks="eeg")

ica.apply(raw, exclude = pick)


#%% Filtering
raw.filter(l_freq=0.5,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
#%% Do epoch based on the events
events,event_id = mne.events_from_annotations(raw)
epoch = mne.Epochs(raw,events,event_id=event_id['8'],tmin=-0.1,tmax=4.5,baseline=(-0.05,0),reject=dict(eeg=100e-6)) # reject=dict(eeg=100e-6) # event_id ['8'] = 7 in events
evoked = epoch.average()
evoked.plot()
#evoked.plot_topomap(times,vlim=(-1,1),show_names=False)



#%% Save out preprocessed file
raw.save(root_path + '/data/RASP_' + participant +'/rasp_' + participant +'_ssp_ica_intrabad_preprocessed.fif',  overwrite=True)

#%% check the stim2 and stimtracker timing
# events_test = events[events[:,2] != 1]
# diff = np.diff(events_test[:,0])/2000*1000 # difference showed in ms
# all_diff = diff[diff<=50]
# plt.hist(all_diff, bins =5)


#%% Drop epoch from training 
#epoch.drop(indices=[0, 1, 2, 3])

# #%% Save fif file 
# epoch.save(root_path + 'Pilot_6/Pilot_6_preprocessed.fif', overwrite=True)





