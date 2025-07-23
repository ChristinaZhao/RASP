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
#from scipy.spatial.transform import Rotation as R
# import curryreader as cr

root_path = '/mnt/storage/RASP/'

participant = '04'

bad_channels = [] ## not sure from looking at runsheet
## load the data with read_raw_curry()
raw = mne.io.read_raw_curry(root_path + '/data/RASP_' + participant +'/Acquisition_rasp_' + participant +'.cdt', preload=True)

## read in montage that was output from EEGLAB reading in pilot_6 file
montage = mne.channels.read_custom_montage(root_path + 'CHN_EEGLAB_channel_location_updated.loc')

## drop ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2'] that are not in montage
raw.drop_channels(['Trigger','F11','F12','FT11','FT12','Cb1', 'Cb2'] + bad_channels)

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
#raw.plot_sensors(show_names = True,sphere="eeglab")
#raw0.plot()

# Plot each channel individually
# raw.plot(n_channels=len(raw.ch_names), scalings='auto', title='EEG Data')
#%% Get events
events,event_id = mne.events_from_annotations(raw)
mne.write_events(root_path + '/data/RASP_' + participant +'/rasp_' + participant +'_eve.fif', events, overwrite=True)
#%% Do some artifact removal
## Repairing with SSP
ecg_evoked = mne.preprocessing.create_ecg_epochs(raw,ch_name='EMG1').average()
ecg_evoked.apply_baseline((None,None))
ecg_evoked.plot_joint()
ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='EMG1', n_jobs = 5)
mne.viz.plot_projs_topomap(ecg_projs,info=raw.info)
fig = mne.viz.plot_projs_joint(ecg_projs, ecg_evoked, picks_trace="O1")
fig.suptitle("ECG projectors")


eog_evoked = mne.preprocessing.create_eog_epochs(raw,ch_name=['VEOG','HEOG','Fp1','Fp2']).average()
eog_evoked.apply_baseline((None,None))
eog_evoked.plot_joint()
eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['VEOG','HEOG'], n_jobs = 5)
mne.viz.plot_projs_topomap(eog_projs,info=raw.info)
fig = mne.viz.plot_projs_joint(eog_projs, eog_evoked, picks_trace ="Fz")
fig.suptitle("EOG projectors")

raw.add_proj(ecg_projs)
raw.add_proj(eog_projs)

raw.apply_proj()
#%% Referencing to the average of Mastoids
raw.set_eeg_reference(ref_channels=['M1','M2'],ch_type='eeg', projection=False)

#%% Filtering
raw.filter(l_freq=0.5,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))

#%% Save out preprocessed file
raw.save(root_path + '/data/RASP_' + participant +'/rasp_' + participant +'_preprocessed.fif',  overwrite=True)

#%% Do epoch based on the events
epoch = mne.Epochs(raw,events,event_id=event_id['8'],tmin=-0.1,tmax=1.5,baseline=(-0.05,0),reject=dict(eeg=100e-6)) # reject=dict(eeg=100e-6) # event_id ['8'] = 7 in events
evoked = epoch.average()
evoked.plot()
#evoked.plot_topomap(times,vlim=(-1,1),show_names=False)
#%% check the stim2 and stimtracker timing
# events_test = events[events[:,2] != 1]
# diff = np.diff(events_test[:,0])/2000*1000 # difference showed in ms
# all_diff = diff[diff<=50]
# plt.hist(all_diff, bins =5)


#%% Drop epoch from training 
#epoch.drop(indices=[0, 1, 2, 3])

# #%% Save fif file 
# epoch.save(root_path + 'Pilot_6/Pilot_6_preprocessed.fif', overwrite=True)





