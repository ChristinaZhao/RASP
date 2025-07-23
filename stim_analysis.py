#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:07:20 2025
RASP
Organize the stimuli files by blocks 1-8
Analyze distribution of stimuli length
Extract envelopes for each block and save them out, they can be rearranged later for each participant during EEG analysis

@author: zhaotc
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import seaborn as sns
import matplotlib.pyplot as plt

#%% add a column of stimulus duration for each trial and then analyze stim duration by conditions
root_path = '/mnt/storage/RASP/stimuli/'
csvFile = pd.read_csv(root_path + 'Wav_Files_by_Block_zc.csv')
dur_trial_TA = []
dur_trial_BA = []
dur_trial_CT = []
for ntrial in np.arange(len(csvFile)):  
    print('Extracting envelope from trial ' + str(ntrial))
    if csvFile.loc[ntrial]['Condition'] == 'TA':
        fs, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + 'alt_' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'ScaledSpeakers/' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
        dur_tar = len(target_audio)/fs
        dur_bg= len(bg_audio)/fs
        larger_value = max(dur_tar,dur_bg)
        dur_trial_TA.append(larger_value)
        
    elif csvFile.loc[ntrial]['Condition'] == 'BA':        
        fs, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'ScaledSpeakers/' + 'alt_' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
        dur_tar = len(target_audio)/fs
        dur_bg = len(bg_audio)/fs
        larger_value = max(dur_tar,dur_bg)
        dur_trial_BA.append(larger_value)
        
    elif csvFile.loc[ntrial]['Condition'] == 'CT':
        fs, target_audio = wavfile.read(root_path + 'ScaledSpeakers/' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'ScaledSpeakers//' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
        dur_tar = len(target_audio)/fs
        dur_bg = len(bg_audio)/fs
        larger_value = max(dur_tar,dur_bg)
        dur_trial_CT.append(larger_value)
    else: print("something is wrong!")
    
#%% plot a figure of stimulus duration across conditions
dur_trial = [dur_trial_TA, dur_trial_CT, dur_trial_BA]
plt.figure()
# Create labels for the data
labels = ['Target Altered', 'Control', 'Background Altered']

# Create the violin plot
sns.violinplot(data=dur_trial)

# Add labels to the x-axis
plt.xticks(ticks=range(len(labels)), labels=labels)

# Add a title
plt.title('Stimulus Duration by Conditions')

# Show the plot
plt.show()

from scipy.stats import f_oneway

# Perform one-way ANOVA
f_stat, p_value = f_oneway(dur_trial_TA, dur_trial_CT, dur_trial_BA)

print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# ntime = np.shape(data)[-1]
# ntr = np.shape(data)[0]
# nch = np.shape(data)[1]
