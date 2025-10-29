#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 13:13:12 2025

Pulling individual data from RASP into groups
Visualization and statistics

@author: zhaotc
"""

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# specify which metric
# loop to read in results
# check the % of trials thrown out, if > 10% then throw out the participant

participants = ['rasp_02','rasp_04','rasp_09','rasp_13','rasp_23','rasp_26','rasp_28','rasp_29','rasp_31','rasp_37','rasp_38','rasp_39','rasp_40','rasp_46','rasp_50', 'rasp_57','rasp_59', 'rasp_61', 'rasp_62','rasp_67']

# remove rasp40
#%% pulling group together
metric = 'plvs'  ## for plvs, cohs and wplis

stream = ['ta','bg'] # 'ta' or 'bg'

root_path = '/mnt/storage/RASP/'

group_ta = pd.DataFrame()
group_bg = pd.DataFrame()
group_diff = pd.DataFrame()

for p in participants:
    file_name_ta = root_path + 'data/RASP_' + p[5:7] + '/'+ p + '_' + metric + '_' + stream[0] + '.csv'
    file_name_bg = root_path + 'data/RASP_' + p[5:7] + '/'+ p + '_' + metric + '_' + stream[1] + '.csv'
    df_ta = pd.read_csv(file_name_ta,index_col=0)
    df_bg = pd.read_csv(file_name_bg,index_col=0)
    pnan = df_ta.iloc [:,0:-1].isna().sum()/(df_ta.iloc[:,0:-1].isna().sum() + df_ta.iloc[:,0:-1].count())
    # check amount of NaN in EEG
    if (pnan <= 0.1).all() == 1:  #only include if it's less than 10% rejection
        #len(bad_channels[p[5:7]])>0:
        diff = df_ta.iloc[:,0:-1]-df_bg.iloc[:,0:-1]  #target stream - background stream
            # df_ta[bad_channels[p[5:7]]] = np.nan #chaged 9/16 as bad channels were intrapolated
            # df_bg[bad_channels[p[5:7]]] = np.nan
            # diff[bad_channels[p[5:7]]] = np.nan
        diff['Condition'] = df_ta['Condition']
        diff['participant'] = p
        df_ta['participant'] = p
        df_bg['participant'] = p
        group_ta = pd.concat([group_ta,df_ta])
        group_bg = pd.concat([group_bg,df_bg])
        group_diff = pd.concat([group_diff,diff])
    else:
        print ('excluding', p)

# group_ta.to_csv(root_path + 'results/group_gamma_' + stream[0] + '_' + metric + '.csv')
# group_bg.to_csv(root_path + 'results/group_gamma_' + stream[1] + '_' + metric + '.csv')
# group_diff.to_csv(root_path + 'results/group_diff_gamma_' + metric + '.csv')

#%% visualization
group_data = group_diff
participant_avg = group_data.groupby(['Condition','participant']).mean()

condition_avg = participant_avg.groupby('Condition').mean()
sem_by_condition = participant_avg.groupby('Condition').sem()

# Prepare data for plotting
conditions = condition_avg.index
mean_scores = condition_avg['CP5'] # Using Cz for now
sem_scores = sem_by_condition['CP5']

# Create the bar plot with error bars
plt.bar(conditions, mean_scores, yerr=sem_scores, capsize=5, ecolor='black')
plt.xlabel('Condition')
plt.ylabel(metric)
plt.title('Target - Background Stream')
plt.show()

#%% topomap
n_channel = 58
ch_names = condition_avg.columns.tolist()
ch_types = ['eeg'] * n_channel
sfreq = 1

evoked_data = condition_avg.to_numpy()

info = mne.create_info(ch_names = ch_names, ch_types = ch_types, sfreq = sfreq)

evoked = mne.EvokedArray(evoked_data.T, info)
montage = mne.channels.read_custom_montage(root_path + 'CHN_EEGLAB_channel_location_updated.loc')
evoked.set_montage(montage)

import matplotlib.pyplot as plt

# Plot topomaps and get the figure and axes
fig = evoked.plot_topomap(times = [0,1,2], scalings = dict(eeg = 1),sphere = 'eeglab',vlim=(0.03,0.045),cnorm=None,colorbar=False)

# Custom titles
titles = ["BA", "CT", "TA"]

# Apply titles to each subplot
for ax, title in zip(fig.axes, titles):
    ax.set_title(title, fontsize=12)

plt.show()


#%% load in the behavior data and correlate with EEG
behavior = pd.read_csv('/mnt/storage/RASP/Behavioral Analysis/Results/participant_by_condition.csv')
behavior.columns = behavior.columns.str.lower()
behavior.rename(columns = {'0':'Condition'}, inplace = True)

participant_columns = [col for col in behavior.columns if 'rasp' in col]

# Unpivot the DataFrame
behavior_melted = pd.melt(behavior,
                    id_vars=['Condition'],  # Columns to keep as identifiers
                    value_vars=participant_columns, # Columns to unpivot
                    var_name='participant', # New column for participant IDs
                    value_name='behaviorscore') # New column for scores

merged = pd.merge(participant_avg, behavior_melted, on = ['participant','Condition'])

#%% scatter plot
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(data=merged, x='CP5', y='behaviorscore', hue='Condition', s=100) # s controls marker size
plt.title('Scatter Plot of C1 vs Behavior by Condition')
plt.xlabel('Difference between Target and Background at Cz')
plt.ylabel('Behavior Accuracy')
plt.legend(title='Condition')
plt.grid(True)
plt.show()

#%% prelim multiple regression
import pandas as pd
from scipy.stats import pearsonr

# Assuming your DataFrame is named df
# Columns: 'participant_id', 'condition', 'EEG', 'behavior'

df = merged_stat
# Loop through each condition
for condition in df['Condition'].unique():
    print(f"\nRegression for condition: {condition}")
    
    # Filter data for the current condition
    subset = df[df['Condition'] == condition]
    
    # Define independent variable (EEG) and dependent variable (behavior)
    X = subset[['Fz']]
    y = subset['behaviorscore']
    
    r, p = pearsonr(X, y)
    print(f"Pearson r: {r:.3f}, p-value: {p:.3f}")


#%% write out data frames for R
merged_stat = merged.reset_index()

# write them out for R
merged_stat.to_csv('/mnt/storage/RASP/results/merged_stat_diff_gamma_plvs.csv')
eeg_stat = participant_avg.reset_index()
eeg_stat.to_csv('/mnt/storage/RASP/results/eeg_stat_diff_gamma_plvs.csv')

#%% trying to do some preliminary statistics
from statsmodels.stats.anova import AnovaRM

for ch in stats_data.columns[2:]:
    aov_table = AnovaRM(data = stats_data, depvar = ch, subject = 'participant', within = ['Condition']).fit()
    print(ch)
    print(aov_table)

#%% Permutation
import mne
import scipy

eeg_stat = participant_avg.reset_index()
X1 = eeg_stat[eeg_stat['Condition'] == 'BA'].iloc[:,2:].to_numpy() # BA
X2 = eeg_stat[eeg_stat['Condition'] == 'CT'].iloc[:,2:].to_numpy() # CT
X3 = eeg_stat[eeg_stat['Condition'] == 'TA'].iloc[:,2:].to_numpy() # TA
X = [X1, X3]

raw = mne.io.Raw('/mnt/storage/RASP/data/RASP_02/rasp_02_ssp_ica_intrabad_preprocessed.fif')
adjacency, ch_names = mne.channels.find_ch_adjacency(raw.info, ch_type = 'eeg')

# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.1

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = len(X)
n_observations = len(X[0])
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# run the cluster based permutation analysis
cluster_stats = mne.stats.spatio_temporal_cluster_test(
    X,
    n_permutations=5000,
    threshold=f_thresh,  # doing TFCE instead
    tail=tail,
    n_jobs=None,
    buffer_size=None,
    adjacency=adjacency,
    seed = 12345
)

F_obs, clusters, p_values, _ = cluster_stats

ch_names_array = np.array(ch_names)
ch_names_array[np.where(p_values<0.05)[0]]

# import pickle

# # Save cluster results
# with open('/mnt/storage/RASP/results/cohs_diff_gamma_clusters.pkl', 'wb') as f:
#     pickle.dump({
#         'F_obs': F_obs,
#         'clusters': clusters,
#         'cluster_p_values': p_values,
#     }, f)

#%% visualization for significant clusters
ch_names = np.array(ch_names)
significant_channels = ch_names[p_values < 0.05]

condition_avg = participant_avg.groupby('Condition').mean()
sem_by_condition = participant_avg.groupby('Condition').sem()

# Prepare data for plotting
for i in significant_channels:
    conditions = condition_avg.index
    mean_scores = condition_avg[i] # Using Cz for now
    sem_scores = sem_by_condition[i]
    
    # Create the bar plot with error bars
    plt.bar(conditions, mean_scores, yerr=sem_scores, capsize=5, ecolor='black')
    plt.xlabel('Condition')
    plt.ylabel(metric)
    plt.title('Target - Background Stream')
    plt.show()
plt.show()


conditions = condition_avg.index
mean_scores = condition_avg[significant_channels].mean(axis = 1) # Using Cz for now
sem_scores = sem_by_condition[significant_channels].mean(axis = 1)

# Create the bar plot with error bars
plt.bar(conditions, mean_scores, yerr=sem_scores, capsize=5, ecolor='black')
plt.xlabel('Condition')
plt.ylabel(metric)
plt.title('Target - Background Stream')
plt.show()