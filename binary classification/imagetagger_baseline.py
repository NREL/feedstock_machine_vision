#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:11:36 2019

@author: dsievers

Modification Log:
    9/4/2019---cgudaval---Process updations and code cleanup

Description:
    Script to determine when a feedstock anomaly is present based on process datalogs
    and label corresponding weighbelt images accordingly. User-selectable process tags
    and thresholds to determine when a feedstock anomaly is actually present.
    Ability to apply time phasing to process data is applied to more accurately identify
    condition times relative to the weighbelt images.

    Processing *by run* to avoid edge effects where runs abut each other, and in case
    screw speeds for runs differ when calculating delays between equipment.
    
NOTES
    WB belt-load and pug mill motor load seem rather uncorrelated with cross feeder and 
    PSF loads for some runs, particularly run 2. We know run 2 had a higher WB belt-load setpoint,
    which helped immensely with stabilizing the mass feed rates but we don't know why someone
    changed this setpoint only for this one run. WB speed seems somewhat correlated and offers
    a method to apply time phasing to accurately label the images given PSF disturbances.
"""

#import sys
#sys.path.insert(1, 'C:\\Users\\cgudaval\\Desktop\\FCIC_Code\\Modules')
#sys.path.insert(1, 'C:\\Users\\cgudaval\\Desktop\\FCIC_Code\\Modules\\engtools')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from FCIC.FY18_metadata import Meta
from FCIC.FCICfunctions import clipper, phaser, alarmWithTD
from engtools import indexconvert
import copy
from timing import Timer
import os
from engtools import df_smooth
import random
from PIL import Image
import csv

# =============================================================================
# Hyper-Parameters
# =============================================================================
#%%
smoothingWindowSize = 15*2  # (seconds) window sixe for smoothing "rawprocess[tags]"
minBeltSpeed = 1.0 # Minimum belt speed to judge that system is running

thresh_qnt = 0.60  # "percentile" of cdf to use as threshold value
debouncetime = 130 # (seconds) proc_data must exceed threshold for this time

slackTime = 10 # (seconds) slack time +/- around image times to search the label flag series

#%%
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

runs = [
        'LT_baseline_1',
        'LT_baseline_2',
        'LT_baseline_3',
        'LT_baseline_4',
        ]

path_rawdata = '/Users/cgudaval/Desktop/FCIC/FCIC_data'
path_dbs = '/Users/cgudaval/Desktop/FCIC/FCIC_dbs'

# =============================================================================
# Load Process data
# =============================================================================
#%%
print('loading process data...')
picklepath = os.path.join(path_dbs, 'LT_baseline_process.pkl')
rawprocess = pd.read_pickle(picklepath)

# tags to use
tags = [
       'LHR weigh-belt speed',
       'LHR pug-mill load',
       'LHR cross-feeder load', 
       'LHR PSF load'
        ]

speedtags = [
        'LHR weigh-belt speed',
        'LHR pug-mill speed',
        'LHR cross-feeder speed',
        'LHR PSF speed'
        ]

# times when cablevey was started
idx = np.nonzero(np.diff(np.array(rawprocess['LHR cablevey load']>0.5,dtype=int))==1)[0]
cablevey_startup = rawprocess.index[idx]

# make new df for filter tags and then smooth
print('smoothing raw process data...')
# smooth filter
t = Timer()
proc_smooth = df_smooth(rawprocess[tags], smoothingWindowSize)
t.split('smoothing complete')

"""
Check speed variations across runs. This may be a problem with applying
phasing, since the pug mill and maybe other equipment were messed with 
in the middle of the runs. The speed settings on the pug mill are 
different for three runs and two are constant, so a run-specific phasing
delay may be applied (see below).
"""
if False:
    plt.close('all')
    fig, axs = plt.subplots(len(speedtags))
    fig.set_size_inches(12,4)
    fig.set_tight_layout(True)
    for run in runs:
        meta = Meta(run)
        clipstart = pd.Timestamp(meta.start_full)
        clipend = pd.Timestamp(meta.end_full)
        df = clipper(rawprocess[speedtags], clipstart, clipend)
        dfh = indexconvert(df, 'h', chopgaps=True, gapthresh=1.0)
        for i, tag in enumerate(speedtags):
            axs[i].title.set_text(tag)
            axs[i].plot(dfh[tag], label=run)
            axs[i].legend()

"""
make mask array for steady state times for each run
including both normal and high-acid times since feedstock should not be affected

some points during some runs there were breakdowns for periods of time
only take times when equipment was running
"""
running = (rawprocess[speedtags] > minBeltSpeed).all(axis=1)  # filter for when process is actually running

steadystate = pd.DataFrame()
for run in runs:
    meta = Meta(run)
    # use images in as wide of possible range
    clipstart = pd.Timestamp(meta.start_official)
    clipend = pd.Timestamp(meta.end_ha)
    ssr = pd.Series((clipstart <= rawprocess.index) & (rawprocess.index <= clipend),
                    index=rawprocess.index, name=run)
    ssrr = pd.Series(ssr & running, index=ssr.index, name=run)
    steadystate = pd.concat([steadystate, ssrr], axis=1)

proc_smooth_ss = proc_smooth[steadystate.any(axis=1)]
process = {}
for run in runs:
    process[run] = proc_smooth[steadystate[run]]

# process dictionary is prepared

#%%
"""
Calculate phasing delays to align weigh belt images with process conditions.
Using a method that optimizes a delay time applied to the two compared series
using the correlation coefficient between the two.
that are nearby.
"""
reftag = 'LHR weigh-belt speed'  # the reference with camera
picklepath = os.path.join(path_dbs, 'LT_baseline_feederdelays.pkl')
if not os.path.isfile(picklepath):
    # only work on tags outside the reference
    tagsdel = tags.copy()
    tagsdel.remove(reftag)
    
    delays = {}
    for run, v in process.items():
        s1 = v[reftag]  # the reference
        delays[run] = {}
        lowerBound = 0
        for tag in tagsdel:
            s2 = v[tag]
            print('calculating phasing delay for', run, tag)
            dt = phaser(s1, s2, bounds=(lowerBound,5))
            lowerBound = dt.x
            delays[run][tag] = dt.x
    delaydf = pd.DataFrame.from_dict(delays, orient='index')  # BUG: doesn't keep original order
    delaydf = delaydf[tagsdel]  # reorder columns to original
    delaydf['LHR cross-feeder load'] = delaydf['LHR cross-feeder load'].mean()
    delaydf.to_pickle(picklepath)
else:
    delaydf = pd.read_pickle(picklepath)
    delaydf['LHR cross-feeder load'] = delaydf['LHR cross-feeder load'].mean()

# Apply delays

procwdelays = {}
for run in runs:
    procdf = copy.deepcopy(process[run][reftag])
    for tag in delaydf.columns:
        pdel = copy.deepcopy(process[run][tag])
        dt = int(delaydf.loc[run][tag] * 60)  # force to int of seconds to avoid precision blowup
        pdel.index = pdel.index - pd.Timedelta(dt, unit='s')
        procdf = pd.concat([procdf, pdel], axis=1, join='outer')
        temp_procdf = pd.concat([procdf, pdel], axis=1, join='inner')
    #procdf.fillna(method='ffill', inplace=True)
    #procdf.dropna(axis=1, how='any', inplace=True)
    procdf = procdf.interpolate(method = 'linear')
    procwdelays[run] = procdf.dropna()

# =============================================================================
# Process Condition Classification
# =============================================================================
#%%
"""
Process variables automatically labeled based on user-set conditions. The
associated timestamps will be used to apply the same labels to the data.
"""

# calculate quantile threshold
# need to do this individually for each run to avoid edge effects where runs meet
#window = pd.Timedelta(10, unit='m')
window = 2*60*60  # this is a long-time average of the quantile, assume 1s intervals in data
# TODO: would be nice to use timedelta as window, but not supported by `center` option
#window = pd.Timedelta(60, unit='m')
proc_qnt = {}
for run, df in procwdelays.items():
    qnt = df.rolling(window, center=True).quantile(thresh_qnt)
    qnt.fillna(method='ffill', inplace=True)
    qnt.fillna(method='bfill', inplace=True)
    proc_qnt[run] = qnt

# =============================================================================
# # HACK: just hypersmooth the process data and determine when in alarm state
# # Needed due to issues with the "alarm" function
# print('hypersmoothing process data for filter...')
# win = 10*60  # s
# t = Timer()
# abnormal = {}
# for run, df in procwdelays.items():
#     prochypersmooth = df_smooth(df, win)
#     abnormal[run] = prochypersmooth > proc_qnt[run]
# t.split('smoothing & filtering complete')
# =============================================================================

abnormal = dict()
for run, df in procwdelays.items():
    t = Timer()
    abnormal[run] = dict()
    for col in df.columns.values:
        
        dataArray = np.array(df[col])
        threshArray = np.array(proc_qnt[run][col])
        index = np.pad((np.diff(np.array(df.index))/(10**9)).astype(int),(0,1),'constant',constant_values=(0, 1))
        
        temp = (alarmWithTD(dataArray,threshArray,index,debouncetime) == 1)
        abnormal[run][col] = temp.transpose()
        
    abnormal[run] = pd.DataFrame(abnormal[run], index=procwdelays[run].index)
    t.split('for smoothing '+run)

#%%
# Identify abnormal feedstocks and plot

tags_filter = [           # tags to filter by using AND logic
#        'LHR weigh-belt belt load',
#        'LHR weigh-belt speed',
#        'LHR pug-mill load',
        'LHR cross-feeder load', 
#        'LHR PSF load',
        ]

anomaly = pd.Series()
plt.close('all')
for run, v in procwdelays.items():
    cols = procwdelays[run].columns
    fig, axs = plt.subplots(len(cols)+1)
    fig.set_size_inches(12,4)
    fig.set_tight_layout(True)
    for i, c in enumerate(cols):
        axs[i].title.set_text(c)
        axs[i].plot(v[c], linewidth=0.5)
        axs[i].plot(proc_qnt[run][c], linestyle='--', color=pltcolors[2], linewidth=0.5)
        maxlim = axs[i].get_ylim()[1]
        axs[i].fill_between(x=abnormal[run].index, y1=abnormal[run][c].values*maxlim, 
           facecolor=pltcolors[1], edgecolor=None, alpha=0.75)

    abflag = abnormal[run][tags_filter].all(axis=1)
    anomaly = anomaly.append(abflag)
    
    axs[-1].fill_between(x=abflag.index, y1=abflag.values, 
       facecolor=pltcolors[4], edgecolor=None, alpha=0.75)
    axs[-1].title.set_text('CONDITION FILTER')
    axs[-1].axes.get_yaxis().set_visible(False)
    for t in cablevey_startup:
        axs[-1].axvline(t, color='k')
    axs[-1].set_xlim(axs[0].get_xlim())
    
# =============================================================================
# Label Images
# =============================================================================
#%%
"""
Must assess each image for valid corresponding process data. Images
dropped if they don't have any process data associated with them, such
as unexpected operation pauses. Those process data have already been
stripped above.
"""
picklepath = os.path.join(path_dbs, 'LT_baseline_img_labeled.pkl')
if True: #not os.path.isfile(picklepath):
    picklepathread = os.path.join(path_dbs, 'LT_baseline_img.pkl')
    impathtime = pd.read_pickle(picklepathread)

    print('labeling images based on process flags...')
    impathtime_labeled = pd.DataFrame(columns=impathtime.columns)
    impathtime_labeled['anomaly'] = None
    # slack time +/- around image times to search the label flag series
    slack = pd.Timedelta(slackTime, unit='s')
    #TODO: refactor out for loop
    tim = Timer()
    for i, t in enumerate(impathtime.index):
        tneg = t - slack
        tpos = t + slack
        f = anomaly[(tneg <= anomaly.index) & (anomaly.index <= tpos)]
        if len(f) == 0:
            print(i, t, 'no tag')
            continue  # skip if nothing found
        else:
            impathtime_labeled = impathtime_labeled.append(impathtime.loc[t])
            if f.any():
                impathtime_labeled['anomaly'][t] = True
            else:
                impathtime_labeled['anomaly'][t] = False
            print(i, t, impathtime_labeled['anomaly'][t])
    tim.split('labeling forloop')
    
    print('{0} images retained out of {1}. {2} images are labeled true.'\
          .format(len(impathtime_labeled),
                  len(impathtime),
                  impathtime_labeled['anomaly'].sum()))
        
    impathtime_labeled.to_pickle(picklepath)

# =============================================================================
# Label Visual Assessment
# =============================================================================
#%%
picklepath = os.path.join(path_dbs, 'LT_baseline_img_labeled.pkl')
if os.path.isfile(picklepath):
    impathtime_labeled = pd.read_pickle(picklepath)
    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(4,6)
    fig.set_tight_layout(True)
#    random.seed(1)
    falses = impathtime_labeled[impathtime_labeled['anomaly']==False]
    trues = impathtime_labeled[impathtime_labeled['anomaly']==True]
    
    def getimage(df):
        idx = random.randint(0, len(df))
        path = os.path.join(path_rawdata, df['path'].iloc[idx])
        img = Image.open(path)
        resize = (0.15 * np.array(np.shape(img)[0:2])).astype(int)
        img = img.resize((resize[1],resize[0]))
        axis.imshow(img)
        axis.set_xlabel(str(df.index[idx]), fontsize='x-small')
        axis.set_xticklabels([])
        axis.set_yticklabels([])
                
    for axis in axs[:,0]:
        getimage(falses)

    for axis in axs[:,1]:
        getimage(trues)

    axs[0][0].set_title('anomaly: False', fontsize='x-large')
    axs[0][1].set_title('anomaly: True', fontsize='x-large')
    
    fileName = 'imLabels_FD_'+str(thresh_qnt*100)[0:2]+'.csv'
    impathtime_labeled.to_csv(fileName, index=True, quoting=csv.QUOTE_NONNUMERIC) 