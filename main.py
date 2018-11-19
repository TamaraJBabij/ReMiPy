# -*- coding: utf-8 -*-
#%%
#import and define
import pandas as pd
import numpy as np
from services import rootTreeOps, configLoader
from dictOps import mapDict
from collections import namedtuple
import itertools
import matplotlib.pyplot as plt
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.stats import norm

Event = namedtuple('Event', ['t1', 't2', 'timesum', 'c1', 'c2', 'groupNumber', 'detector'])

config = configLoader.load("config.json")
print(config)

trees = rootTreeOps.openFromDirectory(r"C:\Users\Tamara\Desktop\ReMi20169121531")
fullData = rootTreeOps.asDataFrame(trees)
fullData["time_ns"] = 32000 - 0.5*fullData[b'Time']
fullData["channel"] = pd.Series(config.channels[x].channelId for x in fullData[b'Channel'])
fullData["detector"] = pd.Series(config.channels[x].detectorId for x in fullData[b'Channel'])
groupByDetector = fullData.groupby("detector")
negData = groupByDetector.get_group(config.detectorId.neg)

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def calculateTimeSums(data, detectorId):
    #Calculates the time sums for each layer of the positive or negative detector
    groupByNumber = data.groupby(b'GroupNumber')
    groups = { name: groupByNumber.get_group(name).groupby('channel') for (name, _) in groupByNumber }
    uData = calculateChannelTimesums(groups, config.channelId.u1, config.channelId.u2)
    vData = calculateChannelTimesums(groups, config.channelId.v1, config.channelId.v2)
    wData = calculateChannelTimesums(groups, config.channelId.w1, config.channelId.w2)
    
    #U DATA
    plt.figure()
    plt.title("U layer")
    n, bins, patches = plt.hist(uData, 80, (100, 140))
    #bins is the edges of each bin, and therefore one extra length
    bins = (bins[:-1] + bins[1:])/2
    #Initial guesses
    a_u = np.amax(n)
    x0_u = bins[np.argmax(n)]
    sigma_u = 2
    popt,pcov = curve_fit(gaus,bins,n,p0=[a_u,x0_u,sigma_u])
    plt.plot(bins, gaus(bins,*popt), 'ro:')
    uFit = [popt]
    plt.show()
    
    #V DATA
    plt.figure()
    plt.title("V layer")
    n, bins, patches = plt.hist(vData, 80, (100, 140))
    #Get midpoint of the bins
    bins = (bins[:-1] + bins[1:])/2
    #Initial guesses
    a_v = np.amax(n)
    x0_v = bins[np.argmax(n)]
    sigma_v = 2
    popt,pcov = curve_fit(gaus,bins,n,p0=[a_v,x0_v,sigma_v])
    plt.plot(bins, gaus(bins,*popt), 'ro:')
    vFit = [popt]
    plt.show()
    
    #W DATA
    plt.figure()
    plt.title("W layer")
    n, bins, patches = plt.hist(wData, 80, (100, 140))
    bins = (bins[:-1] + bins[1:])/2
    #Initial guesses
    a_w = np.amax(n)
    x0_w = bins[np.argmax(n)]
    sigma_w = 2
    popt,pcov = curve_fit(gaus,bins,n,p0=[a_w,x0_w,sigma_w])
    plt.plot(bins, gaus(bins,*popt), 'ro:')
    wFit = [popt]
    plt.show()
    return [np.array(uData), np.array(vData), np.array(wData)], [uFit, vFit, wFit]


def calculateChannelTimesums(data, channel1, channel2):
    #Calculates all possible combinations of channel1 + channel2 within a group
    return list(itertools.chain.from_iterable( calculateGroupTimesums(data, channel1, channel2, groupNumber) 
        for groupNumber, data in data.items() if groupNumber < 10000 ))

def calculateGroupTimesums(groupData, channel1, channel2, groupNumber):
    #Considers all possible MCP and layer combinations
    #If the layer hit is too far in time from the mcp hit, they must not be
    #associated
    try:
        data1 = groupData.get_group(channel1)
        data2 = groupData.get_group(channel2)
        mcpData = groupData.get_group(config.channelId.mcp)
        diffs = [(t1[0] - m[0], t2[0] - m[0]) 
            for t1 in data1.as_matrix(["time_ns"])
            for t2 in data2.as_matrix(["time_ns"])
            for m in mcpData.as_matrix(["time_ns"])]
        return [d1 + d2 for d1, d2 in diffs if d1 > 0 and d1 < 500 and d2 > 0 and d2 < 500]
    except:
        return []
    
def calculateGroupEvents(groupData, channel1, channel2, detectorId, groupNumber):
    try:
        data1 = groupData.get_group(channel1)
        data2 = groupData.get_group(channel2)
        return [Event(t1, t2, t1 + t2, channel1, channel2, groupNumber, detectorId)
            for t1 in data1.as_matrix(["time_ns"]) for t2 in data2.as_matrix(["time_ns"])]
    except:
        return []
    
#%%
#Calculate the timesums for each channel
negSums, fitData = calculateTimeSums(negData, config.detectorId.neg)
print("Peak Height, Peak position, Standard Deviation")
print(fitData)

#%%
#Now we want to calculate the events
#Make sure that the layer information is within 3 standard deviations of the timesum
