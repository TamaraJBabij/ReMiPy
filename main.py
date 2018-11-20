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

Event = namedtuple('Event', ['u', 'v', 'w', 'timesum', 'c1', 'c2', 'groupNumber'])

config = configLoader.load("config.json")
print(config)

trees = rootTreeOps.openFromDirectory(r"C:\Users\Tamara\Desktop\ReMi20169121531")
fullData = rootTreeOps.asDataFrame(trees)
fullData["time_ns"] = 32000 - 0.5*fullData[b'Time']
fullData["channel"] = pd.Series(config.channels[x].channelId for x in fullData[b'Channel'])
fullData["detector"] = pd.Series(config.channels[x].detectorId for x in fullData[b'Channel'])
groupByDetector = fullData.groupby("detector")
negData = groupByDetector.get_group(config.detectorId.neg)

#Negative Detector Constants as given by Dans calibration software
U_neg_pitch = 0.3035*2 
V_neg_pitch = 0.2959*2
W_neg_pitch = 0.2999*2
U_offset = -0.631
V_offset = 0.0124
W_offset = 0.0203
W_gap = 0.0203
U_gap = 8.407
V_gap = 7.2827
W_gap = 7.4789

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
    uFit,pcov = curve_fit(gaus,bins,n,p0=[a_u,x0_u,sigma_u])
    plt.plot(bins, gaus(bins,*uFit), 'ro:')
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
    vFit,pcov = curve_fit(gaus,bins,n,p0=[a_v,x0_v,sigma_v])
    plt.plot(bins, gaus(bins,*vFit), 'ro:')
    
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
    wFit,pcov = curve_fit(gaus,bins,n,p0=[a_w,x0_w,sigma_w])
    plt.plot(bins, gaus(bins,*wFit), 'ro:')
    
    plt.show()
    return (uFit, vFit, wFit)


def calculateEvents(data, uFit, vFit, wFit):  
    u_upper = uFit[1]+3*(uFit[-1]) 
    u_lower = uFit[1]-3*(uFit[-1])
    v_upper = vFit[1]+3*(vFit[-1]) 
    v_lower = vFit[1]-3*(vFit[-1])
    w_upper = wFit[1]+3*(wFit[-1]) 
    w_lower = wFit[1]-3*(wFit[-1])
    
    #Calculates the time sums for each layer of the positive or negative detector
    groupByNumber = data.groupby(b'GroupNumber')
    #print(groupByNumber)
    groups = { name: groupByNumber.get_group(name).groupby('channel') for (name, _) in groupByNumber }
    #print(groups)
    bounds = (u_lower, u_upper, v_lower, v_upper, w_lower, w_upper)
    data = calculateChannelEvents(groups, bounds)
    return data
    
def calculateChannelEvents(data, bounds):
    return list(itertools.chain.from_iterable( calculateGroupEvents(data, groupNumber, bounds) 
        for groupNumber, data in data.items() if groupNumber < 100 ))

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
        #print(diffs)
        return [d1 + d2 for d1, d2 in diffs if d1 > 0 and d1 < 500 and d2 > 0 and d2 < 500]
    except:
        return []
    
def getDiffs(d1, d2, lowerTs, upperTs):
    if d1 > 0 and d1 < 500 and d2 > 0 and d2 < 500 and (d1 + d2) > lowerTs and (d1 + d2) < upperTs:
        return (d1, d2)
    else:
        return None
    
def isReconstructable(u, v, w):
    return ((u == None) + (v == None) + (w == None)) <= 1
    
def calculateGroupEvents(groupData, groupNumber, bounds):
    u_lower, u_upper, v_lower, v_upper, w_lower, w_upper = bounds
    try:
        mcpData = groupData.get_group(config.channelId.mcp)
        
        diffs = [(getDiffs(u1[0] - m[0], u2[0] - m[0], u_lower, u_upper), 
                  getDiffs(v1[0] - m[0], v2[0] - m[0], v_lower, v_upper),
                  getDiffs(w1[0] - m[0], w2[0] - m[0], w_lower, w_upper))
            for u1 in groupData.get_group(config.channelId.u1).as_matrix(["time_ns"])
            for u2 in groupData.get_group(config.channelId.u2).as_matrix(["time_ns"])
            for v1 in groupData.get_group(config.channelId.v1).as_matrix(["time_ns"])
            for v2 in groupData.get_group(config.channelId.v2).as_matrix(["time_ns"])
            for w1 in groupData.get_group(config.channelId.w1).as_matrix(["time_ns"])
            for w2 in groupData.get_group(config.channelId.w2).as_matrix(["time_ns"])
            for m in mcpData.as_matrix(["time_ns"])]
        #EVENTS SHOULD INCLUDE DETECTOR BUT ITS BRoKEN fOR NOW,
        #IMPLEMENT WHEN ANALYSING ALL DATA
        return [Event(u, v, w, groupNumber)
            for u, v, w in diffs if isReconstructable(u, v, w)]
    except:
        return []
    
    #Event = namedtuple('Event', ['t1', 't2', 'timesum', 'c1', 'c2', 'groupNumber', 'detector'])

def convertLayerPosition(Events, X_neg_pitch, X_gap, X_offset):
    #Want to process information for each array in xEvents 
    X_nogap = []
    X_layer = []
    groups = []
    n = len(Events)
    i = 0
    while (i < (n-1)):
        i = i+1
        #u1 - u2
        x = (X_neg_pitch/2)*(Events[i][0] - Events[i][1])+X_offset
        X_nogap.append(x)
        group = Events[i][5]
        groups.append(group)
        if (x < 0):
            X = x - (X_gap/2)
            X_layer.append(X)
        else:
            X = x + (X_gap/2)
            X_layer.append(X)
    return (X_layer,groups)

#calculateCartesianPosition(U_layer, V_layer, W_layer):
    #Calculate the UV UW and VW cooridinates for each group. 
    
   # return
    
#%%
#Calculate the timesums for each channel
uFit, vFit, wFit = calculateTimeSums(negData, config.detectorId.neg)
print("Peak Height, Peak position, Standard Deviation")
print(np.array([uFit, vFit, wFit]))
#Find the limits of our timesums
#For now this is within 3 sd of our timesums
#%%
#Now we want to calculate the events
#Make sure that the layer information is within 3 standard deviations of the timesum    
events = calculateEvents(negData, uFit, vFit, wFit)
#U_layer = convertLayerPosition(uEvents, U_neg_pitch, U_gap, U_offset)
#V_layer = convertLayerPosition(vEvents, V_neg_pitch, V_gap, V_offset)
#W_layer = convertLayerPosition(wEvents, W_neg_pitch, W_gap, W_offset)

#Check layer sums make sense:
#plt.figure()
#print("U_layer")
#plt.hist(U_layer[0], bins=80, label='U')
#print("V_layer")
#plt.hist(V_layer[0], bins=80, label='V')
#print("W_layer")
#plt.hist(W_layer[0], bins=80, label='W')
#plt.xlabel("Position (mm)")
#plt.ylabel("Counts")
#plt.legend()
#plt.show()

#%%
#Convert to cartesian co-ordinates, need group info for this