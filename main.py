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
import math
import os.path
import pickle

#Event = namedtuple('Event', ['u', 'v', 'w', 'timesum', 'c1', 'c2', 'groupNumber'])
Event = namedtuple('Event', ['u', 'v', 'w', 'groupNumber'])
#neg_pitch = namedtuple('Pitch', ['u', 'v', 'w'])
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
        for groupNumber, data in data.items() if groupNumber < 100000 ))

def calculateChannelTimesums(data, channel1, channel2):
    #Calculates all possible combinations of channel1 + channel2 within a group
    return list(itertools.chain.from_iterable( calculateGroupTimesums(data, channel1, channel2, groupNumber) 
        for groupNumber, data in data.items() if groupNumber < 100000 ))

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
    return ((u == None) + (v == None) + (w == None)) <= 0
    
def channelCount(u, v, w):
    return (len(u) > 0) + (len(v) > 0) + (len(w) > 0)

def headOrNone(list):
    if len(list) > 0:
        return list[0]
    else:
        return None

def getEvent(u, v, w, groupNumber):
    return Event(headOrNone(u), headOrNone(v), headOrNone(w), groupNumber)

def getDiffsForMcp(m, groupData, channel1, channel2, lower, upper):
    diffs = [ getDiffs(t1[0] - m, t2[0] - m, lower, upper)
            for t1 in readGroup(groupData, channel1).as_matrix(["time_ns"])
            for t2 in readGroup(groupData, channel2).as_matrix(["time_ns"]) ]
    
    return [ d for d in diffs if d != None ]
    

def readGroup(groupedData, key):
    try:
        return groupedData.get_group(key)
    except:
        return pd.DataFrame()

def calculateGroupEvents(groupData, groupNumber, bounds):
    u_lower, u_upper, v_lower, v_upper, w_lower, w_upper = bounds
    mcpData = readGroup(groupData, config.channelId.mcp)
    
    diffs = [(getDiffsForMcp(m[0], groupData, config.channelId.u1, config.channelId.u2, u_lower, u_upper),
                 getDiffsForMcp(m[0], groupData, config.channelId.v1, config.channelId.v2, v_lower, v_upper),
                 getDiffsForMcp(m[0], groupData, config.channelId.w1, config.channelId.w2, w_lower, w_upper))
        for m in mcpData.as_matrix(["time_ns"])]
    
    return [getEvent(u, v, w, groupNumber) for (u, v, w) in diffs if channelCount(u, v, w) > 2]
    #EVENTS SHOULD INCLUDE DETECTOR BUT ITS BRoKEN fOR NOW,
    #IMPLEMENT WHEN ANALYSING ALL DATA

    
    #Event = namedtuple('Event', ['t1', 't2', 'timesum', 'c1', 'c2', 'groupNumber', 'detector'])

def convertLayerPosition(Events, neg_pitch, gap, offset):
    #Want to process information for each array in xEvents 
    U_layer = []
    V_layer = []
    W_layer = []
    
    #groups = []
    n = len(Events)
    i = 0
    while (i < (n-1)):
        i = i+1
        if events[i].u != None:
            u = (neg_pitch[0]/2)*(Events[i].u[0] - Events[i].u[1])+offset[0]
            #U_nogap.append(u)
            if (u < 0):
                U = u - (gap[0]/2)
                U_layer.append(u)
            else:
                U = u + (gap[0]/2)
                U_layer.append(U)
        else:
            u = 0
            U_layer.append(u)
        if events[i].v != None:
            v = (neg_pitch[1]/2)*(Events[i].v[0] - Events[i].v[1])+offset[1]
            #V_nogap.append(V)
            if (v < 0):
                V = v - (gap[1]/2)
                V_layer.append(v)
            else:
                V = v + (gap[1]/2)
                V_layer.append(V)
        else:
            v = 0
            V_layer.append(v)
        if events[i].w != None:
            w = (neg_pitch[2]/2)*(Events[i].w[0] - Events[i].w[1])+offset[2]
            if (w < 0):
                W = w - (gap[2]/2)
                W_layer.append(w)
            else:
                W = w + (gap[2]/2)
                W_layer.append(W)
        else:
            w = 0
            W_layer.append(w)
            
        #group = Events[i][5]
        #groups.append(group)
        
        
    return (U_layer, V_layer, W_layer)

def convertCartesian(layer_info):
    UV_x = []
    UV_y = []
    UW_x = []
    UW_y = []
    VW_x = []
    VW_y = []
    n = len(layer_info[1])
    i = 0
    while (i < (n-1)):
        i = i+1
        #UV reconstruction
        if layer_info[0][i] != 0 and layer_info[1][i] != 0:
            x = layer_info[0][i]
            UV_x.append(x)
            y = (1/math.sqrt(3))*(-layer_info[0][i]+2*layer_info[1][i])
            UV_y.append(y)
        #UW reconstruction
        if layer_info[0][i] != 0 and layer_info[2][i] != 0:
            x = layer_info[0][i]
            UW_x.append(x)
            y = (1/math.sqrt(3))*(2*layer_info[2][i]+layer_info[0][i])
            UW_y.append(y)
        #VW reconstruction
        if layer_info[1][i] != 0 and layer_info[2][i] != 0:
            x = layer_info[1][i]-layer_info[2][i]
            VW_x.append(x)
            y = (1/math.sqrt(3))*(layer_info[2][i]+layer_info[1][i])
            VW_y.append(y)
            
    UV = (UV_x, UV_y)
    UW = (UW_x, UW_y)
    VW = (VW_x, VW_y)
    
    plt.figure()
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(UV_x, UV_y, 'rx', label='UV')
    plt.plot(UW_x, UW_y, 'bx', label='UW')
    plt.plot(VW_x, VW_y, 'gx', label='VW')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.xlim(-50, 50) 
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(UV_x, UV_y, 'bx', label='UV')
    plt.plot(UW_x, UW_y, 'bx', label='UW')
    plt.plot(VW_x, VW_y, 'bx', label='VW')
    plt.show()
    

    X = np.concatenate((UV_x,UW_x),axis=0)
    X = np.concatenate((X,VW_x),axis=0)
    Y = np.concatenate((UV_y,UW_y),axis=0)
    Y = np.concatenate((Y,VW_y),axis=0)

    
    
    plt.figure()
    plt.hist2d(X, Y, bins=200)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()
    return UV, UW, VW

#calculateCartesianPosition(U_layer, V_layer, W_layer):
    #Calculate the UV UW and VW cooridinates for each group. 
    
   # return
    
#%%
#Calculate the timesums for each channel
#uFit, vFit, wFit = calculateTimeSums(negData, config.detectorId.neg)
uFit, vFit, wFit = ([  18.051664,    119.28900662,    1.6100861 ],
                    [  14.79397714,  120.46929475,    1.975484  ],
                    [  11.07192193,  114.65234937,    2.89525712])
print("Peak Height, Peak position, Standard Deviation")
print(np.array([uFit, vFit, wFit]))
#Find the limits of our timesums
#For now this is within 3 sd of our timesums
#%%
#Now we want to calculate the events
#Make sure that the layer information is within 3 standard deviations of the timesum    
events = None
if os.path.isfile('events.pickle'):
    with open('events.pickle', 'rb') as f:
        events = pickle.load(f)
else:
    events = calculateEvents(negData, uFit, vFit, wFit)
    with open('events.pickle', 'wb') as f:
        pickle.dump(events, f)
        
print(events)

def compareCoords(coord1, coord2, name1, name2, axisName):    
    points = np.array(coord1) - coord2
    plt.figure()
    n, bins, patches = plt.hist(points, bins=80)
    plt.title(name1 + " - " + name2 + " " + axisName)
    plt.xlabel(axisName + " diff (mm)")
    plt.ylabel("Counts")
    
    #bins is the edges of each bin, and therefore one extra length
    midpts = (bins[:-1] + bins[1:])/2
    #Initial guesses
    a_u = np.amax(n)
    x0_u = midpts[np.argmax(n)]
    sigma_u = 10
    uFit,pcov = curve_fit(gaus,midpts,n,p0=[a_u,x0_u,sigma_u])
    plt.plot(midpts, gaus(midpts,*uFit), 'ro:')
    
    plt.show()
    return (uFit[1], uFit[2])
    

def analyseLayerPositions(neg_pitch, neg_offset, gap):
    print(neg_pitch)
    print(gap)
    print(neg_offset)
    layer_info = convertLayerPosition(events, neg_pitch, gap, neg_offset)
    print("converted layer pos")
    
    #Check layer sums make sense:
    plt.figure()
    plt.hist(layer_info[0], bins=80, label='U')
    plt.hist(layer_info[1], bins=80, label='V')
    plt.hist(layer_info[2], bins=80, label='W')
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    
    #%%
    #Convert to cartesian co-ordinates, need group info for this
    
    UV, UW, VW = convertCartesian(layer_info)
    print("converted to cartesian")
        
    return (
        compareCoords(UV[0], VW[0], "UV", "VW", "x"),
        compareCoords(UW[0], VW[0], "UW", "VW", "x"),
        compareCoords(UV[1], UW[1], "UV", "UW", "y"),
        compareCoords(UV[1], VW[1], "UV", "VW", "y"),
        compareCoords(UW[1], VW[1], "UW", "VW", "y")
    )
    
    
#%% loop over co-ordinates
#np.arange(0.2, 1.2, 0.07)
#Negative Detector Constants as given by Dans calibration software
neg_pitches = ([0.3043*2], [0.2963*2] , [0.3003*2])
neg_offsets = ([-1.5], [1], [-2])
gaps = ([8.3894], [7.3063], [7.50289])
#neg_pitch = (0.3043*2, 0.2963*2, 0.3003*2)
#neg_offset = (-0.632, 0.0114, 0.0171)
#gap = (8.3894, 7.3063, 7.50289)
#U_neg_pitch = 0.3035*2 
#V_neg_pitch = 0.2959*2
#W_neg_pitch = 0.2999*2
#U_offset = -0.631
#V_offset = 0.0124
#W_offset = 0.0203
#U_gap = 8.407
#V_gap = 7.2827
#W_gap = 7.4789
print("generating param sets")
param_sets = [((np_0, np_1, np_2), (no_0, no_1, no_2), (g_0, g_1, g_2))
    for np_0 in neg_pitches[0] for np_1 in neg_pitches[1] for np_2 in neg_pitches[2]
    for no_0 in neg_offsets[0] for no_1 in neg_offsets[1] for no_2 in neg_offsets[2]
    for g_0 in gaps[0] for g_1 in gaps[1] for g_2 in gaps[2]]

all_fits = [(analyseLayerPositions(neg_pitch, neg_offset, gap), neg_pitch, neg_offset, gap)
    for neg_pitch, neg_offset, gap in param_sets]

uvvw_x = [std for ((x0, std), _, _, _, _), neg_pitch, neg_offset, gap in all_fits]
uwvw_x = [std for (_, (x0, std), _, _, _), neg_pitch, neg_offset, gap in all_fits]
uvuw_y = [std for (_, _, (x0, std), _, _), neg_pitch, neg_offset, gap in all_fits]
uvvw_y = [std for (_, _, _, (x0, std), _), neg_pitch, neg_offset, gap in all_fits]
uwvw_y = [std for (_, _, _, _, (x0, std)), neg_pitch, neg_offset, gap in all_fits]
xs = [neg_offset[2] for _, neg_pitch, neg_offset, gap in all_fits]

plt.figure()
plt.plot(xs, uvvw_x, label='uvvw x', marker='o')
plt.plot(xs, uwvw_x, label='uwvw x', marker='o')
plt.plot(xs, uvuw_y, label='uvuw y', marker='o')
plt.plot(xs, uvvw_y, label='uvvw y', marker='o')
plt.plot(xs, uwvw_y, marker='o', label='uwvw y')
plt.legend()
plt.show()