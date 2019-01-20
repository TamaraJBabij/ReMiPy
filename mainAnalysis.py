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
from scipy import stats
import math
import os.path
import pickle
from enum import IntEnum

#Event = namedtuple('Event', ['u', 'v', 'w', 'timesum', 'c1', 'c2', 'groupNumber'])
Event = namedtuple('Event', ['u', 'v', 'w', 'mcpTime', 'detector', 'particleType'])
Group = namedtuple('Group', ['events', 'groupNumber'])
ParticleType = IntEnum("ParticleType", ["ElemParticle", "Ion1", "Ion2", "Other"])
#neg_pitch = namedtuple('Pitch', ['u', 'v', 'w'])
config = configLoader.load("config.json")
print(config)

def getParticleType(time):
    if time > config.particleTimes["elemParticle"]["min"] and time < config.particleTimes["elemParticle"]["max"]:
        return ParticleType.ElemParticle
    if time > config.particleTimes["ion1"]["min"] and time < config.particleTimes["ion1"]["max"]:
        return ParticleType.Ion1
    if time > config.particleTimes["ion2"]["min"] and time < config.particleTimes["ion2"]["max"]:
        return ParticleType.Ion2
    return ParticleType.Other
trees = rootTreeOps.openFromDirectory(r"C:\Users\Tamara\Desktop\ReMi20169121531")
fullData = rootTreeOps.asDataFrame(trees)
#add b'Time' for operation on laptop
fullData["time_ns"] = 32000 - 0.5*fullData['Time']
fullData["particle_type"] = pd.Series(int(getParticleType(t)) for t in fullData['time_ns'])
fullData["channel"] = pd.Series(int(config.channels[x].channelId) for x in fullData['Channel'])
fullData["detector"] = pd.Series(int(config.channels[x].detectorId) for x in fullData['Channel'])
groupByDetector = fullData.groupby("detector")
posData = groupByDetector.get_group(config.detectorId.pos)
negData = groupByDetector.get_group(config.detectorId.neg)


plt.figure()
plt.title("pos times")
n, bins, patches = plt.hist(posData["time_ns"], 100)
plt.show()

plt.figure()
plt.title("neg times")
n, bins, patches = plt.hist(negData["time_ns"], 100)
plt.show()

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def calculateTimeSums(data, detectorId):
    #Calculates the time sums for each layer of the positive or negative detector
    groupByNumber = data.groupby('GroupNumber')
    groups = { name: groupByNumber.get_group(name).groupby('channel') for (name, _) in groupByNumber }
    uData = calculateChannelTimesums(groups, config.channelId.u1, config.channelId.u2)
    vData = calculateChannelTimesums(groups, config.channelId.v1, config.channelId.v2)
    wData = calculateChannelTimesums(groups, config.channelId.w1, config.channelId.w2)
    
    #U DATA
    plt.figure()
    plt.title("U layer")
    n, bins, patches = plt.hist(uData, 80, (0, 200))
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
    n, bins, patches = plt.hist(vData, 80, (0, 200))
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
    n, bins, patches = plt.hist(wData, 80, (0, 200))
    bins = (bins[:-1] + bins[1:])/2
    #Initial guesses
    a_w = np.amax(n)
    x0_w = bins[np.argmax(n)]
    sigma_w = 2
    wFit,pcov = curve_fit(gaus,bins,n,p0=[a_w,x0_w,sigma_w])
    plt.plot(bins, gaus(bins,*wFit), 'ro:')
    
    plt.show()
    
    u_upper = uFit[1]+2*(uFit[-1]) 
    u_lower = uFit[1]-2*(uFit[-1])
    v_upper = vFit[1]+2*(vFit[-1]) 
    v_lower = vFit[1]-2*(vFit[-1])
    w_upper = wFit[1]+2*(wFit[-1]) 
    w_lower = wFit[1]-2*(wFit[-1])
    
    return ((u_lower, u_upper), (v_lower, v_upper), (w_lower, w_upper))

def calculateGroups(data, posBounds, negBounds):
    groupByNumber = data.groupby('GroupNumber')
    groups = [ (groupNumber, groupByNumber.get_group(groupNumber).groupby('channel')) for (groupNumber, _) in groupByNumber ]
    return [ Group(calculateGroupEvents(group, groupNumber, posBounds, negBounds), groupNumber) for (groupNumber, group) in groups ]

def calculateEvents(data, bounds):  
    #Calculates the time sums for each layer of the positive or negative detector
    groupByNumber = data.groupby('GroupNumber')
    groups = { name: groupByNumber.get_group(name).groupby('channel') for (name, _) in groupByNumber }
    data = calculateChannelEvents(groups, bounds)
    return data
    
def calculateChannelEvents(data, bounds):
    return list(itertools.chain.from_iterable( calculateGroupEvents(data, groupNumber, bounds) 
        for groupNumber, data in data.items() if groupNumber < 100000 ))

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
        return [d1 + d2 for d1, d2 in diffs if d1 > 0 and d1 < 130 and d2 > 0 and d2 < 130]
    except:
        return []
    
def getDiffs(d1, d2, lowerTs, upperTs):
    if d1 > 0 and d1 < 130 and d2 > 0 and d2 < 130 and (d1 + d2) > lowerTs and (d1 + d2) < upperTs:
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

def getEvent(u, v, w, groupNumber, detectorID, particleType):
    return Event(headOrNone(u), headOrNone(v), headOrNone(w), groupNumber, int(detectorID), int(particleType))

def getDiffsForMcp(m, detector, groupData, channel1, channel2, bounds):
    diffs = [ getDiffs(t1 - m, t2 - m, bounds[0], bounds[1])
            for [t1, d1] in readGroup(groupData, channel1).as_matrix(["time_ns", "detector"])
            for [t2, d2] in readGroup(groupData, channel2).as_matrix(["time_ns", "detector"]) 
            if d1 == detector and d2 == detector]
    
    return [ d for d in diffs if d != None ]
    

def readGroup(groupedData, key):
    try:
        return groupedData.get_group(key)
    except:
        return pd.DataFrame()

def boundsForDet(detectorId, posBounds, negBounds, layerNumber):
    if detectorId == config.detectorId.pos:
        return posBounds[layerNumber]
    else:
        return negBounds[layerNumber]

def calculateGroupEvents(groupData, groupNumber, posBounds, negBounds):
    mcpData = readGroup(groupData, config.channelId.mcp)
    #print(mcpData)
    #if not mcpData.empty:
        #print(mcpData['detector']) 
    diffs = [(getDiffsForMcp(m[0], m[2], groupData, config.channelId.u1, config.channelId.u2, boundsForDet(m[2], posBounds, negBounds, 0)),
                 getDiffsForMcp(m[0], m[2], groupData, config.channelId.v1, config.channelId.v2, boundsForDet(m[2], posBounds, negBounds, 1)),
                 getDiffsForMcp(m[0], m[2], groupData, config.channelId.w1, config.channelId.w2, boundsForDet(m[2], posBounds, negBounds, 2)),
                 m[0], m[1], m[2])
        for m in mcpData.as_matrix(["time_ns", "particle_type", "detector"])]
    
    
    return [getEvent(u, v, w, mcpTime, det, pType) 
        for (u, v, w, mcpTime, pType, det) in diffs if pType == ParticleType.Ion1 or channelCount(u, v, w) > 1]
    #EVENTS SHOULD INCLUDE DETECTOR BUT ITS BRoKEN fOR NOW,
    #IMPLEMENT WHEN ANALYSING ALL DATA

    
    #Event = namedtuple('Event', ['t1', 't2', 'timesum', 'c1', 'c2', 'groupNumber', 'detector'])

def convertLayerPosition(events, neg_pitch, gap, offset):
    #Want to process information for each array in xEvents 
    U_layer = []
    V_layer = []
    W_layer = []
    
    #groups = []
    n = len(events)
    i = 0
    while (i < (n-1)):
        i = i+1
        if events[i].u != None:
            u = (neg_pitch[0]/2)*(events[i].u[0] - events[i].u[1])+offset[0]
            #U_nogap.append(u)
            if (u < -1):
                U = u - (gap[0]/2)
                U_layer.append(U)
            else:
                U = u + (gap[0]/2)
                U_layer.append(U)
        else:
            u = 0
            U_layer.append(u)
        if events[i].v != None:
            v = (neg_pitch[1]/2)*(events[i].v[0] - events[i].v[1])+offset[1]
            #V_nogap.append(V)
            if (v < 1):
                V = v - (gap[1]/2)
                V_layer.append(V)
            else:
                V = v + (gap[1]/2)
                V_layer.append(V)
        else:
            v = 0
            V_layer.append(v)
        if events[i].w != None:
            w = (neg_pitch[2]/2)*(events[i].w[0] - events[i].w[1])+offset[2]
            if (w < -1):
                W = w - (gap[2]/2)
                W_layer.append(W)
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
    plt.plot(0,0)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.xlim(-50, 50) 
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot(UV_x, UV_y, 'bx', label='UV')
    plt.plot(UW_x, UW_y, 'bx', label='UW')
    plt.plot(VW_x, VW_y, 'bx', label='VW')
    plt.plot(0,0)
    plt.show()
    return UV, UW, VW


#calculateCartesianPosition(U_layer, V_layer, W_layer):
    #Calculate the UV UW and VW cooridinates for each group. 
    
   # return
def loadOrCalculate(name, fn):
    if os.path.isfile(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        result = fn()
        with open(name, 'wb') as f:
            pickle.dump(result, f)
        return result
#%%
#Calculate the timesums for each channel
posBounds = loadOrCalculate('posBounds.pickle', lambda: calculateTimeSums(posData, config.detectorId.pos))
negBounds = loadOrCalculate('negBounds.pickle', lambda: calculateTimeSums(negData, config.detectorId.neg))
print("Bounds")
print(np.array(posBounds))
print(np.array(negBounds))
#Find the limits of our timesums
#For now this is within 3 sd of our timesums
#%%
#Now we want to calculate the events
#Make sure that the layer information is within 3 standard deviations of the timesum    
groups = loadOrCalculate('groups.pickle', lambda: calculateGroups(fullData, posBounds, negBounds))

print(groups)

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
    print(uFit)
    return (uFit[1], abs(uFit[2]))
    

def analyseLayerPositions(events, neg_pitch, neg_offset, gap):
    print(neg_pitch)
    print(gap)
    print(neg_offset)
    layer_info = convertLayerPosition(events, neg_pitch, gap, neg_offset)
    print("converted layer pos")
    
    #Check layer sums make sense:
    plt.figure()
    plt.hist(layer_info[0], bins=80, label='U')
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(layer_info[1], bins=80, label='V')
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(layer_info[2], bins=80, label='W')
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    
    #%%
    #Convert to cartesian co-ordinates, need group info for this
    
    UV, UW, VW = convertCartesian(layer_info)
    print("converted to cartesian")
    
