# -*- coding: utf-8 -*-
import json
from enum import Enum
from model.config import Config, Channel

def load(file):
    with open(file) as f:
        data = json.load(f)

    channelIds = set(x["channel"] for x in data["channels"])
    detectorIds = set(x["detector"] for x in data["channels"])
    
    channelId = Enum("ChannelId", list(channelIds))
    detectorId = Enum("DetectorId", list(detectorIds))
    channels = {c["id"]: Channel(channelId[c["channel"]], detectorId[c["detector"]]) for c in data["channels"]}
    
        
    
    return Config(channelId, detectorId, channels)
