# -*- coding: utf-8 -*-
from collections import namedtuple

Channel = namedtuple('Channel', ['channelId', 'detectorId'])
Config = namedtuple('Config', ['channelId', 'detectorId', 'channels', 'properties', 'particleTimes'])