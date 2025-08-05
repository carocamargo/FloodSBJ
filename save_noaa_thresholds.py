#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:51:06 2025

@author: carocamargo
"""
import pickle
from utils_work import noaaFlood
def save_dict(data,path,filename):
    
    outname = "{}{}.pkl".format(path,filename)
    with open(outname, "wb") as f:
        pickle.dump(data, f)
        
stations = [
            '8443970', # boston
#             '8447435',# chatam
            '8449130', # nantucket
            '8447930', # woods hole
            '8447386', # fall river
            '8452660', # Newport
            '8510560', # Montauk
            '8461490', # New London
            '8465705', # New Haven
            '8467150', # brigeport
            '8516945',# kings point
            '8531680', # sandy hook
            '8534720' # ; % Atlantic City, NJ
            ]
path = '/Users/carocamargo/Documents/data/floodSBJ/'

datum='MHHW'
for datum in ['MHHW','MSL']:
    thresholds = {station:noaaFlood(station,datum=datum) for station in stations}
    save_dict(thresholds,path,'noaaFlood_{}'.format(datum))