# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:51:22 2021

@author: ib_hansen
"""

from tablesalt.topology import stationoperators
from tablesalt.topology import pathfinder

OPGETTER = stationoperators.StationOperators(
    'kystbanen', 'local', 'metro', 'suburban', 'fjernregional'
    )

OPGETTER.station_pair(8600654,123999999999,format='operator_id')
