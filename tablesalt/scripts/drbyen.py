# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:46:11 2020

@author: alkj
"""


import numpy as np

from tablesalt import StoreReader
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser


def _get_exception_stations(store, *uic):
    
    reader = StoreReader(store)
    stops = reader.get_data('stops')
    
    tripkeys = stops[np.isin(stops[:, 2], list(uic))][:, 0]
    trips = stops[np.isin(stops[:, 0], tripkeys)]
      
    return trips

def main(store):
    parser = TableArgParser('year')
    args = parser.parse()
    year = args['year']
    
    
    store_loc = find_datastores(r'H://')
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']


       
    return single_output