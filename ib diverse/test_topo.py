# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 00:22:45 2021

@author: ib_hansen
"""
import networkx as nx

from tablesalt.topology import tools

#%% zone neighour graph 

xx = tools.EdgeMaker()
edges = xx.make_edges()
edges.keys()
edges['idx']      # zone -> node
edges['rev_idx']  # node -> zone 
edges['adj_array'].shape

G = nx.from_numpy_array(edges['adj_array'])

#%% Translates StopPointNr to zones for Sjælland 

takstzones = tools.TakstZones()

stop_zone_map = takstzones.stop_zone_map()

StopPointNr_from_zone_map = set(stop_zone_map.keys())
