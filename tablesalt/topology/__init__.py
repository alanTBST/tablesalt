# -*- coding: utf-8 -*-
"""

The topology subpackage contains classes and functions used to interact
with data from the transport network.py


It has:
    * methods to download network and routing information from Rejseplan
    * ways to retrieve operators servicing stations
    * a method to validate whether a given operator can service a trip leg
    * a ZoneProperties class that provides various information regarding
    zones traveled through
    * a class to share zone work between operators on a trip

as well as other topological tools
"""

from .tools import RejseplanLoader
from .zonegraph import ZoneGraph
from .pathfinder import ZoneProperties, BORDER_STATIONS, ZoneSharer
from .stationoperators import StationOperators
