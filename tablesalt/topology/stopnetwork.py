# -*- coding: utf-8 -*-
"""
Module holding the classes to deal with stop points in the Danish public transit network

The main useful class is StopsList.
An instance of a StopsList contains instances of the Stop class 

An instance of a StopsList can be instantiated as:
.. highlight:: python
.. code-block:: python

# create an instance with the default stop network
stop_list = StopsList()

# access the first stop in the list of stops
stop1 = stop_list[0]

# create a new instance of the StopsList class with only railway stations
rail_stops = stop_list.rail_stops()

# create a new instance of the StopsList class with only bus stops
bus_stops = stop_list.bus_stops()
"""

import copy
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import InitVar, asdict, dataclass, field
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import (Any, ClassVar, Dict, List, Optional, Set, Tuple, TypedDict,
                    Union, Iterable)

import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, MultiLineString, Point  # type: ignore

HERE = Path(__file__).parent

class StopElements(TypedDict):
    """
    TypedDict class for type hints in loading stops.json
    """

    stop_id: int
    stop_code: str
    stop_name: str
    stop_desc: str
    stop_lat: float
    stop_lon: float
    location_type: int

    parent_station: Optional[Union[str, int]]
    wheelchair_boarding: Union[str, int]
    platform_code: Union[str, int]

def _load_stopslist(filepath: Union[str, Path]) -> List[StopElements]:
    """
    load the stops from a json file

    :param filepath: the path to the stops.json file
    :type filepath: str
    :return: a list of dictionaries conforming to StopElements
    :rtype: List[StopElements]

    """

    stoplist = []
    try:
        with open(filepath) as fp:
            data = json.load(fp)
    except (JSONDecodeError, UnicodeDecodeError, TypeError):
        raise ValueError("File must be a json file")
    
    for k, v in data.items():
        stop = v.copy()
        stop['stop_id'] = int(k)
        stoplist.append(stop)
    return stoplist


def _load_border_stations():

    fp = HERE / 'borderstations.json'

    with open(fp) as f:
        border_stations = json.load(f)

    return {int(k): v for k, v in border_stations.items()}


def _load_alternate_stations() -> Dict[int, Tuple[int, ...]]:

    fp = HERE / 'alternatenumbers.json'

    with open(fp) as f:
        alternate_numbers = json.load(f)
    alternate_numbers = {int(k): v for k, v in alternate_numbers.items()}

    out = defaultdict(list)

    for k, v in alternate_numbers.items():
        out[v].append(k)

    return {k: tuple(v) for k, v in out.items()} 

# only for denmark
BORDER_STATIONS = _load_border_stations()
ALTERNATE_STATIONS = _load_alternate_stations()

@dataclass
class Stop:
    
    """
    A stop dataclass that holds data from a stop point from gtfs data

    :param stop_id: the uic number or stop number of the stop
    :type stop_id: int

    :param : stop_code: the code of the stop
    :type : TYPE

    """

    stop_id: int
    stop_code: str
    stop_name: str
    stop_desc: str
    stop_lat: float
    stop_lon: float
    location_type: int

    parent_station: Optional[Union[str, int]] = None
    wheelchair_boarding: Optional[Union[str, int]] = None
    platform_code: Optional[Union[str, int]] = None

    operators: InitVar[str] = None
 
    @classmethod
    def from_dict(cls, obj: StopElements) -> 'Stop':
        """
        create a Stop object from a dictionary of StopElements

        :param obj: a dictionary conforming to typeddict StopElements
        :type obj: StopElements
        :return: an instance of a Stop
        :rtype: Stop
        """

        return cls(**obj)

    @property 
    def alternate_stop_ids(self) -> Tuple[int, ...]:
        """Return the possible alternate numbers for the stop.
        an empty tuple is returned if there are not alternate stop 
        numbers

        :return: a tuple of integers of alternate stop numbers
        :rtype: Tuple[int, ...]
        """
        return ALTERNATE_STATIONS.get(self.stop_id, ())

    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        return the coordinates of the stop

        :return: a tuple of latitude, longitude
        :rtype: Tuple[float, float]
        """
        return self.stop_lon, self.stop_lat

    @property
    def is_border(self) -> bool:
        """determine if the stop is on a tariff zone border

        :return: True if the stop is 
        :rtype: bool
        """
        return (self.stop_id in BORDER_STATIONS or 
            any(x in BORDER_STATIONS for x in self.alternate_stop_ids))
    
    def as_point(self) -> Point:
        """return the stop as a shapely Point"""

        return Point(self.coordinates)
    
    @property
    def mode(self) -> str:
        # denmark specific
        # try make it based on route with routes.txt, route_type

        stopid_str = str(self.stop_id)
        if len(stopid_str) == 7 or (stopid_str.startswith('86') and len(stopid_str) == 9):
            return 'rail'
        return 'bus'


    def __hash__(self):

        return hash(self.stop_id) + hash(self.stop_name)



class StopsList(Iterator):

    DEFAULT_PATH: ClassVar[Path] = Path(HERE, 'stops.json')
    
    def __init__(
            self,
            *stops: Stop
            ) -> None:
        """Load a list of Stops from a json file

        :param stops: the path to a stops.json file, defaults to None. If 
            no file is given, the default stops are used
        
        :type stops: Stop
        """

        self.stops: List[Stop]
        self.stops = list(stops)
        self._stops_dict = None

        stop_ids: Set[int] = {
            x.stop_id for x in self.stops
        }
        alternate_stops = set(chain.from_iterable(
            [x.alternate_stop_ids for x in self.stops]
                )
            )
        self._stop_ids = stop_ids.union(alternate_stops)
        

        self._index = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stops={self.stops})"

    def __getitem__(self, index) -> Stop:
        # update to deal with slicing and returning a StopList
        # right now slice returns normal list of Stop
        return self.stops.__getitem__(index)

    def __iter__(self) -> 'StopsList':
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.stops):
            res = self.stops[self._index]
            self._index += 1
            return res
        raise StopIteration

    def __len__(self) -> int:
        return len(self.stops)
    
    def __contains__(self, val):

        return val in self._stop_ids
    
    @property
    def stops_dict(self) -> Dict[int, Stop]:
        """return the StopsList as a dictionary with keys 
        as stop_ids and values as Stop instances

        :return: dictionary of Stops
        :rtype: Dict[int, Stop]
        """

        # TODO include alternate stop numbers!!!!!!!!!!!!!!!!!!!!11
        if self._stops_dict is None:
            d = {}
            for x in self.stops:
                alt_ids = x.alternate_stop_ids
                d[x.stop_id] = x
                if alt_ids:
                    for i in alt_ids:
                        d[i] = x

            self._stops_dict = d

        return self._stops_dict

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> 'StopsList':
        """Create an instance of a StopsList from a json file

        :param filepath: the path to the json file, defaults to None. If filepath is None, 
                         the default Danish stops network is used.
        :type filepath: Optional[str], optional
        :return: an instance of a StopsList
        :rtype: StopsList
        """
        
        filepath = filepath if filepath is not None else StopsList.DEFAULT_PATH
        _data = _load_stopslist(str(filepath))
        stops = [Stop.from_dict(x) for x in _data]

        return cls(*stops) 

    def get_stop(self, stop_id: int) -> Stop:
        """Return a Stop from the list by stop id
        None is returned if the stop does not exist

        :param stop_id: the stop id
        :type stop_id: int
        :return: a stop
        :rtype: Stop
        """

        return self.stops_dict.get(stop_id)


    def rail_stops(self) -> 'StopsList':
        """
        Return a new instance of a StopList with only rail stops

        :return: a new onstance of a StopsList
        :rtype: StopsList
        """
        rail = copy.deepcopy((self))
        rail.stops = [x for x in rail if x.mode == 'rail']

        return rail

    def bus_stops(self) -> 'StopsList':
        """
        return a new instance of a StopsList with only bus stops

        :return: a new instance of a StopsList
        :rtype: StopsList
        """

        bus = copy.deepcopy((self))
        bus.stops = [x for x in bus if x.mode == 'bus']

        return bus

    def border_stations_and_zones(self) -> Dict[int, Tuple[int, ...]]:
        """Return a dictionary of railways stations on tariff zone borders
        with zone numbers

        :return: dictionary of station uic numbers and correcsponding zones
        :rtype: Dict[int, Tuple[int, ...]]
        """
       
        border_stops = {x for x in self if x.is_border}
        border_dict = {}
        for stop in border_stops:
            border_dict[stop.stop_id] = tuple(BORDER_STATIONS[stop.stop_id]['zones'])
            for num in stop.alternate_stop_ids:
                border_dict[num] = tuple(BORDER_STATIONS[stop.stop_id]['zones'])

        return border_dict

    @classmethod
    def update_default_stops(self):
        # update the stops in the stops.json from rejseplan gtfs
        return

    def to_dataframe(self):

        return pd.DataFrame()

    def to_geodataframe(self, crs: Optional[int] = 4326):
        
        stops = pd.DataFrame(self.stops)
        gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat)
            )
        current_proj = int(gdf.crs.to_string().split(':')[1])

        #gdf.crs = crs

        return gdf

class Line:
    def __init__(
        self, 
        line_id: int, 
        line_name: str, 
        *line_stops: Stop
        ):
        """A network line

        :param line_id: the id number of the line
        :type line_id: int
        :param line_name: the name of the line
        :type line_name: str
        :param line_stops: the stops that make up the line
        :type line_stops: Stop
        """
        
        self.line_id = line_id
        self.line_name = line_name
        self.line_stops = list(line_stops)

        stop_ids: Set[int] = {
            x.stop_id for x in self.line_stops
        }
        alternate_stops = set(chain.from_iterable(
            [x.alternate_stop_ids for x in self.line_stops]
                )
            )
        self._stop_ids = stop_ids.union(alternate_stops)


    def __repr__(self) -> str:
        ids = (x.stop_id for x in self.line_stops)
        return (f"{self.__class__.__name__}"
                f"(line_id={self.line_id},"
                f"line_name={self.line_name},"
                f"stops={', '.join(map(str, ids))}")
    
    def __iter__(self) -> 'StopsList':
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.line_stops):
            res = self.line_stops[self._index]
            self._index += 1
            return res
        raise StopIteration
  
    def __getitem__(self, index) -> Stop:
        # update to deal with slicing and returning a StopList
        # right now slice returns normal list of Stop
        return self.line_stops.__getitem__(index)
    
    def __contains__(self, stop_id):

        return stop_id in self._stop_ids
    
    @classmethod
    def from_stopslist(
        cls, 
        stoplist: StopsList, 
        line_id: int, 
        line_name: str, 
        stop_numbers: Optional[Iterable[int]] = None
        ):
        """Create an instance of a Line from an instance of a StopsList

        :param stoplist: an instance of a StopsList
        :type stoplist: StopsList
        :param line_id: the id of the line
        :type line_id: int
        :param line_name: the name of the line
        :type line_name: str
        :param stop_numbers: an iterable of stop ids to inc
        :type stop_numbers: Optional[Iterable[int]], defaults to None
        """
        stop_numbers = stop_numbers if stop_numbers is not None else []
        return 
    def to_simple_linestring():

        return NotImplemented

    def shape_length():

        return NotImplemented
    
    def plot():

        return NotImplemented


class BusLine(Line):

    def __init__(self, line_id: int, line_name: str, *line_stops: Stop):
        super().__init__(line_id, line_name, *line_stops)


class RailLine(Line):

    def __init__(self, line_id: int, line_name: str, *line_stops: Stop):
        super().__init__(line_id, line_name, *line_stops)


class RailNetwork:

    DEFAULT_PATH = HERE / 'rail_lines.json'
    
    def __init__(self, *rail_lines: RailLine) -> None:
        """A Network of rail lines
        """

        self.rail_lines = list(rail_lines)

    def __repr__(self):

        names = [x.line_name for x in self.rail_lines]

        return f"{self.__class__.__name__}(rail_lines=[{', '.join(map(str, names))}])"

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> 'RailNetwork':
        """Load a Rail Network from a json file

        :param filepath: path to a json file, defaults to None. If filepath is None, 
                         the default Danish rail network is used.
        :type filepath: Optional[str], optional
        :return: RailNetwork instance
        :rtype: RailNetwork
        """

        path = filepath if filepath is not None else RailNetwork.DEFAULT_PATH
        
        stoplist = StopsList.from_json().rail_stops()
        with open(path, 'r', encoding='iso-8859-1') as f:
            data = json.load(f)

        network = []
        for line_name, d in data.items():
            line_id = d['line_id']
            stops = (stoplist.get_stop(x) for x in d['stops'])
            stops = [x for x in stops if x is not None]
            if stops:
                line = RailLine(line_id, line_name, *stops)
                network.append(line)
        return cls(*network)


class BusNetwork:

    def __init__(self, *bus_lines: BusLine):

        self.bus_lines = list(bus_lines)

    def plot():

        return NotImplemented


class RegionNetwork:

    def __init__(self) -> None:
        pass


def line_factory():

    return 
