# -*- coding: utf-8 -*-
"""

"""

import copy
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from shapely.geometry import Point  # type: ignore

HERE = Path(__file__).parent

class StopElements(TypedDict):
    """
    TypedDict class for type hints in loading stops.json
    """

    stop_number: int
    stop_code: str
    stop_name: str
    stop_desc: str
    stop_lat: float
    stop_lon: float
    location_type: int

    parent_station: Optional[Union[str, int]]
    wheelchair_boarding: Union[str, int]
    platform_code: Union[str, int]

    is_border: Optional[bool]



def _load_stopslist(filepath: str) -> List[StopElements]:
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
        stop['stop_number'] = int(k)
        stoplist.append(stop)
    return stoplist


def _load_border_stations():

    fp = HERE / 'borderstations.json'

    with open(fp) as f:
        border_stations = json.load(f)

    return {int(k): v for k, v in border_stations.items()}

BORDER_STATIONS = _load_border_stations()

@dataclass
class Stop:
    """
    A stop dataclass that holds data from a stop point from gtfs data

    :param stop_number: the uic number or stop number of the stop
    :type stop_number: int

    :param : stop_code: the code of the stop
    :type : TYPE

    """

    stop_number: int
    stop_code: str
    stop_name: str
    stop_desc: str
    stop_lat: float
    stop_lon: float
    location_type: int

    parent_station: Optional[Union[str, int]] = None
    wheelchair_boarding: Optional[Union[str, int]] = None
    platform_code: Optional[Union[str, int]] = None
    _alternate_stop_number: Optional[int] = None
    

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
    def alternate_stop_number(self):
        return self._alternate_stop_number

    @alternate_stop_number.setter
    def alternate_stop_number(self, value):
        self._alternate_stop_number = value
    @property
    def coordinates(self) -> Tuple[float, float]:
        """
        return the coordinates of the stop

        :return: a tuple of latitude, longitude
        :rtype: Tuple[float, float]

        """

        return self.stop_lat, self.stop_lon
    @property
    def is_border(self):
        return self.stop_number in BORDER_STATIONS
    def as_point(self) -> Point:
        """return the stop as a shapely point"""

        return Point(self.coordinates)

    def __hash__(self):

        return hash(self.stop_number)



class StopsList(Iterator):

    def __init__(
            self,
            filepath: str,
            ) -> None:
        """
        Return a

        :param filepath: the path to the stops.json file
        :type filepath: str

        """

        self._filepath = filepath

        self._data = _load_stopslist(filepath)
        self.border_stations = _load_border_stations()
        self.stops: List[Stop]
        self.stops = [Stop.from_dict(x) for x in self._data]

        self._index = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self._filepath}')"

    def __getitem__(self, index) -> Stop:
        # update to deal with slicing and returning a StopList
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


    def rail_stops(self) -> 'StopsList':
        """
        return a new instance of a StopList with only rail stops

        :return: a new onstance of a StopsList
        :rtype: StopsList

        """
        rail = copy.deepcopy((self))
        rail.stops = [x for x in rail if len(str(x.stop_number)) == 7]

        return rail

    def bus_stops(self) -> 'StopsList':
        """
        return a new instance of a StopsList with only bus stops

        :return: a new onstance of a StopsList
        :rtype: StopsList

        """

        bus = copy.deepcopy((self))
        bus.stops = [x for x in bus if len(str(x.stop_number)) != 7]

        return bus

    @classmethod
    def update_stops(self):
        # update the stops in the stops.json from rejseplan gtfs
        return
