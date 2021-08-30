# -*- coding: utf-8 -*-

from json import JSONDecodeError
from pathlib import Path

import pytest
from shapely.geometry import Point  # type: ignore
from tablesalt.topology.stopnetwork import (
    Stop, 
    StopsList, 
    _load_stopslist, 
    _load_border_stations
    )  # noqa

HERE = Path(__file__).parent


class MockJson:

    @staticmethod
    def load():
        stop1 = {
            'location_type': 0,
            'parent_station': '',
            'platform_code': '',
            'stop_code': '',
            'stop_desc': '',
            'stop_lat': 56.937271416304,
            'stop_lon': 9.753533239882,
            'stop_name': 'Østervang (Øster Hornumvej / Øster Hornum)',
            'stop_id': 845301102,
            'wheelchair_boarding': ''
            }

        stop2 = {
            'location_type': 0,
            'parent_station': '',
            'platform_code': '',
            'stop_code': '',
            'stop_desc': '',
            'stop_lat': 56.083154489147994,
            'stop_lon': 12.468205271773,
            'stop_name': 'Karinebæk St.',
            'stop_id': 8601620,
            'wheelchair_boarding': ''
            }

        return [stop1, stop2]


def test_load_stopslist():
    fp = HERE / 'test_stops.json'
    stoplist = _load_stopslist(fp) 
    assert len(stoplist) == 6


def test_load_stopslist_raises_type():
    fp = HERE / 'test_stops.csv'
    with pytest.raises(TypeError):
        _load_stopslist()


def test_load_stopslist_raises_value():
    fp = HERE / 'test_stops.csv'
    with pytest.raises(ValueError):
        _load_stopslist(fp)


def test_load_border_stations():

    border_stops = _load_border_stations()
    assert isinstance(border_stops, dict)
    assert len(border_stops) == 41 # this is all the border stations. Only Update if zones change!
    assert all(isinstance(x, int) for x in border_stops)


def test_stop_fromdict():

    ds = MockJson().load()
    stop1 = Stop.from_dict(ds[0])
    stop2 = Stop.from_dict(ds[1])

    assert isinstance(stop1, Stop)
    assert isinstance(stop2, Stop)


def test_stop_equality():

    d = MockJson().load()[0]
    stop1 = Stop.from_dict(d)
    stop2 = Stop.from_dict(d)

    assert stop1 == stop2

def test_stop_methods():

    d = MockJson().load()[0]

    stop = Stop.from_dict(d)
    assert stop.coordinates == (9.753533239882, 56.937271416304)
    assert stop.as_point() == Point((9.753533239882, 56.937271416304))

def test_stopslist_from_file():
    
    fp = HERE / 'test_stops.json'
    slist = StopsList.from_json(fp)

    assert isinstance(slist, StopsList)

def test_stoplist_getitem(a_stopslist):

    assert isinstance(a_stopslist[0], Stop)


def test_stoplist_iteration(a_stopslist):
    with pytest.raises(StopIteration):
        while True:
            x = next(a_stopslist)


def test_rail_stops(a_stopslist):

    rail_list = a_stopslist.rail_stops()

    assert isinstance(rail_list, StopsList)
    assert len(rail_list) == 5


def test_bus_stops(a_stopslist):

    bus_list = a_stopslist.bus_stops() 
    assert isinstance(bus_list, StopsList)
    assert len(bus_list) == 1



