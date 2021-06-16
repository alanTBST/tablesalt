# -*- coding: utf-8 -*-

from json import JSONDecodeError
from pathlib import Path

import pytest
from shapely.geometry import Point  # type: ignore
from topo.stopnetwork import Stop, StopsList, _load_stopslist, _load_border_stations  # noqa

HERE = Path(__file__).parent


class MockJson:

    @staticmethod
    def load():
        stop1 = {
            'is_border': None,
            'location_type': 0,
            'parent_station': '',
            'platform_code': '',
            'stop_code': '',
            'stop_desc': '',
            'stop_lat': 56.937271416304,
            'stop_lon': 9.753533239882,
            'stop_name': 'Østervang (Øster Hornumvej / Øster Hornum)',
            'stop_number': 845301102,
            'wheelchair_boarding': ''
            }

        stop2 = {
            'is_border': None,
            'location_type': 0,
            'parent_station': '',
            'platform_code': '',
            'stop_code': '',
            'stop_desc': '',
            'stop_lat': 56.083154489147994,
            'stop_lon': 12.468205271773,
            'stop_name': 'Karinebæk St.',
            'stop_number': 8601620,
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
    assert stop.coordinates == (56.937271416304, 9.753533239882)
    assert stop.as_point() == Point((56.937271416304, 9.753533239882))


def test_stopslist_from_file():
    
    fp = HERE / 'test_stops.json'
    slist = StopsList(fp)

    assert isinstance(slist, StopsList)


def test_stoplist_getitem():

    return


def test_stoplist_iteration():

    return 


def test_rail_stops():

    return 


def test_bus_stops():

    return 

