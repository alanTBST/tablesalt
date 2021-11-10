
import pytest
from tablesalt import transitfeed
from tablesalt.topology.stationoperators import StationOperators



feed = transitfeed.archived_transitfeed('20211011_20220105')
station_getter = StationOperators(feed)

@pytest.mark.parametrize('station_tuple',
    [
        (8603307, 8603308),
        (8603307, 8603309),
        (8603307, 8603310),
        (8603302, 8603311),
        (8603315, 8603311),
        (8600646, 8603303),
        (8600703, 8603317),
        (8603307, 8600736),
        (8603307, 8600703),
        (8603302, 8600856),
        (8603333, 8603339),
        ]
    )
def test_metro_operator(station_tuple):

    operator = station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        )
    assert operator == {'Metroselskabet'}


@pytest.mark.parametrize('station_tuple',
    [
        (8600669, 8600626),
        (8600669, 8603307),
        (8600664, 8600626)
        ]
     )
def test_kyst_operator(station_tuple):
    operator = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert operator == {'DSB'}


@pytest.mark.parametrize('station_tuple',
    [
        (8600669, 8600858),
        (8600669, 8600856),
        (8600664, 8600858),
        (8600858, 8600668)
        ]
     )
def test_kystkastrup_operator(station_tuple):
    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert line == ['first']


@pytest.mark.parametrize('station_tuple',
    [
        (8602645, 8602641),
        (8602623, 8602645)
        ]
     )
def test_lolland_operator(station_tuple):

    oeprator = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert oeprator == ['movia_s']



@pytest.mark.parametrize('station_tuple',
    [
        (8600822, 8600824),
        ]
     )
def test_falster_operator(station_tuple):

    oeprator = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert oeprator == ['dsb']

@pytest.mark.parametrize('station_tuple',
    [
        (8600626, 6548),
        (8600798, 6554),
        (8690798, 6554),
        (8600624, 6551)
        ]
     )
def test_dsb_to_bus_line(station_tuple):

    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert line == ['vestbanen_sjælland']



@pytest.mark.parametrize('station_tuple',
    [
        (8600858, 6551),
        (8600622, 8600669),
        (8600622, 8600858),
        (8690622, 8600858),
        (8600696, 8600668),
        (8600611, 8600807),
        (8601411, 8603003),
        (8603517, 8600723),
        (8600677, 8601711)
     ]
     )
def test_missed_check(station_tuple):
    """a missed check between to stations should
    result in more than one operator or no operators"""
    operator = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert len(operator) != 1


@pytest.mark.parametrize('station_tuple',
    [(8690626, 8600798),
     (8690626, 8600646),
     (8690659, 8600655)]
    )
def test_stog_to_dsb_equals_stog(station_tuple):
    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1])
    assert line == ['s-tog']



@pytest.mark.parametrize('station_tuple',
    [
        (8600626, 8690798),
        (8600626, 8600646),
        (8600626, 8690659)
        ]
    )
def test_dsb_to_stog_equals_dsb(station_tuple):
    operator = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert operator ==  ['dsb']


def test_start_dsb_to_metro():
    return

def test_start_dsb_to_stog():
    return

def test_start_dsb_to_stog():
    return

def test_start_metro_to_dsb():

    return


@pytest.mark.parametrize('station_tuple',
    [
        (8690683, 8600669),
        (8600683, 8600669)
        ]
    )
def test_stog_dsb_equals_local(station_tuple):
    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert line == ['movia_h']


@pytest.mark.parametrize('station_tuple',
    [
        (8690683, 8601411),
        (8690683, 8601715),
        (8690674, 8603003),
        (8600674, 8603003),
        (8600803, 8601025)
    ]
    )
def test_stog_to_local_equals_local(station_tuple):
    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1]
        )
    assert line == ['movia_h'] or line == ['movia_s']



@pytest.mark.parametrize('station_tuple',
    [
        (8690683, 7933),
        (8600683, 7933),
        (8690683, 6118)
        ]
    )
def test_stog_to_localbus_equals_local(station_tuple):
    line = station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='operator')
    assert line == ['movia_h'] or line == ['movia_s']


def test_start_bus_hovedstaden():
    assert True

def test_start_bus_sydsj():
    assert True

def test_start_bus_vestsj():
    assert True





