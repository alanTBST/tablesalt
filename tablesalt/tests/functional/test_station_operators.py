
import pytest
from tablesalt.topology.stationoperators import StationOperators


sj_station_getter = StationOperators(
    'kystkastrup',
    'suburban',
    'sjællandfjernregional',
    'sjællandlocal',
    'metro'
    )

# NOTE: THESE TEST ARE FOR DEFAULT DANISH PASSENGER STATIONS FOR REGION SJÆLLAND

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

    operator = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='operator'
        )
    assert operator == ['metro']


@pytest.mark.parametrize('station_tuple',
    [
        (8603309, 8603315),
        (8603309, 8603312),
        (8603312, 8603317),
        (8603312, 8603301)
    ]
)
def test_metro_line_m1(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    assert line == ['metro_m1']



@pytest.mark.parametrize('station_tuple',
    [
        (8603309, 8603327),
        (8603301, 8603328),
        (8603324, 8603309),
    ]
)
def test_metro_line_m2(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    assert line == ['metro_m2']


@pytest.mark.parametrize('station_tuple',
    [
        (8603301, 8603302),
        (8603302, 8603306),
        (8603302, 8603309),
        (8603302, 8603308)
    ]
)
def test_metro_line_m1_m2(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    setline = set(line)
    assert setline == {'metro_m1', 'metro_m2'}



@pytest.mark.parametrize('station_tuple',
    [
        (8603342, 8603341),
        (8603339, 8603335),
        (8603308, 8603339),
    ]
)
def test_metro_line_m3(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    assert line == ['metro_m3']



@pytest.mark.parametrize('station_tuple',
    [
        (8603346, 8603308),
        (8603308, 8603345),
        (8603345, 8603331),
    ]
)
def test_metro_line_m4(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    assert line == ['metro_m4_1']




@pytest.mark.parametrize('station_tuple',
    [
        (8603308, 8603333),
        (8603333, 8603308)
    ]
)
def test_metro_line_m3_m4(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='line'
        )
    assert set(line) == {'metro_m3', 'metro_m4_1'}


@pytest.mark.parametrize('station_tuple',
    [
        (8600669, 8600626),
        (8600669, 8603307),
        (8600664, 8600626)
        ]
     )
def test_kyst_line(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    assert line == ['kystbanen']



@pytest.mark.parametrize('station_tuple',
    [
        (8600669, 8600626),
        (8600669, 8603307),
        (8600664, 8600626)
        ]
     )
def test_kyst_operator(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['first']



@pytest.mark.parametrize('station_tuple',
    [
        (8600669, 8600858),
        (8600669, 8600856),
        (8600664, 8600858),
        (8600858, 8600668)
        ]
     )
def test_kystkastrup_operator(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['first']


@pytest.mark.parametrize('station_tuple',
    [
        (8602645, 8602641),
        (8602623, 8602645)
        ]
     )
def test_lolland_operator(station_tuple):

    oeprator = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert oeprator == ['movia_s']



@pytest.mark.parametrize('station_tuple',
    [
        (8600822, 8600824),
        ]
     )
def test_falster_operator(station_tuple):

    oeprator = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
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

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
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
    operator = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert len(operator) != 1


@pytest.mark.parametrize('station_tuple',
    [(8690626, 8600798),
     (8690626, 8600646),
     (8690659, 8600655)]
    )
def test_stog_to_dsb_equals_stog(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='operator')
    assert line == ['s-tog']



@pytest.mark.parametrize('station_tuple',
    [
        (8600626, 8690798),
        (8600626, 8600646),
        (8600626, 8690659)
        ]
    )
def test_dsb_to_stog_equals_dsb(station_tuple):
    operator = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='operator'
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
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
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
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
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
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1],
        format='operator')
    assert line == ['movia_h'] or line == ['movia_s']


def test_start_bus_hovedstaden():
    assert True

def test_start_bus_sydsj():
    assert True

def test_start_bus_vestsj():
    assert True





