
import pytest


# NOTE: THESE TEST ARE FOR DEFAULT DANISH PASSENGER STATIONS FOR REGION SJÆLLAND 
@pytest.mark.parametrize('station_tuple',
    [(8603307, 8603308),
    (8603307, 8603309),
    (8603307, 8603310),
    (8603302, 8603311), 
    (8603315, 8603311)])  
def test_easy_metro_station_pair(sj_station_getter, station_tuple):
    """[summary]

    :param sj_station_getter: stationoperators.StationOperators instance
    :type sj_station_getter: [type]
    :param station_tuple: [description]
    :type station_tuple: [type]
    """

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    #op_res = sj_station_getter.station_pair(8603307, 8603308, format='operator')

    assert line == ('metro', )


@pytest.mark.parametrize('station_tuple',
    [(8603307, 8600736),
     (8603307, 8600703), 
     (8603302, 8600856)]) 
def test_harder_metro_station_pair(sj_station_getter, station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    assert line == ('metro', )


@pytest.mark.parametrize('station_tuple',
    [(8600669, 8600626),
     (8600669, 8603307), 
     (8600664, 8600626)]
     ) 

def test_easy_kyst_station_pair(sj_station_getter, station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    assert line == ('kystbanen', )


"""
@pytest.mark.parametrize('station_tuple',
    [(8600626, 8600858),])

def test_harder_kyst_station_pair(sj_station_getter, station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert line == (8, )
"""

@pytest.mark.parametrize('station_tuple',
    [(8600626, 6548),
     (8600626, 6554), 
     (8600626, 6551)]) 
def test_easy_dsb_bus_station_pair(sj_station_getter, station_tuple):

    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')

    assert line == ('fjernregional', )


# missed check at  , (8600858, 6551)
def test_start_dsb_to_metro():
    return 

def test_start_dsb_to_stog():
    return 

def test_start_dsb_to_stog():
    return  

def test_start_metro_to_dsb():

    return 

@pytest.mark.parametrize('station_tuple',
    [(8600626, 8690798),
     (8600626, 8690646), 
     (8600659, 8690798)]       
    ) 
def test_dsb_to_stog_equals_dsb(sj_station_getter, station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert line == ('fjernregional', )

@pytest.mark.parametrize('station_tuple',
    [(8690626, 8600798),
     (8690646, 8600798),
     (8690626, 8600646), 
     (8690659, 8600655)]       
    ) 
def test_stog_to_dsb_equals_stog(sj_station_getter, station_tuple):   
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert line == ('suburban', )


@pytest.mark.parametrize('station_tuple',
    [(8690683, 8600669), 
     (8600683, 8600669)]
    ) 
def test_stog_dsb_equals_local(sj_station_getter, station_tuple):   
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert line == ('local', )

@pytest.mark.parametrize('station_tuple',
    [(8690683, 8601411),
    (8690683, 8601715), 
    (8690674, 8603003), 
    (8600674, 8603003)
    ]
    ) 
def test_stog_to_local_equals_local(sj_station_getter, station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')

    assert line == ('local', )

@pytest.mark.parametrize('station_tuple',
    [(8690683, 7933),
     (8600683, 7933),
     (8690683, 6118)]
    )
def test_stog_to_localbus_equals_local(sj_station_getter, station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert line == ('local', )


def test_start_bus_hovedstaden():
    assert True

def test_start_bus_sydsj():
    assert True

def test_start_bus_vestsj():
    assert True





