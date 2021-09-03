
import pytest
from tablesalt.topology.stationoperators import StationOperators


sj_station_getter = StationOperators(
    'kystbanen',
    'suburban',
    'sjællandfjernregional',
    'sjællandlocal',
    'øresunds_banen',
    'metro'
    )

# NOTE: THESE TEST ARE FOR DEFAULT DANISH PASSENGER STATIONS FOR REGION SJÆLLAND 

@pytest.mark.parametrize('station_tuple',
    [(8603307, 8603308),
    (8603307, 8603309),
    (8603307, 8603310),
    (8603302, 8603311), 
    (8603315, 8603311)])  
def test_easy_metro_operator(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    #op_res = sj_station_getter.station_pair(8603307, 8603308, format='operator')

    assert line == ['metro']




@pytest.mark.parametrize('station_tuple',
    [(8603307, 8600736),
     (8603307, 8600703), 
     (8603302, 8600856)]) 
def test_harder_metro_operator(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['metro']



@pytest.mark.parametrize('station_tuple',
    [(8600669, 8600626),
     (8600669, 8603307), 
     (8600664, 8600626)]
     ) 
def test_easy_kyst_line(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    assert line == ['kystbanen']




@pytest.mark.parametrize('station_tuple',
    [(8600669, 8600626),
     (8600669, 8603307), 
     (8600664, 8600626)]
     ) 
def test_easy_kyst_operator(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['first']


@pytest.mark.parametrize('station_tuple',
    [(8600626, 6548),
     (8600626, 6554), 
     (8600626, 6551)]
     )
def test_easy_dsb_bus_line(station_tuple):

    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='line'
        )
    assert line == ['vestbanen_sjælland']



@pytest.mark.parametrize('station_tuple',
    [(8600858, 6551),
     (8600622, 8600669), 
     (8600622, 8600858), 
     (8690622, 8600858), 
     (8600696, 8600668), 
     (8600611, 8600807), 
     (8601411, 8603003)]
     )
def test_missed_check(station_tuple):
    """a missed check between to stations should 
    result in more than one operator or no operators"""
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert len(line) != 1



@pytest.mark.parametrize('station_tuple',
    [(8600669, 8600858)]
     )
def test_east_coast(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['first']




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
     (8600626, 8600646), 
     (8600659, 8690798)]       
    ) 
def test_dsb_to_stog_equals_dsb(station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert True

@pytest.mark.parametrize('station_tuple',
    [(8690626, 8600798),
     (8690626, 8600646), 
     (8690659, 8600655)]       
    ) 
def test_stog_to_dsb_equals_stog(station_tuple):   
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='operator')
    assert line == ['s-tog']


@pytest.mark.parametrize('station_tuple',
    [(8690683, 8600669), 
     (8600683, 8600669)]
    ) 
def test_stog_dsb_equals_local(station_tuple):   
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['movia_h']
    
@pytest.mark.parametrize('station_tuple',
    [(8690683, 8601411),
    (8690683, 8601715), 
    (8690674, 8603003), 
    (8600674, 8603003)
    ]
    ) 
def test_stog_to_local_equals_local(station_tuple):
    line = sj_station_getter.station_pair(
        station_tuple[0], station_tuple[1], format='operator'
        )
    assert line == ['movia_h']

@pytest.mark.parametrize('station_tuple',
    [(8690683, 7933),
     (8600683, 7933),
     (8690683, 6118)]
    )
def test_stog_to_localbus_equals_local(station_tuple):
    line = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert True


def test_start_bus_hovedstaden():
    assert True

def test_start_bus_sydsj():
    assert True

def test_start_bus_vestsj():
    assert True





