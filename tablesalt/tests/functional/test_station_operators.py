
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

    :param sj_station_getter: Staion
    :type sj_station_getter: [type]
    :param station_tuple: [description]
    :type station_tuple: [type]
    """

    op_id_res = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    #op_res = sj_station_getter.station_pair(8603307, 8603308, format='operator')

    assert op_id_res == ('metro', )


@pytest.mark.parametrize('station_tuple',
    [(8603307, 8600736),
    (8603307, 8600703), 
    (8603302, 8600856)]) 
def test_harder_metro_station_pair(sj_station_getter, station_tuple):

    op_id_res = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert op_id_res == ('metro', )


@pytest.mark.parametrize('station_tuple',
    [(8600669, 8600626),
    (8600669, 8603307), 
    (8600664, 8600626)]) 

def test_easy_kyst_station_pair(sj_station_getter, station_tuple):
    op_id_res = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert op_id_res == ('kystbanen', )
"""
@pytest.mark.parametrize('station_tuple',
    [(8600626, 8600858),])

def test_harder_kyst_station_pair(sj_station_getter, station_tuple):
    op_id_res = sj_station_getter.station_pair(station_tuple[0], station_tuple[1], format='line')
    assert op_id_res == (8, )
"""