


def test_station_pair(sj_station_getter):

    op_id_res = sj_station_getter.station_pair(8603307, 8603308, format='operator_id')
    op_res = sj_station_getter.station_pair(8603307, 8603308, format='operator')

    assert op_id_res == (6, )
    assert op_res == ('metro', )