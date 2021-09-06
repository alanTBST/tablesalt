
"""
Test the results for revenue distribution
using the ZoneSharer and ZoneGraph classes
from the topology module.

NOTE: These tests are for the composite bus and rail graph


"""

import pytest

from tablesalt.topology import ZoneGraph, ZoneSharer


graph = ZoneGraph(region='sjælland')

#ZoneSharer(graph, zones, stops, operators, usage)


#single tests
#@pytest.mark.parametrize('input_tuple',
#    [
#        ((1001, 1001, 1260, 1260), (4790, 8600626, 52098, 41777), (1, 5, 3, 3), (1, 3, 3, 2))
#    ])
#def test_sharer_results(input_tuple):
#
#    sharer = ZoneSharer(graph, *input_tuple)
#    standard = sharer.share()['standard']
#    standard = set(standard)
#
#
#    assert standard == {(0.5, 'movia_h'), (14, 'dsb'), (0.5, 'movias_s')}

def test_faxe_koge_sydhavn():

    sharer = ZoneSharer(
        graph,
        (1270, 1020, 1002),
        (8601013, 8690803, 8600760),
        (2, 4, 4),
        (1, 3, 2)
        )
    standard = sharer.share()['standard']

    assert standard == ((4.5, 'movia_s'), (8.5, 's-tog'))

def test_sydhavn_koge_faxe():

    sharer = ZoneSharer(
        graph,
        (1002, 1020, 1270),
        (8600760, 8690803, 8601013),
        (4, 4, 2),
        (1, 3, 2)
        )
    standard = sharer.share()['standard']

    assert standard == ((4.5, 'movia_s'), (8.5, 's-tog'))


def test_sydhavn_koge_faxe_2():

    sharer = ZoneSharer(
        graph,
        (1002, 1020, 1020, 1270),
        (8600760, 8690803, 8600803, 8601013),
        (4, 4, 4, 2),
        (1, 3, 3, 2)
        )
    standard = sharer.share()['standard']

    assert standard == ((4.5, 'movia_s'), (8.5, 's-tog'))

def test_vestamager_nport_dybbro():

    sharer = ZoneSharer(
        graph,
        (1003, 1001, 1001),
        (8603319, 8600646, 8600634),
        (6, 4, 4),
        (1, 3, 2)
        )
    standard = sharer.share()['standard']

    return standard == ((1.5, 'metro'), (0.5, 's-tog'))



#test one zone one operator


# test one zone two operators


# test one zone three operators


#test
