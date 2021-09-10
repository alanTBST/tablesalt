
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
# operators: 1=movia_h, 2=movia_s, 3=movia_v, 4=stog, 5=dsb, 6=metro, 8=first
# usage:  1=Fi, 2=Co, 2=Su, 4=Tr

#-----------------------------
# individual tests
#-----------------------------
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
    # added extra susu check - should be equal
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
        (8603317, 8600646, 8600634),
        (6, 4, 4),
        (1, 3, 2)
        )
    standard = sharer.share()['standard']

    assert standard == ((1.5, 'metro'), (0.5, 's-tog'))

def test_stenl_dybbro():

    sharer = ZoneSharer(
        graph,
        (1074, 1001),
        (8600711, 8600634),
        (4, 4),
        (1, 2)
        )

    standard = sharer.share()['standard']
    assert standard == ((7, 's-tog'),) or standard == ((7.0, 's-tog'),)

## more border tests

#--------------------------
#test one zone one operator
#--------------------------

def test_stog_zone_1043():

    sharer = ZoneSharer(
        graph,
        (1043, 1054),
        (8600622, 8600621),
        (4, 4),
        (1, 2)
    )
    standard = set(sharer.share()['standard'])

    assert standard == {(1, 's-tog')}


def test_stog_zone_1043_2():
    # double check at albertslund
    sharer = ZoneSharer(
        graph,
        (1043, 1054, 1054),
        (8600622, 8600621, 8600621),
        (4, 4, 4),
        (1, 3, 2)
    )
    standard = set(sharer.share()['standard'])

    assert standard == {(1, 's-tog')}

def test_stog_zone_1043_3():
    # double check in at glostrup
    sharer = ZoneSharer(
        graph,
        (1043, 1043, 1043),
        (8600622, 8600622, 8600621),
        (4, 4, 4),
        (1, 3, 2)
    )
    standard = set(sharer.share()['standard'])

    assert standard == {(1, 's-tog')}


def test_multi_city_ring_metro():

    sharer = ZoneSharer(
        graph,
        (1001, 1001, 1001, 1001),
        (8603305, 8603344, 8603344, 8603302),
        (6, 6, 6, 6),
        (1, 2, 4, 2)
    )

    standard = set(sharer.share()['standard'])

    assert standard == {(1, 'metro')}

#----------------------------
# test one operator two zones
#----------------------------

def test_landlystvej_pbangsvej_rodovrecentrum():

    sharer = ZoneSharer(
        graph,
        (1002, 1002, 1032),
        (7155, 2088, 435),
        (1, 1, 1),
        (1, 3, 2)
    )
    standard = sharer.share()['standard']
    assert set(standard) == {(2, 'movia_h'),} or set(standard) == {(2.0, 'movia_h'),}

#----------------------------
# test one zone two operators
#----------------------------
def test_albertslund_glostrup_Åskellet_Østbrovej():

    sharer = ZoneSharer(
        graph,
        (1043, 1043, 1043),
        (8600621, 2261, 10532),
        (4, 1, 1),
        (1, 3, 2)
    )
    standard = sharer.share()['standard']
    assert set(standard) == {(0.5, 'movia_h'), (0.5, 's-tog')}
    # albertslund is also a border station
def test_albertslund_glostrup_Åskellet_Østbrovej_2():

    sharer = ZoneSharer(
        graph,
        (1054, 1043, 1043),
        (8600621, 2261, 10532),
        (4, 1, 1),
        (1, 3, 2)
    )
    standard = sharer.share()['standard']
    assert set(standard) == {(0.5, 'movia_h'), (0.5, 's-tog')}


#------------------------------
# test one zone three operators
#------------------------------
def test_drbyen_nport_strandboulevarden_norhavn_dybbro():
    #also has a CoTr to be removed and a border stations
    sharer = ZoneSharer(
        graph,
        (1003, 1001, 1001, 1001, 1001),
        (8603311, 28329, 1132, 8600653, 8600634),
        (6, 1, 1, 4, 4),
        (1, 3, 2, 4, 2)
    )

    standard = sharer.share()['standard']
    assert set(standard) == {(0.3333333333333333, 'metro'), (0.3333333333333333, 'movia_h'),(0.3333333333333333, 's-tog')}


"""
#-----------------------------
# smap error due to 8600717, 8600646 not being on the same line
# we use
#-----------------------------
def test_krekilstrup_nport_ncampus():

    sharer = ZoneSharer(
        graph,
        (1115, 1115, 1001, 1001),
        (8602505, 8600717, 52615, 7030),
        (3, 5, 1, 1),
        (1, 3, 3, 2)
    )
    standard = sharer.share()['standard']
    set_standard = set(standard)
    assert set_standard == {(0.5, 'movia_s'), (10, 'dsb'),  (0.5, 'movia_h')}
"""