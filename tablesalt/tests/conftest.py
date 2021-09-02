# place for fixtures: ie setup for tests

import shutil
from pathlib import Path
import pytest
from tablesalt.topology.stopnetwork import StopsList

from tablesalt.topology.stationoperators import StationOperators


HERE = Path(__file__).parent

@pytest.fixture
def a_stopslist():

    fp = HERE / 'unit' / 'test_stops.json'
    slist = StopsList.from_json(fp)
    
    return slist


@pytest.fixture
def a_stop(a_stopslist):

    return a_stopslist[0]

@pytest.fixture
def a_line():

    return 

@pytest.fixture
def a_railLine():

    return 


@pytest.fixture
def lib_dir(tmp_path):
    d = tmp_path / 'rejskortstores'
    yield d
    shutil.rmtree(str(d))

@pytest.fixture
def sj_station_getter():

    return StationOperators(
    'kystbanen', 'local', 'metro', 'suburban', 'fjernregional'
    )
