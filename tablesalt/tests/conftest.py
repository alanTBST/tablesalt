# place for fixtures: ie setup for tests
from pathlib import Path
import pytest
from tablesalt.topology.stopnetwork import StopsList


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