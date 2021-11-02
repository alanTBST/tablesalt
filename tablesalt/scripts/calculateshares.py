# -*- coding: utf-8 -*-
"""
This script uses the zone work model to calculate the zone work
for each operator for each trip in sjælland and writes them to and lmdb key-value store.




USAGE
=====

**delrejsersetup.py must be run before this script**

To run this script for the year 2019:

    python **./path/to/tablesalt/tablesalt/scripts/calculateshares.py -y 2019**


Resultant directory tree structure
==================================

| given_output_directory/
|         |---rejsekortstores/
|                   |------dbs/
|                           |-----**calculated_shares**
|                   |------hdfstores/
|                   |------packs/


calculated_shares
-----------------

This is the main output from the calculateshares.py script.
It is an lmdb key-value store. The keys are bytestrings of tripkeys
and the values are bytestrings of tuples of the form
b'((float, operator), (float, operatr), ...)'

"""

import gc
import os
import pickle
from functools import partial
from itertools import chain, groupby
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Iterator

import numpy as np
from tqdm import tqdm

from tablesalt import StoreReader, transitfeed
from tablesalt.common import make_store
from tablesalt.common.io import mappers
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import db_paths, find_datastores
from tablesalt.running import WindowsInhibitor
from tablesalt.topology import ZoneGraph, ZoneSharer
from tablesalt.topology.stationoperators import StationOperators
from tablesalt.topology.tools import TakstZones


THIS_DIR = Path(__file__).parent

CPU_USAGE = 0.6 # % of processors
DB_START_SIZE = 8 # gb


# this should be in common.io as contractor_to_array
def proc_contractors(contrpack) -> np.ndarray:
    """Return the contractors dict as an array"""
    arr_length = len(tuple(chain(*contrpack.values())))
    arr = np.zeros(shape=(arr_length, 4), dtype=np.int64)
    i = 0
    for k, v in contrpack.items():
        for record in v:
            arr[i, 0] = k
            arr[i, 1] = record[0]
            arr[i, 2] = record[1]
            arr[i, 3] = record[2]
            i += 1
    return arr

# should be place in common.io
def _load_contractor_pack(
    store: str,
    region: str,
    region_contractors: Dict[str, List[str]]
    ) -> Dict[int, Tuple[int, ...]]:
    """Load and process the contractor/operator information

    :param store: the path to and h5 file
    :type store: str
    :param region: the region to include ['hovedstaden', 'sjælland']
    :type region: str
    :param region_contractors: a dictionary of region -> lst_operators
    :type region_contractors: Dict[str, List[str]]
    :return: an operator dictionary tripkey -> tuple of operator ids
    :rtype: Dict[int, Tuple[int, ...]]
    """
    reader = StoreReader(store)
    contractors = reader.get_data('contractors')

    operator_ids = mappers['operator_id']

    contractor_filters = [
        operator_ids[x] for x in region_contractors[region]
        ]

    contractors = proc_contractors(contractors)
    bad_ops = contractors[:, 0][
        ~np.isin(contractors[:, 2], contractor_filters)
        ]

    contractors = contractors[~np.isin(contractors[:, 0], bad_ops)]
    contractors = contractors[
        np.lexsort((contractors[:, 1], contractors[:, 0]))
        ]

    op_dict = {key: tuple(x[2] for x in grp) for key, grp in
               groupby(contractors, key=itemgetter(0))}

    return op_dict

class TripDict(TypedDict):
    """Typed dictionary for static type checking
    for a stop/usage/operator/zone dictionary
    """
    tripkey: int
    trip_values: Tuple[int, ...]

# should be able to return these dictionaries from StoreReader
def _load_store_data(
    store: str,
    region: str,
    zonemap: Dict[int, int],
    region_contractors: Dict[str, List[str]]
    ) -> Tuple[TripDict, TripDict, TripDict, TripDict]:
    """
    Load the stop data from the h5 file and create
    the stop, zone and usage and operator dicts

    :param store: the path to an h5 file
    :type store: str
    :param region: the region to use
    :type region: str
    :param zonemap: a dictionary mapping stopid -> zone number
    :type zonemap: Dict[int, int]
    :param region_contractors: the region contractor dictionary
    :type region_contractors: Dict[str, List[str]]
    :return: a tuple of the stop, zone, usage and operator dictionaries
    :rtype: Tuple[TripDict, TripDict, TripDict, TripDict]
    """


    reader = StoreReader(store)
    stops = reader.get_data('stops')
    stops = stops[np.lexsort((stops[:, 1], stops[:, 0]))]

    usage_dict = {
        key: tuple(x[3] for x in grp) for key, grp in
        groupby(stops, key=itemgetter(0))
        }
    usage_dict = {
        k: v for k, v in usage_dict.items() if
        v[0] == 1 and v[-1] == 2
        }
    stop_dict = {
        key: tuple(x[2] for x in grp) for key, grp in
        groupby(stops, key=itemgetter(0)) if key in usage_dict
        }
    zone_dict = {
        k: tuple(zonemap.get(x) for x in v) for
        k, v in stop_dict.items()
        }
    zone_dict = {
        k:v for k, v in zone_dict.items() if
        all(x for x in v) and all(1000 < y < 1300 for y in v)
        }

    op_dict = _load_contractor_pack(store, region, region_contractors)

    op_dict = {k:v for k, v in op_dict.items() if k in zone_dict}
    stop_dict = {k:v for k, v in stop_dict.items() if k in op_dict}
    zone_dict = {k:v for k, v in zone_dict.items() if k in op_dict}
    usage_dict = {k:v for k, v in usage_dict.items() if k in op_dict}

    return stop_dict, zone_dict, usage_dict, op_dict


def _get_store_num(store: str) -> str:
    """Just get the number of the store file name

    :param store: the path to the h5 file
    :type store: str
    :return: the number in the file name
    :rtype: str
    """
    st = str(store).split('.')[0]
    st = st.split('rkfile')[1]

    return st

def _get_input(
    stop_dict: TripDict,
    zone_dict: TripDict,
    usage_dict: TripDict,
    op_dict: TripDict
    ) -> Iterator[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """Generate trips and associated sequences to input into a ZoneSharer

    :param stop_dict: a dictionary of tripkey -> stop_sequence
    :type stop_dict: TripDict
    :param zone_dict: a dictionary of tripkey -> stop_sequence
    :type zone_dict: TripDict
    :param usage_dict: a dictionary of tripkey -> stop_sequence
    :type usage_dict: TripDict
    :param op_dict: a dictionary of tripkey -> stop_sequence
    :type op_dict: TripDict
    :yield: a tuple of tripkey, zone_sequence, stop_sequence, operator_sequence, usage_sequence
    :rtype: Iterator[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]
    """
    for k, zone_sequence in zone_dict.items():
        yield k, zone_sequence, stop_dict[k], op_dict[k], usage_dict[k]


def chunk_shares(
    store: str,
    year: int,
    graph: ZoneGraph,
    opgetter: StationOperators,
    region: str,
    zonemap: Dict[int, int],
    region_contractors: Dict[str, List[str]]
    ) -> Tuple[Dict[int, Tuple[Tuple[float, str], ...]], Dict[int, Tuple[Tuple[float, str], ...]]]:
    """For all trips in an h5 file, calculate the zone work shares

    :param store: the path to the h5 file
    :type store: str
    :param year: the year of analysis
    :type year: int
    :param graph: a ZoneGraph instance of tariff zones and route edges
    :type graph: ZoneGraph
    :param region: the region to perform the analysis on
    :type region: str
    :param zonemap: a mapping of stopid -> zone number
    :type zonemap: Dict[int, int]
    :param region_contractors: a dictionary of region -> list of contractors
    :type region_contractors: Dict[str, List[str]]
    :return: dict of model one results, dict of model two (solo zoner price) results
    :rtype: Tuple[Dict[int, Tuple[Tuple[float, str], ...]], Dict[int, Tuple[Tuple[float, str], ...]]]
    """

    stopsd, zonesd, usaged, operatorsd = _load_store_data(
        store, region, zonemap, region_contractors
        )

    gen = _get_input(stopsd, zonesd, usaged, operatorsd)

    border_changes = {}
    model_one_shares = {}
    model_two_shares = {} # solo_zone_price
    model_three_shares = {} # bumped
    model_four_shares = {}

    for k, zones, stops, operators, usage in tqdm(gen):
        # k, zones, stops, operators, usage = next(gen)
        try:
            sharer = ZoneSharer(graph, opgetter, zones, stops, operators, usage)
            trip_shares = sharer.share()
            model_one_shares[k] = trip_shares['standard']
        except Exception as e:
            continue

        if sharer.border_trip:
            initial_zone_sequence = sharer.zone_sequence
            model_one_shares[k] = trip_shares['standard']
            final_zone_sequence = sharer.zone_sequence
            if initial_zone_sequence != final_zone_sequence:
                border_changes[k] = final_zone_sequence
            model_two_shares[k] = trip_shares['solo_price']
            model_three_shares[k] = trip_shares['bumped']
            model_four_shares[k] = trip_shares['bumped_solo']
        else:
            model_one_shares[k] = trip_shares['standard']
            model_two_shares[k] = trip_shares['solo_price']
            model_three_shares[k] = trip_shares['bumped']
            model_four_shares[k] = trip_shares['bumped_solo']

    num = _get_store_num(store)

    fp = (THIS_DIR / '__result_cache__'
         / f'{year}' / 'borderzones' / f'bzones{num}.pickle')
    parent_dir = Path(fp).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(fp, 'wb') as f:
        pickle.dump(border_changes, f)

    return model_one_shares, model_two_shares, model_three_shares, model_four_shares

def main():
    """
    Main function to create the operator
    shares for the data in the datastores in parallel
    using CPU_USAGE % processing power
    """

    parser = TableArgParser('year')
    args = parser.parse()

    year = args['year']

    store_loc = find_datastores()
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']

    zones = TakstZones()
    zonemap = zones.stop_zone_map()

    # TODO into config
    region_contractors = {
        'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
        'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
        }

    region = 'sjælland'

    graph = ZoneGraph(region=region)

    allfeeds = transitfeed.available_archives()

    year_archives = [x for x in allfeeds if str(year) in x]

    feed = transitfeed.archived_transitfeed(year_archives[0])
    for archive in year_archives[1:]:
        archive_feed = transitfeed.archived_transitfeed(archive)
        feed = feed + archive_feed


    opgetter = StationOperators(feed)

    pfunc = partial(chunk_shares,
                    year=year,
                    graph=graph,
                    opgetter=opgetter,
                    region=region,
                    zonemap=zonemap,
                    region_contractors=region_contractors)

    with Pool(round(os.cpu_count() * CPU_USAGE)) as pool:
        results = pool.imap(pfunc, stores)
        for model_one, model_two, model_three, model_four in tqdm(results, total=len(stores)):
            make_store(model_one, db_path, start_size=DB_START_SIZE)
            make_store(model_two, db_path + '_model_2', start_size=DB_START_SIZE)
            make_store(model_three, db_path + '_model_3', start_size=DB_START_SIZE)
            make_store(model_four, db_path + '_model_4', start_size=DB_START_SIZE)
            gc.collect()

# if __name__ == "__main__":
#     if os.name == 'nt':
#         INHIBITOR = WindowsInhibitor()
#         INHIBITOR.inhibit()
#         main()
#         INHIBITOR.uninhibit()
#     else:
#         main()



# fp = r'C:\Users\B087115\Documents\GitHub\tablesalt\tablesalt\scripts'

# import gc
# import os
# import pickle
# from functools import partial
# from itertools import chain, groupby
# from multiprocessing import Pool
# from operator import itemgetter
# from pathlib import Path
# from typing import Dict, List, Tuple, TypedDict, Iterator

# import numpy as np
# from tqdm import tqdm

# from tablesalt import StoreReader
# from tablesalt.common import make_store
# from tablesalt.common.io import mappers
# from tablesalt.preprocessing.parsing import TableArgParser
# from tablesalt.preprocessing.tools import db_paths, find_datastores
# from tablesalt.running import WindowsInhibitor
# from tablesalt.topology import ZoneGraph, ZoneSharer
# from tablesalt.topology.tools import TakstZones


# THIS_DIR = Path(fp)

# year = 2019
# store_loc = find_datastores()
# paths = db_paths(store_loc, year)
# stores = paths['store_paths']
# db_path = paths['calculated_stores']

# zones = TakstZones()
# zonemap = zones.stop_zone_map()

# # TODO into config
# region_contractors = {
#     'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
#     'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
#     }

# region = 'sjælland'
# graph = ZoneGraph(region=region)


# store = stores[0]

# stopsd, zonesd, usaged, operatorsd = _load_store_data(
#     store, region, zonemap, region_contractors
#     )

# gen = _get_input(stopsd, zonesd, usaged, operatorsd)

# border_changes = {}
# model_one_shares = {}
# model_two_shares = {} # solo_zone_price
# model_three_shares = {} # bumped
# model_four_shares = {}
# for k, zones, stops, operators, usage in tqdm(gen):
#     # k, zones, stops, operators, usage = next(gen)
#     sharer = ZoneSharer(graph, zones, stops, operators, usage)
#     if any(len(set(x)) == 1 for x in sharer.stop_legs):
#         break
