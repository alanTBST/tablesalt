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
|                           |-----calculated_shares_model_2
|                           |-----calculated_shares_model_3
|                           |-----calculated_shares_model_4
|                   |------hdfstores/
|                   |------packs/


calculated_shares
-----------------

This is the main output from the calculateshares.py script.
It is an lmdb key-value store. The keys are bytestrings of tripkeys
and the values are bytestrings of tuples of the form
b'((float, operator), (float, operatr), ...)'

"""

import os
import pickle
from collections import defaultdict
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
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import db_paths, find_datastores
from tablesalt.running import WindowsInhibitor
from tablesalt.topology import ZoneGraph, ZoneSharer
from tablesalt.topology.stationoperators import StationOperators


THIS_DIR = Path(__file__).parent

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

    contractors = proc_contractors(contractors)

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
    zonemap: Dict[int, int],
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

    op_dict = _load_contractor_pack(store)

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
    zonemap: Dict[int, int],
    ) -> Tuple[Dict[int, Tuple[Tuple[float, str], ...]], Dict[int, Tuple[Tuple[float, str], ...]]]:
    """For all trips in an h5 file, calculate the zone work shares

    :param store: the path to the h5 file
    :type store: strS
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

    stopsd, zonesd, usaged, operatorsd = _load_store_data(store, zonemap)

    gen = _get_input(stopsd, zonesd, usaged, operatorsd)
    border_changes = {}
    model_results = defaultdict(dict)
    errs = {}
    for k, zones, stops, operators, usage in tqdm(gen):
        # k, zones, stops, operators, usage = next(gen)
        try:
            sharer = ZoneSharer(graph, opgetter, zones, stops, usage, takst_suffix=False)
            trip_shares = sharer.share()
        except Exception as e:
            errs[k] = str(e)
            continue
        if sharer.border_trip:
            initial_zone_sequence = sharer.zone_sequence
            final_zone_sequence = sharer.zone_sequence
            if initial_zone_sequence != final_zone_sequence:
                border_changes[k] = final_zone_sequence
        model_results[1][k] = trip_shares['standard']
        model_results[2][k] = trip_shares['solo_price']
        model_results[3][k] = trip_shares['bumped']
        model_results[4][k] = trip_shares['bumped_solo']
        model_results[5][k] = trip_shares['spend']
        model_results[6][k] = trip_shares['spend_solo']

    num = _get_store_num(store)

    fp = (THIS_DIR / '__result_cache__'
         / f'{year}' / 'borderzones' / f'bzones{num}.pickle')
    parent_dir = Path(fp).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(fp, 'wb') as f:
        pickle.dump(border_changes, f)

    return model_results

def main():
    """
    Main function to create the operator
    shares for the data in the datastores in parallel
    using CPU_USAGE % processing power
    """

    parser = TableArgParser('year', 'bus_stop_distance', 'cpu_usage')
    args = parser.parse()
    year = args['year']
    store_loc = find_datastores()
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']
    bus_distance = args['bus_stop_distance']

    # allfeeds = transitfeed.available_archives()
    # year_archives = [x for x in allfeeds if str(year) in x]
    feed = transitfeed.transitfeed_from_zip(
        r'C:\Users\b087115\Desktop\gtfs2019\sept11.zip'
        )
    f2 = transitfeed.transitfeed_from_zip(
        r'C:\Users\b087115\Desktop\gtfs2019\may23_2019.zip'
        )
    feed = feed + f2

    zonemap = feed.stop_zone_map()
    graph = ZoneGraph(feed, region='sjælland')
    # for archive in tqdm(year_archives[-2:], f'merging transit feeds for {year}'):
    #     archive_feed = transitfeed.archived_transitfeed(archive)
    #     feed = feed + archive_feed
    opgetter = StationOperators(
        feed,
        bus_distance_cutoff=bus_distance,
        allow_operator_legs=True    # allow for revenue
        )

    pfunc = partial(chunk_shares,
                    year=year,
                    graph=graph,
                    opgetter=opgetter,
                    zonemap=zonemap)

    cpu_usage = args['cpu_usage']
    processors = int(round(os.cpu_count() * cpu_usage))

    with Pool(processors) as pool:
        results = pool.imap(pfunc, stores)
        for result in tqdm(results, total=len(stores)):
            for i in [1, 2, 3, 4, 5, 6]:
                make_store(
                    result[i],
                    db_path + f'_model_{i}',
                    start_size=DB_START_SIZE
                    )

if __name__ == "__main__":
    from datetime import datetime
    st = datetime.now()

    INHIBITOR = None
    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
    main()

    if INHIBITOR:
        INHIBITOR.uninhibit()

    print(datetime.now() - st)
