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
from tablesalt.common.io.datastores import make_store
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import db_paths, find_datastores
from tablesalt.running import WindowsInhibitor
from tablesalt.topology import ZoneGraph, ZoneSharer
from tablesalt.topology.stationoperators import StationOperators

def load(store, zonemap):

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

    return stop_dict, zone_dict, usage_dict


def analyse_trips(store):
    stop_dict, usage_dict = load(store)
    return

def main():


    feed = transitfeed.transitfeed_from_zip(
        r'H:\datastores\GTFSstores\20190911_20191204.zip'
        )
    zonemap = feed.stop_zone_map()

    parser = TableArgParser('year')
    return
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

