import ast
import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from urllib.request import urlopen

import pandas as pd
from pandas.core.frame import DataFrame
from tablesalt.resources.config import load_config

log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
TEMP_DIR = Path(THIS_DIR, 'temp_gtfs_data')
ARCHIVE_DIR = Path(THIS_DIR, 'gtfs_archive')

CONFIG = load_config()

def _download_new_gtfs(
    write_text_files: Optional[bool] = False
    ) -> Dict[str, pd.core.frame.DataFrame]:
    """download the latest gtfs data from rejseplan,
    write the text files to a folder named temp_gtfs_data

    :param write_text_files: write the .txt files to disk, defaults to False
    :type write_text_files: Optional[bool], optional
    :raises Exception: If we cannot download the data for any reason
    :return: a dictionary of dataframes for the gtfs txt files
    :rtype: Dict[str, pd.core.frame.DataFrame]
    """


    gtfs_url = CONFIG['rejseplanen']['gtfs_url']
    resp = urlopen(gtfs_url)

    if resp.code == 200:
        log.info("GTFS response success")
    else:
        log.critical(f"GTFS download failed - error {resp.code}")
        raise Exception("Could not download GTFS data")

    gtfs_data: Dict[str, pd.core.frame.DataFrame] = {}

    with zipfile.ZipFile(BytesIO(resp.read())) as zfile:
        for x in zfile.namelist():
            df = pd.read_csv(zfile.open(x), low_memory=False)
            gtfs_data[x] = df

    if write_text_files:
        for filename, df in gtfs_data.items():
            fp = TEMP_DIR / filename
            df.to_csv(fp, encoding='iso-8859-1')

    return gtfs_data

def _load_route_types() -> Tuple[Dict[int, int], Set[int], Set[int]]:
    """load the rail and bus route types from the config.ini file

    Unfortunately rejseplan does not fully implement the extended GTFS
    route types yet, so we return a mapping to the new codes as well

    :return: a tuple of (old_route_codes_map, rail_route_types, bus_route_types)
    :rtype: Tuple[Dict[int, int], Set[int], Set[int]]
    """

    old_route_types = dict(CONFIG['old_route_type_map'])

    old_route_types_map: Dict[int, int] = {
        int(k): int(v) for k, v in old_route_types.items()
        }

    rail_route_types = set(ast.literal_eval(
        CONFIG['current_rail_types']['codes']
        ))
    bus_route_types = set(ast.literal_eval(
        CONFIG['current_bus_types']['codes']
        ))
    return old_route_types_map, rail_route_types, bus_route_types



class _FeedObject:
    pass

class Agency(_FeedObject):

    def __init__(self, filepath: Optional[str] = None) -> None:

        # self, usedata='' (archive, new)
        self.filepath = filepath if filepath is not None else ARCHIVE_DIR / 'agency.json'

    def to_dict(self) -> None:

        return

    def read_json(self) -> None:

        return

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> None:

        return

class Stops(_FeedObject):
    def __init__(self) -> None:
        pass

class Routes(_FeedObject):

    def __init__(self) -> None:
        pass

class Trips(_FeedObject):

    def __init__(self) -> None:
        pass

class Transfers(_FeedObject):
    def __init__(self) -> None:
        pass


class TransitFeed:
    def __init__(self, *feeddata: _FeedObject) -> None:
        pass




