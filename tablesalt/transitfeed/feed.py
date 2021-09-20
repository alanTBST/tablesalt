
"""
This module deals with transit feed data provided by Rejseplanen.
More specifically it is the static GTFS data.


"""

import ast
import json
import logging
import zipfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import AbstractSet, ClassVar, Dict, Optional, Tuple, Set, TypeVar, Any
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd
from pandas.core.frame import DataFrame
from tablesalt.resources.config import load_config

log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent
TEMP_DIR = Path(THIS_DIR, 'temp_gtfs_data')
ARCHIVE_DIR = Path(THIS_DIR, 'gtfs_archive')

CONFIG = load_config()

TNum = TypeVar('TNum', int, float)

def _download_latest_feed(
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
        raise URLError("Could not download GTFS data")

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
    return (
        old_route_types_map,
        rail_route_types,
        bus_route_types
        )


def _load_process_trips(
    gtfs_dir: Path,
    rail_route_ids: Set[str],
    route_agency: Dict[str, str]
    ) -> Tuple[Set[int], Dict[int, str]]:

    trips_fp = list(gtfs_dir.glob('trips*'))[0]
    trips = pd.read_csv(trips_fp, usecols=['trip_id', 'route_id'])

    rail_trips = trips.query("route_id in @rail_route_ids").copy()
    rail_trips['agency_name'] = rail_trips.loc[:, 'route_id'].replace(route_agency)

    trip_agency = dict(zip(rail_trips['trip_id'], rail_trips.loc[:, 'agency_name']))
    rail_trip_ids = set(rail_trips.trip_id)

    return rail_trip_ids, trip_agency

def _load_process_stop_times(
    gtfs_dir: Path,
    rail_trip_ids: Set[int],
    trip_agency: Dict[int, str]
    ) -> DataFrame:

    stoptimes_fp = list(gtfs_dir.glob('stop_times*'))[0]
    stop_times = pd.read_csv(stoptimes_fp, usecols=['trip_id', 'stop_id'])

    rail_stop_times = stop_times.query("trip_id in @rail_trip_ids").copy()

    trip_stops = rail_stop_times.groupby('trip_id')['stop_id'].apply(tuple)
    trip_stops = trip_stops.reset_index()
    trip_stops = trip_stops.drop_duplicates('stop_id')
    trip_stops.loc[:, 'agency_name'] = \
        trip_stops.loc[:, 'trip_id'].replace(trip_agency)

    return trip_stops

class TransitFeedObject(ABC):

    @classmethod
    @abstractmethod
    def from_archive(self) -> 'TransitFeedObject':
        pass

    @classmethod
    @abstractmethod
    def latest(self, latest_data: DataFrame) -> 'TransitFeedObject':
        pass


class Agency(TransitFeedObject):

    def __init__(
        self,
        agency_data: Dict[int, str]
        ) -> None:
        self._data = agency_data

    def __getitem__(self, item: int) -> str:

        return self._data.__getitem__(item)

    def get(
        self,
        item: int,
        default: Optional[str] = None
        ) -> Optional[str]:
        return self._data.get(item, default)

    @classmethod
    def from_archive(cls) -> 'Agency':
        with open(ARCHIVE_DIR / 'agency.json', 'r') as f:
            agency_data = json.load(f)
        return cls(agency_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Agency':

        agency_data = dict(
            zip(
                latest_data.loc[:, 'agency_id'],
                latest_data.loc[:, 'agency_name']
            )
        )
        return cls(agency_data)


class Stops(TransitFeedObject):
    def __init__(
        self,
        stops_data: Dict[int, Dict[str, Any]]
        ) -> None:

        self._data = stops_data


    @classmethod
    def from_archive(cls) -> 'Stops':
        with open(ARCHIVE_DIR / 'stops.json', 'r') as f:
            stop_data = json.load(f)
        return cls(stop_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Stops':

        latest_data = latest_data.fillna('')
        stops_data = latest_data.set_index('stop_id').T.to_dict()

        return cls(stops_data)

    def _update_archive(self) -> None:

        return


class Routes(TransitFeedObject):

    OLD_ROUTE_MAP: ClassVar[Dict[int, int]]
    BUS_ROUTE_TYPES: ClassVar[Set[int]]
    RAIL_ROUTE_TYPES: ClassVar[Set[int]]

    OLD_ROUTE_MAP, RAIL_ROUTE_TYPES, BUS_ROUTE_TYPES = _load_route_types()

    def __init__(
        self,
        routes_data: Dict[str, Dict[str, Any]]
        ) -> None:

        self._data: Dict[str, Dict[str, Any]] = routes_data

        self._rail_routes: Optional[Dict[str, Dict[str, Any]]] = None
        self._bus_routes: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def rail_routes(self) -> Dict[str, Dict[str, Any]]:
        if self._rail_routes is not None:
            return self._rail_routes
        self._rail_routes = {
            route_id: info for route_id, info in self._data.items() if
            info['route_type'] in self.RAIL_ROUTE_TYPES
            }
        return self._rail_routes

    @rail_routes.setter
    def rail_routes(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._rail_routes = value

    @property
    def bus_routes(self) -> Dict[str, Dict[str, Any]]:
        if self._bus_routes is not None:
            return self._bus_routes
        self._bus_routes = {
            route_id: info for route_id, info in self._data.items() if
            info['route_type'] in self.BUS_ROUTE_TYPES
        }
        return self._bus_routes


    @classmethod
    def from_archive(cls) -> 'Routes':
        with open(ARCHIVE_DIR / 'routes.json', 'r') as f:
            routes_data = json.load(f)
        return cls(routes_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Routes':

        latest_data.loc[:, 'route_type'] = \
            latest_data.loc[:, 'route_type'].replace(cls.OLD_ROUTE_MAP)

        routes_data = latest_data.set_index('route_id').T.to_dict()
        return cls(routes_data)
    def _update_archive(self) -> None:

        return

class Trips(TransitFeedObject):
    def __init__(self, trips_data) -> None:
        self._data = trips_data

    @classmethod
    def from_archive(cls) -> 'Trips':

        with open(ARCHIVE_DIR / 'trips.json', 'r') as f:
            trips_data = json.load(f)

        return cls(trips_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Trips':

        latest_data.set_index('trip_id').T.to_dict()

        trips_data = latest_data.set_index('trip_id').T.to_dict()
        return cls(trips_data)


class StopTimes(TransitFeedObject):

    def __init__(self, stoptimes_data) -> None:
        self._data = stoptimes_data

    @classmethod
    def from_archive(cls) -> 'StopTimes':
        with open(ARCHIVE_DIR / 'stop_times.json', 'r') as f:
            trips_data = json.load(f)
        return cls(trips_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'StopTimes':


        routes_data = latest_data.set_index('').T.to_dict()
        return cls(routes_data)


class TransitFeed:

    def __init__(
        self,
        agency: Agency,
        routes: Routes,
        stops: Stops
        ) -> None:

        self.agency = agency
        self.routes = routes
        self.stops = stops

    def rail_routes(self) ->  Dict[str, Dict[str, Any]]:
        routes = self.routes.rail_routes

        rail_routes = {}
        for route_id, info in routes.items():
            info['agency_name'] = self.agency.get(info['agency_id'], '')
            rail_routes[route_id] = info

        return routes

def update_archive(dataset: str) -> None:

    return


def latest_transitfeed(update_archive: bool = False) -> TransitFeed:
    """Factory function that returns the latest available
    transfeet data as a TransitFeed object

    :param update_archive: [description], defaults to True
    :type update_archive: bool, optional
    """
    latest_gtfs_data = _download_latest_feed()
    agency_df = latest_gtfs_data['agency.txt']
    agency = Agency.latest(agency_df)

    routes_df = latest_gtfs_data['routes.txt']
    routes = Routes.latest(routes_df)

    stops_df = latest_gtfs_data['stops.txt']
    stops = Stops.latest(stops_df)

    stop_times_df = latest_gtfs_data['stop_times.txt']

    trips_df = latest_gtfs_data['trips.txt']

    transfers_df = latest_gtfs_data['transfers.txt']

    calendar = latest_gtfs_data['calendar.txt']

    calendar_dates = latest_gtfs_data['calendar_dates.txt']


    return TransitFeed(agency, routes, stops)

def archived_transitfed() -> None:

    return