
"""
This module deals with transit feed data provided by Rejseplanen.
More specifically it is the static GTFS data.


"""

import ast
import json
import logging
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from io import BytesIO, TextIOWrapper
from itertools import groupby
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any, ClassVar, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Iterable
from urllib.error import URLError
from urllib.request import urlopen

import msgpack
import pandas as pd
from msgpack.exceptions import ExtraData
from pandas.core.frame import DataFrame
from shapely import geometry, wkt
from shapely.geometry import LineString
from tablesalt.resources.config import load_config

log = logging.getLogger(__name__)

THIS_DIR = Path('C:\\Users\\B087115\\Documents\\GitHub\\tablesalt\\tablesalt\\transitfeed') #(__file__).parent
ARCHIVE_DIR = Path(THIS_DIR, '__gtfs_archive__')

CONFIG = load_config()

TNum = TypeVar('TNum', int, float)


class TransitFeedError(Exception):
    pass

def download_latest_feed() -> Dict[str, pd.core.frame.DataFrame]:
    """download the latest gtfs data from rejseplan,
    write the text files to a folder named temp_gtfs_data

    :raises Exception: If we cannot download the data for any reason
    :return: a dictionary of dataframes for the gtfs txt files
    :rtype: Dict[str, pd.core.frame.DataFrame]
    """
    required = {
        'agency.txt',
        'stops.txt',
        'routes.txt',
        'trips.txt',
        'stop_times.txt',
        'calendar.txt',
        'calendar_dates.txt',
    }

    gtfs_url = CONFIG['rejseplanen']['gtfs_url']
    resp = urlopen(gtfs_url)

    if resp.code == 200:
        log.info("GTFS response success")
    else:
        log.critical(f"GTFS download failed - error {resp.code}")
        raise URLError("Could not download GTFS data")

    gtfs_data: Dict[str, pd.core.frame.DataFrame] = {}

    with zipfile.ZipFile(BytesIO(resp.read())) as zfile:
        names = zfile.namelist()
        missing = {x for x in required if x not in names}
        if missing:
            raise TransitFeedError(
                f"Mandatory dataset/s [{', '.join(missing)}] are missing from the feed."
                )
        for x in names:
            df = pd.read_csv(zfile.open(x), low_memory=False)
            gtfs_data[x] = df

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

class _TransitFeedObject(ABC):

    @classmethod
    @abstractmethod
    def from_archive(self) -> '_TransitFeedObject':
        pass

    @classmethod
    @abstractmethod
    def latest(self, latest_data: DataFrame) -> '_TransitFeedObject':
        pass


class Agency(_TransitFeedObject):

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


class Stops(_TransitFeedObject):
    def __init__(
        self,
        stops_data: Dict[int, Dict[str, Any]]
        ) -> None:

        self._data = stops_data

    def __getitem__(self, item: int) -> Dict[str, Any]:

        return self._data.__getitem__(item)

    def get(
        self,
        item: int,
        default: Optional[Any] = None
        ) -> Optional[Any]:

        return self._data.get(item, default)

    @classmethod


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


class Routes(_TransitFeedObject):

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

    def __getitem__(self, item: str) -> Dict[str, Any]:

        return self._data.__getitem__(item)

    def get(
        self,
        item: str,
        default: Optional[str] = None
        ) -> Any:

        return self._data.get(item, default)

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

class Trips(_TransitFeedObject):

    def __init__(
        self,
        trips_data: Dict[int, Dict[str, Any]]
        ) -> None:

        self._data = trips_data
        self._trip_route_map: Optional[Dict[int, str]] = None
        self._route_trip_map: Optional[Dict[str, Tuple[int, ...]]] = None

    def __getitem__(self, item: int) -> Dict[str, Any]:

        return self._data.__getitem__(item)

    def get(
        self,
        item: int,
        default: Optional[Dict[str, Any]] = None
        ) -> Optional[Dict[str, Any]]:
        return self._data.get(item, default)

    @property
    def trip_route_map(self) -> Optional[Dict[int, str]]:

        return self._trip_route_map

    @trip_route_map.setter
    def trip_route_map(self, value: Dict[int, str]) -> None:
        self._trip_route_map = value

    @property
    def route_trip_map(self) -> Optional[Dict[str, Tuple[int, ...]]]:
        return self._route_trip_map

    @route_trip_map.setter
    def route_trip_map(self, value: Dict[str, Tuple[int, ...]]) -> None:
        self._route_trip_map = value


    @classmethod
    def from_archive(cls) -> 'Trips':

        with open(ARCHIVE_DIR / 'trips.json', 'r') as f:
            trips_data = json.load(f)

        return cls(trips_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Trips':

        triproute = dict(zip(latest_data.loc[:, 'trip_id'], latest_data.loc[:, 'route_id']))

        route_trips = zip(latest_data.loc[:, 'route_id'], latest_data.loc[:, 'trip_id'])
        route_tripsmap = {key:tuple(x[1] for x in grp) for
                          key, grp in groupby(route_trips, key=itemgetter(0))}

        trips_data = latest_data.set_index('trip_id').T.to_dict()

        trip = cls(trips_data)
        trip.trip_route_map = triproute
        trip.route_trip_map = route_tripsmap

        return trip


class StopTimes(_TransitFeedObject):

    def __init__(
        self,
        stoptimes_data: Dict[int, Tuple[Dict[str, Any], ...]]
        ) -> None:
        self._data = stoptimes_data

    @classmethod
    def from_archive(cls) -> 'StopTimes':
        with open(ARCHIVE_DIR / 'stop_times.json', 'r') as f:
            stoptimes_data = json.load(f)
        return cls(stoptimes_data)


    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'StopTimes':
        df = latest_data.fillna('0')

        # this deals with older format gtfs from Rejseplan
        if not 'int' in df.loc[:, 'stop_id'].dtype.name:
            df.loc[:, 'stop_id'] = \
                df.loc[:, 'stop_id'].astype(str).str.strip('G').astype(int)

        df = df.sort_values(['trip_id', 'stop_sequence']) # assure trip order

        vals = df.itertuples(name='StopTimes', index=False)

        stoptimes_data = {key: tuple(x._asdict() for x in grp) for
             key, grp in groupby(vals, key=attrgetter('trip_id'))}

        return cls(stoptimes_data)

class Transfers(_TransitFeedObject):

    def __init__(self, transfers_data: List[Dict[str, Any]]) -> None:
        self._data = transfers_data

    @classmethod
    def from_archive(cls) -> 'Transfers':
        with open(ARCHIVE_DIR / 'transfers.json', 'r') as f:
            trips_data = json.load(f)
        return cls(trips_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Transfers':

        null_columns = []
        for col in latest_data.columns:
            if latest_data.loc[:, col].isnull().all():
                null_columns.append(col)
        new_cols = [x for x in latest_data.columns if x not in null_columns]
        df = latest_data.loc[:, new_cols]
        for col in df.columns:
                if col in ('transfer_type', 'min_transfer_time'):
                    df.loc[:, col] = df.loc[:, col].fillna(0).astype(int)
                else:
                    df.loc[:, col] = df.loc[:, col].fillna('')

        transfers_data = df.itertuples(name='Transfer', index=False)
        transfers_data = [x._asdict() for x in transfers_data]

        return cls(transfers_data)

class Calendar(_TransitFeedObject):

    def __init__(self, calender_data: Dict[int, Dict[str, int]]) -> None:
        self._data = calender_data

    def __getitem__(self, service_id: int) -> Dict[str, int]:
        return self._data.__getitem__(service_id)

    @classmethod
    def from_archive(cls) -> 'Calendar':
        with open(ARCHIVE_DIR / 'calendar.json', 'r') as f:
            calender_data = json.load(f)
        return cls(calender_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Calendar':
        calendar_data = latest_data.set_index('service_id').T.to_dict()
        return cls(calendar_data)

    def period(self) -> Tuple[datetime, datetime]:

        values = self._data.values()
        values_it = iter(values)
        first = next(values_it)

        start = first['start_date']
        end = first['end_date']

        start_time = datetime.strptime(str(start), '%Y%m%d')
        end_time = datetime.strptime(str(end), '%Y%m%d')

        return (start_time, end_time)


class CalendarDates(_TransitFeedObject):

    def __init__(
        self,
        calendar_dates_data: Dict[int, Tuple[Tuple[int, int], ...]]
        ) -> None:
        self._data = calendar_dates_data

    @classmethod
    def from_archive(cls) -> 'CalendarDates':
        with open(ARCHIVE_DIR / 'calendar_dates.json', 'r') as f:
            calender_date_data = json.load(f)
        return cls(calender_date_data)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'CalendarDates':

        latest_data = latest_data.sort_values('service_id')

        calendar_tuples = zip(latest_data['service_id'],
                              latest_data['date'],
                              latest_data['exception_type'])

        calendar_dates_data = {
            key: tuple((x[1], x[2]) for x in grp) for key, grp in
            groupby(calendar_tuples, key=itemgetter(0))
        }
        return cls(calendar_dates_data)



class Shapes(_TransitFeedObject):

    def __init__(self, shapes_data: Dict[int, LineString]) -> None:
        self._data = shapes_data


    def __getitem__(self, item: int) -> LineString:

        return self._data.__getitem__(item)

    def get(
        self,
        item: int,
        default: Optional[LineString] = None
        ) -> Optional[LineString]:
        return self._data.get(item, default)

    @classmethod
    def from_archive(cls) -> 'Shapes':

        return cls()

    @staticmethod
    def _group_to_linestring(grp: Iterable[Tuple[TNum, ...]]) -> LineString:

        string = str(tuple(str(x[2]) + ' ' + str(x[1]) for x in grp)).replace("'", "")

        return wkt.loads('LINESTRING ' + string)

    @classmethod
    def latest(cls, latest_data: DataFrame) -> 'Shapes':

        latest_data = latest_data.sort_values(['shape_id', 'shape_pt_sequence'])

        shapes = latest_data.itertuples(name=None, index=False)

        shape_lines = {
            key: cls._group_to_linestring(grp) for
            key, grp in groupby(shapes, key=itemgetter(0))
        }
        return cls(shape_lines)

    def to_geojson(self, filepath: str) -> None:
        """Write the shapes data to a geojson file

        :param filepath: the path of the file to write to
        :type filepath: str
        """

        features: List[Any] = []
        for k, v in self._data.items():
            geom = geometry.mapping(v)

            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "shape_id": k
                    }
                }
            features.append(feature)

        for_geojson = {
            "type": "FeatureCollection",
            "features": features
            }
        with open(filepath, 'w') as fp:
            json.dump(for_geojson, fp)


class TransitFeed:

    def __init__(
        self,
        agency: Agency,
        stops: Stops,
        routes: Routes,
        trips: Trips,
        stop_times: StopTimes,
        calendar: Calendar,
        calendar_dates: CalendarDates,
        transfers: Optional[Transfers] = None,
        shapes: Optional[Shapes] = None
        ) -> None:


        self.agency = agency
        self.routes = routes
        self.stops = stops
        self.trips = trips
        self.stop_times = stop_times
        self.calendar = calendar
        self.calendar_dates = calendar_dates

        self.transfers = transfers
        self.shapes = shapes

    def rail_routes(self) ->  Dict[str, Dict[str, Any]]:
        """Return only the rail routes of the feed

        :return: a dictionary of routes
        :rtype: Dict[str, Dict[str, Any]]
        """

        rail_routes = {}
        for route_id, info in self.routes.rail_routes.items():
            info['agency_name'] = self.agency.get(info['agency_id'], '')
            rail_routes[route_id] = info

        return rail_routes

    def bus_routes(self) -> Dict[str, Dict[str, Any]]:
        """Return only the bus routes of the feed

        :return: a dictionary of routes
        :rtype: Dict[str, Dict[str, Any]]
        """

        bus_routes = {}
        for route_id, info in self.routes.bus_routes.items():
            info['agency_name'] = self.agency.get(info['agency_id'], '')
            bus_routes[route_id] = info

        return bus_routes

    def feed_period(self) -> Tuple[datetime, datetime]:
        """Return the time period that the feed is valid for

        :return: a tuple of start_time, end_time
        :rtype: Tuple[datetime, datetime]
        """

        return self.calendar.period()

    def to_archive(self) -> None:
        """Add the transitfeed to the package archive
        """

        period = self.feed_period()
        period_string = '_'.join(datetime.strftime(x, '%Y%m%d') for x in period)
        path = ARCHIVE_DIR / period_string
        path.mkdir(parents=True, exist_ok=True)

        for dset, klass in self.__dict__.items():
            filepath = path / dset
            with open(filepath, 'wb') as f:
                try:
                    msgpack.pack(klass._data, f)
                except TypeError:
                    klass.to_geojson(filepath)
                else:
                    pass

        shutil.make_archive(str(path), 'zip', path)
        shutil.rmtree(path)

def latest_transitfeed(
    add_transfers: bool = True,
    add_shapes: bool = True,
    add_to_archive: bool = False
    ) -> TransitFeed:
    """Factory function that returns the latest available transit
    feed data fro Rejseplan as a TransitFeed object

    :param add_transfers: [description], defaults to True
    :type add_transfers: bool, optional
    :param add_shapes: [description], defaults to True
    :type add_shapes: bool, optional
    :param add_to_archive: [description], defaults to False
    :type add_to_archive: bool, optional
    :return: [description]
    :rtype: TransitFeed
    """


    # order matches TransitFeed args
    required = [
        ('agency.txt', Agency.latest),
        ('stops.txt', Stops.latest),
        ('routes.txt', Routes.latest),
        ('trips.txt', Trips.latest),
        ('stop_times.txt', StopTimes.latest),
        ('calendar.txt', Calendar.latest),
        ('calendar_dates.txt', CalendarDates.latest)
    ]

    latest_gtfs_data = download_latest_feed()
    req_classes = tuple(v(latest_gtfs_data[k]) for k, v in required)

    transfers: Optional[Transfers]
    shapes: Optional[Shapes]

    if add_transfers:
        transfers_df = latest_gtfs_data['transfers.txt']
        transfers = Transfers.latest(transfers_df)
    else:
        transfers = None

    if add_shapes:
        shapes_df = latest_gtfs_data['shapes.txt']
        shapes = Shapes.latest(shapes_df)
    else:
        shapes = None

    feed = TransitFeed(*req_classes, transfers=transfers, shapes=shapes)
    if add_to_archive:
        feed.to_archive()

    return feed


def available_archives() -> List[str]:
    """List the available transitfeed archives

    :return: a list of period_strings
    :rtype: List[str]
    """

    available = ARCHIVE_DIR.glob('*.zip')

    return [x.stem for x in available]

def archived_transitfeed(period_string: str) -> TransitFeed:

    archives = available_archives()
    if period_string not in archives:
        raise FileNotFoundError(
            f"{period_string} is not in the archive. Available periods are {archives}"
            )

    fp = ARCHIVE_DIR / (period_string + '.zip')

    data: DefaultDict[str, Any]
    data = defaultdict()

    with zipfile.ZipFile(fp, 'r') as zfile:
        names = zfile.namelist()
        for name in names:
            with zfile.open(name) as f:
                try:
                    d = msgpack.unpack(f, strict_map_key=False)
                except ExtraData: # we know this is the shapes geojson
                    jsonbytes = zfile.read(name)
                    d = json.loads(jsonbytes.decode('utf-8'))
                data[name] = d

    agency = Agency(data['agency'])
    stops = Stops(data['stops'])
    routes = Routes(data['routes'])
    trips = Trips(data['trips'])
    stop_times = StopTimes(data['stop_times'])
    calendar = Calendar(data['calendar'])
    calendar_dates = CalendarDates(data['calendar_dates'])

    transfers = Transfers(data['transfers'])
    shapes = Shapes(data['shapes'])

    feed = TransitFeed(
        agency,
        stops,
        routes,
        trips,
        stop_times,
        calendar,
        calendar_dates,
        transfers=transfers,
        shapes=shapes
        )

    return feed
