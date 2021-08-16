"""
Functions and methods to interact with
the Delrejser Rejsekort Data.

These classes read data from lmdb key-value
stores that are created by running ingestors.py
"""
import inspect
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from operator import attrgetter
from pathlib import Path
from typing import (
    Dict,
    Any,
    Generator,
    DefaultDict,
    Iterable,
    Set,
    Optional,
    Union,
    List,
    Tuple,
    Sequence,
    TypeVar
    )

import lmdb  # type: ignore
import msgpack  # type: ignore
from numpy import arange  # type: ignore

# from iomapping import Mappers
from .records import (
    StopRecord,
    TimeRecord,
    TripOrderError,
    PassengerRecord,
    OperatorRecord,
    TripRecord,
    PriceRecord
    )

R = TypeVar('R', StopRecord, TimeRecord, PassengerRecord, PriceRecord, OperatorRecord)


def _make_key_val(
    _dict: Dict[Any, Any],
    db_path: str,
    map_size: float = None
    ) -> None:
    """
    Create a lmdb key-value store

    :param _dict: the dictionary to store.
    :type _dict: Dict[Any, Any]
    :param db_path: path to the store to create.
    :type db_path: str
    :param map_size: The memory mappping size of the datastore, defaults to None
    :type map_size: float, optional
    :return: None
    :rtype: None

    """

    with lmdb.open(db_path, map_size=map_size) as env:
        with env.begin(write=True) as txn:
            for key, val in _dict.items():
                if not isinstance(key, bytes):
                    if not isinstance(key, str):
                        key = str(key)
                    key = bytes(key, 'utf-8')
                test_key = txn.get(key)
                if test_key:
                    valb = msgpack.unpackb(test_key, strict_map_key=False)
                    try:
                        val = {**valb, **val}
                    except TypeError:
                        continue
                val = msgpack.packb(val)
                txn.put(key, val)

def make_store(
    d: Dict[Any, Any],
    db_path: str,
    start_size: float = 0.5,
    size_limit: float = 30
    ) -> None:
    """
    Create an lmdb database from a dictionary

    :param d:  A Dictionary to write to the key-value
    :type d: Dict[Any, Any]
    :param db_path: the path location of the store to create.
    :type db_path: str
    :param start_size: The starting size of the datastore in GB.
                       The default is 1, defaults to 0.5
    :type start_size: float, optional
    :param size_limit: The maximum size to try and make the key-value store in GB, defaults to 30
    :type size_limit: float, optional
    :raises a: ValueError if a store cannot be created with the
        given size_limit
    :raises ValueError:
    :return:
    :rtype: None


    :Example:

    """

    map_sizes = {x: x * 1024 * 1024 * 1024 for
                 x in arange(start_size , size_limit, 0.25)}  # try increaments of 256mb

    i = start_size
    while True:
        try:
            _make_key_val(d, db_path, map_size=map_sizes[i])
            break
        except lmdb.MapFullError:
            i += 1
        if i >= size_limit:
            raise ValueError(
                f"failed to make key-value store: "
                f"size {db_path} > {size_limit}gb limit"
                )


def _delrejser_query():

    return 



class DelrejserStore:
    """DelrejserStore"""

    def __init__(self, path: str) -> None:
        """
        Access the delrejser data in the lmdb stores

        :param path: path of the delrejser store
        :type path: str
        :return: ''
        :rtype: None

        :Example:

        """


        self.path = Path(path)
        self.stop_store = StopStore(self.path / 'stops')
        self.time_store = TimeStore(self.path / 'time')
        self.passenger_store = PassengerStore(self.path / 'pas')
        self.trip_user_store = TripUserStore(self.path / 'tripcard')
        self.operator_store = OperatorStore(self.path / 'operator')
        self.price_store = PriceStore(self.path / 'price')



        self._store_arguments = {
            'stops': set(inspect.signature(self.stop_store.query).parameters),
            'time': set(inspect.signature(self.time_store.query).parameters),
            'passenger': set(inspect.signature(self.passenger_store.query).parameters),
            'operator': set(inspect.signature(self.operator_store.query).parameters),
            'price': set(inspect.signature(self.operator_store.query).parameters)
            }
        
        self._store_functions = {
            'stops': self.stop_store.query,
            'time': self.time_store.query,
            'passenger': self.passenger_store.query,
            'operator': self.operator_store.query,
            'price': self.passenger_store.query
            }
    
    def _determine_first_query(self, **kwargs):

        query_arguments = set(kwargs.keys())

        counts = ((k, len(v.intersection(query_arguments))) for 
                   k, v in self._store_arguments.items())
    
        return max(counts, key=lambda x: x[1])[0]


    def _query_remaining_stores():
        # put the other functions in a threadpool and call

        return 
    
    def query(self, **kwargs) -> Generator[List[TripRecord], None, None]:
        """[summary]

        :yield: a list of TripRecords of the given chunksize
        :rtype: Generator[List[TripRecord], None, None]
        """
        
        init_chunksize = 1_000
        chunksize = kwargs.pop('chunksize', 1_000)
        
        
        users = kwargs.pop('users', None)
        if users is not None:
            pass
        functions = self._store_functions.copy()

        first_query = self._determine_first_query(**kwargs)
        first_query_function = functions.pop(first_query)
        first_query_kws = {k: v for k, v in kwargs.items() if 
                           k in self._store_arguments[first_query]}
        
        first_values = first_query_function(chunksize=init_chunksize, **first_query_kws)
        
        trip_chunks = []
        for records in first_values:
            tripkeys={x.tripkey for x in records}
            new_records, first_records = self._get_other_records(
                kwargs, functions, records, tripkeys
                )
            
            for n in zip(first_records, *new_records):
                trip_chunks.append(TripRecord(*n))
                if len(trip_chunks) == chunksize:
                    yield trip_chunks
                    trip_chunks.clear()
            else:
                yield trip_chunks

    def _get_other_records(self, kwargs, functions, records, tripkeys):
        
        common_keys = set()           
        new_records = []
        for store, store_query in functions.items():
            store_query_kws = {k: v for k, v in kwargs.items() if 
                                   k in self._store_arguments[store]}
            next_store_records = list(store_query(tripkeys=tripkeys, **store_query_kws))[0]
            if not common_keys:
                common_keys = tripkeys
            new_records.append(next_store_records)
            common_keys.intersection_update({rec.tripkey for rec in next_store_records})
            
        new_records = ([y for y in rec if y.tripkey in common_keys] for rec in new_records)
        first_records = (x for x in records if x.tripkey in common_keys)
        new_records = (sorted(x, key=attrgetter('tripkey')) for x in new_records)
        first_records = sorted(first_records, key=attrgetter('tripkey'))
        
        return new_records,first_records
                
# ABC_QueryableBaseStore
# @abstractmethod
# def query():
#     pass

class _BaseStore:
    def __init__(self, path: Path) -> None:
        """
        Base class for individual store classes
        :param path: path to the key-value store
        :type path: Path
        :return:
        :rtype: None

        """

        self.path = path
        self.env = None

    def __enter__(self):
        self.env = lmdb.open(str(self.path))
        return self.env

    def __exit__(self, _1, _2, _3):
        self.env.close()

    def __repr__(self):

        return f'{self.__class__.__name__}({self.path!r})'

    def _check_conditions(self, record, **kwargs):

        passenger_kws = (
            'adult',
            'child',
            'youth',
            'pensioner',
            'handicap',
            'bike',
            'dog'
            )

        isin_kws = (
            'route',
            'uses_operator',
            'visits',
            'start_hour',
            # 'weekday',
            )

        start_kws = (
            'origin',
            'startswith'
            )

        end_kws = (
            'destination',
            'endswith'
            )
        #todo
        # other_kws = ('trip_duration', 'date', 'weekday', 'longest_leg')

        flags: List[bool] = []

        for k, v in kwargs.items():
            if k in passenger_kws:
                if v is not None:
                    if v:
                        flags.append(k in record)
                    else:
                        flags.append(k not in record)
            elif k in isin_kws:
                if v is not None:
                    if isinstance(v, (tuple, list)) :
                        flags.append(any(x in record for x in v))
                    elif isinstance(v, set):
                        flags.append(all(x in record for x in v))
                    else:
                        flags.append(v in record)
            elif k in start_kws:
                if v is not None:
                    try:
                        if isinstance(v, (tuple, list, set)):
                            flags.append(any(x == record.start for x in v))
                        else:
                            flags.append(v == record.start)
                    except TripOrderError:
                        pass
            elif k in end_kws:
                if v is not None:
                    try:
                        if isinstance(v, (tuple, list, set)):
                            flags.append(any(x == record.end for x in v))
                        else:
                            flags.append(v == record.end)
                    except TripOrderError:
                        pass
            else:
                raise ValueError(f"argument {k} not recognised")

        if not flags:
            return True

        return all(x for x in flags)

    def _get_defined_keys(self, tripkeys: Iterable[bytes], cursor_obj: lmdb.Cursor, obj: R):

        vals = cursor_obj.getmulti(tripkeys)

        return [obj(x[0], msgpack.unpackb(x[1], strict_map_key=False)) for x in vals]

    def query(self, obj: R, **kwargs) -> Generator[List[R], None, None]:
        """
        Query the data store

        :param obj: the record class to use
        :type obj: TypeVar('R', StopRecord, TimeRecord, PassengerRecord, PriceRecord, OperatorRecord)
        :param **kwargs: query filters

        - passenger_kws = (
            'adult',
            'child',
            'youth',
            'pensioner',
            'handicap',
            'bike',
            'dog'
            )

        - isin_kws = (
            'route',
            'uses_operator',
            'visits',
            'start_hour',
            )

        - start_kws = (
            'origin',
            'startswith'
            )

        - end_kws = (
            'destination',
            'endswith'

        :type **kwargs: Union[int, str, bool]
        :raises AttributeError: if query method is not supported
        :yield: a list of Records
        :rtype: Generator[List[Record], None, None]

        """

        if isinstance(self, TripUserStore):
            raise AttributeError(
                f"{self.__class__.__name__} does not support queries"
                )

        chunksize = kwargs.pop('chunksize', 10_000)
        tripkeys = kwargs.pop('tripkeys', None)
        chunk = []
        with self as store:
            with store.begin() as txn:
                cursor = txn.cursor()
                if tripkeys is not None:
                    if tripkeys is not None:
                        records = self._get_defined_keys(tripkeys, cursor, obj)
                        for record in records:
                            if self._check_conditions(record, **kwargs):
                                chunk.append(record)
                                if len(chunk) == chunksize:
                                    yield chunk
                                    chunk.clear()
                        else:
                            yield chunk    
                
                else:
                    for k, v in cursor:
                        try:
                            val = msgpack.unpackb(v, strict_map_key=False)
                        except TypeError:
                            continue
                        record = obj(k, val)
                        if self._check_conditions(record, **kwargs):
                            chunk.append(record)
                            if len(chunk) == chunksize:
                                yield chunk
                                chunk.clear()
                    else:
                        yield chunk


class TripUserStore(_BaseStore):
    """TripUserStore"""
    def __init__(self, path: Path) -> None:
        """
        Class for reading tripkeys by yser for the key-value store

        :param path: the path to the TripUserStore
        :type path: Path
        :return:
        :rtype: None

        :Example:
            Insert example here


        """


        self.path = path
        super().__init__(path)

    def get_tripkeys(
            self,
            users: Optional[Sequence[str]] = None,
            ) -> DefaultDict[str, List[int]]:
        """
        Return the tripkeys for the given user card numbers

        :param users: The encrypted card numbers of the users, defaults to None
        :type users: Optional[Sequence[str]], optional
        :return: A dictionary with the user card number as key and their trips
        :rtype: DefaultDict[str, int]

        """

        if users is not None:
            userids = set(users)
        user_trips = defaultdict(list)
        with self as store:
            with store.begin() as txn:
                cursor = txn.cursor()
                for k, v in cursor:
                    v_dec = msgpack.unpackb(v)
                    if users:
                        if v_dec in userids:
                            user_trips[v_dec].append(k)
                    else:
                        user_trips[v_dec].append(k)

        return user_trips


class StopStore(_BaseStore):
    """StopStore"""

    def __init__(self, path: Path) -> None:
        """
        Class for reading StopRecords for the key-value store

        :param path: the path to the StopStore
        :type path: Path
        :return:
        :rtype: None

        """


        self.path = path
        super().__init__(path)


    def query(
        self,
        origin: Optional[int] = None,
        destination: Optional[int] = None,
        visits: Optional[Union[Tuple[int, ...], List[int], Set[int]]] = None,
        tripkeys: Optional[bytes] = None,
        chunksize: int = 1_000
        ) -> Generator[List[R], None, None]:
        """
         Get stop values from the key-value store.

        :param origin: The stopid of the origin, defaults to None
        :type origin: Optional[int], optional
        :param destination: the stop id of the destination, defaults to None
        :type destination: Optional[int], optional
        :param visits: the stopids of visted stops on the trips, defaults to None
        :type visits: Optional[Union[Tuple[int, ...], List[int], Set[int]]], optional
        :param chunksize: length of list to return with each iteration, defaults to 10_000
        :type chunksize: int, optional
        :return: a list of StopRecords of given chunksize
        :rtype: Generator[List[StopRecord], None, None]

        """

        return super().query(
            StopRecord,
            origin=origin,
            destination=destination,
            visits=visits,
            chunksize=chunksize, 
            tripkeys=tripkeys
            )

class WeekdayDateError(Exception):
    """Error for data and weekday error"""

class TripDurationError(Exception):
    """Error for invalid trip duration"""

class TimeStore(_BaseStore):
    """TimeStore"""

    def __init__(self, path: Path) -> None:
        """
        Class for reading TimeRecords for the key-value store

        :param path: the path to the TimeStore
        :type path: Path
        :return:
        :rtype: None

        """

        self.path = path
        super().__init__(path)
    def query(
        self,
        start_hour: Optional[Union[int, Tuple[int, ...]]] = None,
        # weekday: Optional[Union[int, Tuple[int, ...]]] = None,
        tripkeys: Iterable[bytes] = None,
        chunksize: int = 10_000
        )  -> Generator[List[R], None, None]:
        """
        Read TimeRecords from the key value store

        :param start_hour: the starting hour of the trip, defaults to None
        :type start_hour: Optional[Union[int, Tuple[int, ...]]], optional
        :param # weekday: the weekday (integer 0->6) of the trip, defaults to None
        :type # weekday: Optional[Union[int, Tuple[int, ...]]], optional
        :param chunksize: length of list to return with each iteration, defaults to 10_000
        :type chunksize: int, optional
        :return: a generator object that returns lists of Records
        :rtype: Generator[List[TimeRecord], None, None]

        """

        return super().query(
            TimeRecord,
            start_hour=start_hour,
            # weekday=weekday,
            tripkeys=tripkeys,
            chunksize=chunksize
            )

class PassengerStore(_BaseStore):
    """PassengerStore"""

    def __init__(self, path: Path) -> None:
        """
        Class for reading PassengerRecords for the key-value store.

        :param path: the path to the passenger store
        :type path: Path
        :return:
        :rtype: None

        """

        self.path = path
        super().__init__(path)

    def query(
        self,
        adult: Optional[bool] = None,
        child: Optional[bool] = None,
        youth: Optional[bool] = None,
        pensioner: Optional[bool] = None,
        handicap: Optional[bool] = None,
        bike: Optional[bool] = None,
        dog: Optional[bool] = None,
        tripkeys: Iterable[bytes] = None,
        chunksize: int = 10_000
        ) -> Generator[List[R], None, None]:
        """
        Read PassengerRecords from the key value store.

        :param adult: If an adult is on the trip, defaults to None
        :type adult: Optional[bool], optional
        :param child: If a child is on the trip, defaults to None
        :type child: Optional[bool], optional
        :param youth: If a youth is on the trip, defaults to None
        :type youth: Optional[bool], optional
        :param pensioner: If a pensioner is on the trip, defaults to None
        :type pensioner: Optional[bool], optional
        :param handicap: If a handicapped user is on the trip, defaults to None
        :type handicap: Optional[bool], optional
        :param bike: If a bicyle is on the trip, defaults to None
        :type bike: Optional[bool], optional
        :param dog: If a dog is on the trip, defaults to None
        :type dog: Optional[bool], optional
        :param chunksize: length of list to return with each iteration, defaults to 10_000
        :type chunksize: int, optional
        :return: a generator that yields lists of Records of the given chunksize
        :rtype: Generator[List[PassengerRecord]], None, None]

        """


        return super().query(
            PassengerRecord,
            adult=adult,
            child=child,
            youth=youth,
            pensioner=pensioner,
            handicap=handicap,
            bike=bike,
            dog=dog,
            tripkeys=tripkeys,
            chunksize=chunksize
            )

class OperatorStore(_BaseStore):
    def __init__(self, path: Path) -> None:
        """
        Class for reading OperatorRecords for the key-value store.

        :param path: the path to the OperatorStore
        :type path: Path
        :return:
        :rtype: None

        """

        self.path = path
        super().__init__(path)


    def query(
        self,
        route: Optional[Union[str, Tuple[str, ...], Set[str]]] = None,
        uses_operator: Optional[Union[str, Tuple[str, ...], Set[str]]] = None,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        tripkeys: Iterable[bytes] = None,
        chunksize: int = 10_000
        ) -> Generator[List[R], None, None]:
        """
        Read OperatorRecords from the key value store.

        :param route: a route number/id, defaults to None
        :type route: Optional[Union[str, Tuple[str, ...], Set[str]]], optional
        :param uses_operator: one of the operators, defaults to None
        :type uses_operator: Optional[Union[str, Tuple[str, ...], Set[str]]], optional
        :param startswith: if the trip starts with operator, defaults to None
        :type startswith: Optional[str], optional
        :param endswith: if the trip ends with operator, defaults to None
        :type endswith: Optional[str], optional
        :param chunksize:  length of list to return with each iteration, defaults to 10_000
        :type chunksize: int, optional
        :return: a generator that yields lists of len chunksize
        :rtype: Generator[List[OperatorRecord]], None, None]

        """


        return super().query(
            OperatorRecord,
            route=route,
            uses_operator=uses_operator,
            startswith=startswith,
            endswith=endswith,
            tripkeys=tripkeys,
            chunksize=chunksize
            )


class PriceStore(_BaseStore):
    def __init__(self, path: Path) -> None:
        """
        Class for reading PriceRecords for the key-value store.

        :param path: path the the price store
        :type path: Path
        :return: None
        :rtype: None

        """

        self.path = path
        super().__init__(path)

    def query(self, tripkeys: Iterable[bytes] = None):
        return super().query(
            PriceRecord,
            tripkeys=tripkeys
            )

    def get_zeros_trips(self) -> Set[bytes]:
        """
        Get the trips that have price of 0.

        :return: a set of tripkeys
        :rtype: Set[bytes]

        """


        zero_keys = set()
        with self as store:
            with store.begin() as txn:
                cursor = txn.cursor()
                for k, v in cursor:
                    try:
                        val = msgpack.unpackb(v, strict_map_key=False)
                    except TypeError:
                        continue
                    price = PriceRecord(k, val)
                    if price.paid == 0:
                        zero_keys.add(k)
        return zero_keys
