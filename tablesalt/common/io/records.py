# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:14:53 2021

@author: alkj


The data in the key-value stores is stored in bytes.

This data has been serialized using msgpack

This module contains the classes that allow that data
to be queried

"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Union, Tuple, Sequence, TypeVar, Set

class TripOrderError(Exception):
    """An error raised when the applicationsequence and the
    model (check type) to not match"""
    pass


HERE = Path(__file__).parent

T = TypeVar('T', int, str, datetime)


# TODO: this function should be in a different module
def _legify(seq: Sequence[T]) -> Tuple[Tuple[T, T], ...]:
    """
    'Legify' the input sequence as a tuple of leg tuples.

    :param seq: the sequence to transform
    :type seq: Sequence[T]
    :return: the legified sequence as a tuple of legs
    :rtype: Tuple[Tuple[T, T], ...]

    :Example:
        (1, 2, 3, 4, 5) ---> ((1, 2), (2, 3), (3, 4), (4, 5))

    """
    return tuple(zip(seq, seq[1::]))


S = Dict[int, Dict[str, int]]


def _ordered(obj):
    "order a record object by application sequence"
    return tuple(zip(*sorted(obj.data.items())))[1]


class Record:
    pass

class StopRecord(Record):
    """StopRecord"""
    def __init__(
            self,
            tripkey: Union[bytes, str, int],
            data: S
            ) -> None:
        """
        Class to add functionality to the value in the key-value store

        :param tripkey: the unique tripkey of the trip
        :type tripkey: Union[bytes, str, int]
        :param data: a dictionary value returned from the key-value store
        :type data: S -> Dict[int, Dict[str, int]]
        :return:
        :rtype: None

        """

        self.tripkey = tripkey
        self.data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({int(self.tripkey)}, {self.data})'

    def __hash__(self):
        return hash(self.tripkey)

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return self.stop_ids == other.stop_ids

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, stopid) -> bool:
        return stopid in self.stop_ids

    @property
    def origin(self) -> int:
        """
        Return the origin of the stop data.

        :raises TripOrderError: If there is a problem with the trip sequenc
        :return: The stopid of the origin
        :rtype: int

        """

        key = self.data[min(self.data)]
        if not key['model'] == 1:
            raise TripOrderError("Applicationsequence/Model mismatch")
        return key['stop']

    @property
    def destination(self) -> int:
        """
        Return the destination of the stop record

        :raises TripOrderError: If there is a problem with the trip sequence
        :return: The stopid of the destination
        :rtype: int

        """

        key = self.data[max(self.data)]
        if not key['model'] == 2:
            raise TripOrderError("Applicationsequence/Model mismatch")
        return key['stop']

    @property
    def stop_ids(self) -> Tuple[int, ...]:
        """
        Get the stopids on the trip.

        :return: A tuple of stopids on the trip.
        :rtype: Tuple[int, ...]

        """

        return tuple(x['stop'] for x in self.data.values())

    @property
    def transfers(self) -> Tuple[int, ...]:
        """
        Get the stopids of the transfers on the trip

        :return: A tuple of the transfer stopids
        :rtype: Tuple[int, ...]

        """

        orig, dest = min(self.data), max(self.data)
        inner_checks = {k: v for k, v in self.data.items() if k not in (orig, dest)}
        return tuple(x['stop'] for x in inner_checks.values())

    def models(self) -> Tuple[int, ...]:
        """
        Return the model variables of the trip

        :return: a tuple of the model variables.
        :rtype: Tuple[int, ...]

        """

        return tuple(x['model'] for x in self.data.values())

    def model_legs(self) -> Tuple[Tuple[int, int], ...]:
        """
        Return the model varriables in leg form.

        :return: A tuple of tuple legs.
        :rtype: Tuple[Tuple[int, int], ...]

        """
        return _legify(self.models())

    def legs(self) -> Tuple[Tuple[T, T], ...]:
        """
        Return the stopids in leg format

        :return: a tuple of leg tuples
        :rtype: Tuple[Tuple[T, T], ...]

        """
        return _legify(self.stop_ids)

    def static_legs(self) -> Dict[int, int]:
        """
        Return the legs where the user checks in/out at the same location

        Returns
        -------
        Dict[int, int]
            A dictionary of leg number (starting from zero) and the stopid
            that the static leg occurs at
        """
        legs = self.legs()
        return {i: j[0] for i, j in enumerate(legs) if j[0]==j[1]}
    start = origin
    end = destination


class TimeRecord(Record):
    """TimeRecord"""
    def __init__(
            self,
            tripkey: Union[bytes, str, int],
            data: Dict[int, str]
            ) -> None:
        """
        Class to add functionality to the value in the key-value store

        :param tripkey: The tripkey in the key-value store
        :type tripkey: Union[bytes, str, int]
        :param data: time data returned from the key-value store.
        :type data: Dict[int, str]
        :return:
        :rtype: None

        """

        self.tripkey = tripkey
        self.data = data
        self.first_check_in: datetime = \
            datetime.fromisoformat(self.data[min(self.data)])

    def __repr__(self):
        return f'{self.__class__.__name__}({int(self.tripkey)}, {self.data})'

    def __hash__(self):
        return hash(self.tripkey)

    def __contains__(self, hour: int) -> bool:
        return hour == self.start_hour

    @property
    def start_hour(self) -> int:
        """
        Return the starting hour of the trip.

        :return: hour (24 hour clock)
        :rtype: int

        """

        return self.first_check_in.hour

    def trip_date(self) -> date:
        """
        Return the date on which the trip occured.

        :return: The date of the start of the trip
        :rtype: date

        """

        return self.first_check_in.date()

    def weekday(self) -> int:
        """
        Get the weekday on which the trip occured.

        :return: weekday (int form)
        :rtype: int

         - 0 --> 'Monday'
         - 1 --> 'Tuesday'

         - 6 --> 'Sunday'

        """
        return self.first_check_in.weekday()

    def trip_duration(self) -> float:
        """
        Return the total trip duration in minutes

        :return: the trip duration
        :rtype: float

        """

        final_check_out = datetime.fromisoformat(
            self.data[max(self.data)]
            )

        return (final_check_out - self.first_check_in).seconds / 60

    def _order_times(self) -> Tuple[datetime, ...]:

        return tuple(datetime.fromisoformat(x) for x in _ordered(self))

    def leg_start_times(self) -> Dict[int, datetime]:
        """
        Return the start time of each leg

        Returns
        -------
        Dict[int, datetime]
            the leg number as key and a datetime as value.

        """
        return {i: j for i, j in enumerate(sorted(_ordered(self))[:-1])}

    def legs(self):
        """
        Return the time sequence as a tuple of legs.

        :return: a tuple of time legs
        :rtype: TYPE

        """

        times = self._order_times()
        return _legify(times)

    def leg_durations(self) -> Dict[int, float]:
        """
        Return the duration of each leg on the trip

        Returns
        -------
        Dict[int, float]
            the leg number as key and a time in minutes as value.

        """

        return {i: (j[1] - j[0]).seconds / 60 for
                i, j in  enumerate(self.legs())}

fp = HERE / 'passengertypes.json'
with open(fp, 'r') as f:
    PAS_TYPES = {k: set(v) for k, v in json.load(f).items()}

CARD_TYPE = {
    1: "personal",
    2: "flex",
    3: "anonymous"
    }

class PassengerRecord(Record):
    """PassengerRecord"""
    def __init__(
            self,
            tripkey: Union[bytes, str, int],
            data: Dict[str, int]
            ) -> None:
        """
        Class to add functionality to the value in the passenger key-value store.

        :param tripkey: The tripkey in the key-value store
        :type tripkey: Union[bytes, str, int]
        :param data: passenger data returned from the key-value store.
        :type data: Dict[str, int]
        :return:
        :rtype: None

        """
        self.tripkey = tripkey
        self.data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({int(self.tripkey)}, {self.data})'

    def __hash__(self):
        return hash(self.tripkey) + hash(self.card_type)

    def __contains__(self, val):
        val = val.lower()
        try:
            return PAS_TYPES[val].intersection(self.passenger_types)
        except KeyError:
            return False

    @property
    def total_passengers(self) -> int:
        """
        The total number of people//things checked in on a trip

        :return: total passengers
        :rtype: int

        """

        return self.data['pt']

    @property
    def passenger_types(self) -> Set[int]:

        return {v for k, v in self.data.items()
                if k in ('t1', 't2', 't3')}
    @property
    def card_type(self) -> str:
        return CARD_TYPE[self.data['c']]

    def _has_passenger(self, pastype: str) -> bool:
        return any(x in PAS_TYPES[pastype] for
                   x in self.passenger_types)

    @property
    def has_bike(self) -> bool:
        return self._has_passenger('bicycle')
    @property
    def has_dog(self) -> bool:
        return self._has_passenger('dog')
    @property
    def has_child(self) -> bool:
        return self._has_passenger('child')

    @property
    def has_youth(self) -> bool:
        return self._has_passenger('youth')
    @property
    def has_adult(self) -> bool:
        return self._has_passenger('adult')
    @property
    def has_pensioner(self) -> bool:
        return self._has_passenger('pensioner')
    @property
    def has_handicap(self) -> bool:
        return self._has_passenger('handicap')

    def adult_and_child(self) -> bool:

        return self.has_adult and self.has_child

    def people_count(self):
        raise NotImplementedError("oops")


class OperatorRecord(Record):
    """OperatorRecord"""
    def __init__(
            self,
            tripkey: Union[bytes, str, int],
            data: S
            ) -> None:
        """
        Class to add functionality to the operator value in the key-value store.

        :param tripkey: The tripkey in the datastore
        :type tripkey: Union[bytes, str, int]
        :param data: a dictionary value returned from the key-value store
        :type data: Dict[int, Dict[str, int]]
        :return: ''
        :rtype: None

        """

        self.tripkey = tripkey
        self.data = data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({int(self.tripkey)}, {self.data})'

    def __hash__(self):
        return hash(self.tripkey)

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.all_operators == other.all_operators) and \
            (self.routes == other.routes)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __contains__(self, val) -> bool:
        val = val.lower()
        return (val in self.all_operators) or (val in self.routes)

    @property
    def start_operator(self) -> str:
        """
        Get the first operator on the trip.

        :return: an operator string
        :rtype: str

        """
        return self.data[min(self.data)]['operator'].lower()
    @property
    def end_operator(self) -> str:
        """
        Get the last operator on the trip.

        :return: an operator string
        :rtype: str

        """
        return self.data[max(self.data)]['operator'].lower()

    @property
    def all_operators(self) -> Tuple[str, ...]:
        """
        Get the sequence of operators on the trip.

        :return: a tuple of operator strings
        :rtype: Tuple[str, ...]

        """
        return tuple(x['operator'].lower() for x in _ordered(self))

    @property
    def is_kombi(self) -> bool:
        """
        Determine if the trip has more than one operator.

        :return: 1 of the trip has more than one operator, 0 otherwise
        :rtype: bool

        """

        return len(self.all_operators) == 1
    @property
    def routes(self) -> Tuple[str, ...]:
        """
        Get all of the routes used on the trip.

        :return: tuple of route strings
        :rtype: Tuple[str, ...]

        """
        return tuple(x['route'].lower() for x in self.data.values())

    def legs(self) -> Tuple[Tuple[T, T], ...]:
        """
        Return the operator sequence as legs.

        :return: a tuple of operator legs
        :rtype: Tuple[Tuple[T, T], ...]

        """


        return _legify(self.all_operators)

    def start(self):
        return self.start_operator
    def end(self):
        return self.end_operator

class PriceRecord(Record):

    def __init__(
            self,
            tripkey: Union[bytes, str, int],
            data: Dict[str, Union[int, float]]
            ) -> None:
        """
        Class to add functionality to the value in the price store

        :param tripkey: the tripkey in the datastore
        :type tripkey: Union[bytes, str, int]
        :param data: the data value returned
        :type data: Dict[str, Union[int, float]]
        :return: ''
        :rtype: None

        """

        self.tripkey = tripkey
        self.data = data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({int(self.tripkey)}, {self.data})'

    def __hash__(self):
        return hash(self.tripkey)

    @property
    def paid(self) -> float:
        """
        Return the total paid for the trip.

        :return: the price paid in kroner
        :rtype: float

        """

        return self.data['price']


class TripRecord:
    """TripRecord"""

    def __init__(
            self,
            tripkey: Union[bytes, str, int]
            ) -> None:
        """
        Class to unify the Records class

        :param tripkey: The tripkey in the datastore
        :type tripkey: Union[bytes, str, int]
        :return: ''
        :rtype: None

        """
        self.tripkey = tripkey
