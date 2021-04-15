# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:42:36 2020

@author: alkj

Classes for manipulating Pendler Product data sets
"""

import os
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator,
                    List, Optional, Sequence,
                    Set, Tuple, TypeVar, Union)

import msgpack  # type: ignore
import pandas as pd  # type: ignore
import pkg_resources

ZONE = Dict[str, Union[int, str]]
ZONES = Sequence[ZONE]
RELATION = Union[str, int, float, ZONES]

def _load_zonerelations() -> Dict[int, Dict[str, RELATION]]:
    """load the zone relations data"""
    fpath = pkg_resources.resource_filename(
        'tablesalt',
        os.path.join(
            'resources',
            'revenue',
            'zone_relations.msgpack'
            )
        )
    with open(fpath, 'rb') as f:
        zonerelations = msgpack.load(f, strict_map_key=False)
    return zonerelations


def _zones_to_paid(
    zonerel: Dict[int, Dict[str, RELATION]]
    ) -> Dict[Tuple[int, ...], int]:
    """create zones -> npaid dict
    from the zone relation dictionary"""
    outdict: Dict[Tuple[int, ...], int] = {}
    # zones: ZONE
    for _, v in zonerel.items():
        try:
            zones = tuple(x['ZoneID'] for x in v['Zones'])
            paid = v['PaidZones']
            outdict[zones] = paid
        except KeyError:
            pass

    return outdict

def _split_period_indices(group: List[Any]) -> List[int]:
    """find the indices of the group that indicate a change in valid zones
    from one period to the next"""
    s_indx = []
    for i, j in enumerate(group):
        try:
            if i != 0:
                if j[2] != group[i-1][2]:
                    s_indx.append(i)
        except IndexError:
            pass
    return s_indx


def _process_static_cards(date_dict):
    """process the cards that have the same set of valid zones for
    all of their periods. This are called static cards

    :param date_dict: [description]
    :type date_dict: [type]
    :return: [description]
    :rtype: [type]
    """
    stat_dict = {}
    for key, group in groupby(date_dict, key=itemgetter(0)):
        grp = list(group)
        p_dict = {}
        for _, j in enumerate(grp):
            p_dict[j[1]] = {'start': j[3], 'end':j[4], 'zones': j[2]}
        stat_dict[key] = p_dict

    return stat_dict


def _process_dynamic_cards(date_dict):
    """process the cards that have differing sets of valid zones for their
    periods. These are called dynamic cards

    :param date_dict: [description]
    :type date_dict: [type]
    :return: [description]
    :rtype: [type]
    """
    dyn_dict = {}

    for key, group in groupby(date_dict, key=itemgetter(0)):

        grp = list(group)
        p_dict = {}
        period_indices = _split_period_indices(grp)
        n_periods = len(period_indices) + 1

        for i in range(n_periods):
            if i == 0:
                records = grp[: period_indices[i]]
                p_dict[i] = {'start': min([x[3] for x in records]),
                             'end': max([x[4] for x in records]),
                             'zones': list(set(x[2] for x in records))[0]}
                continue
            p_indx = period_indices[i-1]
            if i < len(period_indices):
                records = grp[p_indx:p_indx + 1]
            else:
                records = grp[p_indx:]
            p_dict[i] = {'start': min([x[3] for x in records]),
                         'end': max([x[4] for x in records]),
                         'zones': list(set(x[2] for x in records))[0]}
        dyn_dict[key] = p_dict

    return dyn_dict


def _process_cards(prodzones,
                   product_dates,
                   dynamic: bool = False): # -> Dict[int, Dict]:
    """
    process the cards to return a dictionary
    return a  dictionary that has the cardnum as the key
    and values as dictionary:
            0: {'start': timestamp0, 'end':timestamp0, 'zones0': valid zones for period 0},
            1: {'start': timestamp1, 'end':timestamp1, 'zones1': valid zones for period 1},
            ....
    """
    dynamic_dates = {
        k: val for k, val in product_dates.items() if k
        in prodzones
        }

    dynamic_dates = {
        k: tuple(pd.to_datetime(x, dayfirst=True) for x in val) for
        k, val in dynamic_dates.items()
        }

    dates = (
        (k[0], k[1], val, dynamic_dates[k][0][0], dynamic_dates[k][0][1])
        for k, val in prodzones.items()
        )
    dates = sorted(dates, key=itemgetter(0, 3, 4))

    if not dynamic:
        return _process_static_cards(dates)

    return _process_dynamic_cards(dates)


class _PendlerInput():

    def __init__(self,
                 year: int,
                 products_path: str,
                 product_zones_path: str,
                 min_valid_days: Optional[int] = 14) -> None:
        """Class that loads and processes the input pendlerkombi data
        for the pendler revenue distribution models

        :param year: the year of analysis
        :type year: int
        :param products_path: the path to the rejsekort pendler products csv
        :type products_path: str
        :param product_zones_path: the path to the rejskort pendler valid zones csv
        :type product_zones_path: str
        :param min_valid_days: the minimum validity in days to use, defaults to 14
        :type min_valid_days: Optional[int], optional
        """

        self.year = year
        self.min_valid_days = min_valid_days
        self.products_path: str = products_path
        self.product_zones_path: str = product_zones_path
        self.paid_zones = _zones_to_paid(_load_zonerelations())

        self.products = self._load_valid_pendler_data()
        self.product_zones = self._get_product_zones(self.products)
        self.product_dates = self._get_product_dates(self.products)


    def _load_valid_pendler_data(self) -> pd.core.frame.DataFrame:
        """load the required pendler kombi products data

        :return: a dataframe with valid pendler kombi products for the year
        :rtype: pd.core.frame.DataFrame
        """

        col1 = ['EncryptedCardEngravedID', 'SeasonPassID',
                'Fareset', 'PsedoFareset',
                'SeasonPassName', 'SeasonPassZones',
                'ValidityStartDT', 'ValidityEndDT',
                'ValidDays', 'FromZoneNr',
                'ToZoneNr', 'PassagerType']

        try:
            pendler_product = pd.read_csv(
                self.products_path,
                encoding='iso-8859-1',
                sep=';',
                usecols= col1,
                dtype={'EncryptedCardEngravedID': str, 'SeasonPassID':int}
                )
        except ValueError:
            pendler_product = pd.read_csv(
                self.products_path,
                encoding='utf-8',
                sep=',',
                usecols= col1,
                dtype={'EncryptedCardEngravedID':str, 'SeasonPassID':int}
                )

        pendler_product = pendler_product.fillna(0)
        pendler_product.loc[:, ('FromZoneNr', 'ToZoneNr')] = \
        pendler_product.loc[:, ('FromZoneNr', 'ToZoneNr')].astype(int)

        pendler_product = pendler_product.query(
            "ValidDays >= @self.min_valid_days"
            )

        pendler_product = \
            pendler_product.query(
                "ValidityEndDT != '.' and ValidityStartDT != '.'"
                )
        pendler_product = \
            pendler_product.loc[
                pendler_product.loc[:, 'SeasonPassName'].str.lower().str.contains('kombi')
            ]

        return pendler_product.loc[
            pendler_product.loc[:, 'ValidityEndDT'].str.contains(str(self.year))
            ]

    def _get_product_zones(
            self,
            valid_pendler: pd.core.frame.DataFrame
        ): #-> Dict:
        """load the pendler valid zone data

        :param valid_pendler: dataframe of valid pendler data
        :type valid_pendler: pd.core.frame.DataFrame
        :return: a mapping of cardnum-seasonpass -> valid zones
        :rtype: Dict
        """
        pendler_zones = pd.read_csv(
            self.product_zones_path,
            sep=';'
            )
        pendler_zones['key'] = list(zip(
            pendler_zones['EncryptedCardEngravedID'],
            pendler_zones['SeasonPassID']
            ))

        valid_pendler['key'] = list(zip(
            valid_pendler['EncryptedCardEngravedID'],
            valid_pendler['SeasonPassID']))

        pendler_zones = pendler_zones.loc[
            pendler_zones.loc[:, 'key'].isin(valid_pendler['key'])
            ]

        pendler_zones = \
        pendler_zones.sort_values(
            ['EncryptedCardEngravedID', 'SeasonPassID']
            )
        pendler_zones = \
        pendler_zones.itertuples(name=None, index=False)

        return {key: tuple(x[2] for x in group) for key, group in
                groupby(pendler_zones, key=itemgetter(0, 1))}

    @staticmethod
    def _get_product_dates(valid_pendler: pd.core.frame.DataFrame): # -> Dict[Tuple[str, int], Tuple[]]:
        """transform the 'ValidityStartDT' and 'ValidityEndDT'
        to a dictionary for each season pass

        :param valid_pendler: [description]
        :type valid_pendler: [type]
        :return: pd.core.frame.DataFrame
        :rtype: Dict[Tuple[str, int], Tuple[]]
        """
        product_dates = valid_pendler.loc[
            :, ('EncryptedCardEngravedID', 'SeasonPassID',
                'ValidityStartDT', 'ValidityEndDT')
                ]
        product_dates = product_dates.sort_values(
            ['EncryptedCardEngravedID', 'SeasonPassID']
            )
        product_dates = product_dates.itertuples(name=None, index=False)

        return {key: tuple((x[2], x[3]) for x in group) for key, group in
                groupby(product_dates, key=itemgetter(0, 1))}

    @staticmethod
    def _find_dynamic_validity(vzone_dict: Dict[Tuple[str, int], Tuple[int, ...]]) -> Set[str]:
        """return the set of card numbers that have changing sets of valid zones

        :param vzone_dict: a dictionary of the valid zones for the period cards
        :type vzone_dict: Dict[Tuple[str, int], Tuple[int, ...]]
        :return: a set of card numbers
        :rtype: Set[str]
        """
        cards_seen = set()
        cardszones_seen = set()
        exceptions = set()
        for k, val in vzone_dict.items():
            card = k[0]
            cardzone = card, val
            if card in cards_seen and not cardzone in cardszones_seen:
                exceptions.add(card)
            cards_seen.add(card)
            cardszones_seen.add(cardzone)

        return exceptions

    @staticmethod
    def _split_static_dynamic(product_zones, dynamic_exceptions):
        """
        split the valid zones dictionary into two.

        static and dynamic. static cards have the same valid zone combinations
        for all of their periods and dynamic cards change at some point
        """

        static_card_zones = {k: val for k, val in product_zones.items()
                             if k[0] not in dynamic_exceptions}
        dynamic_card_zones = {k: val for k, val in product_zones.items()
                              if k[0] in dynamic_exceptions}

        return static_card_zones, dynamic_card_zones

    def get_user_data(self, users: str = 'all'): #Dict[]
        """return the desired user card data

        :param users: the subset of users to return, defaults to 'all'
            options ['all', 'static', 'dynamic']
        :type users: str, optional
        :raises ValueError: if users is not in ['all', 'static', 'dynamic']
        :return: user card data
        :rtype: Dict
        """
        if users not in ['all', 'static', 'dynamic']:
            raise ValueError(
                "users argument must be one of: 'all', 'static', 'dynamic'"
                )

        if users == 'all':

            return _process_cards(
                self.product_zones,
                self.product_dates,
                dynamic=False
                )

        static_card_zones, dynamic_card_zones = \
            self._split_static_dynamic(
                self.product_zones,
                self._find_dynamic_validity(self.product_zones)
                )

        if users == 'dynamic':
            return _process_cards(dynamic_card_zones, self.product_dates, dynamic=True)

        return _process_cards(static_card_zones,
                              self.product_dates,
                              dynamic=False)

class PendlerKombiUsers():
    """
    Class that adds functionality to the
    input from PendlerInput()

    It allows for easier subsetting of users
    based on valid zones, valid time periods/seasonpass ids, etc.
    """

    REGIONS = {
        'sydsjælland': (1200, 1300),
        'vestsjælland': (1100, 1200),
        'hovedstad': (1000, 1100),
        'sjælland': (1000, 1300)
        }
    def __init__(
        self,
        year: int,
        products_path: str,
        product_zones_path: str,
        min_valid_days = 14,
        user_group = 'all'
        ): # -> None:
        """class to interact with the pendler kombi user data

        :param year: the year of analysis
        :type year: int
        :param products_path: the path to the PeriodeProdukt.csv
        :type products_path: str
        :param product_zones_path: the path to the Zoner.csv
        :type product_zones_path: str
        :param min_valid_days: the minimum valid days for a card to be used,
                defaults to 14
        :type min_valid_days: int, optional
        :param user_group: 'all', 'static or 'dynamic', defaults to 'all'
        :type user_group: str, optional
        """
        self.input_data = _PendlerInput(
            year, products_path=products_path,
            product_zones_path=product_zones_path,
            min_valid_days=min_valid_days
            )

        self.users = self.input_data.get_user_data(users=user_group)

    @staticmethod
    def _valid_season_zones(pprod):
        """return all unique combinations of valid zones
        """
        return set(pprod.loc[:, 'SeasonPassZones'].unique())


    def _paid_filter(self, nzones: int) -> Set[Tuple[int, ...]]:
        """get the set of valid zones for the given nzones

        :param nzones: the number of paid zones
        :type nzones: int
        :return: a set of all valid zones where paid = nzones
        :rtype: Set[Tuple[int, ...]]
        """

        return {k for k, v in self.input_data.paid_zones.items()
                if v == nzones}

    def _filter_paid_zones(self, nzones: int) -> Dict[Tuple[str, int], Tuple[int,...]]:
        """filter the season passes by the number of paid zones

        :return: theseason passes that have the desired paid zones
        :rtype: Diict[Tuple[str, int], Tuple[int,...]]
        """
        zones = self._paid_filter(nzones)
        valid_paid_keys = {
            k:v for k, v in self.input_data.product_zones.items()
            if v in zones
            }

        return valid_paid_keys

    def _filter_ptype(self, ptype: str) -> Set[Tuple[str, int]]:
        """return the set of season passes that match the passenger type

        :param ptype: the passenger type
        :type ptype: str
        :raises ValueError: if the given passenger type is not in the dataset
        :return: a set of season passes
        :rtype: Set[Tuple[str, int]]
        """
        valid_ptype = set(self.input_data.products['PassagerType'])

        if not ptype in valid_ptype:
            raise ValueError(
                f"dataset does not have {ptype} passenger types"
                f"valid passenger types are {valid_ptype}"
                )

        passenger_type_keys = set(self.input_data.products.query(
            "PassagerType == @ptype"
            )['key'])

        return passenger_type_keys

    def _find_users_with_zones(self, chosen_zones: Iterable[int] = None):
        """
        for the input chosen_zones, find the users
        that have those valid zones

        parameter
        ---------
        chosen_zones:
            a set of tariff zones for the desired users
            if chosen_zones is None, returns users that
            have all zones

        returns a set of tuples (cardum, seasonpass num)
        """
        if not isinstance(chosen_zones, set):
            chosen_zones = set(chosen_zones)
        if chosen_zones:
            zones_userset = set()
            for k, v in self.users.items():
                for k1, v1 in v.items():
                    if set(v1['zones']) == chosen_zones:
                        zones_userset.add((k, k1))
            return zones_userset
        zones_userset = set()
        for k, v in self.users.items():
            for k1, v1 in v.items():
                if len(v1['zones']) > 90: # for all zone users
                    zones_userset.add((k, k1))

        return zones_userset

    def _subset_users(self, chosen_zones):
        """

        create a subset of the users dictionary for
        the users and the periods of they have the chosen zones

        returns a three tuple: user_subset, n_users, n_period_cards
        """

        zones_userset = self._find_users_with_zones(chosen_zones)

        zones_userset = sorted(
            zones_userset, key=itemgetter(0, 1)
            )
        zones_userset = {
            key: tuple(x[1] for x in grp) for
            key, grp in groupby(zones_userset, key=itemgetter(0))
            }

        user_subset = {}
        for k, v in self.users.items():
            if k not in zones_userset:
                continue
            user_d = {k1: v1 for k1, v1 in v.items()
                      if k1 in zones_userset[k]}
            user_subset[k] = user_d

        return (user_subset, len(user_subset),
                sum(len(x) for x in user_subset.values()))

    def _zone_in_region(self, zonenum: int, region: str) -> bool:
        x, y = self.REGIONS[region]
        return x < zonenum < y

    def _subset_takst(self, prodzones, takst):
        # *takst - list(takst) for mulit takstset

        if takst in self.REGIONS:
            count_func = lambda x: Counter(x)[True] == 1
        else:
            count_func = lambda x: Counter(x)[True] > 1

        takst_prods = {}
        for k, v in prodzones.items():
            try:
                in_takst = any(self._zone_in_region(x, takst) for x in v)
            except KeyError:
                in_takst = True
            if not in_takst:
                continue
            all_takst = (any(self._zone_in_region(x, reg) for x in v)
                         for reg in self.REGIONS)

            if count_func(all_takst):
                takst_prods[k] = v
        return takst_prods

    def get_data(
        self,
        paid_zones: Optional[int] = None,
        ptype = None,
        takst: Optional[str] = None,
        user_group: Optional[str] = 'all'
        ):

        prodzones = self.input_data.product_zones

        if paid_zones is not None:
            paid_valid = self._filter_paid_zones(paid_zones)
            prodzones = {
                k: v for k, v in prodzones.items() if k in paid_valid
                }
        if ptype is not None:
            pas_valid = self._filter_ptype(ptype)
            prodzones = {
                k: v for k, v in prodzones.items() if k in pas_valid
                }
        if takst is not None:
            prodzones = self._subset_takst(prodzones, takst)

        if user_group == 'all':
            return _process_cards(
                        prodzones,
                        self.input_data.product_dates,
                        dynamic=False
                        )
        static_card_zones, dynamic_card_zones = \
            self.input_data._split_static_dynamic(
                self.input_data.product_zones,
                self.input_data._find_dynamic_validity(prodzones)
                )
        stat = _process_cards(
            static_card_zones,
            self.input_data.product_dates,
            dynamic=False
            )

        if user_group == 'static':
            return stat

        if user_group == 'dynamic':
            dyn = _process_cards(dynamic_card_zones, self.input_data.product_dates, dynamic=True)

            return dyn

        raise ValueError(f'usergroup: {user_group} not recognised')

    def subset_time(self):
        """
        subset the user dictionary based on the
        validity time period
        """

        raise NotImplementedError('Not available yet')

    def subset_zones(self, zones=None):
        """
        subset the user dictionary based on
        the input zones
        parameters
        -----------
        zones: set, tuple, list
            the valid zones of wanted users

        """
        # if not stats:
        #     return self._subset_users()
        return self._subset_users(chosen_zones=zones)

