"""

Author: Alan Jones alkj@tbst.dk

"""
from collections import defaultdict

import numpy as np


def add_dicts(dict1, dict2):
    """


    Parameters
    ----------
    dict1 : TYPE
        DESCRIPTION.
    dict2 : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """

    return {key: dict1.get(key, 0) + dict2.get(key, 0)
              for key in set(dict1) | set(dict2)}


def _legify(v):

    return tuple((v[i], v[i+1]) for i in range(len(v)-1))

def sep_legs(val):
    """


    Parameters
    ----------
    val : TYPE
        DESCRIPTION.

    Returns
    -------
    trip : TYPE
        DESCRIPTION.

    """
    """
    Description:
        Input:
        Output:
    """
    trip = []
    for i, j in enumerate(val):
        try:
            leg = (j, val[i+1])
            trip.append(leg)
        except IndexError:
            break
    trip = tuple(trip)
    return trip

def legify():
    return

def create_legs(ddict):
    """


    Parameters
    ----------
    ddict : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """


    """
    creates tuples of the rejsekort checks that elucidate the individual
    legs on the journey

    Input: dictionary of...
    Output: dictionary of...
    """
    for_keys = []
    for val in ddict.values():
        trip = sep_legs(val)
        for_keys.append(trip)
    for_keys = tuple(for_keys)

    return dict(zip(ddict.keys(), for_keys))

def check_zone_dist(leg, distances):
    """


    Parameters
    ----------
    leg : TYPE
        DESCRIPTION.
    distances : TYPE
        DESCRIPTION.

    Returns
    -------
    check_zone_dist : TYPE
        DESCRIPTION.

    """

    """
    check the distance travelled on a leg
    """

    check_zone_dist = tuple(distances[x] for x in leg[1])
    check_zone_dist = (leg[0], tuple(zip(leg[1], check_zone_dist)))

    return check_zone_dist

def op_count(leg):
    """


    Parameters
    ----------
    leg : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    """
    counts the distinct operators per leg in the leg tuple
    """
    count = tuple(len(set(x)) for x in leg)

    return tuple(zip(leg, count))

def divide_dict(dictionary, chunk_size):
    """
    Divide one dictionary into several dictionaries
    Return a list, each item is a dictionary

    Parameters
    ----------
    dictionary : TYPE
        DESCRIPTION.
    chunk_size : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    count_ar = np.linspace(0, len(dictionary), chunk_size + 1, dtype= int)
    group_lst = []
    temp_dict = defaultdict(lambda : None)
    i = 1
    for key, value in dictionary.items():
        temp_dict[key] = value
        if i in count_ar:
            group_lst.append(temp_dict)
            temp_dict = defaultdict(lambda : None)
        i += 1
    return [dict(x) for x in group_lst]



def split_list(alist, wanted_parts=1):
    """


    Parameters
    ----------
    alist : TYPE
        DESCRIPTION.
    wanted_parts : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    list
        DESCRIPTION.

    """

    """
    A list splitting function that separates a large list into the
    desired number of sublists given in wanted_parts
    Used mostly in the if __name__ == "__main__" conditional
    to split input lists of files for use in parallel processing
    """

    length = len(alist)

    return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
                  for i in range(wanted_parts)]





