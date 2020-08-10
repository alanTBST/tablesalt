# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:23:33 2019

@author: alkj

functions for analysing and distributing
the zone shares for kombi operator rejsekort
trips

"""
from itertools import groupby
from collections import Counter


from tablesalt.revenue import operatorvalidation
from tablesalt.common.io import mappers
from tablesalt.common import triptools


def property_dicts(operator_legs, usage_legs, multi_zone_properties):
    """
    generator that yields the input dictionary for the
    calculation

    parameters
    ----------
    operator_legs:
          the dictionary of legified operators
    zone_legs:
          dict of legified zones
    stop_legs:
          dict of legified stop ids
    multi_zone_properties:
          the dict of zone properties of kombi trips
    touched_zones:
          the dict of zones touched for each trip
    """

    all_keys = operator_legs.keys()

    for tripkey in all_keys:

        try:
            stop_legs = triptools.sep_legs(
                multi_zone_properties[tripkey]['stop_sequence']
                )
            zone_legs = triptools.sep_legs(
                multi_zone_properties[tripkey]['zone_sequence']
                )
            try:
                border_legs = multi_zone_properties[tripkey]['border_legs']
            except KeyError:
                border_legs = []
            yield {
                'tripkey': tripkey,
                'stop_legs': stop_legs,
                'op_legs': operator_legs[tripkey],
                'usage_legs': usage_legs[tripkey],
                'zone_legs': zone_legs,
                'border_legs': border_legs,
                'touched_zones': multi_zone_properties[tripkey]['touched_zones'],
                'visited_zones': multi_zone_properties[tripkey]['visited_zones'],
                'total_zones': multi_zone_properties[tripkey]['total_travelled_zones']
                }
        except KeyError:
            continue

def aggregated_zone_operators(vals):
    """

    perform the aggregation

    parameter
    ---------
    vals:
        a list or tuple of tuples of the form:
            ((n_zones[0], op_id[0]), (n_zones[1], op_id[1]),....(n_zones[n], op_id[n]))
    """

    vals = list(vals.values())

    out_list = []
    for x in vals:
        if isinstance(x[0], int):
            out_list.append(x)
            continue
        for y in x:
            out_list.append(y)

    out_list = sorted(out_list, key=lambda x: x[1])

    return tuple(((sum(x[0] for x in grp), key)) for
                 key, grp in groupby(out_list, key=lambda x: x[1]))


def share_calculation(operator_legs, usage_legs,
                      zone_properties, border_stations):
    """
    calculate the shares for
    """

    properties = property_dicts(
        operator_legs, usage_legs, zone_properties
        )
    processed_properties = \
        operatorvalidation.procprop(properties, border_stations)

    for_share_assignment = {}
    track_back = {}
    n = 0
    for val in processed_properties:
        out = {}
        zone_counts = Counter(val['visited_zones'])
        if max(zone_counts.values()) == 1:
            # this works for
            for i, imputed_leg in enumerate(val['imputed_zone_legs']):
                for zone in imputed_leg:
                    if zone in val['nlegs_in_touched']:
                        if val['nlegs_in_touched'][zone] == 1:
                            out[zone] = 1, val['new_op_legs'][i][0]
                        else:
                            counts = Counter(val['ops_in_touched'][zone])
                            try:
                                out[zone] = tuple((v/val['nlegs_in_touched'][zone], k) for
                                                  k, v in counts.items())
                            except ZeroDivisionError:
                                out[zone] = 1, val['new_op_legs'][i][0]
                    else:
                        out[zone] = 1, val['new_op_legs'][i][0]
            for_share_assignment[val['tripkey']] = aggregated_zone_operators(out)

        else:
            n += 1 # track back
            track_back[val['tripkey']] = val
        if out:
            for_share_assignment[val['tripkey']] = aggregated_zone_operators(out)

    return for_share_assignment
