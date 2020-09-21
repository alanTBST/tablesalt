"""
Working file for operator leg validation

"""

from itertools import groupby, chain
from collections import Counter

from tablesalt.common.io import mappers
from tablesalt.common import triptools


def operators_in_touched_(tzones, zonelegs, oplegs, border_zones=None):
    """
    determine the operators in the zones that are touched
    returns a dictionary:

    parameters
    -----------
    tzones:
        a tuple of the zones touched by
        a rejsekort tap on a trip
    zonelegs:
        a legified tuple of zones
    oplegs:
        a legified tuple of operators

    """
    ops_in_touched = {}
    for tzone in tzones:
        ops = []
        for i, j in enumerate(zonelegs):
            if tzone in j:
                l_ops = list(oplegs[i])
                ops.extend(l_ops)
        # modulo:  only put in one operator value per leg
        ops_in_touched[tzone] = \
        tuple(j for i, j in enumerate(ops) if i % 2 == 0)
    if not border_zones:
        return ops_in_touched

    return {k:v if k not in border_zones else (v[0],)
            for k, v in ops_in_touched.items()}

def impute_leg(zone_leg, vis_zones):
    """
    for the two touched zones on a zone leg,
    fill in the zones that the leg travels through

    parameters
    ----------
    zone_leg:
        the tuple of zones to impute
    vis_zones:
        the tuple of visited_zones

    """

    vis_zones_ = list(vis_zones)

    visited_idxs = [vis_zones_.index(int(zone)) for zone in zone_leg]

    return vis_zones[visited_idxs[0]: visited_idxs[1] + 1]


def impute_zone_legs(trip_zone_legs, visited_zones):

    return tuple(impute_leg(leg, visited_zones)
                 for leg in trip_zone_legs)

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

def needs_assignment_check(oplegs):
    """
    return boolean
    True if there needs to be an operator check
    False otherwise
    """
    return any(len(set(leg)) > 1 for leg in oplegs)

def removeCoTr(val, CoTr):
    """
    remove the checkout - checkin again (CoTr) legs
    """
    # (2, 4) corresponds to (Co, Tr) currently
    if CoTr not in val['usage_legs']:
        return val
    CoTr_idxs = []
    for i, j in enumerate(val['usage_legs']):
        if j == CoTr:
            CoTr_idxs.append(i)
    
    new_val = val.copy()
    for x in ('stop_legs', 'op_legs', 'zone_legs', 'border_legs'):
        try:
            new_val[x] = tuple(
                j for i, j in enumerate(new_val[x]) if i not in CoTr_idxs
                )
        except (KeyError, ValueError, TypeError):
            pass
    return new_val


def contains_border(stoplegs, border_stations):
    """
    parameters
    ----------
    stoplegs:
        tuple of legified stop id legs
    border_stations:
        dict of border stations
    returns boolean
    True if
    """

    return any(x in border_stations for x in chain(*stoplegs))

def _no_borders(val):

    val['nlegs_in_touched'] = {
        tzone: len([x for x in val['zone_legs'] if tzone in x])
        for tzone in val['touched_zones']
        }
    val['ops_in_touched'] = operators_in_touched_(
        val['touched_zones'], val['zone_legs'], val['new_op_legs']
        )
    val['imputed_zone_legs'] = \
    impute_zone_legs(val['zone_legs'], val['visited_zones'])

    return val


def _with_borders(val, border_stations):

    val['imputed_zone_legs'] = \
    impute_zone_legs(val['zone_legs'], val['visited_zones'])

    nlegs_in_touched = {}

    bstations = set(border_stations)
    bleg_stops = set(chain(*[j for i, j in enumerate(val['stop_legs'])
                     if i in val['border_legs']]))
    try:
        border_station_zones = \
        border_stations[list(bstations.intersection(bleg_stops))[0]]
    except IndexError:
        border_station_zones = []

    for tzone in val['touched_zones']:
        tzone_count = 0
        for i, j in enumerate(val['zone_legs']):
            if tzone in j:
                if i not in val['border_legs'] or \
                tzone not in border_station_zones:
                    tzone_count += 1
                elif i in val['border_legs']:
                    imputed = val['imputed_zone_legs'][i]
                    if all(x in imputed for x in border_station_zones):
                        tzone_count += 1

        nlegs_in_touched[tzone] = tzone_count
    val['nlegs_in_touched'] = nlegs_in_touched

    val['ops_in_touched'] = operators_in_touched_(
        val['touched_zones'], val['zone_legs'], val['new_op_legs'],
        border_zones=border_station_zones)

    return val

def legops(new_legs):
    """
    just legify the output from determin_operator
    """

    out = []
    for x in new_legs:
        if len(x) == 1:
            out.append((x[0], x[0]))
        else:
            out.append((x[0], x[1]))
    return tuple(out)



def procprop(properties, border_stations):
    """
    process_properties frim multisharing
    """
    rev_model_dict = {v:k for k, v in mappers['model_dict'].items()}
    co_tr_tuple = (rev_model_dict['Co'], rev_model_dict['Tr'])
    # missed_check_point = []
    # output = []
    # op_erros = []
    # other_errors = []
    for val in properties:
        try:
            val = removeCoTr(val, co_tr_tuple)
            bordercheck = contains_border(val['stop_legs'], border_stations)
            try:
                new_legs = tuple(
                    OPGETTER.station_pair(*x, format='operator') for x in val['stop_legs']
                    )
                if not all(x for x in new_legs):
                    # missed_check_point.append(val)
                    continue
                val['new_op_legs'] = legops(new_legs)
            except (KeyError, ValueError):
                # op_erros.add(val['tripkey'])
                continue
 
            if bordercheck:
                val = _with_borders(val, border_stations)
            else:
                val = _no_borders(val)
            yield val

        except Exception:
            # other_errors.append(val)
            continue

