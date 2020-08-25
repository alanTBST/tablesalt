# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 03:58:53 2020

@author: alkj
"""

import os
import subprocess
import ast
import pkg_resources

import msgpack
import pandas as pd

def _load_zone_relations():
    """


    Returns
    -------
    zonerelations : TYPE
        DESCRIPTION.

    """


    if os.path.isfile('zone_relations.msgpack'):
        with open('zone_relations.msgpack', 'rb') as fp:
            zonerelations = msgpack.load(fp)

    else:
        # TODO make sure this returns the actual result
        zonerelations = subprocess.run("python zonerelations.py",
                                       shell=True,
                                       capture_output=True)
    return zonerelations


def _load_old_relations():
    """


    Returns
    -------
    zonerelations_old : TYPE
        DESCRIPTION.

    """

    fpath = 'zone_relations.msgpack'
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as fp:
            zonerelations_old = msgpack.load(fp, strict_map_key=False)
    return zonerelations_old


def _find_differences(zonerelations, zonerelations_old):
    """


    Parameters
    ----------
    zonerelations : TYPE
        DESCRIPTION.
    zonerelations_old : TYPE
        DESCRIPTION.

    Returns
    -------
    diff : TYPE
        DESCRIPTION.

    """


    diff = {}
    for k, v in zonerelations.items():
        if zonerelations_old[k] == v:
            continue

        if ((v[b'StartZone'] == zonerelations_old[k][b'StartZone']) and
           (v[b'DestinationZone'] == zonerelations_old[k][b'DestinationZone']) and
           (v[b'Zones'] == zonerelations_old[k][ b'Zones'])):
            continue
        diff[k] = v
    return diff

def _proc_zone_relations(zone_rels: dict):
    """


    Parameters
    ----------
    zone_rels : dict
        DESCRIPTION.

    Returns
    -------
    zone_rels : TYPE
        DESCRIPTION.

    """


    wanted_keys = ('StartZone',
                   'DestinationZone',
                   'PaidZones',
                   'ValidityZones',
                   'Zones')

    zone_rels = {k: {k1: v1 for k1, v1 in v.items() if k1 in wanted_keys}
                 for k, v in zone_rels.items()}
    zone_rels = list(zone_rels.values())

    for x in zone_rels:
        x['ValidZones'] = _unpack_valid_zones(x['Zones'])


    return zone_rels

def _load_kombi_results():
    """


    Returns
    -------
    dict
        DESCRIPTION.

    """


    results = pd.read_csv('sjælland/pendlerkeys2019.csv', index_col=0)
    results = results.to_dict(orient='index')

    return {ast.literal_eval(k): v for k, v in results.items()}

def _unpack_valid_zones(zonelist):

    return tuple(x['ZoneID'] for x in zonelist)

def main():
    """


    Returns
    -------
    None.

    """

    zone_rels = _load_old_relations()
    zone_rels = _proc_zone_relations(zone_rels)
    results = _load_kombi_results()

    out = {}
    for i, x in enumerate(zone_rels):
        x = {k: v for k, v in x.items() if k != 'Zones'}
        matched_result = results.get(x['ValidZones'])
        if matched_result is not None:
            val = {**x, **matched_result}
        else:
            val = x
        out[i] = val

    frame = pd.DataFrame.from_dict(out, orient='index')
    cp = frame.copy(deep=True)
    cp.columns = ['DestinationZone', 'StartZone', 'PaidZones', 'ValidityZones',
       'ValidZones', 'first', 'metro', 'movia', 'stog', 'dsb', 'n_users',
       'n_period_cards', 'n_trips']
    df = pd.concat([frame, cp])
    # add new frame d=o and o=d
    df.to_csv('sjælland/zone_relation_keys2019.csv', index=False)
    return frame







