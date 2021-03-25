# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:10:21 2020

@author: alkj
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:51:11 2019

@author: alkj, email: alkj@tbst.dk, alanksjones@gmail.com




This script calls the api that creates the
zonemaps on the DOT site for pendler data

it produces a msgpack file named
zone_relations.msgpack

This data is necessary for pendler revenue distribution

Be aware that the contents of the msgpack file is in bytes,
convert as required

"""

import urllib
import json
from itertools import combinations, permutations
from multiprocessing.pool import ThreadPool
import pandas as pd 

import msgpack
from tqdm import tqdm


def call_api(szdz):
    """
    Call the DOT API.

    This is the api that delivers the frontend at:
    "https://map.dtinf.dk/?mapmode=pendler&amp;lfs=12&amp;zoom=9&amp;opaque=0.6&amp;comp=true"

    Parameters
    ----------
    szdz : tuple/list
        this is a two tuple of (startzone, destinationzone)
            using the national style zone numbers
            eg zone '01' or '1' is zone 1001
            it can also be a two element list of integers.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    response : dict
        a dictionary of {'DestinationZone': dz,
                         'StartZone': sz,
                         'Zones': validzones}

    """
    if not isinstance(szdz, (tuple, list)):
        raise TypeError("szdz must be a tuple or a list" )
    sz = int(szdz[0])
    dz = int(szdz[1])
    try:
        response = urllib.request.urlopen(
            f'https://map.dtinf.dk/lib/relation.php?o={sz}&d={dz}')
        response = json.load(response)
        # breakpoint()
        response['DestinationZone'] = dz
        response['StartZone'] = sz
        response['Zones'] = tuple(response['Zones'])
        return response

    except Exception as e:
        response = {'DestinationZone': dz,
                    'StartZone': sz,
                    'Error': str(e)}
        return response

def main(maxpairs=None):
    """
    main function that makes threaded calls to the DOT
    API. creates the zone_relations msgpack file
    for use in the pendler revenue distribution
    program
    # TODO!!!! Make this for takst sjælland
    """

    hovedstad = list(range(1001, 1300))
    perms_old = list(combinations(hovedstad, 2))[:maxpairs]
    perms = list(permutations(hovedstad, 2))[:maxpairs]
    with ThreadPool(50) as pool:

        results = pool.imap(call_api, perms)
        res_dict = {}
        for i, res in tqdm(enumerate(results), total=len(perms)):
            res_dict[i] = res
        
        missed = {k:v for k, v in res_dict.items() if len(v) < 4}
        print('\nmissed: ', len(missed),flush=True)

        # new_missed = {}
        # for k, v in tqdm(missed.items()):
        #     res = call_api((v['StartZone'], v['DestinationZone']))
        #     new_missed[k] = res

        # res_dict.update(new_missed)

    # TODO: CHANGE LOCATION
    # with open(r'zone_relations.msgpack', 'wb') as f:
    #     msgpack.pack(res_dict, f)
    
        return res_dict
def ibsmain(maxpairs=100 )  :  
    ''' KJust a very slow experiment'''
    hovedstad = list(range(1001, 1300))
    perms = list(combinations(hovedstad, 2))[:maxpairs]
    perms = list(combinations(hovedstad, 2))[:maxpairs]
    
    res_dict={p : call_api(p) for p in perms }      
    missed = {k:v for k, v in res_dict.items() if len(v) < 4}
    print('missed: ', len(missed))
    # breakpoint()

    new_missed = {k : call_api((v['StartZone'], v['DestinationZone'])) for  k,v in missed.items()}

    # TODO: CHANGE LOCATION
    # with open(r'zone_relations.msgpack', 'wb') as f:
    #     msgpack.pack(res_dict, f)
    
    return res_dict,new_missed 


if __name__ == "__main__":

    res_dict = main(None)
    import pickle
    ZoneDistanceDict = {k : v for k,v in res_dict.items() if len(v) >=4}
    all_routes = {(v['StartZone'],v['DestinationZone']):v for k,v in res_dict.items() if len(v) >= 4}
    ZonesToDf = [[v['StartZone'],v['DestinationZone'], v['DirectDistance'],v['PaidZones'],v['DirectDistance']-v['PaidZones']] for k,v in all_routes.items()]
    ZonesDf = pd.DataFrame(ZonesToDf,columns= ['StartZone','DestinationZone','DirectDistance','PaidZones','DiffZones'])
    from1001 = ZonesDf.query('StartZone == 1001')
    to1001   = ZonesDf.query('DestinationZone == 1001')
    #%% pickle
    with open('ZoneDistanceDict.pc', 'wb') as handle:
       pickle.dump(ZoneDistanceDict, handle)
    #%%   
    with open('ZoneDistanceDict.pc', 'rb') as handle:
       ZoneDIstanceTest = pickle.load(handle)

