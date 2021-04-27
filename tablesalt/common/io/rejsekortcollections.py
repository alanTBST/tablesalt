
"""
Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com
"""

import json
import pkg_resources
from pathlib import Path

from typing import Dict, List, Union


col_dict: Dict[str, Union[List[int], List[List[int]], Dict[str, int], Dict[str, str]]] = { ... }

THIS_DIR = Path(__file__).parent

def _load_collection() -> col_dict:
    """Load the json file containing the rejsekort collections

    Returns
    -------
    Dict[str, Union[List[int], List[List[int]], Dict[str, int], Dict[str, str]]]
        A dictionary of the json file
    """

    fp = THIS_DIR.parent.parent / 'resources' / 'rejsekortcollections.json'
    with open(fp,
        'r', encoding='utf-8') as f:
        collection = json.load(f)

    return collection


def proc_collection(
    collection: col_dict
    ) -> col_dict:
    """process the dictionary in more python like types and
    add some extra keys needed for analysis

    Parameters
    ----------
    collection : Dict[str, Union[List[int], List[List[int]], Dict[str, int], Dict[str, str]]]
        the dictionary returned from _load_collection

    Returns
    -------
    Dict
        a processed dictionary of rejsekort specific objects
    """

    model_bytes = {
        bytes(k.encode('iso-8859-1')): bytes(v.encode('iso-8859-1')) for
        k, v in collection['model_dict'].items()
        }

    rev_model = {v: k for k, v in collection['model_dict'].items()}

    rev_model_bytes = {
        bytes(k.encode('iso-8859-1')): bytes(v.encode('iso-8859-1')) for
        k, v in rev_model.items()
        }

    collection['model_dict'] = {
        int(k): v for k, v in collection['model_dict'].items()
        }
    collection['model_dict'].update(model_bytes)
    collection['model_dict'].update(rev_model)
    collection['model_dict'].update(rev_model_bytes)

    collection['metro_map'] = {
        int(k): int(v) for k, v in collection['metro_map'].items()
        }
    collection['metro_map_names'] = {
        int(k): v for  k, v in collection['metro_map_names'].items()
        }

    collection['unmapped_metro_main_uic'] = [
        int(x) for x in collection['unmapped_metro_main_uic']
        ]

    to_int = [
        'm_uic', 's_uic', 'm_s_uic',
        's_m_uic', 'm_d_uic', 'd_m_uic'
        ]
    for key in to_int:
        collection[key] = [int(x) for x in collection[key]]

    collection['operator_id'] = {
        k: int(v) for k, v in collection['operator_id'].items()
        }
    collection['card_id'] = {
        k: int(v) for k, v in collection['card_id'].items()
        }
    collection['contractor_id'] = {
        k: int(v) for k, v in collection['contractor_id'].items()
        }
    collection['extra_drop'] = [
        tuple(int(x) for x in y) for y in collection['extra_drop']
        ]
    collection['weekday_dict'] = {
        int(k): v for k, v in collection['weekday_dict'].items()
        }
    collection['metro_map_rev'] = {
        v: k for k, v in collection['metro_map'].items()
        }

    return collection


OPERATOR_INTERESECT_EXCEPTIONS = {
    'MD': {
        (8600858, 8603315):'D',
        (8603315, 8600858):'D',
        (8600856, 8603328):'D',
        (8603328, 8600856):'D'
        },
    'LS': {
        (8600683, 8600674):'S',
        (8600674, 8600683):'S',
        (8600683, 8600669):'S',
        (8600669, 8600683):'S'
        }
    }

# Connection Points
# =============================================================================
# add new metro stops
LINE_CONNECTORS = {
    8600001: ('vendsyssel_banen', 'skagens_banen'),
    8600009: ('hjørring_hirtshals_banen', 'vendsyssel_banen'),
    8600020: ('vendsyssel_banen', 'randers_aalborg_banen'),
    8600040: ('aarhus_randers', 'randers_aalborg_banen'),
    8600044: ('langå_struer_banen', 'aarhus_randers'),
    8600053: ('aarhus_randers', 'fredericia_aarhus_banen',
                'grenaa_banen', 'odder_banen'),
    8600061: ('fredericia_aarhus_banen', 'skanderborg_skjern_banen'),
    8600073: ('fredericia_aarhus_banen', 'vejle_holstebro_banen'),
    8600079: ('fynske_hovedbane', 'fredericia_aarhus_banen',
                'fredericia_vamdrup_banen'),
    8600086: ('fredericia_vamdrup_banen', 'lunderskov_esbjerg_banen'),
    8600087: ('fredericia_vamdrup_banen', 'vamdrup_padborg_banen'),
    8600097: ('vamdrup_padborg_banen', 'sønderborg_banen'),
    8600189: ('langå_struer_banen', 'thisted_struer',
                'vestjyske_længdebane'),
    8600192: ('vejle_holstebro_banen', 'vejle_holstebro_banen'),
    8600196: ('lemvig_banen', 'vestjyske_længdebane'),
    8600205: ('vestjyske_længdebane', 'skanderborg_skjern_banen'),
    8600212: ('nebelbanen', 'vestjyske_længdebane'),
    8600215: ('vestjyske_længdebane', 'lunderskov_esbjerg_banen'),
    8600219: ('bramming_tønder_banen', 'lunderskov_esbjerg_banen'),
    8600275: ('skanderborg_skjern_banen', 'vejle_holstebro_banen'),
    8600512: ('svendborg_banen', 'fynske_hovedbane'),
    8600518: ('fynske_hovedbane', 'storbælt'),
    8600526: ('fynske_hovedbane', 'svendborg_banen'),
    8600601: ('storbælt', 'vestbanen_sjælland'),
    8600611: ('vestbanen_sjælland', 'sydbanen'),
    8600617: ('vestbanen_sjælland', 'lillesyd', 'nordvestbanen'),
    8600624: ('vestbanen_sjælland', 'b_line', 'c_line'),
    8600626: ('vestbanen_sjælland', 'a_line', 'b_line', 'c_line',
                'kystbanen', 'øresunds_banen'),
    8600631: ('b_line', 'c_line'),
    8600634: ('a_line', 'b_line', 'c_line'),
    8600644: ('a_line', 'f_line'),
    8600645: ('a_line', 'b_line', 'c_line'),
    8600646: ('a_line', 'b_line', 'c_line',
                'kystbanen', 'øresunds_banen'),
    8600650: ('a_line', 'b_line', 'c_line',
                'kystbanen', 'øresunds_banen'),
    8600653: ('a_line', 'b_line', 'c_line'),
    8600654: ('a_line', 'b_line', 'c_line'),
    8600655: ('b_line', 'c_line', 'f_line',
                'kystbanen', 'øresunds_banen'),
    8600659: ('c_line', 'kystbanen', 'øresunds_banen'),
    8600661: ('kystbanen', 'øresunds_banen'),
    8600662: ('kystbanen', 'øresunds_banen'),
    8600663: ('kystbanen', 'øresunds_banen'),
    8600664: ('kystbanen', 'øresunds_banen'),
    8600665: ('kystbanen', 'øresunds_banen'),
    8600666: ('kystbanen', 'øresunds_banen'),
    8600667: ('kystbanen', 'øresunds_banen'),
    8600668: ('kystbanen', 'øresunds_banen', 'lillenord'),
    8600669: ('kystbanen', 'hornbækbanen', 'lillenord',
                'øresunds_banen'),
    8600674: ('nærumbanen', 'b_line'),
    8600683: ('frederiksværk_banen', 'gribskovbanen_fork_1',
                'gribskovbanen_fork_2','lillenord',
                'b_line'),
    8600703: ('metro_m1', 'metro_m2', 'c_line'),
    8600717: ('nordvestbanen', 'tølløse_banen'),
    8600719: ('odsherreds_banen', 'nordvestbanen'),
    8600736: ('c_line', 'f_line', 'metro_m1', 'metro_m2'),
    8600742: ('b_line', 'f_line'),
    8600783: ('a_line', 'f_line'),
    8600792: ('a_line', 'lillesyd'),
    8600798: ('b_line', 'vestbanen_sjælland'),
    8600803: ('a_line', 'lillesyd'),
    8600810: ('lillesyd', 'sydbanen'),
    8600821: ('orehoved_banen', 'sydbanen'),
    8600822: ('orehoved_banen', 'sydbanen'),
    8600824: ('sydbanen', 'orehoved_banen', 'lollandsbanen'),
    8600901: ('østbanen_fork_1', 'østbanen_fork_2'),
    8600902: ('østbanen_fork_1', 'østbanen_fork_2'),
    8600903: ('østbanen_fork_1', 'østbanen_fork_2'),
    8600904: ('østbanen_fork_1', 'østbanen_fork_2'),
    8600905: ('østbanen_fork_1', 'østbanen_fork_2'),
    8601403: ('gribskovbanen_fork_2', 'hornbækbanen'),
    8601407: ('gribskovbanen_fork_1', 'gribskovbanen_fork_2'),
    8601421: ('gribskovbanen_fork_1', 'gribskovbanen_fork_2'),
    8601433: ('gribskovbanen_fork_1', 'gribskovbanen_fork_2'),
    8603301: ('metro_m1', 'metro_m2', 'c_line'),
    8603302: ('metro_m1', 'metro_m2', 'c_line', 'f_line'),
    8603307: ('a_line', 'b_line', 'c_line',
                'kystbanen', 'øresunds_banen', 'metro_m1',
                'metro_m2'),
    8603315: ('øresunds_banen', 'metro_m1'),
    8603328: ('øresunds_banen', 'metro_m2')}

REGION_MAP_NAMES = {
    'sjælland':'sj' ,
    'jylland': 'jyl',
    'fyn': 'fyn',
    'storebæltbroen': 'sj/fyn'
    }

LINE_REGION_MAP = {
    'a_line': 'sj',
    'aarhus_randers':'jyl',
    'b_line':'sj',
    'bramming_tønder_banen':'jyl',
    'c_line':'sj',
    'f_line':'sj',
    'fredericia_aarhus_banen':'jyl',
    'fredericia_vamdrup_banen':'jyl',
    'frederiksværkbanen':'sj',
    'fynske_hovedbane':'fyn',
    'grenaa_banen':'jyl',
    'gribskovbanen_fork_1':'sj',
    'gribskovbanen_fork_2':'sj',
    'hjørring_hirtshals_banen':'jyl',
    'hornbækbanen':'sj',
    'kystbanen':'sj',
    'langå_struer_banen':'jyl',
    'lemvig_banen':'jyl',
    'lillenord':'sj',
    'lillesyd':'sj',
    'lollandsbanen':'sj',
    'lunderskov_esbjerg_banen':'jyl',
    'metro_m1':'sj',
    'metro_m2':'sj',
    'metro_m3':'sj',
    'metro_m4':'sj',
    'nebelbanen':'jyl',
    'nordvestbanen':'sj',
    'nærumbanen':'sj',
    'odder_banen':'jyl',
    'odsherredsbanen':'sj',
    'orehoved_banen':'sj',
    'randers_aalborg_banen':'jyl',
    'skagens_banen':'jyl',
    'skanderborg_skjern_banen':'jyl',
    'svendborg_banen':'fyn',
    'sydbanen':'sj',
    'sønderborg_banen':'jyl',
    'thisted_struer':'jyl',
    'tølløsebanen':'jyl',
    'vamdrup_padborg_banen':'jyl',
    'vejle_holstebro_banen':'jyl',
    'vendsyssel_banen':'jyl',
    'vestbanen_sjælland':'sj',
    'vestjyske_længdebane':'jyl',
    'øresunds_banen': 'sj',
    'østbanen_fork_1':'sj',
    'østbanen_fork_2':'sj',
    'storbælt': 'sj/fyn'
    }



