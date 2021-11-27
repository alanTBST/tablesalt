# -*- coding: utf-8 -*-
"""

What does it do?
================

This script merges and processes sales data received from the operators.


USAGE
=====

Place all of the delivered sales data files in one folder/directory.

Eg ./path/to/salesdata

The name of the operator should be in the filename. EG 'moviasales2020.csv', 'dsbomsætning.xlsx'

To run the script

    python ./path/to/tablesalt/tablesalt/scripts/salesdatamerge.py -y 2019 -d ./path/to/salesdata

This will place the file in a result cache directory in tablesalt to be used
in the salesoutput script later.
"""

import ast
import difflib
import glob
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, AnyStr, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

THIS_DIR: Path = Path(__file__).parent

SUPPORTED_FILES: Set[str] = {
    'csv',
    'xlsx',
    'xls'
    }

MERGE_COLUMNS: List[str] = [
    'salgsvirksomhed',
    'indtægtsgruppe',
    'salgsår',
    'salgsmåned',
    'takstsæt',
    'produktgruppe',
    'produktnavn',
    'kundetype',
    'salgsmedie',
    'betaltezoner',
    'startzone',
    'slutzone',
    'valgtezoner',
    'omsætning',
    'antal'
    ]

STRAIGHT_MAP: Dict[str, str] = {
    'aar': 'salgsår',
    'afsætning': 'antal',
    'summen af antal': 'antal',
    'summen af omsætning (fuld pris)': 'omsætning',
    'amtzfra': 'startzone',
    'amtztil': 'slutzone',
    'zoneliste': 'valgtezoner',
    'kildesystem': 'salgsmedie'
    }

KNOWN_NOT_NEEDED: Set[str] = {
    'smedie',
    'antalzoner'
    }

INTEGER_COLUMNS: Set[str] = {
    'salgsår',
    'salgsmåned',
    'betaltezoner',
    }

STRING_COLUMNS: Set[str] = {
    'salgsvirksomhed',
    'takstsæt',
    'produktgruppe',
    'produktnavn',
    'kundetype' ,
    'salgsmedie',
    'valgtezoner'
    }

FLOAT_COLUMNS: Set[str] = {
    'omsætning',
    'antal'
    }

def parseargs() -> Dict:
    """
    Parse the command line arguments

    Returns
    -------
    Dict
        a dictionary of the arguments.

    """
    desc = ("Script for merging the sales input data "
            "from Movia, DSB and Movia. "
            "Merges csv or excel files into a single "
            "file")
    parser = ArgumentParser(description=desc)
    dhelp = ("The path to the directory containing "
             "the sales data from the companies.")

    parser.add_argument('-d', '--directory', help=dhelp, type=Path)

    ohelp = ("The name of the output file. NOTE: with file extension. "
             "The default output filename is mergedsales.csv")

    parser.add_argument(
        '-y', '--year', type=int
        )
    return vars(parser.parse_args())


def directory_contents(directory: AnyStr) -> List[AnyStr]:
    """Get the contents of the directory

    :param directory: the path to the directory containing the sales data.
    :type directory: AnyStr
    :raises TypeError: if any of the files in the directory are not csv or excel files.
    :return: a list of the files in the given directory.
    :rtype: List[AnyStr]
    """
    contents = glob.glob(os.path.join(directory, '*'))
    extensions = {x.split('.')[-1] for x in contents}
    if not all(x in SUPPORTED_FILES for x in extensions):
        raise TypeError(
            f"only file types {', '.join(SUPPORTED_FILES)} supported"
            )

    return contents


def _op_id(filename: AnyStr) -> Tuple[str, str]:

    if 'movia' in filename.lower():
        return 'movia', filename

    if 'dsb' in filename.lower():
        return 'dsb', filename

    if 'metro' in filename.lower():
        return 'metro', filename

    raise ValueError(f"could not find operator in {filename}")



def identify_operator(dir_contents: List[AnyStr]) -> Dict[str, str]:
    """Identify the operator for each filename

    :param dir_contents: list of filenames
    :type dir_contents: List[AnyStr]
    :raises ValueError: if any of the operators are not in the filenames.
    :return: a dictionary of operator -> filename
    :rtype: Dict[str, str]
    """

    vals = {}
    for file in dir_contents:
        try:
            k, v = _op_id(file)
            vals[k] = v
        except Exception as e:
            raise ValueError from e

    return vals

def _get_separator(file: AnyStr) -> str:
    """find the separator of the file

    :param file: the filepath
    :type file: AnyStr
    :return: a comma or a semicolon
    :rtype: str
    """
    with open(file, 'r') as f:
        firstline = f.readline()
    commas = firstline.count(',')
    semi = firstline.count(';')

    if commas == 0:
        return ';'
    if semi == 0:
        return ','
    if semi > commas:
        return ';'
    if commas > semi:
        return ','
    return NotImplemented


def _read_utf8(file: AnyStr) -> Tuple[str, ...]:

    return tuple(pd.read_csv(
        file, nrows=0,
        sep=_get_separator(file)
        ).columns)


def _read_iso(file: AnyStr) -> Tuple[str, ...]:

    return tuple(pd.read_csv(
        file, nrows=0, encoding='iso-8859-1',
        sep=_get_separator(file)
        ).columns)

def _check_chars(columns: Tuple) -> bool:

    nordic = {'å', 'æ', 'ø'}
    colstring = set(''.join(columns).lower())

    inters = nordic.intersection(colstring)

    if not inters:
        return False
    return True


def _read_header(file: AnyStr) -> Tuple[str, ...]:

    ext = file.split('.')[-1]
    if ext == 'csv':
        try:
            cols = _read_iso(file)
            if _check_chars(cols):
                return cols
            return _read_utf8(file)
        except Exception as e:
            raise e
    if ext in ('xlsx', 'xls'):
        return tuple(pd.read_excel(file, nrows=0).columns)
    raise TypeError(f"cannot use filetype {ext}")

def find_columns(identified_operators: Dict) -> Dict:

    cols = {}
    for operator, filename in identified_operators.items():
        try:
            columns = _read_header(filename)
            cols[operator] = columns
        except TypeError as e:
            raise e

    return cols


def match_columns(columndict: Dict[str, Tuple[str, ...]]) -> Dict[str, Dict[str, str]]:
    """
    Match the input columns/field names given to the wanted
    column names in MERGE_COLUMNS

    Parameters
    ----------
    columndict : dictionary of {'operator': ('col1', 'col2'...)}
        DESCRIPTION.

    Returns
    -------
    Dict
        DESCRIPTION.

    """
    opcols = {}
    for operator, colnames in columndict.items():
        usecols = {}
        for col in colnames:
            if col in KNOWN_NOT_NEEDED:
                continue
            directmatch = STRAIGHT_MAP.get(col.lower())
            if directmatch is not None:
                usecols[col] = directmatch
                continue
            else:
                closematches = difflib.get_close_matches(col, MERGE_COLUMNS)
                if not closematches:
                    if 'omsætning' in col.lower():
                        usecols[col] = 'omsætning'
                else:
                    usecols[col] = closematches[0]
        opcols[operator] = {k: v for k, v in usecols.items() if v}

    return opcols

def read_and_merge(
        files: Dict, matched_columns: Dict
        ) -> pd.core.frame.DataFrame:
    """
    Read each of the files and merge them into a single dataframe

    """

    frames = []
    for operator, file in files.items():

        ext = file.split('.')[-1]
        if ext == 'csv':
            sep = _get_separator(file)
            try:
                frame = pd.read_csv(
                    file, sep=sep, usecols=list(matched_columns[operator])
                    )
            except UnicodeDecodeError:
                 frame = pd.read_csv(
                    file, sep=sep, encoding='iso-8859-1',
                    usecols=list(matched_columns[operator])
                )
            frame.columns = [matched_columns[operator][x] for x in frame.columns]
            frames.append(frame)

        if ext in ('xlsx', 'xls'):
            frame = pd.read_excel(file, usecols=list(matched_columns[operator]))
            frame.columns = [matched_columns[operator][x] for x in frame.columns]
            frames.append(frame)

    new_frames = []
    for df in frames:
        missing = [x for x in MERGE_COLUMNS if x not in df.columns]
        for col in missing:
            df.loc[:, col] = np.nan
        df = df[MERGE_COLUMNS]
        new_frames.append(df)

    df = pd.concat(new_frames, ignore_index=True)
    df.index.name = 'NR'
    df = df.reset_index()

    return df


def _check_types(column: pd.core.series.Series) -> Set:
    """get distinct types in column"""

    return set(type(x) for x in column)


def _all_string(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """process columns that should contain only strings"""

    for col in STRING_COLUMNS:
        distinct_types = _check_types(frame[col])
        if distinct_types == {str}:
            frame.loc[:, col] = frame.loc[:, col].str.lower()
            continue
        valtypes = {x: type(x) for x in set(frame[col])}
        bad_values = {k for k, v in valtypes.items() if v != str}
        repl = {x: '' for x in bad_values}
        frame.loc[:, col] = frame.loc[:, col].replace(repl).str.lower()

    frame.loc[:, 'takstsæt'] = frame.loc[:, 'takstsæt'].str.replace('ht', 'th')

    return frame

def _paidzones(val: Any) -> int:

    if isinstance(val, float):
        if val > 100 or val < 0:
            return 99
        if not any(x.isdigit() for x in str(val)):
            return 99
        return int(val)
    if isinstance(val, int):
        if val > 100 or val < 0:
            return 99
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return 99
    return 99

def _find_region_candidates(zonevals: Set) -> Set:

    candidates = {'0', '00', '999', '998', '997'}
    for zonenr in zonevals:
        if len(zonenr) in (2, 4) and all(x.isdigit() for x in zonenr):
            continue
        if not any(x.isdigit() for x in zonenr):
            candidates.add(zonenr)
        if '-' in zonenr:
            candidates.add(zonenr)


    return candidates

def _assign_region_nr(
        frame: pd.core.frame.DataFrame, col: str
        ) -> pd.core.frame.DataFrame:

    distinct = set(frame.loc[:, col])
    candidates = _find_region_candidates(distinct)

    frame.loc[
        (frame.loc[:, col].isin(candidates)) &
        (frame.loc[:, 'takstsæt'] == 'th'), col
        ] = '1000'  # hovedstaden zones
    frame.loc[
        (frame.loc[:, col].isin(candidates)) &
        (frame.loc[:, 'takstsæt'] == 'tv'), col
        ] = '1100'  # vestsjælland
    frame.loc[
        (frame.loc[:, col].isin(candidates)) &
        (frame.loc[:, 'takstsæt'] == 'ts'), col
        ] = '1200'  # sydsjælland
    # frame.loc[
    #     (frame.loc[:, col].isin(candidates)) &
    #     (frame.loc[:, 'takstsæt'] == 'dsb'), col
    #     ] = '1300' # any sjælland

    return frame


def _zone_string(string: str, prefix: str) -> str:

    if r'/' in string:
        return string
    return prefix + string.zfill(4)[2:]


def _assign_shortzone(frame: pd.core.frame.DataFrame, col: str) -> pd.core.frame.DataFrame:

    frame.loc[(frame.loc[:, 'takstsæt'] == 'th'), col] = \
        frame.loc[(frame.loc[:, 'takstsæt'] == 'th'), col].apply(
            lambda x: _zone_string(x, '10')
            )
    frame.loc[(frame.loc[:, 'takstsæt'] == 'tv'), col] = \
        frame.loc[(frame.loc[:, 'takstsæt'] == 'tv'), col].apply(
            lambda x: _zone_string(x, '11')
            )
    frame.loc[(frame.loc[:, 'takstsæt'] == 'ts'), col] = \
        frame.loc[(frame.loc[:, 'takstsæt'] == 'ts'), col].apply(
            lambda x: _zone_string(x, '12')
            )

    # TODO - deal with takstsæt == 'dsb'
    return frame


def _startend_zone(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

    cols = ('startzone', 'slutzone')
    for col in cols:
        frame.loc[:, col] = frame.loc[:, col].astype(str)
        frame = _assign_region_nr(frame, col)
        frame = _assign_shortzone(frame, col)


    return frame


def _all_int(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """process columns that should contain only integers"""

    for col in INTEGER_COLUMNS:
        distinct_types = _check_types(frame[col])
        if distinct_types == {int}:
            continue
        if col == 'betaltezoner':
            frame.loc[:, col] = frame.loc[:, col].apply(_paidzones)
            continue

        frame.loc[:, col] = frame[:, col].fillna(0)
        frame.loc[~frame.loc[:, col].apply(lambda x: isinstance(x, int)), col] = 0

    return frame

def _all_float(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """process columns that should contain only floats"""

    for col in FLOAT_COLUMNS:
        distinct_types = _check_types(frame[col])
        if str not in distinct_types:
            continue

        frame.loc[frame.loc[:, col].apply(lambda x: isinstance(x, str)), col] = \
            frame.loc[frame.loc[:, col].apply(
                lambda x: isinstance(x, str)
                ), col].str.replace(',', '.')

    for col in FLOAT_COLUMNS:
        frame.loc[:, col] = frame.loc[:, col].astype(float)

    return frame


def _correct_chosen_form(string: str) -> bool:

    if not any(x.isdigit() for x in string):
        return False

    return (all(x.isdigit() or x == ',' or
                x == ' ' for x in string)
            and ',' in string)


def _eval_literal(val: Any) -> Tuple:

    try:
        value = ast.literal_eval(val)
        if isinstance(value, tuple):
            return value
        return ()
    except Exception as e:
        return ()

def _format_tuple(val: Tuple) -> Tuple[int, ...]:

    return tuple(int('1' + str(x).zfill(4)[1:]) for x in val)


def _valgtezoner(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:


    frame.loc[:, 'valgtezoner'] = \
        frame.loc[:, 'valgtezoner'].apply(lambda x: _eval_literal(x))

    frame.loc[:, 'valgtezoner'] = \
        frame.loc[:, 'valgtezoner'].apply(lambda x: _format_tuple(x))

    return frame


def clean_frame(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """perform all the cleaning on the given frame

    :param frame: a frame to clean
    :type frame: pd.core.frame.DataFrame
    :return: a cleaned dataframe of sales data
    :rtype: pd.core.frame.DataFrame
    """
    frame = _all_string(frame)
    frame = _all_int(frame)
    frame = _startend_zone(frame)
    frame = _all_float(frame)
    frame = _valgtezoner(frame)

    return frame


def main(dirpath, year):
    """main function"""


    contents = directory_contents(dirpath)
    operators = identify_operator(contents)
    given_columns = find_columns(operators)
    columns_matched = match_columns(given_columns)
    df = read_and_merge(operators, columns_matched)
    df = clean_frame(df)

    outdir = Path(THIS_DIR, '__result_cache__', f'{year}', 'preprocessed')
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    fp = outdir / 'mergedsales.csv'
    df.to_csv(fp, index=False)


if __name__ == "__main__":
    from datetime import datetime
    args = parseargs()
    dir_path = args['directory']
    year = args['year']
    st = datetime.now()
    main(dir_path, year)
    print(datetime.now() - st)
