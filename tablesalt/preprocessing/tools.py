"""
Module conatining functions to find location of datastores, setup needed
directories and inspect zipfiles of dejrejser data.
"""
import glob
import os
import sys
import socket
import zipfile
from pathlib import Path
from typing import (
    IO,
    Any,
    Optional,
    Iterable,
    Sequence,
    List,
    Tuple,
    Dict,
    Union
    )

import pandas as pd #type: ignore

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

def find_datastores(start_dir: Optional[str] = None) -> str:
    """
    Find the location of the processed rejsekort datastores

    Parameters
    ----------
    start_dir : Optional[AnyStr], optional
        The directory to start searching from . The default is None.

    Raises
    ------
    FileNotFoundError
        If no rejsekort datastores can be found.

    Returns
    -------
    str
        The directory of the rejsekort datastores.

    """

    if start_dir is None:
        if socket.gethostname() == "tsdw03": # TBST server
            start_dir = r'H:\\'
        else:
            start_dir = os.path.splitdrive(sys.executable)[0] # drive letter of your python installation
            start_dir = os.path.join(start_dir, r'/')
    for dirpath, subdirs, _ in os.walk(start_dir):
        if 'rejsekortstores' in subdirs:
            return dirpath
    raise FileNotFoundError(
        "cannot find a rejsekort datastores location"
        )

def _hdfstores(store_loc: str, year: int) -> List[str]:
    """Return path of the hdf5 files"""
    return glob.glob(
        os.path.join(
            store_loc,
            'rejsekortstores',
            f'{year}DataStores',
            'hdfstores',
            '*.h5'
            )
        )


def setup_directories(year: int, dstores: Optional[str] = None) -> List[str]:
    """
    Setup the directories needed for the chosen year

    Parameters
    ----------
    dstores : path like object
        the directory path of the datastores.

    """
    if not dstores:
        dstores = os.path.join(
            Path(THIS_DIR).parent,
            'datastores',
            'rejsekortstores'
            )
    else:
        dstores = os.path.join(
            dstores,
            'rejsekortstores'
            )
    if not os.path.isdir(dstores):
        os.makedirs(dstores)

    dstore_paths = ('hdfstores', 'dbs', 'packs')

    new_paths = {}
    for d in dstore_paths:
        new_path = os.path.join(dstores, f'{year}DataStores', d)
        new_paths[d] = new_path

    result_paths = ['other', 'pendler', 'single', 'preprocessed']
    for d in result_paths:
        new_path =  os.path.join(
            Path(THIS_DIR).parent,
            'scripts',
            '__result_cache__',
            f'{year}',
            d
            )
        new_paths[d] = new_path

    for _, path in new_paths.items():
        if not os.path.isdir(path):
            os.makedirs(path)

    return new_paths

def db_paths(store_location: str, year: int) -> Dict[str, Union[str, List[str]]]:
    """
    Return a dictionary of the paths to the databases

    Parameters
    ----------
    store_location : str
        the directory location of the datastores.
    year : int
        the analysis year.

    Returns
    -------
    Dict[str, Union[str, List[str]]]
        dict of database name and their paths.

    """

    kv_names = [
        'user_trips_db',
        'kombi_dates_db',
        'kombi_valid_trips',
        'trip_card_db',
        'calculated_stores'
        ]

    out: Dict[str, Union[str, List[str]]] = {}
    for name in kv_names:
        path = os.path.join(
            store_location,
            'rejsekortstores',
            f'{year}DataStores',
            'dbs',
            name
            )
        out[name] = path

    out['store_paths'] = _hdfstores(store_location, year)

    return out

def _get_sub_zips(lstzips: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """get the names of the files in each zipfile"""
    return {zipf: tuple(zipfile.ZipFile(zipf).namelist()) for zipf in lstzips}

def get_zips(path: str) -> List[Tuple[str, str]]:
    """get a list of the zipfiles in the path"""
    if os.path.isdir(path):
        zip_in_path = glob.glob(os.path.join(path, '*.zip'))
        if not zip_in_path:
            raise FileNotFoundError(
                f"There are no zip files present in {path}"
                )
    else:
        raise OSError(f"{path} is not a direcetory")
    zips = _get_sub_zips(zip_in_path)

    all_files: List[Tuple[str, str]] = []
    for k, v in zips.items():
        for f in v:
            if '.csv' in f:
                all_files.append((k, f))

    if not all_files:
        raise FileNotFoundError(
            "Could not find any csv files"
            )
    return all_files


def get_columns(
        zfile: Union[str, 'os.PathLike[Any]', IO[bytes]],
        content: Union[str, zipfile.ZipInfo]
        ) -> List[str]:
    """return a list of the file headers"""
    df_0 = pd.read_csv(
        zipfile.ZipFile(zfile).open(content),
        nrows=0, encoding='iso-8859-1'
        )
    file_columns = df_0.columns
    return [x.lower() for x in file_columns]

def check_all_file_headers(
        file_list: Sequence[Union[str, 'os.PathLike[Any]', IO[bytes]]]
        ) -> bool:
    """test to see if all file headers are like the first"""
    if len(file_list) == 1:
        return True

    zfile, content = file_list[0]
    first_headers = get_columns(zfile, content)
    for file in file_list[1:]:
        zfile, content = file
        headers = get_columns(zfile, content)
        if headers != first_headers:
            raise ValueError(
                f"{file} contains column headers "
                "that do not match the expected headers "
                f"found in {file_list[0]}"
                )

    return True

def col_index_dict(
        file_columns: Sequence[str]
        ) -> Tuple[Dict[str, int], Dict[int, Union[str, int, float]]]:
    """get the index values of the input file_columns"""

    try:
        kortnr = file_columns.index('kortnrkrypt')
    except ValueError:
        kortnr = file_columns.index('kortnr')

    wanted = [
        ('msgreportdate', str), ('contractorid', str),
        ('nyudfører', str), ('tildelrejse', str),
        ('fradelrejse', str), ('passagerantal1', float),
        ('passagerantal2', float), ('passagerantal3', float),
        ('passagertype1', str), ('passagertype2', str),
        ('passagertype3', str), ('tidsrabat', str),
        ('ruteid', str), ('applicationtransactionsequencenu', float),
        ('stoppointnr', int), ('stoppointid', str),
        ('numberscoveredzone', float), ('takstsaet', str),
        ('model', str), ('zonerrejst', float), ('rejsepris', str),
        ('turngl', int), ('korttype', str),
        ('rabattrin', float)
        ]

    colindices: Dict[str, int] = {}
    colindices['kortnr'] = kortnr

    for col, *_ in wanted:
        try:
            colindices[col] = file_columns.index(col)
        except ValueError:
            pass
    coltypes: Dict[int, Union[str, int, float]]
    coltypes = {x[0]: x[1] for x in wanted if x[0] in file_columns}
    coltypes['kortnr'] = str
    coltypes = {colindices[k]: v for k, v in coltypes.items()}

    return colindices, coltypes

def blocks(files: IO[bytes]) -> Iterable[bytes]:
    """yield the bytes of the specified file"""
    while True:
        b = files.read(65536)
        if not b:
            break
        yield b


def sumblocks(zfile: IO[bytes], content: Union[str, zipfile.ZipInfo]) -> int:
    """
    Get the total number of new lines in the zipfile content

    Parameters
    ----------
    zfile : AnyStr
        a path to a zipfile.
    content : AnyStr
        the subfile contents of the given zfile.

    Returns
    -------
    n_lines : int
        returns the number of lines in the file.

    """
    with zipfile.ZipFile(zfile).open(content) as f:
        n_lines = sum(
            bl.count(b"\n") for bl in blocks(f)
            )
    return n_lines
