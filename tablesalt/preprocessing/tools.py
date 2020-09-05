import glob
import os
import sys
import socket
import zipfile
from typing import (
    Optional,
    AnyStr,
    List,
    Tuple,
    Dict,
    Generator,
    Union
    )

def find_datastores(start_dir: Optional[AnyStr] = None) -> AnyStr:

    if start_dir is None:
        if socket.gethostname() == "tsdw03":
            start_dir='H:\\'
        else:
            start_dir = os.path.splitdrive(sys.executable)[0]
            start_dir = os.path.join(start_dir, '\\')
    for dirpath, subdirs, _ in os.walk(start_dir):
        if 'rejsekortstores' in subdirs:
            return dirpath
    raise FileNotFoundError(
        "cannot find a rejsekort datastores location"
        )

def _hdfstores(store_loc, year):

    return glob.glob(
        os.path.join(
            store_loc, 'rejsekortstores',
            f'{year}DataStores', 'hdfstores', '*.h5'
            )
        )


def setup_directories(year: int, dstores: Optional[AnyStr] = None) -> Tuple:
    """
    Setup the directories needed for the chosen year

    Parameters
    ----------
    dstores : path like object
        the directory path of the datastores.

    """
    if not dstores:
        dstores = os.path.join(
            Path(THIS_DIR).parent, 'datastores',
            'rejsekortstores'
            )
    else:
        dstores = os.path.join(
            dstores, 'rejsekortstores'
            )
    if not os.path.isdir(dstores):
        os.makedirs(dstores)

    new_paths = (
        os.path.join(dstores, f'{year}DataStores', 'dbs'),
        os.path.join(dstores, f'{year}DataStores', 'hdfstores'),
        os.path.join(dstores, f'{year}DataStores', 'packs')
        )

    for path in new_paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    return new_paths

def db_paths(store_location: AnyStr, year: int) -> Dict[str, AnyStr]:

    kv_names = [
        'user_trips_db',
        'kombi_dates_db',
        'kombi_valid_trips',
        'trip_card_db',
        'calculated_stores'
        ]

    out = {}
    for name in kv_names:
        path = os.path.join(
            store_location, 'rejsekortstores', f'{year}DataStores',
            'dbs', name
            )
        out[name] = path

    out['store_paths'] = _hdfstores(store_location, year)

    return out

def _get_sub_zips(lstzips: List) -> Dict:
    """get the names of the files in each zipfile"""
    return {zipf: tuple(zipfile.ZipFile(zipf).namelist()) for zipf in lstzips}

def get_zips(path: AnyStr) -> Optional[List[Tuple]]:
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

    all_files = []
    for k, v in zips.items():
        for f in v:
            if '.csv' in f:
                all_files.append((k, f))

    if not all_files:
        raise FileNotFoundError(
            "Could not find any csv files"
            )
    return all_files


def get_columns(zfile: AnyStr, content: AnyStr) -> List[str]:
    """return a list of the file headers"""
    df_0 = pd.read_csv(
        zipfile.ZipFile(zfile).open(content),
        nrows=0, encoding='iso-8859-1'
        )
    file_columns = df_0.columns
    return [x.lower() for x in file_columns]

def check_all_file_headers(file_list: List[Tuple]) -> Optional[bool]:
    """test to see if all file headers are like the first"""
    if len(file_list) == 1:
        return True
    first_headers = get_columns(*file_list[0])
    for file in file_list[1:]:
        headers = get_columns(*file)
        if headers != first_headers:
            raise ValueError(
                f"{file} contains column headers "
                "that do not match the expected headers "
                f"found in {file_list[0]}"
                )

    return True

def col_index_dict(
        file_columns: List[str]
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

    colindices = {}
    colindices['kortnr'] = kortnr
    for col, *_ in wanted:
        try:
            colindices[col] = file_columns.index(col)
        except ValueError:
            pass
    coltypes = {x[0]: x[1] for x in wanted if x[0] in file_columns}
    coltypes['kortnr'] = str
    coltypes = {colindices[k]: v for k, v in coltypes.items()}

    return colindices, coltypes

def blocks(files: AnyStr, size: Optional[int] = 65536) -> Generator:
    """yield the bytes of the specified file"""
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def sumblocks(zfile, content) -> int:
    """
    Get the total number of new lines in the zipfile content

    Parameters
    ----------
    zfile : str
        a path to a zipfile.
    content : TYPE
        DESCRIPTION.

    Returns
    -------
    n_lines : int
        returns the number of lines in the file.

    """
    with zipfile.ZipFile(zfile).open(content) as f:
        n_lines = sum(
            bl.count(b"\n") for bl in blocks(f, size=655360)
            )
    return n_lines
