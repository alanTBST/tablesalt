# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:24:11 2020

@author: alkj
"""


import os
from pathlib import Path
from threading import Thread
from typing import (
    Tuple,
    Dict,
    Optional,
    Generator,
    AnyStr,
    Iterator,
    Sequence,
    List,
    )
import zipfile

import lmdb
import pandas as pd


from tablesalt.preprocessing.tools import (
    find_datastores,
    get_zips,
    check_all_file_headers,
    col_index_dict,
    get_columns,
    sumblocks
    )

__ALLOWED_STORES = frozenset(['tripcard'])

def setup_directories(
        year: int, dstores: AnyStr
        ) -> Tuple[AnyStr, AnyStr, AnyStr]:
    """
    Setup the directories needed for the chosen year

    Parameters
    ----------
    dstores : path like object
        the directory path of the datastores.

    """

    dstores = os.path.join('datastores', 'rejsekortstores')

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

class RawProcessor:

    """
    Process the zipfiles delivered by Rejsedata
    """

    def __init__(self, year: int,
                 input_dir: AnyStr,
                 output_dir: AnyStr,
                 chunksize: Optional[int] = 500_000,
                 ) -> None:
        """


        Parameters
        ----------
        year : int
            DESCRIPTION.
        input_dir : AnyStr
            DESCRIPTION.
        output_dir : AnyStr
            DESCRIPTION.
        chunksize : Optional[int], optional
            DESCRIPTION. The default is 500_000.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """

        self.year = year
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunksize = chunksize
        self._threads: List = []
        self.info: Dict[str, str] = {}

    def _resolve(self):

        # location = find_datastores(r'H://')

        tc_store_path, hdf_path, cont_path = \
            setup_directories(self.year, self.output_dir)

        zips = get_zips(self.input_dir)

        check_all_file_headers(zips)
        file_columns = get_columns(*zips[0])
        col_indices, _ = col_index_dict(file_columns)

        return

    def _card_trip_generator(
            ziplist, col_indices, chunksize, skiprows=0
        ) -> Iterator:
        """
        Parameters
        ----------
        ziplist : list
            A list of the zipfiles and contents
        col_indices : dict
            a dictionary of column names and corresponding indices.
        chunksize : int, optional
            the chunksize to use. The default is 1000000.
        skiprows : int, optional
            the number of rows to skip in the file. The default is 0.

        Yields
        -------
        dict :
            the generator for the lmbd database.

        """
        wanted_columns = [col_indices['kortnr'], col_indices['turngl']]

        for zfile, content in ziplist:
            df_gen = pd.read_csv(
                zipfile.ZipFile(zfile).open(content),
                encoding='iso-8859-1', usecols=wanted_columns,
                chunksize=chunksize, skiprows=skiprows,
                error_bad_lines=False, low_memory=False
                )
            for chunk in df_gen:
                chunk_dict = dict(set(chunk.itertuples(name=None, index=False)))
                byte_dict = {bytes(str(k), 'utf-8'): bytes(str(v), 'utf-8')
                             for k, v in chunk_dict.items()}
                yield byte_dict

    def _make_kv_store(self, keys, values, path):


        return

    def run(self):
        for thread in self._threads:
            thread.start()

        for thread in self._threads:
            thread.join()
