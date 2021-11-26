"""
    This script ingests the rejsekort delrejser data and converts it into a set
    of lmdb key-value stores. You will need to use this script if you wish to make use of
    the DelrejserStore class in the tablesalt.common.io.datastores module

WHAT does it do?
================

    Given a path to a directory of compressed zip files of rejsekort delrejser
    data and a path for an output directory where the resulting datastores
    should be placed. It will setup stops, time, operator, price, pas and trip_card
    databases that can then be accessed using the DelrejserStore giving only the path to
    the parent folder.
    This class can then be used to query the rejsekort delrejser dataset.
"""


from pathlib import Path

from tablesalt.common.io.ingestors import delrejser_setup
from tablesalt.preprocessing.parsing import TableArgParser

if __name__ == "__main__":

    parser = TableArgParser(
        'year',
        'input_dir',
        'output_dir',
        'chunksize'
        )

    args = parser.parse()
    year = args['year']
    dpath = args['input_dir']
    opath = args['output_dir']
    chunksize = args['chunksize']

    delrejser_setup(
        dpath,
        opath,
        chunksize
        )
