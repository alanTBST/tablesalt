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
