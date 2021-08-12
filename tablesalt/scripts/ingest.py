
"""This script sets up the databases on disk for use with the tablesalt package
and rejsekort delrejser.

"""


import os

from tablesalt.common.io.ingestors import delrejser_setup
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.running import WindowsInhibitor

if __name__ == "__main__":
    parser = TableArgParser('input_dir', 'output_dir', 'chunksize')
    args = parser.parse()
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    chunksize = args['chunksize']
    
    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
        delrejser_setup(input_dir, output_dir, chunksize)
        INHIBITOR.uninhibit()
    else:
        delrejser_setup(input_dir, output_dir, chunksize)
