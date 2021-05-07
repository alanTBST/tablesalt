
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.common.io.ingestors import delrejser_setup

if __name__ == "__main__":
    parser = TableArgParser('input_dir', 'output_dir')
    args = parser.parse()
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    delrejser_setup(input_dir, output_dir)
