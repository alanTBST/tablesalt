
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )
from pathlib import Path

def parse_args():
    """parse the cl arguments"""
    DESC = ("Setup all the key-value stores needed \n"
            "for the pendler card revenue distribution \n"
            "for takstsjælland.")

    parser = ArgumentParser(
        description=DESC,
        formatter_class=RawTextHelpFormatter
        )
    parser.add_argument(
        '-y', '--year',
        help='year to unpack',
        type=int,
        required=True
        )
    parser.add_argument(
        '-z', '--zones',
        help='path to input zones csv',
        type=Path,
        required=True
        )
    parser.add_argument(
        '-p', '--products',
        help='path to input pendler products csv',
        type=Path,
        required=True
        )

    args = parser.parse_args()

    return vars(args)
    
class TableArgParser:
    def __init__(self, *args):
        self.arglist = list(args)
    
    
   