
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )
from typing import AnyStr, NamedTuple, Union
import pathlib 

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

class ArgTuple(NamedTuple):
    
    short: str
    long: str
    help: str
    required: bool
    type: Union[str, int, pathlib.Path]
        
class TableArgParser:
    
    ARGUMENTS = {
        'products': ArgTuple(
            '-p', 
            '--products', 
            'path to input pendler products csv', 
            True,
            pathlib.Path
            ), 
        'zones': ArgTuple(
            '-z', 
            '--zones', 
            'path to input zones csv', 
            True,
            pathlib.Path
            ), 
        'year': ArgTuple(
            '-y', 
            '--year', 
            'year to analyse', 
            True,
            int
            )
        }

    def __init__(self, *args: str, description: AnyStr = None) -> None:
        self.arglist = list(args)
    
        self.arglist = [x.lower() for x in args]
        self.description = description
        
        self.parser = ArgumentParser(
            description=self.description,
            formatter_class=RawTextHelpFormatter
            )
        for arg in self.arglist:
            opt = self.ARGUMENTS[arg]
            self.parser.add_argument(
                opt.short, opt.long,
                help=opt.help, 
                type=opt.type, 
                required=opt.required
                )
    def parse(self):
        
        return vars(self.parser.parse_args())    
   