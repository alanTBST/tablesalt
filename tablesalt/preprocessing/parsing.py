
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )
from typing import AnyStr, NamedTuple, Union, Any
import pathlib 


class ArgTuple(NamedTuple):
    
    short: str
    long: str
    help: str
    required: bool
    type: Union[str, int, pathlib.Path]
    default: Any
        
class TableArgParser:
    
    ARGUMENTS = {
        'products': ArgTuple(
            '-p', 
            '--products', 
            'path to input pendler products csv', 
            True,
            pathlib.Path,
            None
            ), 
        'zones': ArgTuple(
            '-z', 
            '--zones', 
            'path to input zones csv', 
            True,
            pathlib.Path,
            None
            ), 
        'year': ArgTuple(
            '-y', 
            '--year', 
            'year to analyse', 
            True,
            int,
            None
            ), 
        'chunksize': ArgTuple(
           '-c', 
           '--chunksize', 
           'The chunksize to read from the data in rows',
           False, 
           int,
           500_000
           ), 
        'input_dir': ArgTuple(
           '-i', 
           '--input_dir', 
           'path to input directory of zip files',
           True, 
           pathlib.Path,
           None
           ),
        'output_dir': ArgTuple(
           '-o', 
           '--output_dir', 
           'path to out directory for the datastoresø',
           True, 
           pathlib.Path,
           None
           )                 
        }

    def __init__(self, *args: str, description: AnyStr = None) -> None:
        
        self.arglist = list(args)
        
        odd_args = set(self.arglist).symmetric_difference(set(self.ARGUMENTS))
        if odd_args:
            raise ValueError(
                f"{list(map(str, odd_args))} not supported"
                )
    
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
                default=opt.default
                )
    def parse(self):
        
        return vars(self.parser.parse_args())    
   