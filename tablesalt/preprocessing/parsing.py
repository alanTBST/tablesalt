
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )
from typing import Dict, NamedTuple, Union, Any, Optional, Type

import pathlib


class ArgTuple(NamedTuple):

    short: str
    long: str
    help_: str
    required: bool
    type_: Union[Type[str], Type[int], Type[pathlib.Path]]
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
           'path to out directory for the datastores',
           True,
           pathlib.Path,
           None
           ),
        'rabattrin': ArgTuple(
           '-r',
           '--rabattrin',
           'path to out directory for the datastores',
           False,
           int,
           None
           ),
        'passenger_type': ArgTuple(
           '-t',
           '--passenger_type',
           'the type of passenger to analyse',
           False,
           str,
           None
           ),
        'model': ArgTuple(
           '-m',
           '--model',
           'the model number',
           False,
           int,
           1
           )

        }

    def __init__(self, *args: str, description: Optional[str] = None) -> None:
        """
        A Basic argument parser for scripts in tablesalt. Allows only a
        specific subset of arguments that make sense in this context.

        Parameters
        ----------
        *args : str
            The arguments allowed in the argparser.
        description : Optional[str], optional
            A description of the argument. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None

        """
        

        self.arglist = list(args)

        if not all(x in self.ARGUMENTS for x in self.arglist):
            odd_args = {x for x in self.arglist if x not in self.ARGUMENTS}
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
                help=opt.help_,
                type=opt.type_,
                required=opt.required,
                default=opt.default
                )
    def parse(self) -> Dict[str, Union[int, pathlib.Path, str]]:
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        

        return vars(self.parser.parse_args())
