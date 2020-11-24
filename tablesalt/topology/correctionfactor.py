# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:44:17 2020

@author: alkj
"""

import os
import sys
from pathlib import Path

import pandas as pd
from pandas.core.frame import DataFrame


THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

def load_model_results(year: int, model: int) -> DataFrame:

    base = THIS_DIR.parent

    fp = os.path.join(
        base,
        'scripts',
        '__result_cache__',
        f'{year}',
        'output',
        f'takst_sjælland{year}_model_{model}.csv'
        )

    df = pd.read_csv(fp)

    return df


def load_base_factors():

    return

def main(model: int) -> None:

    return

if __name__ == "__main__":

    main(sys.argv[1])