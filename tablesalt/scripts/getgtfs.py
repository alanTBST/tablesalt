

"""Build a GTFS archive from transitfeeds.com

"""

import sys
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request
from datetime import datetime
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path
from tablesalt.preprocessing.tools import find_datastores
from tablesalt.transitfeed.feed import _load_gtfs_response_zip
from multiprocessing.pool import ThreadPool

# args
# period to find
# start date, end date default end date is today's date
# what format to write to tablesalt feed-archive, zips, etc

year = int(sys.argv[1])
month = 1
day = 1

start = datetime(year=year, month=month, day=day)
dates = pd.date_range(start=start, freq='D', periods=365)

datastores = Path(find_datastores())
BASE_DIR = datastores / 'rejsekortstores'/ f'{year}DataStores' / 'gtfsstores'
BASE_DIR.mkdir(parents=True, exist_ok=True)

def get_write(date):
    datestring = date.strftime('%Y%m%d')

    url = fr'https://transitfeeds.com/p/rejseplanen/705/{datestring}/download'
    request_object = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urlopen(request_object)
        gtfs_data = _load_gtfs_response_zip(resp)
    except HTTPError:
        return
    year = datestring[:4]
    new_dir = BASE_DIR / datestring
    new_dir.mkdir(exist_ok=True, parents=True)

    for name, data in gtfs_data.items():
        fp = new_dir / name
        if not fp.is_file():
            data.to_csv(fp, index=False)

with ThreadPool(4) as pool:
    pool.map(get_write, dates)
