

"""Build a GTFS archive from transitfeeds.com

"""

import sys
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request
from datetime import datetime
import pandas as pd
#from pandas.core.frame import DataFrame
from pathlib import Path
from tablesalt.transitfeed import feed
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

BASE_DIR = feed.ARCHIVE_DIR
BASE_DIR.mkdir(parents=True, exist_ok=True)

def get_write(date):
    datestring = date.strftime('%Y%m%d')

    url = fr'https://transitfeeds.com/p/rejseplanen/705/{datestring}/download'
    request_object = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urlopen(request_object)
        data = feed._load_gtfs_response_zip(resp)
    except HTTPError:
        return

    agency = feed.Agency.from_dataframe(data['agency.txt'])
    stops = feed.Stops.from_dataframe(data['stops.txt'])
    routes = feed.Routes.from_dataframe(data['routes.txt'])
    trips = feed.Trips.from_dataframe(data['trips.txt'])
    stop_times = feed.StopTimes.from_dataframe(data['stop_times.txt'])
    calendar = feed.Calendar.from_dataframe(data['calendar.txt'])
    calendar_dates = feed.CalendarDates.from_dataframe(data['calendar_dates.txt'])

    transfers = feed.Transfers.from_dataframe(data['transfers.txt'])
    shapes = feed.Shapes.from_dataframe(data['shapes.txt'])

    feed = feed.TransitFeed(
        agency,
        stops,
        routes,
        trips,
        stop_times,
        calendar,
        calendar_dates,
        transfers=transfers,
        shapes=shapes
        )
    feed.to_archive()

with ThreadPool(4) as pool:
    pool.map(get_write, dates)
