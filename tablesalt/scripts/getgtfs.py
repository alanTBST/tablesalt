

"""Build a GTFS archive from transitfeeds.com

"""

import sys
from datetime import datetime
from multiprocessing import Pool
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
#from pandas.core.frame import DataFrame
from tablesalt.transitfeed import load_gtfs_response_zip
from tablesalt.transitfeed.feed import (ARCHIVE_DIR, Agency, Calendar,
                                        CalendarDates, Routes, Shapes, Stops,
                                        StopTimes, Transfers, TransitFeed,
                                        Trips)

# args
# period to find
# start date, end date default end date is today's date
# what format to write to tablesalt feed-archive, zips, etc


BASE_DIR = ARCHIVE_DIR
BASE_DIR.mkdir(parents=True, exist_ok=True)

def get_write(date):
    datestring = date.strftime('%Y%m%d')

    url = fr'https://transitfeeds.com/p/rejseplanen/705/{datestring}/download'
    request_object = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urlopen(request_object)
        data = load_gtfs_response_zip(resp)
    except HTTPError:
        return

    agency = Agency.from_dataframe(data['agency.txt'])
    stops = Stops.from_dataframe(data['stops.txt'])
    routes = Routes.from_dataframe(data['routes.txt'])
    trips = Trips.from_dataframe(data['trips.txt'])
    stop_times = StopTimes.from_dataframe(data['stop_times.txt'])
    calendar = Calendar.from_dataframe(data['calendar.txt'])
    calendar_dates = CalendarDates.from_dataframe(data['calendar_dates.txt'])

    transfers = Transfers.from_dataframe(data['transfers.txt'])
    shapes = Shapes.from_dataframe(data['shapes.txt'])

    tfeed = TransitFeed(
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
    tfeed.to_archive()

if __name__ == "__main__":


    year = int(sys.argv[1])
    month = 1
    day = 1

    start = datetime(year=year, month=month, day=day)
    dates = pd.date_range(start=start, freq='D', periods=365)

    with Pool(4) as pool:
        pool.map(get_write, dates)
