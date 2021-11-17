"""


GTFS Feed from Rejseplanen
==========================
This subpackage deals with loading and using GTFS data provided by Rejseplan.


"""

from .feed import (
    latest_transitfeed,
    archived_transitfeed,
    available_archives,
    download_latest_feed,
    transitfeed_from_zip
    )