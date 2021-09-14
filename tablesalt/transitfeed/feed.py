
from urllib.request import urlopen
import logging
from tablesalt.resources.config import load_config



log = logging.getLogger(__name__)


def _load_and_unzip_gtfs() -> None:

    conf = load_config()
    gtfs_url = conf['rejseplanen']['gtfs_url']


    return