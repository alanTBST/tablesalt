import configparser
import pkg_resources
from pathlib import Path



def load_config():


    location = Path(r'resources/config/config.ini')
    config = configparser.ConfigParser()
    ini_file = pkg_resources.resource_filename(
        'tablesalt', location
    )
    config.read(ini_file)

    return config