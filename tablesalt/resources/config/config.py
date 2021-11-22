import configparser
import pkg_resources
from pathlib import Path



def load_config() -> configparser.ConfigParser:

    config = configparser.ConfigParser()
    ini_file = pkg_resources.resource_filename(
        'tablesalt', r'resources/config/config.ini'
    )
    config.read(ini_file, encoding='utf8')

    return config


def load_revenue_config() -> configparser.ConfigParser:

    config = configparser.ConfigParser()
    ini_file = pkg_resources.resource_filename(
        'tablesalt', r'resources/config/config_revenue.ini'
    )
    config.read(ini_file, encoding='utf8')


    return config

