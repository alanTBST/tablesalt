    """
    This module is to create linestrings for each rail segment 
    between any two stations

    """

from pathlib import Path
import pkg_resources
import geopandas as gpd

def _load_railways_shapefile():

    path = Path(r'resources/networktopodk/DKrail//denmarkl-railways-shape/railways.shp')
    
    fp = pkg_resources.resource_filename(
        'tablesalt', str(path)
    )

    df = gpd.read_file(fp)
    return df
