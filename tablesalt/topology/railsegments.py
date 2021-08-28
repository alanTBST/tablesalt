"""This module is to create linestrings for each rail segment 
between any two stations

"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import geopandas as gpd
import pkg_resources
from shapely.geometry import LineString, Point
from tablesalt.topology.stopnetwork import StopsList, RailNetwork


class LineDict(TypedDict):

    name: Optional[str]
    type: str
    geometry: LineString
    points: List[Point]


def _load_railways_shapefile() -> gpd.GeoDataFrame:
    
    fp = pkg_resources.resource_filename(
        'tablesalt', os.path.join(
            'resources',
            'networktopodk',
            'DKrail', 
            'railways', 
            'railways.shp'
            )
    )

    df = gpd.read_file(fp, encoding='utf8')
    return df


def _convert_linestrings_to_points(gdf: gpd.GeoDataFrame) -> Dict[int, LineDict]:

    gdf.loc[:, 'points'] = gdf.apply(
        lambda x: [Point(y) for y in x['geometry'].coords], axis=1
        )
    gdf = gdf.set_index('osm_id')
    records_dict = gdf.T.to_dict()

    return records_dict



class RailLineStringCreator:


    def __init__(
        self, stopfilepath: Optional[str] = None, 
        railinefailepath: Optional[str] = None
        ):

        self.stopslist = StopsList.from_json().rail_stops()
        self.stopsdict = self.stopslist.stops_dict
        self.network = RailNetwork.from_json()
    
    def create_linestring(self, start_stop_id, end_stop_id):

        start_point = self.stopsdict.get(start_stop_id).as_point()
        end_point = self.stopsdict.get(end_stop_id).as_point()

        return start_point, end_point
    
    



