"""
This module is to create linestrings for each rail segment 
between any two stations. The LineStrings created can be used
to create dedicated plots for each segment between stations 

"""

import os
from itertools import chain, repeat, combinations
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import geopandas as gpd
import pkg_resources
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from tablesalt.topology.stopnetwork import StopsList, RailNetwork
from tablesalt.topology.pathfinder import to_legs

class LineDict(TypedDict):

    name: Optional[str]
    type: str
    geometry: LineString
    points: List[Point]


def _load_railwaysegments_shapefile() -> gpd.GeoDataFrame:
    
    fp = pkg_resources.resource_filename(
        'tablesalt', os.path.join(
            'resources',
            'networktopodk',
            'DKrail', 
            'railways', 
            'railsegments.shp'
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

        self.stopslist = StopsList.from_json(stopfilepath).rail_stops()
        self.stopsdict = self.stopslist.stops_dict
        self.network = RailNetwork.from_json(railinefailepath)

        self._shapes_frame = _load_railwaysegments_shapefile()
        
        self._points_dict = _convert_linestrings_to_points(self._shapes_frame)
        self._stops_frame = self.stopslist.to_geodataframe()
        
        self._nearest = self._ckdnearest()
    
    def create_linestring(
        self, 
        start_stop_id: int, 
        end_stop_id: int
        ) -> LineString:
        """Create a LineString object from a given start and end uic

        :param start_stop_id: uic number of the start of the linestring
        :type start_stop_id: int
        :param end_stop_id: uic number of the end of the linestring
        :type end_stop_id: int
        :return: a LineString between the given stop ids
        :rtype: LineString
        """
       
        start_stop = self.stopsdict.get(start_stop_id)
        start_id = start_stop.stop_id
        
        start_point = start_stop.as_point()
        end_point = self.stopsdict.get(end_stop_id).as_point()

        linestring_id = int(self._nearest.query("stop_id==@start_id")['osm_id'])
        # problem if start and end aren't on same linestring

        shape_points = self._points_dict[linestring_id]['points']

        start_distances = [start_point.distance(pt) for pt in shape_points]
        end_distances = [end_point.distance(pt) for pt in shape_points]

        points = shape_points[np.argmin(start_distances): np.argmin(end_distances)]

        return LineString(points)
    

    
    def _ckdnearest(self) -> gpd.GeoDataFrame:
        """Find the closest linestring to each Stop 

        :return: a geodataframe of the stop points and osm_id of the closest linestring
        :rtype: gpd.GeoDataFrame
        """

        # https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
        A = np.concatenate(
            [np.array(geom.coords) for geom in self._stops_frame.geometry.to_list()]
            )
        B = [np.array(geom.coords) for geom in self._shapes_frame.geometry.to_list()]
        B_ix = tuple(chain.from_iterable(
            [repeat(i, x) for i, x in enumerate(list(map(len, B)))])
            )
        B = np.concatenate(B)
        ckd_tree = cKDTree(B)
        dist, idx = ckd_tree.query(A, k=1)
        idx = itemgetter(*idx)(B_ix)
        gdf = pd.concat(
            [self._stops_frame, self._shapes_frame.loc[idx, ['osm_id']].reset_index(drop=True),
            pd.Series(dist, name='dist')], axis=1
            )
        
        return gdf
