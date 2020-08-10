# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:21:26 2020

@author: alkj
"""


from setuptools import setup, find_packages

setup(name='tablesalt',
      version='0.1',
      description='Package for Rejsekort analysis at TBST',
      author='Alan Jones Trafik, Bygge of Boligstyrelsen: Plan & Klima',
      author_email='alkj@tbst.dk, alanksjones@gmail.com',
      package_dir = {'tablesalt':'tablesalt'},
      packages = find_packages(),
      package_data={
          'resources': [
              'networktopodk/*.json', 
              'networktopodk/*.csv',
              'networktopodk/*.xlsx',
              'networktopodk/*.txt',
              'networktopodk/*.zip',
              'config/*.json',
              '*.h5',
              '*.json'
              'networktopodk/DKTariffZones/*.shp', 
              'networktopodk/DKTariffZones/*.prj', 
              'networktopodk/DKTariffZones/*.shx', 
              'networktopodk/DKTariffZones/*.sbx', 
              'networktopodk/DKTariffZones/*.sbn', 
              'networktopodk/DKTariffZones/*.cpg', 
              'networktopodk/DKTariffZones/*.dbf',
              'networktopodk/DKTariffZones/takstsjaelland/*.shp', 
              'networktopodk/DKTariffZones/takstsjaelland/*.prj', 
              'networktopodk/DKTariffZones/takstsjaelland/*.shx', 
              'networktopodk/DKTariffZones/takstsjaelland/*.sbx', 
              'networktopodk/DKTariffZones/takstsjaelland/*.sbn', 
              'networktopodk/DKTariffZones/takstsjaelland/*.cpg', 
              'networktopodk/DKTariffZones/takstsjaelland/*.dbf'
              ]
          },
      install_requires=['pandas', 'shapely', 'lmdb', 'msgpack',
                        'requests', 'pyodbc', 'h5py', 'geopandas',
                        'pyproj', 'setuptools-git', 'networkx', 
                        'xlrd'], 
      include_package_data=True,
      zip_safe=True)

