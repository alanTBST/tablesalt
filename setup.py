# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:21:26 2020

@author: alkj
"""


from setuptools import setup, find_packages
import versioneer

with open('README.rst', 'r') as f:
    long_description = f.read()
    
    
setup(name='tablesalt',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Package for Transit and Rejsekort analysis at Trafikstyrelsen',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      author='Alan Jones, Trafikstyrelsen: Plan & Klima',
      author_email='alkj@tbst.dk, alkj@trafikstyrelsen.dk, alanksjones@gmail.com',
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
              'networktopodk/DKrail/denmark-railways-shape/*.shp',
              'networktopodk/DKrail/denmark-railways-shape/*.prj',
              'networktopodk/DKrail/denmark-railways-shape/*.shx',
              'networktopodk/DKrail/denmark-railways-shape/*.sbx',
              'networktopodk/DKrail/denmark-railways-shape/*.sbn',
              'networktopodk/DKrail/denmark-railways-shape/*.cpg',
              'networktopodk/DKrail/denmark-railways-shape/*.dbf',
              'networktopodk/DKTariffZones/takstsjaelland/*.shp',
              'networktopodk/DKTariffZones/takstsjaelland/*.prj',
              'networktopodk/DKTariffZones/takstsjaelland/*.shx',
              'networktopodk/DKTariffZones/takstsjaelland/*.sbx',
              'networktopodk/DKTariffZones/takstsjaelland/*.sbn',
              'networktopodk/DKTariffZones/takstsjaelland/*.cpg',
              'networktopodk/DKTariffZones/takstsjaelland/*.dbf',
              'networktopodk/gtfs/*.txt'
              ]
          },
      install_requires=['pandas', 'shapely', 'python-lmdb', 'msgpack-python',
                        'requests', 'h5py', 'geopandas',
                        'pyproj', 'setuptools-git', 'networkx',
                        'xlrd', 'openpyxl'],
      include_package_data=True,
      zip_safe=True)

