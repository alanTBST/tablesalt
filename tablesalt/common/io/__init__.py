"""
io contains the functions and classes used to load package data and
rejsekort delrejser data
"""

from .rejsekortcollections import _load_collection, proc_collection
mappers = proc_collection(_load_collection())
from .storereader import StoreReader
# from .raw import RawReader
