"""
TBST Trafik, Bygge, og Bolig -styrelsen
Package for data analysis at TBST
Includes Rejsekort and GTFS analysis

Author: Alan Jones alkj@tbst.dk

"""

from .rejsekortcollections import _load_collection, proc_collection
mappers = proc_collection(_load_collection())
from .storereader import StoreReader
# from .raw import RawReader